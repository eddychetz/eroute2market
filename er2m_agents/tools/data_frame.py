import re
import os
from datetime import datetime

# --- channel ID constants ---
PRIMARY_CHANNEL_ID   = 19775654
SECONDARY_CHANNEL_ID = 197715655

# brand → tertiary channel id rules
# special case: caltex/astron/freshstop → TWO rows (handled in function logic)
TERTIARY_MAP_SINGLE = {
    "engen": 192327178,
    "sasol": 192327180,     # Sasol Delight
    "bp":    192327176,
    "shell": 1867298,
    "total": 192327181
}
# special dual-channel case:
DUAL_BRANDS = {"caltex", "astron", "freshstop"}
DUAL_TERTIARY_IDS = (244742793, 297002250)   # Caltex Starmart, Astron Freshstop


def _brand_token_from_promo_name(promo_name: str) -> str:
    """
    Extract brand token from text BEFORE '-' and normalize to lowercase trimmed.
    Example: 'BP - Summer Promo' -> 'bp'
    """
    token = str(promo_name).split('-', 1)[0].strip().lower()
    # collapse double spaces
    while "  " in token:
        token = token.replace("  ", " ")
    return token


def get_promo_tables(
    raw: pd.DataFrame,
    manufacturer_id: int | None = None,
    buyer_group_id: int | None = None,
    user: str = "automation",
    date_added: str | None = None
):
    """
    Build three normalized tables from the raw file:
    1) df_promo: one row per promo_name with start/end dates and IDs
    2) df_promo_channel: rows per promo_name with primary/secondary and brand-aware tertiary
    3) df_promo_product: product list per promo_name (Variant as product_description)

    Parameters
    ----------
    raw : pd.DataFrame
        Expected columns: ['Banner Group', 'Brand', 'Variant', 'Promo Start', 'Promo Finish']
    manufacturer_id : int|None
    buyer_group_id  : int|None
    user            : str
    date_added      : str|None  (YYYY-MM-DD); defaults to today if None

    Returns
    -------
    dict with:
    {
        "df_promo": pd.DataFrame[cols: promo_name, start_date, end_date, manufacturer_id, buyer_group_id, user, date_added],
        "df_promo_channel": pd.DataFrame[cols: promo_name, primary_channel_id, secondary_channel_id, tertiary_channel_id],
        "df_promo_product": pd.DataFrame[cols: promo_name, product_description, start_date, end_date]
    }
    """
    df = raw.copy()

    # --- Basic selection & standard fields ---
    required = ['Banner Group', 'Brand', 'Variant', 'Promo Start', 'Promo Finish']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input: {missing}")

    df['promo_name'] = (
        df['Banner Group'].astype(str).str.strip()
        + " - " +
        df['Brand'].astype(str).str.strip()
    )
    df['product_description'] = df['Variant'].astype(str).str.strip()
    df['start_date'] = pd.to_datetime(df['Promo Start'], dayfirst=True, errors='coerce')
    df['end_date']   = pd.to_datetime(df['Promo Finish'], dayfirst=True, errors='coerce')

    # Keep a slim table for downstream
    df1 = df[['promo_name', 'product_description', 'start_date', 'end_date']].copy()

    # --- df_promo: one row per promo_name ---
    # Choose earliest start and latest end per promo_name as the promo window
    promo_agg = df1.groupby('promo_name', as_index=False).agg(
        start_date=('start_date', 'min'),
        end_date=('end_date', 'max')
    )

    # annotate IDs & audit fields (fill if not provided)
    if date_added is None:
        date_added = pd.Timestamp.today().strftime("%Y-%m-%d")

    promo_agg['manufacturer_id'] = manufacturer_id
    promo_agg['buyer_group_id']  = buyer_group_id
    promo_agg['user']            = user
    promo_agg['date_added']      = date_added

    df_promo = promo_agg[['promo_name', 'start_date', 'end_date', 'manufacturer_id', 'buyer_group_id', 'user', 'date_added']].copy()

    # --- df_promo_channel: brand-aware tertiary per promo_name ---
    channel_rows = []
    for _, row in df_promo.iterrows():
        p_name = row['promo_name']
        brand = _brand_token_from_promo_name(p_name)

        if any(b in brand for b in DUAL_BRANDS):
            # special case: insert two rows
            for ter_id in DUAL_TERTIARY_IDS:
                channel_rows.append({
                    'promo_name': p_name,
                    'primary_channel_id':   PRIMARY_CHANNEL_ID,
                    'secondary_channel_id': SECONDARY_CHANNEL_ID,
                    'tertiary_channel_id':  ter_id
                })
        else:
            # resolve single
            # accept loose matches: exact map key within brand token
            ter_id = None
            for k, v in TERTIARY_MAP_SINGLE.items():
                if k in brand:
                    ter_id = v
                    break
            if ter_id is None:
                # you can either skip or raise
                # skip silently:
                #   continue
                # or be explicit for data quality:
                raise ValueError(f"Unrecognized brand token for promo_name='{p_name}' (derived brand='{brand}')")

            channel_rows.append({
                'promo_name': p_name,
                'primary_channel_id':   PRIMARY_CHANNEL_ID,
                'secondary_channel_id': SECONDARY_CHANNEL_ID,
                'tertiary_channel_id':  ter_id
            })

    df_promo_channel = pd.DataFrame(channel_rows)

    # --- df_promo_product: all item rows per promo_name ---
    # Keep start/end for lineage; these can help when you bulk-load
    df_promo_product = df1[['promo_name', 'product_description', 'start_date', 'end_date']].drop_duplicates().reset_index(drop=True)

    return {
        "df_promo": df_promo.reset_index(drop=True),
        "df_promo_channel": df_promo_channel.reset_index(drop=True),
        "df_promo_product": df_promo_product.reset_index(drop=True)
    }

# Split a dataframe into a dict of {promo_name: subframe}
def split_tables_by_promo_name(df: pd.DataFrame, promo_col: str = 'promo_name') -> dict[str, pd.DataFrame]:
    """
    Utility to split a dataframe into a dict of {promo_name: subframe}.
    Helpful when you need a 'table per promo' artifact (e.g., saving per file).
    """
    result = {}
    for p_name, sub in df.groupby(promo_col):
        result[p_name] = sub.reset_index(drop=True)
    return result

# Save each promo's product list to CSV
def save_products_per_promo(df_promo_product: pd.DataFrame, out_dir: str = "./data"):
    """
    Optionally save each promo's product list to CSV for auditing or handoff:
    ./data/<safe_promo_name>.csv
    """
    os.makedirs(out_dir, exist_ok=True)

    for p_name, sub in df_promo_product.groupby('promo_name'):
        # safe filename
        safe = re.sub(r'[^A-Za-z0-9._-]+', '_', p_name)
        sub.to_csv(os.path.join(out_dir, f"{safe}_products.csv"), index=False)