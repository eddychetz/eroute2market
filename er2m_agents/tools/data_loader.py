# from langchain.tools import tool

import pandas as pd
import re
import os
from datetime import datetime
from typing_extensions import Tuple, List, Optional
import prefect
from prefect import task, flow
ALLOW_UNSAFE_PICKLE_ENV_VAR = "ALLOW_UNSAFE_PICKLE"
DEFAULT_MAX_MB = 20  # cap file size we attempt to load
DEFAULT_MAX_ROWS = 5000  # cap rows read per file to avoid OOM
DEFAULT_MAX_ENTRIES = 1000  # cap directory recursion output
DEFAULT_MAX_DEPTH = 5  # cap directory recursion depth

POSSIBLE_FMTS = [
        "%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y", "%d-%m-%y",
        "%d %b %Y", "%d %b '%y", "%d %B %Y", "%d %B '%y"
    ]

DASH_PATTERN = re.compile(r"\s*[\-–—]\s*")  # hyphen, en dash, em dash
DATE_TOKEN = re.compile(
    r"(\d{1,2}[./]\d{1,2}[./]\d{2,4}|\d{1,2}\s*[A-Za-z]{3,9}\s*'\d{2,4}|\d{1,2}\s*[A-Za-z]{3,9}\s*\d{2,4})"
                        )
# =========================
# Planner readers (CSV/XLSX)
# =========================
@task(name="get_latest_file")
def get_latest_file(
    folder_path: str,
    name_contains: Optional[str] = None,
    allowed_exts: Tuple[str, ...] = (".xlsx", ".csv"),
) -> str:
    """Return full path of the newest matching file in a SharePoint-synced folder."""
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Folder not found: {folder_path}")

    candidates: List[Tuple[str, float]] = []
    for fname in os.listdir(folder_path):
        lower = fname.lower()
        if not lower.endswith(allowed_exts):
            continue
        if name_contains and name_contains.lower() not in lower:
            continue
        fpath = os.path.join(folder_path, fname)
        if os.path.isfile(fpath):
            candidates.append((fpath, os.path.getmtime(fpath)))

    if not candidates:
        raise FileNotFoundError(
            f"No matching files in {folder_path} "
            f"(filters: name_contains={name_contains}, exts={allowed_exts})"
        )

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]

# Date Parser for output standardization

# Define a function to parse a single date
def _parse_single_date(token):
    token = str(token).strip().replace("\u2009", " ")
    for fmt in POSSIBLE_FMTS:
        try:
            dt = datetime.strptime(token, fmt)
            if dt.year < 100:  # normalize 2-digit years to 2000s
                dt = dt.replace(year=2000 + dt.year)
            return pd.Timestamp(dt)
        except Exception:
            pass
    return pd.to_datetime(token, dayfirst=True, errors="coerce")

# Define a function to split a range of date strings
def _parse_split_range(cell):
    """
    Return (start, finish) parsed from a range string or single date.
    """

    if pd.isna(cell):
        return (pd.NaT, pd.NaT)
    s = str(cell).strip()
    if not s:
        return (pd.NaT, pd.NaT)

    # 1) Try splitting by any dash
    parts = DASH_PATTERN.split(s)
    if len(parts) == 2:
        return (_parse_single_date(parts[0]), _parse_single_date(parts[1]))

    # 2) Otherwise, take the first 1–2 recognizable date tokens
    tokens = DATE_TOKEN.findall(s)
    if len(tokens) >= 2:
        return (_parse_single_date(tokens[0]), _parse_single_date(tokens[1]))
    if len(tokens) == 1:
        return (_parse_single_date(tokens[0]), pd.NaT)
    return (pd.NaT, pd.NaT)

# Header detection
def _detect_header_row(df0: pd.DataFrame, must_have=("Banner Group","Brand","Variant")) -> int:
    """Locate the header row (tables often have logos/notes above)."""
    scan_rows = min(100, len(df0))
    target = [m.lower() for m in must_have]
    for i in range(scan_rows):
        row_vals = df0.iloc[i].astype(str).str.strip().str.lower().tolist()
        if all(any(m == v for v in row_vals) for m in target):
            return i
    return 0  # fallback

# Read CSVs with fallback encodings
def _load_file_with_fallbacks(path: str) -> pd.DataFrame:
    for enc in (None, "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"Failed to read CSV: {path}")

# Read planner
@task(name="Get Promo Planner")
def get_promo_planner(path: str, sheet_name: str = "PFM Promo Planner") -> pd.DataFrame:
    """
    Read the planner:
    - detect header row (Excel),
    - apply headers,
    - **forward-fill merged cells in `Banner Group` and `Brand`**,
    - ensure `Variant` exists,
    - keep only meaningful rows.
    """
    if path.lower().endswith(".csv"):
        raw = _load_file_with_fallbacks(path)
        # ---- Forward-fill fix for CSVs as well (in case of blank repeats) ----
        for col in ("Banner Group", "Brand"):
            if col in raw.columns:
                raw[col] = raw[col].ffill()
        if "Variant" not in raw.columns:
            raw["Variant"] = pd.NA
        keep = ~raw[["Brand","Variant"]].isna().all(axis=1)
        return raw.loc[keep].reset_index(drop=True)

    # Excel workbook
    df0 = pd.read_excel(path, sheet_name=sheet_name, header=None, engine="openpyxl")
    hdr = _detect_header_row(df0)
    _headers = df0.iloc[hdr].astype(str).str.strip().tolist()
    raw = df0.iloc[hdr + 1 :].copy()
    raw.columns = _headers
    raw = raw.reset_index(drop=True)

    # ---- Forward-fill fix for merged cells (your requested change) ----
    for col in ("Banner Group", "Brand"):
        if col in raw.columns:
            raw[col] = raw[col].ffill()

    # Make sure Variant exists (some tabs omit it)
    if "Variant" not in raw.columns:
        raw["Variant"] = pd.NA

    # Keep rows that have at least Brand or Variant
    keep = ~raw[["Brand","Variant"]].isna().all(axis=1)
    return raw.loc[keep].reset_index(drop=True)

# =========================
# Period extraction
# =========================
def _coalesce(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    for c in candidates:
        if c in df.columns:
            return df[c]
    return None

def _extract_periods(raw: pd.DataFrame):
    """Support combined range (one column) or Start/Finish (two columns)."""
    buy_combined  = ["Buy In Period", "Buy-in Period", "Buyin Period"]
    buy_start     = ["Buy In Start", "Buy-in Start", "Start (Buy In)", "Buy In - Start", "Buy In: Start", "Start - Buy In", "Start"]
    buy_finish    = ["Buy In Finish", "Buy-in Finish", "Finish (Buy In)","Buy In - Finish", "Buy In: Finish", "Finish - Buy In", "Finish"]

    promo_combined = ["Promotional Period", "Promo Period", "Promotion Period"]
    promo_start    = ["Promotional Start", "Promo Start", "Promotion Start","Start (Promo)", "Start - Promo", "Start (Promotional Period)", "Start"]
    promo_finish   = ["Promotional Finish", "Promo Finish", "Promotion Finish","Finish (Promo)", "Finish - Promo", "Finish (Promotional Period)", "Finish"]

    def _extract(df, combined_cols, start_cols, finish_cols):
        s_col, f_col = _coalesce(df, start_cols), _coalesce(df, finish_cols)
        if s_col is not None or f_col is not None:
            s = pd.to_datetime(s_col, errors="coerce", dayfirst=True) if s_col is not None else pd.Series(pd.NaT, index=df.index)
            f = pd.to_datetime(f_col, errors="coerce", dayfirst=True) if f_col is not None else pd.Series(pd.NaT, index=df.index)
            return s, f

        comb = _coalesce(df, combined_cols)
        if comb is None:
            return pd.Series(pd.NaT, index=df.index), pd.Series(pd.NaT, index=df.index)

        s_vals, f_vals = zip(*comb.apply(_parse_split_range))
        return pd.Series(list(s_vals), index=df.index), pd.Series(list(f_vals), index=df.index)

    buy_s, buy_f = _extract(raw, buy_combined,  buy_start,  buy_finish)
    pro_s, pro_f = _extract(raw, promo_combined, promo_start, promo_finish)
    return buy_s, buy_f, pro_s, pro_f


# =========================
# Build final result
# =========================
@tool(response_format="content_and_artifact")
def get_raw_data(raw: pd.DataFrame) -> pd.DataFrame:
    buy_s, buy_f, pro_s, pro_f = _extract_periods(raw)

    res = pd.DataFrame({
        "Banner Group": raw.get("Banner Group", pd.Series([pd.NA]*len(raw))),
        "Brand":        raw.get("Brand", pd.Series([pd.NA]*len(raw))),
        "Variant":      raw.get("Variant", pd.Series([pd.NA]*len(raw))),
        "Buy In Start": buy_s,
        "Buy In Finish":buy_f,
        "Promo Start":  pro_s,
        "Promo Finish": pro_f
    })

    # Keep rows that have a Variant OR at least one date (avoid spacer rows)
    date_cols = ["Buy In Start","Buy In Finish","Promo Start","Promo Finish"]
    keep = (~res["Variant"].isna()) | (~pd.isna(res[date_cols]).all(axis=1))
    res = res.loc[keep].reset_index(drop=True)

    # Clean text fields
    for c in ["Banner Group","Brand","Variant"]:
        if c in res.columns:
            res[c] = res[c].astype(str).str.strip().replace({"nan": "", "None": ""})

    return res

# =========================
# Transformations
# =========================

@tool(response_format="content_and_artifact")
def get_promo_mechanics(df: pd.DataFrame) -> pd.DataFrame:
    # Final structure
    field_list = [
        'name',
        'start_date',
        'end_date',
        'manufacturer_id',
        'buyer_group_id',
        'user',
        'date_added'
    ]

    promo_config = pd.DataFrame(columns=field_list)

    # Base fields
    promo_config['name'] = df['Banner Group'].astype(str).str.strip() + " - " + df['Brand'].astype(str).str.strip()
    promo_config['start_date'] = pd.to_datetime(df['Promo Start'], dayfirst=True, errors='coerce')
    promo_config['end_date']   = pd.to_datetime(df['Promo Finish'], dayfirst=True, errors='coerce')
    promo_config['manufacturer_id'] = 206575077
    promo_config['user'] = 'bevco_master'
    promo_config['date_added'] = datetime.now().strftime('%Y-%m-%d')

    # -----------------------
    # Buyer Group Mapping
    # -----------------------
    mapping = {
        "Sasol": 177268,
        "TotalEnergies": 178524,
        "BP Express": 1864328,
        "Engen": 1865071,
        "Shell": 1867297,
        "Freshstop /Astron / Caltex": 172785,
        "Pick 'n Pay Express": 1864328
    }

    # Normalize Banner Group for matching
    cleaned = df['Banner Group'].astype(str).str.strip()

    buyer_ids = []
    for val in cleaned:
        assigned = None
        # Attempt direct match
        if val in mapping:
            assigned = mapping[val]
        else:
            # Attempt substring match (e.g. "Caltex (Astron) Caltex" etc.)
            for key in mapping:
                if key in val:
                    assigned = mapping[key]
                    break

        buyer_ids.append(assigned if assigned is not None else "")

    promo_config['buyer_group_id'] = buyer_ids

    return promo_config

def get_promo_variants(raw: pd.DataFrame):
    """
    Get Banner Group and Production per each promotion and return a dataframe per each banner group
    """
    df = raw[['Banner Group', 'Brand', 'Variant', 'Promo Start', 'Promo Finish']]
    df['promo_name'] = df['Banner Group'].astype(str).str.strip() + " - " + df['Brand'].astype(str).str.strip()
    df['product_description'] = df['Variant'].astype(str).str.strip()
    df['start_date'] = pd.to_datetime(df['Promo Start'], dayfirst=True, errors='coerce')
    df['end_date']   = pd.to_datetime(df['Promo Finish'], dayfirst=True, errors='coerce')
    df1 = df.drop(['Banner Group', 'Brand', 'Variant', 'Promo Start', 'Promo Finish'], axis=1, inplace=False)

    return df1


def main():
    print(f"Data loading and transformation started at {datetime.now()}")

    raw = get_raw_data()
    print(f"Raw data loaded at {datetime.now()}")

    df = get_promo_mechanics(raw)
    print(f"Promo Mechanics loaded at {datetime.now()}")