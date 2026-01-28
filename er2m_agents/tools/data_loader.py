from langchain.tools import tool

import pandas as pd
import os
from pathlib import Path

from typing_extensions import Tuple, List, Dict, Optional

ALLOW_UNSAFE_PICKLE_ENV_VAR = "ALLOW_UNSAFE_PICKLE"
DEFAULT_MAX_MB = 20  # cap file size we attempt to load
DEFAULT_MAX_ROWS = 5000  # cap rows read per file to avoid OOM
DEFAULT_MAX_ENTRIES = 1000  # cap directory recursion output
DEFAULT_MAX_DEPTH = 5  # cap directory recursion depth

# =========================
# Planner readers (CSV/XLSX)
# =========================

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
def _read_csv_with_fallbacks(path: str) -> pd.DataFrame:
    for enc in (None, "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"Failed to read CSV: {path}")

@tool(response_format="content_and_artifact")
def read_planner(path: str, sheet_name: str = "PFM Promo Planner") -> pd.DataFrame:
    """
    Read the planner:
      - detect header row (Excel),
      - apply headers,
      - **forward-fill merged cells in `Banner Group` and `Brand`**,
      - ensure `Variant` exists,
      - keep only meaningful rows.
    """
    if path.lower().endswith(".csv"):
        raw = _read_csv_with_fallbacks(path)
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
    hdr = detect_header_row(df0)
    headers = df0.iloc[hdr].astype(str).str.strip().tolist()
    raw = df0.iloc[hdr + 1 :].copy()
    raw.columns = headers
    raw = raw.reset_index(drop=True)


def get_promo_mechanics():

    pass

def get_promo_products():

    pass