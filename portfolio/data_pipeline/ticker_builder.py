from __future__ import annotations

import pandas as pd


MIC_SUFFIX_MAP = {
    "XTSE": ".TO",
    "NEOE": ".NE",
}


def build_yahoo_tickers(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    unsupported = sorted(set(result["mic"].dropna()) - set(MIC_SUFFIX_MAP.keys()))
    if unsupported:
        raise ValueError(f"Unsupported exchange MIC(s): {', '.join(unsupported)}")

    if result["mic"].isna().any():
        raise ValueError("Unsupported exchange MIC(s): <missing>")

    normalized_symbol = result["symbol"].astype(str).str.replace(".", "-", regex=False)
    result["ticker"] = normalized_symbol + result["mic"].map(MIC_SUFFIX_MAP)
    return result
