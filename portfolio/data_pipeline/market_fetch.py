from __future__ import annotations

import pandas as pd
import yfinance as yf


OUTPUT_COLUMNS = [
    "ticker",
    "current_price",
    "previous_close",
    "return_1d",
    "return_1w",
    "return_1m",
    "return_1q",
    "return_ytd",
]
LOOKBACK_MAP = {
    "return_1d": 1,
    "return_1w": 5,
    "return_1m": 21,
    "return_1q": 63,
}


def _extract_close_series(batch_df: pd.DataFrame, ticker: str, multi_ticker: bool) -> pd.Series:
    if batch_df.empty:
        return pd.Series(dtype=float)

    if multi_ticker:
        if ticker not in batch_df.columns.get_level_values(0):
            return pd.Series(dtype=float)
        if "Close" not in batch_df[ticker].columns:
            return pd.Series(dtype=float)
        return batch_df[ticker]["Close"].dropna()

    if "Close" not in batch_df.columns:
        return pd.Series(dtype=float)
    return batch_df["Close"].dropna()


def _compute_return(closes: pd.Series, lookback_sessions: int) -> float | None:
    if len(closes) <= lookback_sessions:
        return None
    current = closes.iloc[-1]
    base = closes.iloc[-(lookback_sessions + 1)]
    if pd.isna(current) or pd.isna(base) or base == 0:
        return None
    return float((current - base) / base)


def _compute_ytd_return(closes: pd.Series) -> float | None:
    if closes.empty:
        return None
    latest_dt = closes.index[-1]
    if not hasattr(latest_dt, "year"):
        return None
    year_mask = closes.index.year == latest_dt.year
    year_closes = closes[year_mask]
    if year_closes.empty:
        return None
    base = year_closes.iloc[0]
    current = closes.iloc[-1]
    if pd.isna(current) or pd.isna(base) or base == 0:
        return None
    return float((current - base) / base)


def fetch_market_data(tickers: list[str]) -> pd.DataFrame:
    unique_tickers = list(dict.fromkeys(tickers))
    if not unique_tickers:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    raw = yf.download(
        unique_tickers,
        period="2y",
        interval="1d",
        group_by="ticker",
        progress=False,
        auto_adjust=False,
    )

    multi_ticker = isinstance(raw.columns, pd.MultiIndex)
    rows: list[dict] = []

    for ticker in unique_tickers:
        closes = _extract_close_series(raw, ticker, multi_ticker)

        current_price = None
        previous_close = None

        if len(closes) >= 1:
            current_price = float(closes.iloc[-1])
        if len(closes) >= 2:
            previous_close = float(closes.iloc[-2])

        return_values = {
            key: _compute_return(closes, lookback_sessions)
            for key, lookback_sessions in LOOKBACK_MAP.items()
        }
        return_values["return_ytd"] = _compute_ytd_return(closes)

        rows.append(
            {
                "ticker": ticker,
                "current_price": current_price,
                "previous_close": previous_close,
                **return_values,
            }
        )

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
