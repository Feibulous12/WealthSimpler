from __future__ import annotations

import pandas as pd


OUTPUT_COLUMNS = [
    "ticker",
    "symbol",
    "account_type",
    "quantity",
    "book_value",
    "current_price",
    "market_value",
    "unrealized_pnl",
    "return_pct",
    "daily_return_pct",
]


def compute_position_metrics(
    positions_df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = positions_df.merge(market_df, on="ticker", how="left")

    merged["market_value"] = merged["quantity"] * merged["current_price"]
    merged["unrealized_pnl"] = merged["market_value"] - merged["book_value"]
    merged["return_pct"] = merged["unrealized_pnl"] / merged["book_value"]
    merged["daily_return_pct"] = (
        (merged["current_price"] - merged["previous_close"]) / merged["previous_close"]
    )

    dynamic_return_cols = [col for col in market_df.columns if col.startswith("return_")]
    result_cols = OUTPUT_COLUMNS + [c for c in dynamic_return_cols if c not in OUTPUT_COLUMNS]
    return merged[result_cols]
