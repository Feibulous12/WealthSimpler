from __future__ import annotations

import pandas as pd


OUTPUT_COLUMNS = [
    "account_type",
    "total_market_value",
    "total_book_value",
    "total_unrealized_pnl",
    "return_pct",
]


def aggregate_by_account(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("account_type", dropna=False, as_index=False)
        .agg(
            total_market_value=("market_value", "sum"),
            total_book_value=("book_value", "sum"),
            total_unrealized_pnl=("unrealized_pnl", "sum"),
        )
    )
    grouped["return_pct"] = grouped["total_unrealized_pnl"] / grouped["total_book_value"]

    all_row = pd.DataFrame(
        {
            "account_type": ["ALL"],
            "total_market_value": [df["market_value"].sum()],
            "total_book_value": [df["book_value"].sum()],
            "total_unrealized_pnl": [df["unrealized_pnl"].sum()],
        }
    )
    all_row["return_pct"] = all_row["total_unrealized_pnl"] / all_row["total_book_value"]

    result = pd.concat([grouped, all_row], ignore_index=True)
    return result[OUTPUT_COLUMNS]
