from __future__ import annotations

from collections import defaultdict

import pandas as pd


INDEX_BUCKET_MAP = {
    "VFV.TO": "S&P_500",
    "VSP.TO": "S&P_500",
    "ZSP.TO": "S&P_500",
    "QQC.TO": "NASDAQ_100",
    "HXQ.TO": "NASDAQ_100",
    "XQQ.TO": "NASDAQ_100",
    "TECH.TO": "NASDAQ_100",
    "VUN.TO": "US_TOTAL_MARKET",
    "XUU.TO": "US_TOTAL_MARKET",
    "ZAG.TO": "CAD_BONDS",
    "XBB.TO": "CAD_BONDS",
    "ZST-L.TO": "CASH_DEFENSIVE",
    "CASH.TO": "CASH_DEFENSIVE",
    "ZMMK.TO": "CASH_DEFENSIVE",
    "VDY.TO": "CANADA_DIVIDEND",
    "ZDY.TO": "US_DIVIDEND",
    "ZEB.TO": "CANADA_FINANCIALS",
    "XEG.TO": "CANADA_ENERGY",
    "XEQT.TO": "GLOBAL_EQUITY",
    "VEQT.TO": "GLOBAL_EQUITY",
    "ZEA.TO": "INTL_EQUITY",
    "ARTI.TO": "NASDAQ_100",
    "CIAI.TO": "NASDAQ_100",
    "FINN.NE": "NASDAQ_100",
    "HDIV.TO": "CANADA_DIVIDEND",
    "ZLB.TO": "CANADA_DIVIDEND",
}

BUCKET_CATEGORY_MAP = {
    "S&P_500": "Core Index",
    "NASDAQ_100": "Tech Growth",
    "US_TOTAL_MARKET": "Core Index",
    "CAD_BONDS": "Defensive/Cash",
    "CASH_DEFENSIVE": "Defensive/Cash",
    "CANADA_DIVIDEND": "Value/Dividend",
    "US_DIVIDEND": "Value/Dividend",
    "CANADA_FINANCIALS": "Value/Dividend",
    "CANADA_ENERGY": "Value/Dividend",
    "GLOBAL_EQUITY": "Core Index",
    "INTL_EQUITY": "Core Index",
    "INDIVIDUAL_STOCK": "Individual Stock",
    "UNMAPPED": "Individual Stock",
}


def build_index_buckets(positions_df_with_ticker: pd.DataFrame) -> pd.DataFrame:
    df = positions_df_with_ticker.copy()
    df["index_bucket"] = df["ticker"].map(INDEX_BUCKET_MAP).fillna("INDIVIDUAL_STOCK")
    df["bucket_provider"] = df["ticker"].map(lambda t: "STATIC_MAP" if t in INDEX_BUCKET_MAP else "UNMAPPED")
    df["confidence"] = df["ticker"].map(lambda t: 1.0 if t in INDEX_BUCKET_MAP else 0.0)
    return df[["ticker", "symbol", "index_bucket", "bucket_provider", "confidence"]].drop_duplicates()


def aggregate_by_bucket(
    positions_metrics_df: pd.DataFrame,
    bucket_map_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = positions_metrics_df.copy()
    if "index_bucket" not in merged.columns:
        merged = merged.merge(
            bucket_map_df[["ticker", "index_bucket"]].drop_duplicates(),
            on="ticker",
            how="left",
        )
    merged["index_bucket"] = merged["index_bucket"].fillna("INDIVIDUAL_STOCK")
    merged["today_pnl_amount"] = merged["market_value"] * merged["daily_return_pct"]

    grouped = (
        merged.groupby("index_bucket", as_index=False)
        .agg(
            total_market_value=("market_value", "sum"),
            daily_pnl=("today_pnl_amount", "sum"),
            member_count=("ticker", "nunique"),
        )
        .sort_values("total_market_value", ascending=False)
    )

    total_mv = grouped["total_market_value"].sum()
    grouped["weight_pct"] = grouped["total_market_value"] / total_mv if total_mv else 0.0
    grouped["overlap_flag"] = grouped.apply(
        lambda r: "High"
        if r["member_count"] >= 2 and r["weight_pct"] >= 0.15
        else ("Medium" if r["member_count"] >= 2 else "None"),
        axis=1,
    )
    return grouped[["index_bucket", "total_market_value", "weight_pct", "daily_pnl", "overlap_flag", "member_count"]]


def detect_overlap(
    bucket_agg_df: pd.DataFrame,
    positions_metrics_df: pd.DataFrame,
    bucket_map_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = positions_metrics_df.copy()
    if "index_bucket" not in merged.columns:
        merged = merged.merge(
            bucket_map_df[["ticker", "index_bucket"]].drop_duplicates(),
            on="ticker",
            how="left",
        )
    merged["index_bucket"] = merged["index_bucket"].fillna("INDIVIDUAL_STOCK")

    bucket_tickers: dict[str, set[str]] = defaultdict(set)
    bucket_accounts: dict[str, set[str]] = defaultdict(set)
    for row in merged[["index_bucket", "ticker", "account_type"]].dropna().itertuples(index=False):
        bucket_tickers[row.index_bucket].add(row.ticker)
        bucket_accounts[row.index_bucket].add(row.account_type)

    rows: list[dict] = []
    for row in bucket_agg_df.itertuples(index=False):
        rows.append(
            {
                "index_bucket": row.index_bucket,
                "member_tickers": sorted(bucket_tickers.get(row.index_bucket, set())),
                "member_accounts": sorted(bucket_accounts.get(row.index_bucket, set())),
                "total_weight_pct": float(row.weight_pct),
                "overlap_flag": row.overlap_flag,
            }
        )

    return pd.DataFrame(rows)


def bucket_category(index_bucket: str) -> str:
    return BUCKET_CATEGORY_MAP.get(index_bucket, "Individual Stock")
