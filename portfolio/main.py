from __future__ import annotations

import json
from pathlib import Path

from portfolio.data_pipeline.aggregate import aggregate_by_account
from portfolio.data_pipeline.compute import compute_position_metrics
from portfolio.data_pipeline.market_fetch import fetch_market_data
from portfolio.data_pipeline.parser import parse_portfolio_json
from portfolio.data_pipeline.ticker_builder import build_yahoo_tickers


def load_json(path: str = "raw.json") -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    raw_json = load_json()

    positions_df = parse_portfolio_json(raw_json)
    positions_df = build_yahoo_tickers(positions_df)

    tickers = positions_df["ticker"].unique().tolist()

    market_df = fetch_market_data(tickers)

    positions_enriched = compute_position_metrics(
        positions_df,
        market_df,
    )

    account_summary = aggregate_by_account(positions_enriched)

    print(positions_enriched.head())
    print(account_summary)


if __name__ == "__main__":
    main()
