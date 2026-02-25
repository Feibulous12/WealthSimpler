from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd

from portfolio import dashboard_report


class DashboardReportTests(unittest.TestCase):
    def test_generate_findings_list_handles_empty_and_zero_denominators(self) -> None:
        positions = pd.DataFrame(columns=["account_type", "unrealized_pnl", "book_value", "index_bucket", "market_value"])
        overview = {
            "total_unrealized_pnl": 0.0,
            "total_book_value": 0.0,
            "total_market_value": 0.0,
        }
        findings = dashboard_report.generate_findings_list(positions, overview)
        self.assertEqual(findings, [])

    def test_generate_concerns_list_handles_empty_positions(self) -> None:
        positions = pd.DataFrame(columns=["symbol", "index_bucket", "market_value"])
        overlap = pd.DataFrame(columns=["overlap_flag", "member_tickers", "member_count", "index_bucket", "total_weight_pct"])
        concerns = dashboard_report.generate_concerns_list(positions, overlap)
        self.assertEqual(concerns, [])

    def test_render_full_report_handles_empty_data(self) -> None:
        original_st = dashboard_report.st
        dashboard_report.st = SimpleNamespace(
            markdown=lambda *args, **kwargs: None,
            dataframe=lambda *args, **kwargs: None,
            info=lambda *args, **kwargs: None,
        )
        try:
            views = {
                "overview": {
                    "total_market_value": 0.0,
                    "total_unrealized_pnl": 0.0,
                    "total_book_value": 0.0,
                    "hhi": 0.0,
                    "top5_share": 0.0,
                    "estimated_beta": 0.0,
                    "volatility_band": "Low",
                    "growth_share": 0.0,
                    "defensive_share": 0.0,
                    "price_coverage": 0.0,
                    "selected_window_pnl": 0.0,
                    "period_label": "1D",
                },
                "top_holdings": pd.DataFrame(
                    columns=[
                        "symbol",
                        "index_bucket",
                        "market_value",
                        "weight",
                        "unrealized_pnl",
                        "return_pct",
                        "action",
                        "thesis_status",
                        "rationale",
                    ]
                ),
                "bucket_alloc": pd.DataFrame(columns=["index_bucket", "current_weight", "target_weight", "drift"]),
                "rebalancing": pd.DataFrame(columns=["priority", "action", "index_bucket", "drift", "trade_value_cad"]),
                "action_items": pd.DataFrame(columns=["priority", "item", "detail"]),
                "account_summary": pd.DataFrame(),
            }
            positions = pd.DataFrame(columns=["account_type", "index_bucket", "market_value", "symbol"])
            overlap = pd.DataFrame(columns=["overlap_flag", "member_tickers", "index_bucket", "total_weight_pct"])
            dashboard_report.render_full_report(views, positions, overlap)
        finally:
            dashboard_report.st = original_st


if __name__ == "__main__":
    unittest.main()
