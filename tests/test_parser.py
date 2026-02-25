from __future__ import annotations

import unittest

from portfolio.data_pipeline.parser import REQUIRED_COLUMNS, parse_portfolio_json


class ParserTests(unittest.TestCase):
    def test_parse_portfolio_json_returns_empty_with_required_columns(self) -> None:
        df = parse_portfolio_json({})
        self.assertListEqual(list(df.columns), REQUIRED_COLUMNS)
        self.assertTrue(df.empty)

    def test_parse_portfolio_json_extracts_expected_fields(self) -> None:
        raw = {
            "data": {
                "identity": {
                    "financials": {
                        "current": {
                            "positions": {
                                "edges": [
                                    {
                                        "node": {
                                            "security": {"stock": {"symbol": "VFV", "primaryMic": "XTSE"}},
                                            "accounts": [{"id": "tfsa-123"}],
                                            "quantity": "10",
                                            "bookValue": {"amount": "100.5"},
                                            "averagePrice": {"amount": "10.05"},
                                        }
                                    }
                                ]
                            }
                        }
                    }
                }
            }
        }
        df = parse_portfolio_json(raw)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["symbol"], "VFV")
        self.assertEqual(df.iloc[0]["mic"], "XTSE")
        self.assertEqual(df.iloc[0]["account_type"], "TFSA")
        self.assertAlmostEqual(df.iloc[0]["quantity"], 10.0)
        self.assertAlmostEqual(df.iloc[0]["book_value"], 100.5)
        self.assertAlmostEqual(df.iloc[0]["average_price"], 10.05)


if __name__ == "__main__":
    unittest.main()
