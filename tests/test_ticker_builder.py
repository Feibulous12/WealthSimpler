from __future__ import annotations

import unittest

import pandas as pd

from portfolio.data_pipeline.ticker_builder import build_yahoo_tickers


class TickerBuilderTests(unittest.TestCase):
    def test_build_yahoo_tickers_builds_supported_symbols(self) -> None:
        df = pd.DataFrame(
            {
                "symbol": ["VFV", "BRK.B"],
                "mic": ["XTSE", "NEOE"],
            }
        )
        out = build_yahoo_tickers(df)
        self.assertListEqual(out["ticker"].tolist(), ["VFV.TO", "BRK-B.NE"])

    def test_build_yahoo_tickers_raises_for_unsupported_mic(self) -> None:
        df = pd.DataFrame({"symbol": ["SPY"], "mic": ["XNYS"]})
        with self.assertRaisesRegex(ValueError, "Unsupported exchange MIC"):
            build_yahoo_tickers(df)

    def test_build_yahoo_tickers_raises_for_missing_mic(self) -> None:
        df = pd.DataFrame({"symbol": ["VFV"], "mic": [pd.NA]})
        with self.assertRaisesRegex(ValueError, "Unsupported exchange MIC"):
            build_yahoo_tickers(df)


if __name__ == "__main__":
    unittest.main()
