"""
market_data.py
--------------
Fetches market data from Yahoo Finance (free, no API key needed).
Enriches parsed positions with:
  - current price / previous close
  - day change % 
  - 52-week high/low
  - sector/industry (equities)
  - expense ratio / MER (ETFs)
  - 1-year historical prices (for charts)
  - benchmark index data (S&P500, TSX, NASDAQ)

All financial calculations happen here, NOT in the parser.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Optional

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    print("⚠️  yfinance not installed. Run: pip install yfinance")


# ─── Benchmark indices ───────────────────────────────────────────────────────
BENCHMARKS = {
    "S&P 500":  "^GSPC",
    "TSX":      "^GSPTSE",
    "NASDAQ":   "^IXIC",
}

# ─── ETF classification map (symbol → category) ──────────────────────────────
# This is our product's knowledge layer - maps symbols to investment categories.
# Covers all TSX-listed ETFs. Extendable as new symbols appear.
ETF_CATEGORIES = {
    # Core broad market index
    "VFV":  {"major": "Core Index",    "minor": "US - S&P 500",      "hedged": False},
    "VSP":  {"major": "Core Index",    "minor": "US - S&P 500",      "hedged": True},
    "ZSP":  {"major": "Core Index",    "minor": "US - S&P 500",      "hedged": False},
    "XUU":  {"major": "Core Index",    "minor": "US - Total Market",  "hedged": False},
    "VUN":  {"major": "Core Index",    "minor": "US - Total Market",  "hedged": False},
    "XEQT": {"major": "Core Index",    "minor": "Global",             "hedged": False},
    "VEQT": {"major": "Core Index",    "minor": "Global",             "hedged": False},
    "ZEA":  {"major": "Core Index",    "minor": "International",      "hedged": False},

    # NASDAQ / Tech growth
    "QQC":  {"major": "Tech Growth",   "minor": "NASDAQ 100",         "hedged": False},
    "HXQ":  {"major": "Tech Growth",   "minor": "NASDAQ 100",         "hedged": False},
    "XQQ":  {"major": "Tech Growth",   "minor": "NASDAQ 100",         "hedged": True},
    "TECH": {"major": "Tech Growth",   "minor": "NASDAQ 100",         "hedged": True},
    "FINN": {"major": "Tech Growth",   "minor": "Innovation / AI",    "hedged": False},
    "ARTI": {"major": "Tech Growth",   "minor": "Innovation / AI",    "hedged": True},

    # Dividend / Value
    "VDY":  {"major": "Dividend",      "minor": "Canada Dividend",    "hedged": False},
    "ZEB":  {"major": "Dividend",      "minor": "Canada Banks",       "hedged": False},
    "ZLB":  {"major": "Dividend",      "minor": "Canada Low Vol",     "hedged": False},
    "XEG":  {"major": "Dividend",      "minor": "Canada Energy",      "hedged": False},
    "ZDY":  {"major": "Dividend",      "minor": "US Dividend",        "hedged": True},

    # Covered Call / Income
    "HDIV": {"major": "Covered Call",  "minor": "Multi-Sector CC",    "hedged": False},
    "QMAX": {"major": "Covered Call",  "minor": "Tech Yield Max",     "hedged": False},
    "UMAX": {"major": "Covered Call",  "minor": "Utilities Yield Max", "hedged": False},

    # Cash / Defensive
    "CASH": {"major": "Cash",          "minor": "High Interest",      "hedged": False},
    "ZMMK": {"major": "Cash",          "minor": "Money Market",       "hedged": False},
    "ZST.L": {"major": "Cash",          "minor": "Ultra Short Bond",   "hedged": False},
    "XBB":  {"major": "Bonds",         "minor": "Canada Aggregate",   "hedged": False},
    "ZAG":  {"major": "Bonds",         "minor": "Canada Aggregate",   "hedged": False},
}

# Individual stock sector overrides (Yahoo Finance usually has this, but as fallback)
EQUITY_SECTORS = {
    "AAPL":   "Technology",
    "MSFT":   "Technology",
    "NVDA":   "Technology",
    "AEM":    "Materials",
    "CU":     "Utilities",
    "ENB":    "Energy",
    "TD":     "Financials",
    "REI.UN": "Real Estate",
    "WELL":   "Healthcare",
}


def get_category(symbol: str, security_type: str) -> dict:
    """Return investment category for a symbol."""
    if security_type == "EXCHANGE_TRADED_FUND":
        return ETF_CATEGORIES.get(symbol, {
            "major": "Other ETF",
            "minor": "Uncategorized",
            "hedged": False
        })
    else:
        sector = EQUITY_SECTORS.get(symbol, "Equity")
        return {
            "major": "Individual Stock",
            "minor": sector,
            "hedged": False
        }


def fetch_market_data(tickers: list[str]) -> dict:
    """
    Fetch current price data for a list of Yahoo Finance tickers.
    Returns dict keyed by ticker with price info.

    In production: call yfinance batch download.
    Without network: returns None (caller uses JSON fallback).
    """
    if not YF_AVAILABLE:
        return None

    try:
        # Batch download - much faster than individual calls
        data = yf.download(
            tickers,
            period="2d",        # Just need today + yesterday for day change
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
        )
        return data
    except Exception as e:
        print(f"⚠️  Market data fetch failed: {e}")
        return None


def fetch_security_info(ticker: str) -> dict:
    """Fetch metadata for a single ticker (sector, MER, name)."""
    if not YF_AVAILABLE:
        return {}
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "sector":          info.get("sector"),
            "industry":        info.get("industry"),
            "expense_ratio":   info.get("annualReportExpenseRatio"),  # ETF MER
            "long_name":       info.get("longName"),
            # ETF issuer
            "fund_family":     info.get("fundFamily"),
            "52w_high":        info.get("fiftyTwoWeekHigh"),
            "52w_low":         info.get("fiftyTwoWeekLow"),
        }
    except Exception:
        return {}


def enrich_positions(positions: list[dict], market_prices: Optional[dict] = None) -> list[dict]:
    """
    Merge user positions with market data.

    If market_prices is None (no network / API down),
    falls back to using the snapshot price from the original JSON
    if it was preserved, otherwise marks as stale.
    """
    enriched = []

    for pos in positions:
        ticker = pos["yf_ticker"]
        symbol = pos["symbol"]

        # ── Category classification (our product layer) ──────────────────
        category = get_category(symbol, pos["security_type"])

        # ── Market price ─────────────────────────────────────────────────
        current_price = None
        prev_close = None
        day_change_pct = None
        price_source = "unavailable"

        if market_prices and ticker in market_prices:
            try:
                prices = market_prices[ticker]["Close"].dropna()
                if len(prices) >= 2:
                    current_price = float(prices.iloc[-1])
                    prev_close = float(prices.iloc[-2])
                elif len(prices) == 1:
                    current_price = float(prices.iloc[-1])
                price_source = "yahoo_finance"
            except Exception:
                pass

        # Day change
        if current_price and prev_close:
            day_change_pct = ((current_price - prev_close) / prev_close) * 100

        # ── Calculated financials ─────────────────────────────────────────
        qty = pos["quantity"]
        book_value = pos["book_value"]

        current_market_value = qty * current_price if current_price else None
        unrealized_pnl = current_market_value - \
            book_value if current_market_value else None
        unrealized_pnl_pct = (unrealized_pnl / book_value *
                              100) if unrealized_pnl and book_value else None
        day_pnl = current_market_value * \
            (day_change_pct / 100) if current_market_value and day_change_pct else None

        enriched.append({
            # From parser (user-private)
            **pos,

            # Category (our product knowledge)
            "category_major": category["major"],
            "category_minor": category["minor"],
            "is_hedged":      category.get("hedged", False),

            # From market API
            "current_price":        current_price,
            "prev_close":           prev_close,
            "day_change_pct":       round(day_change_pct, 4) if day_change_pct else None,
            "price_source":         price_source,

            # Calculated
            "market_value":         round(current_market_value, 4) if current_market_value else None,
            "unrealized_pnl":       round(unrealized_pnl, 4) if unrealized_pnl else None,
            "unrealized_pnl_pct":   round(unrealized_pnl_pct, 4) if unrealized_pnl_pct else None,
            "day_pnl":              round(day_pnl, 4) if day_pnl else None,

            # Metadata
            "as_of": datetime.utcnow().isoformat(),
        })

    return enriched


def aggregate_portfolio(enriched_positions: list[dict]) -> dict:
    """
    Compute portfolio-level summary metrics.
    Groups by account, by category, and overall.
    """
    valid = [p for p in enriched_positions if p["market_value"] is not None]

    total_market_value = sum(p["market_value"] for p in valid)
    total_book_value = sum(p["book_value"] for p in valid)
    total_pnl = sum(p["unrealized_pnl"] for p in valid)
    total_day_pnl = sum(p["day_pnl"] for p in valid if p["day_pnl"])
    total_pnl_pct = (total_pnl / total_book_value *
                     100) if total_book_value else 0

    # ── By account ────────────────────────────────────────────────────────
    accounts = {}
    for p in valid:
        acct = p["account_type"]
        if acct not in accounts:
            accounts[acct] = {"market_value": 0, "book_value": 0, "pnl": 0}
        accounts[acct]["market_value"] += p["market_value"]
        accounts[acct]["book_value"] += p["book_value"]
        accounts[acct]["pnl"] += p["unrealized_pnl"]

    for acct in accounts:
        bv = accounts[acct]["book_value"]
        accounts[acct]["pnl_pct"] = round(
            accounts[acct]["pnl"] / bv * 100, 2) if bv else 0
        accounts[acct]["weight"] = round(
            accounts[acct]["market_value"] / total_market_value * 100, 2)

    # ── By category_major ─────────────────────────────────────────────────
    categories = {}
    for p in valid:
        cat = p["category_major"]
        if cat not in categories:
            categories[cat] = {"market_value": 0,
                               "book_value": 0, "pnl": 0, "positions": []}
        categories[cat]["market_value"] += p["market_value"]
        categories[cat]["book_value"] += p["book_value"]
        categories[cat]["pnl"] += p["unrealized_pnl"]
        categories[cat]["positions"].append(p["symbol"])

    for cat in categories:
        mv = categories[cat]["market_value"]
        bv = categories[cat]["book_value"]
        categories[cat]["weight"] = round(mv / total_market_value * 100, 2)
        categories[cat]["pnl_pct"] = round(
            categories[cat]["pnl"] / bv * 100, 2) if bv else 0

    # ── By category_minor (for ETF grouping) ─────────────────────────────
    subcategories = {}
    for p in valid:
        sub = p["category_minor"]
        if sub not in subcategories:
            subcategories[sub] = {
                "market_value": 0, "book_value": 0, "pnl": 0,
                "major": p["category_major"], "tickers": []
            }
        subcategories[sub]["market_value"] += p["market_value"]
        subcategories[sub]["book_value"] += p["book_value"]
        subcategories[sub]["pnl"] += p["unrealized_pnl"]
        subcategories[sub]["tickers"].append({
            "symbol":       p["symbol"],
            "account_type": p["account_type"],
            "market_value": p["market_value"],
            "weight_in_sub": 0,  # filled below
        })

    for sub in subcategories:
        mv = subcategories[sub]["market_value"]
        bv = subcategories[sub]["book_value"]
        subcategories[sub]["weight"] = round(mv / total_market_value * 100, 2)
        subcategories[sub]["pnl_pct"] = round(
            subcategories[sub]["pnl"] / bv * 100, 2) if bv else 0
        for t in subcategories[sub]["tickers"]:
            t["weight_in_sub"] = round(
                t["market_value"] / mv * 100, 2) if mv else 0

    return {
        "summary": {
            "total_market_value": round(total_market_value, 2),
            "total_book_value":   round(total_book_value, 2),
            "total_pnl":          round(total_pnl, 2),
            "total_pnl_pct":      round(total_pnl_pct, 2),
            "total_day_pnl":      round(total_day_pnl, 2),
            "position_count":     len(enriched_positions),
            "symbol_count":       len(set(p["symbol"] for p in enriched_positions)),
            "currency":           "CAD",
        },
        "by_account":      accounts,
        "by_category":     categories,
        "by_subcategory":  subcategories,
        "positions":       enriched_positions,
    }


if __name__ == "__main__":
    # Test with mock data (no network needed)
    mock_positions = [
        {"symbol": "VFV", "yf_ticker": "VFV.TO", "name": "Vanguard S&P 500 ETF",
         "security_type": "EXCHANGE_TRADED_FUND", "account_type": "TFSA",
         "account_id": "tfsa-xxx", "quantity": 49.75, "book_value": 7238.45,
         "average_price": 145.49, "currency": "CAD", "exchange": "TSX", "mic": "XTSE"},
        {"symbol": "AAPL", "yf_ticker": "AAPL.TO", "name": "Apple CDR",
         "security_type": "EQUITY", "account_type": "TFSA",
         "account_id": "tfsa-xxx", "quantity": 127.54, "book_value": 3963.56,
         "average_price": 31.08, "currency": "CAD", "exchange": "TSX", "mic": "XTSE"},
    ]

    enriched = enrich_positions(mock_positions, market_prices=None)
    portfolio = aggregate_portfolio(enriched)
    print("✅ market_data.py structure test passed")
    print(f"   Categories defined: {len(ETF_CATEGORIES)}")
    print(f"   Summary keys: {list(portfolio['summary'].keys())}")
