from __future__ import annotations

from datetime import datetime
import json
import subprocess
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from portfolio.data_pipeline.aggregate import aggregate_by_account
from portfolio.data_pipeline.compute import compute_position_metrics
from portfolio.data_pipeline.exposure import (
    aggregate_by_bucket,
    build_index_buckets,
    detect_overlap,
)
from portfolio.data_pipeline.market_fetch import (
    OUTPUT_COLUMNS as MARKET_OUTPUT_COLUMNS,
    fetch_market_data,
)
from portfolio.data_pipeline.parser import parse_portfolio_json
from portfolio.data_pipeline.ticker_builder import build_yahoo_tickers
from portfolio.dashboard_report import render_full_report


BUCKET_DISPLAY_MAP = {
    "S&P_500": "S&P 500",
    "NASDAQ_100": "NASDAQ 100",
    "US_TOTAL_MARKET": "US Total Market",
    "CAD_BONDS": "CAD Bonds",
    "CASH_DEFENSIVE": "Cash Defensive",
    "CANADA_DIVIDEND": "Canada Dividend",
    "US_DIVIDEND": "US Dividend",
    "CANADA_FINANCIALS": "Canada Financials",
    "CANADA_ENERGY": "Canada Energy",
    "GLOBAL_EQUITY": "Global Equity",
    "INTL_EQUITY": "Intl Equity",
    "INDIVIDUAL_STOCK": "Individual Stocks",
}
RETURN_WINDOWS = {
    "1D": "return_1d",
    "1W": "return_1w",
    "1M": "return_1m",
    "1Q": "return_1q",
    "YTD": "return_ytd",
}
RETURN_COLOR_RANGE = {
    "1D": (-0.02, 0.02),
    "1W": (-0.05, 0.05),
    "1M": (-0.1, 0.1),
    "1Q": (-0.2, 0.2),
    "YTD": (-0.4, 0.4),
}
BENCHMARK_TICKERS = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "TSX Composite": "^GSPTSE",
}
BUCKET_SEMANTIC_COLORS = {
    "NORTH_AMERICA": "#5A8DEE",   # blue
    "TECH": "#E05A5A",            # red
    "DEFENSIVE": "#4AAE74",       # green
    "DIVIDEND": "#F0A35E",        # orange
    "BONDS": "#8B93A6",           # gray
    "STOCKS": "#9B7BD0",          # purple
}
SYMBOL_INFO_MAP = {
    "ZSP": ("BMO S&P 500 Index ETF", "0.09%"),
    "VFV": ("Vanguard S&P 500 Index ETF", "0.09%"),
    "VSP": ("Vanguard S&P 500 Index ETF (CAD Hedged)", "0.09%"),
    "QQC": ("Invesco Nasdaq 100 Index ETF", "0.20%"),
    "HXQ": ("Global X Nasdaq 100 Index ETF", "0.28%"),
    "XQQ": ("iShares NASDAQ 100 Index ETF", "0.39%"),
    "VUN": ("Vanguard U.S. Total Market Index ETF", "0.17%"),
    "XUU": ("iShares Core S&P U.S. Total Market Index ETF", "0.07%"),
    "ZAG": ("BMO Aggregate Bond Index ETF", "0.09%"),
    "XBB": ("iShares Core Canadian Universe Bond Index ETF", "0.10%"),
    "CASH": ("Global X High Interest Savings ETF", "0.11%"),
    "ZMMK": ("BMO Money Market Fund ETF Series", "0.14%"),
    "VDY": ("Vanguard FTSE Canadian High Dividend Yield Index ETF", "0.22%"),
    "ZDY": ("BMO US Dividend ETF", "0.35%"),
    "ZLB": ("BMO Low Volatility Canadian Equity ETF", "0.39%"),
    "ZEB": ("BMO Equal Weight Banks Index ETF", "0.28%"),
}
PRIORITY_RANK = {"Immediate": 0, "High": 1, "Medium": 2, "Low": 3}
TARGET_BUCKET_WEIGHTS = {
    "S&P_500": 0.22,
    "NASDAQ_100": 0.15,
    "US_TOTAL_MARKET": 0.18,
    "GLOBAL_EQUITY": 0.10,
    "INTL_EQUITY": 0.05,
    "CAD_BONDS": 0.15,
    "CASH_DEFENSIVE": 0.05,
    "CANADA_DIVIDEND": 0.05,
    "US_DIVIDEND": 0.03,
    "CANADA_FINANCIALS": 0.02,
}
BUCKET_BETA_MAP = {
    "S&P_500": 1.00,
    "NASDAQ_100": 1.20,
    "US_TOTAL_MARKET": 1.00,
    "GLOBAL_EQUITY": 1.00,
    "INTL_EQUITY": 1.00,
    "CANADA_DIVIDEND": 0.85,
    "US_DIVIDEND": 0.85,
    "CANADA_FINANCIALS": 1.10,
    "CANADA_ENERGY": 1.20,
    "CAD_BONDS": 0.25,
    "CASH_DEFENSIVE": 0.05,
    "INDIVIDUAL_STOCK": 1.10,
}


def load_json(path: str = "raw.json") -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def format_cad(v: float | int | None) -> str:
    if v is None or pd.isna(v):
        return "N/A"
    return f"${v:,.2f}"


def format_pct(v: float | int | None) -> str:
    if v is None or pd.isna(v):
        return "N/A"
    return f"{v * 100:.2f}%"


def format_pct_signed(v: float | int | None) -> str:
    if v is None or pd.isna(v):
        return "N/A"
    return f"{v * 100:+.2f}%"


def safe_divide(
    numerator: pd.Series | float | int,
    denominator: pd.Series | float | int,
) -> pd.Series | float:
    if isinstance(numerator, pd.Series):
        n = pd.to_numeric(numerator, errors="coerce")
        if isinstance(denominator, pd.Series):
            d = pd.to_numeric(denominator, errors="coerce")
            return n.div(d.where(d != 0))
        d_scalar = pd.to_numeric(pd.Series([denominator]), errors="coerce").iloc[0]
        if pd.isna(d_scalar) or d_scalar == 0:
            return pd.Series(float("nan"), index=n.index)
        return n / float(d_scalar)

    if isinstance(denominator, pd.Series):
        d = pd.to_numeric(denominator, errors="coerce")
        n_scalar = pd.to_numeric(pd.Series([numerator]), errors="coerce").iloc[0]
        if pd.isna(n_scalar):
            return pd.Series(float("nan"), index=d.index)
        return pd.Series(float(n_scalar), index=d.index).div(d.where(d != 0))

    n = pd.to_numeric(pd.Series([numerator]), errors="coerce").iloc[0]
    d = pd.to_numeric(pd.Series([denominator]), errors="coerce").iloc[0]
    if pd.isna(n) or pd.isna(d) or d == 0:
        return float("nan")
    return float(n / d)


def normalize_positions_for_dashboard(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["book_value", "current_price", "market_value", "unrealized_pnl", "return_pct", "daily_return_pct"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "market_value" in out.columns and "book_value" in out.columns:
        out["market_value"] = out["market_value"].where(out["market_value"].notna(), out["book_value"])
    if "unrealized_pnl" in out.columns and "market_value" in out.columns and "book_value" in out.columns:
        out["unrealized_pnl"] = out["unrealized_pnl"].where(
            out["unrealized_pnl"].notna(),
            out["market_value"] - out["book_value"],
        )
    if "return_pct" in out.columns and "unrealized_pnl" in out.columns and "book_value" in out.columns:
        out["return_pct"] = out["return_pct"].where(
            out["return_pct"].notna(),
            safe_divide(out["unrealized_pnl"], out["book_value"]),
        )
    if "daily_return_pct" in out.columns:
        out["daily_return_pct"] = out["daily_return_pct"].fillna(0.0)
    return out


def display_bucket_name(bucket: str) -> str:
    return BUCKET_DISPLAY_MAP.get(bucket, bucket.replace("_", " ").title())


def build_window_view(df: pd.DataFrame, return_col: str) -> pd.DataFrame:
    view = df.copy()
    if return_col not in view.columns:
        view[return_col] = pd.NA
    view["selected_return_pct"] = pd.to_numeric(view[return_col], errors="coerce")
    view["selected_pnl_amount"] = pd.to_numeric(
        view["market_value"] * view["selected_return_pct"], errors="coerce"
    )
    return view


def build_benchmark_view(df: pd.DataFrame, return_col: str) -> pd.DataFrame:
    view = df.copy()
    if return_col not in view.columns:
        view[return_col] = pd.NA
    view["selected_return_pct"] = pd.to_numeric(view[return_col], errors="coerce")
    return view


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    v = hex_color.lstrip("#")
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def shade_hex(hex_color: str, factor: float) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    f = max(0.0, min(1.0, factor))
    rr = int(r + (255 - r) * f)
    gg = int(g + (255 - g) * f)
    bb = int(b + (255 - b) * f)
    return _rgb_to_hex(rr, gg, bb)


def bucket_color_group(index_bucket: str) -> str:
    if index_bucket in {"S&P_500", "US_TOTAL_MARKET"}:
        return "NORTH_AMERICA"
    if index_bucket in {"NASDAQ_100"}:
        return "TECH"
    if index_bucket in {"CASH_DEFENSIVE"}:
        return "DEFENSIVE"
    if index_bucket in {"CANADA_DIVIDEND", "US_DIVIDEND", "CANADA_FINANCIALS", "CANADA_ENERGY", "GLOBAL_EQUITY", "INTL_EQUITY"}:
        return "DIVIDEND"
    if index_bucket in {"CAD_BONDS"}:
        return "BONDS"
    return "STOCKS"


def render_sticky_brand(name: str = "WealthSimpler") -> None:
    st.markdown(
        f"""
        <style>
        .ws-sticky-wrap {{
            position: sticky;
            top: 0;
            z-index: 999;
            margin: -8px 0 8px 0;
        }}
        .ws-sticky-bar {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(6px);
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 10px 14px;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08);
            font-weight: 700;
            color: #1f2937;
        }}
        .ws-dot {{
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: #0ea5e9;
            display: inline-block;
        }}
        </style>
        <div class="ws-sticky-wrap">
            <div class="ws-sticky-bar"><span class="ws-dot"></span>{name}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _output_path(out_dir: Path, dataset: str) -> Path | None:
    parquet_path = out_dir / f"{dataset}.parquet"
    csv_path = out_dir / f"{dataset}.csv"
    if parquet_path.exists():
        return parquet_path
    if csv_path.exists():
        return csv_path
    return None


def _required_outputs_stale(project_root: Path, out_dir: Path) -> bool:
    raw_path = project_root / "raw.json"
    if not raw_path.exists():
        return True

    required = [
        out_dir / "positions_enriched.parquet",
        out_dir / "positions_enriched.csv",
        out_dir / "account_summary.parquet",
        out_dir / "account_summary.csv",
    ]
    if not any(p.exists() for p in required[:2]) or not any(p.exists() for p in required[2:]):
        return True

    existing = [p for p in required if p.exists()]
    oldest_output_mtime = min(p.stat().st_mtime for p in existing)
    return raw_path.stat().st_mtime > oldest_output_mtime


def _data_cache_key(project_root: Path) -> str:
    out_dir = project_root / "outputs"
    parts: list[str] = []
    for name in ["positions_enriched.parquet", "positions_enriched.csv", "account_summary.parquet", "account_summary.csv"]:
        p = out_dir / name
        if p.exists():
            parts.append(f"{name}:{p.stat().st_mtime_ns}")
    raw_path = project_root / "raw.json"
    if raw_path.exists():
        parts.append(f"raw.json:{raw_path.stat().st_mtime_ns}")
    return "|".join(parts)


def _refresh_outputs_with_skill(project_root: Path, force: bool = False) -> tuple[bool, str]:
    script_path = project_root / "scripts" / "refresh_outputs_with_skill.sh"
    if not script_path.exists():
        return False, f"Missing skill refresh script: {script_path}"

    cmd = [str(script_path)]
    if force:
        cmd.append("--force")
    try:
        proc = subprocess.run(
            cmd,
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return False, f"Failed to execute skill refresh script: {exc}"

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return False, err or "Unknown error while refreshing outputs."

    out = (proc.stdout or "").strip()
    return True, out if out else "Artifacts refreshed."


def _empty_market_data(tickers: list[str]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=MARKET_OUTPUT_COLUMNS)
    return pd.DataFrame(
        {
            "ticker": tickers,
            "current_price": [pd.NA] * len(tickers),
            "previous_close": [pd.NA] * len(tickers),
            "return_1d": [pd.NA] * len(tickers),
            "return_1w": [pd.NA] * len(tickers),
            "return_1m": [pd.NA] * len(tickers),
            "return_1q": [pd.NA] * len(tickers),
            "return_ytd": [pd.NA] * len(tickers),
        },
        columns=MARKET_OUTPUT_COLUMNS,
    )


def _build_benchmark_df(allow_online_fetch: bool = True) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "name": list(BENCHMARK_TICKERS.keys()),
            "ticker": list(BENCHMARK_TICKERS.values()),
        }
    )
    if not allow_online_fetch:
        return base.merge(_empty_market_data(list(BENCHMARK_TICKERS.values())), on="ticker", how="left")
    try:
        benchmark_raw = fetch_market_data(list(BENCHMARK_TICKERS.values()))
    except Exception:
        benchmark_raw = _empty_market_data(list(BENCHMARK_TICKERS.values()))
    return base.merge(benchmark_raw, on="ticker", how="left")


def _load_outputs_dataset(
    project_root: Path,
    allow_online_fetch: bool = True,
) -> dict[str, pd.DataFrame]:
    out_dir = project_root / "outputs"
    positions_path = _output_path(out_dir, "positions_enriched")
    account_path = _output_path(out_dir, "account_summary")
    if not positions_path or not account_path:
        raise FileNotFoundError("Required outputs not found (positions_enriched/account_summary).")

    positions = (
        pd.read_parquet(positions_path)
        if positions_path.suffix == ".parquet"
        else pd.read_csv(positions_path)
    )
    positions = normalize_positions_for_dashboard(positions)
    account_summary = (
        pd.read_parquet(account_path)
        if account_path.suffix == ".parquet"
        else pd.read_csv(account_path)
    )

    if "index_bucket" not in positions.columns:
        bucket_map_df = build_index_buckets(positions[["ticker", "symbol"]].drop_duplicates())
        positions = positions.merge(
            bucket_map_df[["ticker", "index_bucket"]].drop_duplicates(),
            on="ticker",
            how="left",
        )
    positions["index_bucket"] = positions["index_bucket"].fillna("INDIVIDUAL_STOCK")
    positions["today_pnl_amount"] = pd.to_numeric(
        positions["market_value"] * positions["daily_return_pct"], errors="coerce"
    )

    market_df = (
        positions[["ticker", "current_price"]]
        .drop_duplicates()
        .assign(previous_close=pd.NA)
        .reset_index(drop=True)
    )
    benchmark_df = _build_benchmark_df(allow_online_fetch=allow_online_fetch)
    bucket_path = _output_path(out_dir, "bucket_exposure")
    overlap_path = _output_path(out_dir, "overlap")
    bucket_exposure = (
        (pd.read_parquet(bucket_path) if bucket_path.suffix == ".parquet" else pd.read_csv(bucket_path))
        if bucket_path
        else pd.DataFrame()
    )
    overlap = (
        (pd.read_parquet(overlap_path) if overlap_path.suffix == ".parquet" else pd.read_csv(overlap_path))
        if overlap_path
        else pd.DataFrame()
    )

    return {
        "positions": positions,
        "account_summary": account_summary,
        "market": market_df,
        "benchmarks": benchmark_df,
        "bucket_exposure": bucket_exposure,
        "overlap": overlap,
    }


@st.cache_data(show_spinner=False)
def build_dataset(
    path: str = "raw.json",
    prefer_outputs: bool = True,
    cache_key: str = "",
    allow_online_fetch: bool = True,
) -> dict[str, pd.DataFrame]:
    _ = cache_key
    project_root = _project_root()
    if prefer_outputs:
        # Default path: use skill artifacts and only refresh when stale/missing.
        out_dir = project_root / "outputs"
        if _required_outputs_stale(project_root, out_dir):
            ok, message = _refresh_outputs_with_skill(project_root, force=False)
            if not ok:
                st.warning(f"Skill artifacts refresh failed; falling back to direct pipeline. Details: {message}")
        try:
            return _load_outputs_dataset(project_root, allow_online_fetch=allow_online_fetch)
        except Exception as exc:
            st.warning(f"Could not load outputs artifacts; falling back to direct pipeline. Details: {exc}")

    raw_json = load_json(path)
    positions_df = build_yahoo_tickers(parse_portfolio_json(raw_json))
    tickers = positions_df["ticker"].dropna().unique().tolist()
    if allow_online_fetch:
        try:
            market_df = fetch_market_data(tickers)
        except Exception as exc:
            st.warning(f"Live market fetch failed; continuing with local/offline data. Details: {exc}")
            market_df = _empty_market_data(tickers)
    else:
        market_df = _empty_market_data(tickers)
    benchmark_df = _build_benchmark_df(allow_online_fetch=allow_online_fetch)
    positions_enriched = compute_position_metrics(positions_df, market_df)
    positions_enriched = normalize_positions_for_dashboard(positions_enriched)

    bucket_map_df = build_index_buckets(positions_df[["ticker", "symbol"]].drop_duplicates())
    account_summary = aggregate_by_account(positions_enriched)
    bucket_exposure = aggregate_by_bucket(positions_enriched, bucket_map_df)
    overlap = detect_overlap(bucket_exposure, positions_enriched, bucket_map_df)

    positions = positions_enriched.merge(
        bucket_map_df[["ticker", "index_bucket"]].drop_duplicates(),
        on="ticker",
        how="left",
    )
    positions["index_bucket"] = positions["index_bucket"].fillna("INDIVIDUAL_STOCK")
    positions["today_pnl_amount"] = pd.to_numeric(
        positions["market_value"] * positions["daily_return_pct"], errors="coerce"
    )

    return {
        "positions": positions,
        "account_summary": account_summary,
        "market": market_df,
        "benchmarks": benchmark_df,
        "bucket_exposure": bucket_exposure,
        "overlap": overlap,
    }


def render_kpis(df: pd.DataFrame, period_label: str) -> None:
    total_mv = df["market_value"].sum(min_count=1)
    total_book = df["book_value"].sum(min_count=1)
    total_pnl = df["unrealized_pnl"].sum(min_count=1)
    pnl_pct = (total_pnl / total_book) if pd.notna(total_book) and total_book != 0 else float("nan")
    period_pnl = df["selected_pnl_amount"].sum(min_count=1)
    base_denominator = 1 + pd.to_numeric(df["selected_return_pct"], errors="coerce")
    period_base = safe_divide(df["market_value"], base_denominator).sum(min_count=1)
    period_pct = safe_divide(period_pnl, period_base)

    def color_class(v: float | int | None) -> str:
        if v is None or pd.isna(v):
            return "neutral"
        return "up" if v > 0 else ("down" if v < 0 else "neutral")

    as_of_text = datetime.now().strftime("%Y-%m-%d %H:%M")

    st.markdown(
        """
        <style>
        .kpi-card {padding: 8px 6px 2px 0;}
        .kpi-label {font-size: 14px; color: #6b7280; margin-bottom: 6px;}
        .kpi-value {font-size: 46px; font-weight: 700; line-height: 1.08; white-space: nowrap;}
        .kpi-value.up {color: #0f9f57;}
        .kpi-value.down {color: #d64545;}
        .kpi-value.neutral {color: #2f3646;}
        .kpi-sub {font-size: 15px; color: #6b7280; margin-top: 6px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"<div class='kpi-card'><div class='kpi-label'>Total Market Value (CAD)</div>"
            f"<div class='kpi-value neutral'>{format_cad(total_mv)}</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='kpi-card'><div class='kpi-label'>Unrealized PnL</div>"
            f"<div class='kpi-value {color_class(total_pnl)}'>{format_cad(total_pnl)}</div>"
            f"<div class='kpi-sub'>{format_pct(pnl_pct)}</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='kpi-card'><div class='kpi-label'>{period_label} PnL</div>"
            f"<div class='kpi-value {color_class(period_pnl)}'>{format_cad(period_pnl)}</div>"
            f"<div class='kpi-sub'>{format_pct(period_pct)}</div></div>",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"<div class='kpi-card'><div class='kpi-label'>As of</div>"
            f"<div class='kpi-value neutral'>{as_of_text}</div></div>",
            unsafe_allow_html=True,
        )


def render_data_quality(market_df: pd.DataFrame) -> None:
    total = len(market_df)
    missing = int(market_df["current_price"].isna().sum())
    if missing == 0:
        st.caption("Market data status: all prices loaded")
    else:
        st.warning(f"{missing}/{total} tickers unavailable; related metrics shown as N/A")


def render_treemap(df: pd.DataFrame, period_label: str, period_key: str) -> None:
    if df.empty:
        st.info("No bucket data available")
        return

    chart_df = df.copy()
    chart_df["symbol_label"] = chart_df["symbol"].fillna(chart_df["ticker"]).astype(str)

    leaf_df = (
        chart_df.groupby(["index_bucket", "symbol_label"], as_index=False)
        .agg(
            total_market_value=("market_value", "sum"),
            period_pnl=("selected_pnl_amount", "sum"),
        )
        .sort_values("total_market_value", ascending=False)
    )
    leaf_df["bucket_label"] = leaf_df["index_bucket"].map(display_bucket_name)

    total_mv = leaf_df["total_market_value"].sum(min_count=1)
    leaf_df["weight_pct"] = (
        leaf_df["total_market_value"] / total_mv if pd.notna(total_mv) and total_mv else 0.0
    )
    leaf_df["period_return_pct"] = safe_divide(leaf_df["period_pnl"], leaf_df["total_market_value"])
    leaf_df["period_return_pct"] = leaf_df["period_return_pct"].replace(
        [float("inf"), float("-inf")], pd.NA
    ).fillna(0)

    bucket_df = (
        leaf_df.groupby(["index_bucket", "bucket_label"], as_index=False)
        .agg(
            total_market_value=("total_market_value", "sum"),
            period_pnl=("period_pnl", "sum"),
        )
        .sort_values("total_market_value", ascending=False)
    )
    bucket_df["weight_pct"] = (
        bucket_df["total_market_value"] / total_mv if pd.notna(total_mv) and total_mv else 0.0
    )
    bucket_df["period_return_pct"] = safe_divide(bucket_df["period_pnl"], bucket_df["total_market_value"])
    bucket_df["period_return_pct"] = bucket_df["period_return_pct"].replace(
        [float("inf"), float("-inf")], pd.NA
    ).fillna(0)

    labels: list[str] = []
    parents: list[str] = []
    ids: list[str] = []
    values: list[float] = []
    marker_colors: list[float] = []
    customdata: list[list[float]] = []
    text_values: list[str] = []

    for b in bucket_df.itertuples(index=False):
        bucket_id = f"bucket::{b.index_bucket}"
        labels.append(str(b.bucket_label))
        parents.append("")
        ids.append(bucket_id)
        values.append(float(b.total_market_value) if pd.notna(b.total_market_value) else 0.0)
        marker_colors.append(float(b.period_return_pct) if pd.notna(b.period_return_pct) else 0.0)
        customdata.append(
            [
                float(b.weight_pct) if pd.notna(b.weight_pct) else 0.0,
                float(b.period_return_pct) if pd.notna(b.period_return_pct) else 0.0,
                float(b.period_pnl) if pd.notna(b.period_pnl) else 0.0,
                float(b.total_market_value) if pd.notna(b.total_market_value) else 0.0,
            ]
        )
        text_values.append("")

        bucket_leaf = leaf_df[leaf_df["index_bucket"] == b.index_bucket]
        for r in bucket_leaf.itertuples(index=False):
            labels.append(str(r.symbol_label))
            parents.append(bucket_id)
            ids.append(f"{bucket_id}::symbol::{r.symbol_label}")
            values.append(float(r.total_market_value) if pd.notna(r.total_market_value) else 0.0)
            marker_colors.append(float(r.period_return_pct) if pd.notna(r.period_return_pct) else 0.0)
            customdata.append(
                [
                    float(r.weight_pct) if pd.notna(r.weight_pct) else 0.0,
                    float(r.period_return_pct) if pd.notna(r.period_return_pct) else 0.0,
                    float(r.period_pnl) if pd.notna(r.period_pnl) else 0.0,
                    float(r.total_market_value) if pd.notna(r.total_market_value) else 0.0,
                ]
            )
            text_values.append(
                format_pct_signed(float(r.period_return_pct)) if pd.notna(r.period_return_pct) else ""
            )

    cmin, cmax = RETURN_COLOR_RANGE.get(period_key, (-0.02, 0.02))
    heatmap_scale = [
        [0.00, "#ff3b3b"],
        [0.20, "#c94952"],
        [0.40, "#724e5f"],
        [0.50, "#3a404f"],
        [0.60, "#3a5f56"],
        [0.80, "#2f9f5b"],
        [1.00, "#1ec96b"],
    ]

    fig = go.Figure(
        go.Treemap(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            marker=dict(
                colors=marker_colors,
                colorscale=heatmap_scale,
                cmin=cmin,
                cmax=cmax,
                cmid=0,
                colorbar=dict(
                    title=dict(text=f"{period_label} Return (%)", font=dict(color="#e8edf7", size=13)),
                    tickformat=".1%",
                    ticksuffix="",
                    bgcolor="#202636",
                    outlinecolor="#3b4458",
                    tickfont=dict(color="#e8edf7", size=11),
                ),
                line=dict(color="#2b3345", width=1.2),
            ),
            customdata=customdata,
            text=text_values,
            texttemplate="<b>%{label}</b><br><span style='font-size:1.15em'><b>%{text}</b></span>",
            textposition="middle center",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Weight: %{customdata[0]:.2%}<br>"
                f"{period_label} Return: %{{customdata[1]:.2%}}<br>"
                f"{period_label} PnL: $%{{customdata[2]:,.2f}}<br>"
                "Market Value: $%{customdata[3]:,.2f}<extra></extra>"
            ),
        )
    )
    fig.update_traces(
        textfont=dict(color="#f5f7fb"),
        root_color="#171b26",
        pathbar=dict(visible=False),
        tiling=dict(pad=2),
    )
    fig.update_layout(
        margin=dict(t=10, l=0, r=0, b=0),
        height=620,
        paper_bgcolor="#171b26",
        plot_bgcolor="#171b26",
        font=dict(color="#eef2ff", size=14, family="Trebuchet MS, Segoe UI, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)


def render_weight_sunburst(df: pd.DataFrame, period_label: str) -> None:
    if df.empty:
        st.info("No allocation data available")
        return

    if "allocation_depth" not in st.session_state:
        st.session_state["allocation_depth"] = 2

    chart_df = df.copy()
    chart_df["symbol_label"] = chart_df["symbol"].fillna(chart_df["ticker"]).astype(str)
    chart_df["selected_return_pct"] = pd.to_numeric(chart_df["selected_return_pct"], errors="coerce")
    chart_df["selected_pnl_amount"] = pd.to_numeric(chart_df["selected_pnl_amount"], errors="coerce")

    leaf_df = (
        chart_df.groupby(["index_bucket", "symbol_label"], as_index=False)
        .agg(
            market_value=("market_value", "sum"),
            selected_pnl_amount=("selected_pnl_amount", "sum"),
        )
        .sort_values(["index_bucket", "market_value"], ascending=[True, False])
    )
    total_mv = leaf_df["market_value"].sum(min_count=1)
    leaf_df["portfolio_weight"] = leaf_df["market_value"] / total_mv if pd.notna(total_mv) and total_mv else 0.0
    leaf_df["period_return"] = safe_divide(leaf_df["selected_pnl_amount"], leaf_df["market_value"])
    leaf_df["period_return"] = leaf_df["period_return"].replace([float("inf"), float("-inf")], pd.NA)

    bucket_df = (
        leaf_df.groupby("index_bucket", as_index=False)
        .agg(
            market_value=("market_value", "sum"),
            selected_pnl_amount=("selected_pnl_amount", "sum"),
        )
        .sort_values("market_value", ascending=False)
    )
    bucket_df["bucket_label"] = bucket_df["index_bucket"].map(display_bucket_name)
    bucket_df["portfolio_weight"] = bucket_df["market_value"] / total_mv if pd.notna(total_mv) and total_mv else 0.0
    bucket_df["period_return"] = safe_divide(bucket_df["selected_pnl_amount"], bucket_df["market_value"])
    bucket_df["period_return"] = bucket_df["period_return"].replace([float("inf"), float("-inf")], pd.NA)

    ids: list[str] = []
    labels: list[str] = []
    parents: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    text_values: list[str] = []
    customdata: list[list[str]] = []

    for bucket in bucket_df.itertuples(index=False):
        group = bucket_color_group(bucket.index_bucket)
        base_color = BUCKET_SEMANTIC_COLORS[group]
        bucket_id = f"bucket::{bucket.index_bucket}"

        ids.append(bucket_id)
        labels.append(display_bucket_name(bucket.index_bucket))
        parents.append("")
        values.append(float(bucket.market_value) if pd.notna(bucket.market_value) else 0.0)
        colors.append(base_color)
        text_values.append(f"{display_bucket_name(bucket.index_bucket)}<br>{bucket.portfolio_weight:.1%}")
        customdata.append(
            [
                f"{bucket.portfolio_weight:.1%}",
                "100.0%",
                format_cad(bucket.market_value),
                format_pct_signed(bucket.period_return),
                "N/A",
                f"{display_bucket_name(bucket.index_bucket)} · Bucket",
            ]
        )

        members = leaf_df[leaf_df["index_bucket"] == bucket.index_bucket].copy()
        members["bucket_weight"] = safe_divide(members["market_value"], bucket.market_value)

        small_mask = members["portfolio_weight"] < 0.02
        small_members = members[small_mask].copy()
        large_members = members[~small_mask].copy()

        for i, row in enumerate(large_members.itertuples(index=False)):
            shade = min(0.55, 0.12 + i * 0.09)
            color = shade_hex(base_color, shade)
            symbol = row.symbol_label
            full_name, mer = SYMBOL_INFO_MAP.get(symbol, (symbol, "N/A"))
            node_id = f"{bucket_id}::sym::{symbol}"
            ids.append(node_id)
            labels.append(symbol)
            parents.append(bucket_id)
            values.append(float(row.market_value) if pd.notna(row.market_value) else 0.0)
            colors.append(color)
            text_values.append(f"{symbol}<br>{row.portfolio_weight:.1%}")
            customdata.append(
                [
                    f"{row.portfolio_weight:.1%}",
                    f"{row.bucket_weight:.1%}",
                    format_cad(row.market_value),
                    format_pct_signed(row.period_return),
                    mer,
                    f"{symbol} · {full_name}",
                ]
            )

        if not small_members.empty:
            others_mv = small_members["market_value"].sum(min_count=1)
            others_pnl = small_members["selected_pnl_amount"].sum(min_count=1)
            others_pw = others_mv / total_mv if pd.notna(total_mv) and total_mv else 0.0
            others_bw = others_mv / bucket.market_value if bucket.market_value else 0.0
            others_ret = safe_divide(others_pnl, others_mv)
            others_id = f"{bucket_id}::others"
            ids.append(others_id)
            labels.append("Others")
            parents.append(bucket_id)
            values.append(float(others_mv) if pd.notna(others_mv) else 0.0)
            colors.append(shade_hex(base_color, 0.72))
            text_values.append(f"Others<br>{others_pw:.1%}")
            customdata.append(
                [
                    f"{others_pw:.1%}",
                    f"{others_bw:.1%}",
                    format_cad(others_mv),
                    format_pct_signed(others_ret),
                    "N/A",
                    "Others · Small slices",
                ]
            )

            # Depth 3 children: hidden until "Expand all buckets"
            for j, row in enumerate(small_members.itertuples(index=False)):
                color = shade_hex(base_color, min(0.85, 0.78 + j * 0.03))
                symbol = row.symbol_label
                full_name, mer = SYMBOL_INFO_MAP.get(symbol, (symbol, "N/A"))
                node_id = f"{others_id}::sym::{symbol}"
                ids.append(node_id)
                labels.append(symbol)
                parents.append(others_id)
                values.append(float(row.market_value) if pd.notna(row.market_value) else 0.0)
                colors.append(color)
                text_values.append(f"{symbol}<br>{row.portfolio_weight:.1%}")
                customdata.append(
                    [
                        f"{row.portfolio_weight:.1%}",
                        f"{row.bucket_weight:.1%}",
                        format_cad(row.market_value),
                        format_pct_signed(row.period_return),
                        mer,
                        f"{symbol} · {full_name}",
                    ]
                )

    fig = go.Figure(
        go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            marker=dict(colors=colors, line=dict(color="#ffffff", width=1.4)),
            text=text_values,
            textinfo="text",
            insidetextorientation="radial",
            hovertemplate=(
                "<b>%{customdata[5]}</b><br>"
                "─────────────────────────────<br>"
                "Portfolio weight&nbsp;&nbsp;&nbsp;&nbsp;%{customdata[0]}<br>"
                "Bucket weight&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;%{customdata[1]}<br>"
                "Market value&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;%{customdata[2]}<br>"
                f"{period_label} return&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;%{{customdata[3]}}<br>"
                "MER&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;%{customdata[4]}<extra></extra>"
            ),
            customdata=customdata,
            maxdepth=st.session_state["allocation_depth"],
        )
    )
    fig.update_layout(
        margin=dict(t=6, l=0, r=0, b=6),
        height=560,
        paper_bgcolor="#ffffff",
        font=dict(color="#2f3646", size=16, family="Trebuchet MS, Segoe UI, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)

    b1, b2 = st.columns(2)
    if b1.button("Expand all buckets", use_container_width=True, key="allocation_expand_btn"):
        st.session_state["allocation_depth"] = 3
    if b2.button("Collapse to buckets", use_container_width=True, key="allocation_collapse_btn"):
        st.session_state["allocation_depth"] = 1


def render_top_movers_strip(
    df: pd.DataFrame,
    period_label: str,
    benchmark_df: pd.DataFrame,
    return_col: str,
) -> None:
    rank_df = df.copy()
    rank_df["selected_pnl_amount"] = pd.to_numeric(rank_df["selected_pnl_amount"], errors="coerce")
    rank_df["selected_return_pct"] = pd.to_numeric(rank_df["selected_return_pct"], errors="coerce")
    rank_df = rank_df.dropna(subset=["selected_pnl_amount"])
    if rank_df.empty:
        st.info("No mover data available")
        return

    gains = rank_df.nlargest(3, "selected_pnl_amount")[["symbol", "selected_pnl_amount", "selected_return_pct"]]
    losses = rank_df.nsmallest(3, "selected_pnl_amount")[["symbol", "selected_pnl_amount", "selected_return_pct"]]
    net_pnl = rank_df["selected_pnl_amount"].sum(min_count=1)
    total_mv = rank_df["market_value"].sum(min_count=1)
    net_pct_of_mv = (net_pnl / total_mv) if pd.notna(total_mv) and total_mv else float("nan")

    bm = build_benchmark_view(benchmark_df, return_col).copy()
    bm["selected_return_pct"] = pd.to_numeric(bm["selected_return_pct"], errors="coerce")

    max_abs_move = max(
        float(pd.to_numeric(gains["selected_pnl_amount"], errors="coerce").abs().max() or 0.0),
        float(pd.to_numeric(losses["selected_pnl_amount"], errors="coerce").abs().max() or 0.0),
        1.0,
    )
    max_abs_bm = max(float(pd.to_numeric(bm["selected_return_pct"], errors="coerce").abs().max() or 0.0), 0.0001)

    st.markdown(
        """
        <style>
        .mv-title {font-size: 16px; font-weight: 700; color: #4b5563; margin: 0 0 10px 0;}
        .mv-list {display: grid; gap: 10px;}
        .mv-item {
            background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 10px 12px;
        }
        .mv-row {display: flex; justify-content: space-between; align-items: center; gap: 8px;}
        .mv-symbol {font-size: 21px; font-weight: 800; color: #111827;}
        .mv-pnl {font-size: 28px; font-weight: 800;}
        .mv-pnl.up {color: #119b57;}
        .mv-pnl.down {color: #d64545;}
        .mv-ret {font-size: 15px; color: #6b7280; font-weight: 600; margin-top: 2px;}
        .mv-ret.up {color: #119b57;}
        .mv-ret.down {color: #d64545;}
        .mv-bar {margin-top: 8px; height: 6px; border-radius: 999px; background: #eef2f7; overflow: hidden;}
        .mv-fill {height: 100%;}
        .mv-fill.up {background: #19a45f;}
        .mv-fill.down {background: #d64545;}

        .snap-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 12px;
        }
        .snap-title {font-size: 16px; font-weight: 700; color: #4b5563; margin-bottom: 10px;}
        .snap-grid {display: grid; grid-template-columns: 1fr 1fr; gap: 10px;}
        .snap-kpi {border: 1px solid #e5e7eb; border-radius: 10px; padding: 10px; background: #fff;}
        .snap-kpi-label {font-size: 12px; color: #6b7280; font-weight: 600;}
        .snap-kpi-value {font-size: 30px; font-weight: 800; margin-top: 4px;}
        .snap-kpi-value.up {color: #119b57;}
        .snap-kpi-value.down {color: #d64545;}
        .bench-title {font-size: 13px; font-weight: 700; color: #6b7280; margin: 12px 0 8px 0;}
        .bench-list {display: grid; gap: 8px;}
        .bench-row {
            display: grid; grid-template-columns: 1.1fr .9fr; gap: 8px;
            border: 1px solid #e5e7eb; border-radius: 10px; padding: 8px 10px; background: #fff;
        }
        .bench-name {font-size: 14px; color: #1f2937; font-weight: 600;}
        .bench-value {font-size: 18px; font-weight: 800; text-align: right;}
        .bench-value.up {color: #119b57;}
        .bench-value.down {color: #d64545;}
        .bench-bar {grid-column: 1 / -1; height: 5px; border-radius: 999px; background: #eef2f7; overflow: hidden;}
        .bench-fill {height: 100%;}
        .bench-fill.up {background: #19a45f;}
        .bench-fill.down {background: #d64545;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    def render_mover_html(title: str, frame: pd.DataFrame, positive: bool) -> str:
        blocks = [f"<div class='mv-title'>{title}</div><div class='mv-list'>"]
        for row in frame.itertuples(index=False):
            amount = float(row.selected_pnl_amount)
            ret = float(row.selected_return_pct) if pd.notna(row.selected_return_pct) else float("nan")
            level = "up" if amount >= 0 else "down"
            width = min(100.0, max(5.0, abs(amount) / max_abs_move * 100.0))
            blocks.append(
                (
                    "<div class='mv-item'>"
                    "<div class='mv-row'>"
                    f"<div class='mv-symbol'>{row.symbol}</div>"
                    f"<div class='mv-pnl {level}'>{format_cad(amount)}</div>"
                    "</div>"
                    f"<div class='mv-ret {level}'>{format_pct_signed(ret)}</div>"
                    "<div class='mv-bar'>"
                    f"<div class='mv-fill {level}' style='width:{width:.1f}%;'></div>"
                    "</div>"
                    "</div>"
                )
            )
        blocks.append("</div>")
        return "".join(blocks)

    def render_snapshot_html() -> str:
        net_level = "up" if pd.notna(net_pnl) and net_pnl >= 0 else "down"
        ret_level = "up" if pd.notna(net_pct_of_mv) and net_pct_of_mv >= 0 else "down"
        rows = [
            "<div class='snap-card'>",
            f"<div class='snap-title'>Net {period_label} Snapshot</div>",
            "<div class='snap-grid'>",
            "<div class='snap-kpi'>",
            "<div class='snap-kpi-label'>Net PnL (CAD)</div>",
            f"<div class='snap-kpi-value {net_level}'>{format_cad(net_pnl)}</div>",
            "</div>",
            "<div class='snap-kpi'>",
            "<div class='snap-kpi-label'>Net Return</div>",
            f"<div class='snap-kpi-value {ret_level}'>{format_pct_signed(net_pct_of_mv)}</div>",
            "</div>",
            "</div>",
            "<div class='bench-title'>Benchmark Comparison</div>",
            "<div class='bench-list'>",
        ]
        for row in bm.itertuples(index=False):
            val = float(row.selected_return_pct) if pd.notna(row.selected_return_pct) else float("nan")
            level = "up" if pd.notna(val) and val >= 0 else "down"
            width = min(100.0, max(4.0, abs(val) / max_abs_bm * 100.0))
            rows.extend(
                [
                    "<div class='bench-row'>",
                    f"<div class='bench-name'>{row.name}</div>",
                    f"<div class='bench-value {level}'>{format_pct_signed(val)}</div>",
                    "<div class='bench-bar'>",
                    f"<div class='bench-fill {level}' style='width:{width:.1f}%;'></div>",
                    "</div>",
                    "</div>",
                ]
            )
        rows.extend(["</div>", "</div>"])
        return "".join(rows)

    c1, c2, c3 = st.columns([1.15, 1.15, 0.95])
    with c1:
        st.markdown(render_mover_html(f"Top 3 Gainers ({period_label}, CAD)", gains, positive=True), unsafe_allow_html=True)
    with c2:
        st.markdown(render_mover_html(f"Bottom 3 Losers ({period_label}, CAD)", losses, positive=False), unsafe_allow_html=True)
    with c3:
        st.markdown(render_snapshot_html(), unsafe_allow_html=True)


def render_period_panel(df: pd.DataFrame, period_label: str) -> None:
    st.subheader(f"{period_label} Contribution")
    left, right = st.columns([2, 1])

    with left:
        bucket_df = (
            df.groupby("index_bucket", as_index=False)
            .agg(
                total_market_value=("market_value", "sum"),
                period_pnl=("selected_pnl_amount", "sum"),
            )
            .sort_values("total_market_value", ascending=False)
        )
        wf_df = bucket_df.nlargest(8, "total_market_value").copy()
        other = bucket_df.iloc[8:]["period_pnl"].sum() if len(bucket_df) > 8 else 0.0
        if len(bucket_df) > 8:
            wf_df = pd.concat(
                [
                    wf_df,
                    pd.DataFrame(
                        [{"index_bucket": "Other", "period_pnl": other, "total_market_value": 0.0}]
                    ),
                ],
                ignore_index=True,
            )

        y_values = pd.to_numeric(wf_df["period_pnl"], errors="coerce").fillna(0).tolist()
        fig = go.Figure(
            go.Waterfall(
                name=f"{period_label} PnL",
                x=wf_df["index_bucket"].tolist() + ["Total"],
                y=y_values + [sum(y_values)],
                measure=["relative"] * len(wf_df) + ["total"],
            )
        )
        fig.update_layout(margin=dict(t=10, l=0, r=0, b=0), height=380)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        rank_df = df.copy()
        rank_df["selected_pnl_amount"] = pd.to_numeric(rank_df["selected_pnl_amount"], errors="coerce")
        rank_df["selected_return_pct"] = pd.to_numeric(rank_df["selected_return_pct"], errors="coerce")
        rank_df = rank_df.dropna(subset=["selected_pnl_amount"])
        missing_rank = len(df) - len(rank_df)
        st.caption(f"Excluded from ranking due to missing prices: {missing_rank}")

        top5 = rank_df.nlargest(5, "selected_pnl_amount")[["symbol", "selected_pnl_amount", "selected_return_pct"]]
        bot5 = rank_df.nsmallest(5, "selected_pnl_amount")[["symbol", "selected_pnl_amount", "selected_return_pct"]]
        st.write("Top 5")
        st.dataframe(
            top5.rename(columns={"selected_pnl_amount": f"{period_label} PnL", "selected_return_pct": f"{period_label} %"}),
            hide_index=True,
        )
        st.write("Bottom 5")
        st.dataframe(
            bot5.rename(columns={"selected_pnl_amount": f"{period_label} PnL", "selected_return_pct": f"{period_label} %"}),
            hide_index=True,
        )


def render_holdings_table(df: pd.DataFrame, period_label: str) -> None:
    st.subheader("Holdings Table")

    c1, c2, c3 = st.columns(3)
    account = c1.selectbox("Account", ["ALL", "TFSA", "FHSA"], index=0)
    bucket_options = sorted(df["index_bucket"].dropna().unique().tolist())
    selected_buckets = c2.multiselect("Bucket", bucket_options)
    pnl_status = c3.selectbox("PnL Status", ["ALL", "GAIN", "LOSS"], index=0)

    view = df.copy()
    if account != "ALL":
        view = view[view["account_type"] == account]
    if selected_buckets:
        view = view[view["index_bucket"].isin(selected_buckets)]
    if pnl_status == "GAIN":
        view = view[view["unrealized_pnl"] > 0]
    elif pnl_status == "LOSS":
        view = view[view["unrealized_pnl"] < 0]

    total_mv = df["market_value"].sum(min_count=1)
    view["portfolio_weight"] = safe_divide(view["market_value"], total_mv)

    show = view[[
        "symbol",
        "account_type",
        "index_bucket",
        "market_value",
        "selected_pnl_amount",
        "selected_return_pct",
        "unrealized_pnl",
        "return_pct",
        "portfolio_weight",
    ]].rename(
        columns={
            "symbol": "Symbol",
            "account_type": "Account",
            "index_bucket": "Bucket",
            "market_value": "Market Value",
            "selected_pnl_amount": f"{period_label} PnL (amt)",
            "selected_return_pct": f"{period_label} Return (%)",
            "unrealized_pnl": "Unrealized PnL (amt)",
            "return_pct": "Unrealized PnL (%)",
            "portfolio_weight": "Portfolio Weight",
        }
    )
    st.dataframe(show.sort_values("Market Value", ascending=False), use_container_width=True, hide_index=True)


def render_account_compare(df: pd.DataFrame) -> None:
    st.subheader("Account Compare")
    accounts = ["TFSA", "FHSA"]
    cols = st.columns(2)

    for i, acct in enumerate(accounts):
        acct_df = df[df["account_type"] == acct].copy()
        if acct_df.empty:
            cols[i].info(f"{acct}: no data")
            continue

        mv = acct_df["market_value"].sum(min_count=1)
        pnl = acct_df["unrealized_pnl"].sum(min_count=1)
        bucket_weights = (
            acct_df.groupby("index_bucket", as_index=False)["market_value"]
            .sum()
            .sort_values("market_value", ascending=False)
        )
        total = bucket_weights["market_value"].sum()
        top3 = bucket_weights.head(3)["index_bucket"].tolist()

        growth = bucket_weights[bucket_weights["index_bucket"].isin(["NASDAQ_100", "S&P_500", "US_TOTAL_MARKET", "GLOBAL_EQUITY"])]["market_value"].sum()
        defensive = bucket_weights[bucket_weights["index_bucket"].isin(["CAD_BONDS", "CASH_DEFENSIVE"])]["market_value"].sum()

        cols[i].markdown(f"**{acct}**")
        cols[i].write(f"Market Value: {format_cad(mv)}")
        cols[i].write(f"Unrealized PnL: {format_cad(pnl)}")
        cols[i].write(f"Top 3 Buckets: {', '.join(top3) if top3 else 'N/A'}")
        cols[i].write(f"Growth-bucket share: {format_pct(growth / total if total else float('nan'))}")
        cols[i].write(f"Defensive-bucket share: {format_pct(defensive / total if total else float('nan'))}")


def _priority_from_drift(abs_drift: float) -> str:
    if abs_drift >= 0.08:
        return "Immediate"
    if abs_drift >= 0.05:
        return "High"
    if abs_drift >= 0.03:
        return "Medium"
    return "Low"


def _thesis_status(ret: float) -> str:
    if pd.isna(ret):
        return "Unknown"
    if ret >= 0.15:
        return "Strengthening"
    if ret >= -0.08:
        return "Intact"
    if ret >= -0.20:
        return "Weakening"
    return "Broken"


def build_portfolio_manager_views(
    scoped_df: pd.DataFrame,
    full_df: pd.DataFrame,
    account_summary: pd.DataFrame,
    market_df: pd.DataFrame,
    period_label: str,
) -> dict[str, pd.DataFrame | dict[str, float | str]]:
    df = scoped_df.copy()
    total_mv = float(df["market_value"].sum(min_count=1) or 0.0)
    if total_mv <= 0:
        total_mv = 1e-9
    df["weight"] = safe_divide(df["market_value"], total_mv)

    top_holdings = (
        df.groupby(["ticker", "symbol", "index_bucket"], as_index=False)
        .agg(
            market_value=("market_value", "sum"),
            unrealized_pnl=("unrealized_pnl", "sum"),
            return_pct=("return_pct", "mean"),
            selected_return_pct=("selected_return_pct", "mean"),
            current_price=("current_price", "mean"),
        )
        .sort_values("market_value", ascending=False)
    )
    top_holdings["weight"] = safe_divide(top_holdings["market_value"], total_mv)
    top_holdings["thesis_status"] = top_holdings["return_pct"].apply(_thesis_status)
    top_holdings["action"] = "HOLD"
    top_holdings.loc[top_holdings["weight"] >= 0.15, "action"] = "TRIM"
    top_holdings.loc[(top_holdings["return_pct"] <= -0.20) & (top_holdings["weight"] >= 0.05), "action"] = "SELL"
    top_holdings.loc[
        (top_holdings["weight"] <= 0.03) & (top_holdings["index_bucket"].isin(["S&P_500", "US_TOTAL_MARKET", "CAD_BONDS"])),
        "action",
    ] = "ADD"
    top_holdings["confidence"] = top_holdings["current_price"].notna().map({True: "High", False: "Low"})
    top_holdings["rationale"] = top_holdings.apply(
        lambda r: "Position size exceeds concentration threshold"
        if r["action"] == "TRIM"
        else (
            "Drawdown and size indicate thesis impairment"
            if r["action"] == "SELL"
            else ("Core bucket underweight position" if r["action"] == "ADD" else "Thesis and sizing acceptable")
        ),
        axis=1,
    )

    bucket_alloc = (
        df.groupby("index_bucket", as_index=False)["market_value"]
        .sum()
        .sort_values("market_value", ascending=False)
    )
    bucket_alloc["current_weight"] = safe_divide(bucket_alloc["market_value"], total_mv).fillna(0.0)
    bucket_alloc["target_weight"] = bucket_alloc["index_bucket"].map(TARGET_BUCKET_WEIGHTS).fillna(0.0)
    bucket_alloc["drift"] = bucket_alloc["current_weight"] - bucket_alloc["target_weight"]

    missing_targets = sorted(set(TARGET_BUCKET_WEIGHTS.keys()) - set(bucket_alloc["index_bucket"]))
    if missing_targets:
        bucket_alloc = pd.concat(
            [
                bucket_alloc,
                pd.DataFrame(
                    {
                        "index_bucket": missing_targets,
                        "market_value": 0.0,
                        "current_weight": 0.0,
                        "target_weight": [TARGET_BUCKET_WEIGHTS[b] for b in missing_targets],
                        "drift": [-TARGET_BUCKET_WEIGHTS[b] for b in missing_targets],
                    }
                ),
            ],
            ignore_index=True,
        )

    rebal = bucket_alloc.copy()
    rebal["priority"] = rebal["drift"].abs().apply(_priority_from_drift)
    rebal["action"] = rebal["drift"].apply(lambda d: "REDUCE" if d > 0 else "INCREASE")
    rebal["trade_value_cad"] = rebal["drift"].abs() * total_mv
    rebal = rebal[rebal["drift"].abs() >= 0.02].sort_values(
        by=["priority", "trade_value_cad"], key=lambda s: s.map(PRIORITY_RANK) if s.name == "priority" else s, ascending=[True, False]
    )

    hhi = float((top_holdings["weight"].fillna(0.0) * 100).pow(2).sum())
    top5_share = float(top_holdings["weight"].head(5).sum())
    growth_share = float(
        bucket_alloc[bucket_alloc["index_bucket"].isin(["NASDAQ_100", "S&P_500", "US_TOTAL_MARKET", "GLOBAL_EQUITY"])]["current_weight"].sum()
    )
    defensive_share = float(
        bucket_alloc[bucket_alloc["index_bucket"].isin(["CAD_BONDS", "CASH_DEFENSIVE"])]["current_weight"].sum()
    )
    estimated_beta = float(
        (bucket_alloc["current_weight"] * bucket_alloc["index_bucket"].map(BUCKET_BETA_MAP).fillna(1.0)).sum()
    )
    volatility_band = "Low" if estimated_beta < 0.8 else ("Moderate" if estimated_beta <= 1.05 else "High")
    price_coverage = 1.0 - (float(market_df["current_price"].isna().mean()) if len(market_df) else 1.0)

    concentration_actions: list[dict[str, str | float]] = []
    largest = top_holdings.head(1)
    if not largest.empty and float(largest.iloc[0]["weight"]) > 0.15:
        concentration_actions.append(
            {
                "priority": "Immediate",
                "item": f"Trim {largest.iloc[0]['symbol']} concentration",
                "detail": f"Single position at {float(largest.iloc[0]['weight']):.1%}; target < 12-15%.",
            }
        )
    if top5_share > 0.40:
        concentration_actions.append(
            {
                "priority": "High",
                "item": "Reduce top-5 concentration",
                "detail": f"Top-5 holdings are {top5_share:.1%} of portfolio.",
            }
        )

    rebal_actions = [
        {
            "priority": row.priority,
            "item": f"{row.action} {display_bucket_name(row.index_bucket)}",
            "detail": f"Drift {row.drift:+.1%}, rebalance about {format_cad(row.trade_value_cad)}.",
        }
        for row in rebal.head(8).itertuples(index=False)
    ]
    action_items = pd.DataFrame(concentration_actions + rebal_actions)
    if action_items.empty:
        action_items = pd.DataFrame(
            [{"priority": "Low", "item": "No major drift detected", "detail": "Maintain current allocation and review next cycle."}]
        )
    action_items["priority_order"] = action_items["priority"].map(PRIORITY_RANK).fillna(99)
    action_items = action_items.sort_values(["priority_order", "item"]).drop(columns=["priority_order"])

    overview = {
        "total_market_value": float(full_df["market_value"].sum(min_count=1) or 0.0),
        "total_book_value": float(full_df["book_value"].sum(min_count=1) or 0.0),
        "total_unrealized_pnl": float(full_df["unrealized_pnl"].sum(min_count=1) or 0.0),
        "selected_window_pnl": float(full_df["selected_pnl_amount"].sum(min_count=1) or 0.0),
        "price_coverage": price_coverage,
        "hhi": hhi,
        "top5_share": top5_share,
        "growth_share": growth_share,
        "defensive_share": defensive_share,
        "estimated_beta": estimated_beta,
        "volatility_band": volatility_band,
        "period_label": period_label,
    }

    return {
        "overview": overview,
        "top_holdings": top_holdings,
        "bucket_alloc": bucket_alloc.sort_values("current_weight", ascending=False),
        "rebalancing": rebal,
        "action_items": action_items,
        "account_summary": account_summary,
    }


def render_portfolio_manager_web_report(
    scoped_df: pd.DataFrame,
    full_df: pd.DataFrame,
    account_summary: pd.DataFrame,
    market_df: pd.DataFrame,
    period_label: str,
) -> None:
    views = build_portfolio_manager_views(scoped_df, full_df, account_summary, market_df, period_label)
    overview = views["overview"]
    top_holdings = views["top_holdings"]
    bucket_alloc = views["bucket_alloc"]
    rebalancing = views["rebalancing"]
    action_items = views["action_items"]

    st.markdown("---")
    st.header("Portfolio Manager")
    st.caption("Skill-aligned web report for allocation, diversification, risk, position actions, and rebalancing priorities.")

    t1, t2, t3, t4, t5 = st.tabs(
        ["Executive Summary", "Allocation & Diversification", "Risk & Performance", "Position Analysis", "Rebalancing & Action Items"]
    )

    with t1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Market Value", format_cad(overview["total_market_value"]))
        c2.metric("Unrealized PnL", format_cad(overview["total_unrealized_pnl"]))
        c3.metric(f"{overview['period_label']} PnL", format_cad(overview["selected_window_pnl"]))
        c4.metric("Price Coverage", format_pct(overview["price_coverage"]))
        st.subheader("Data Quality")
        st.write(
            f"- Source: `outputs/` artifacts (with fallback).  \n"
            f"- Price coverage: {format_pct(overview['price_coverage'])}.  \n"
            f"- Missing price metrics are shown as N/A and excluded where required."
        )
        st.subheader("Holdings Overview")
        show = top_holdings.head(10)[["symbol", "index_bucket", "market_value", "weight", "unrealized_pnl", "return_pct"]]
        show = show.rename(
            columns={
                "symbol": "Symbol",
                "index_bucket": "Bucket",
                "market_value": "Market Value",
                "weight": "Weight",
                "unrealized_pnl": "Unrealized PnL",
                "return_pct": "Return %",
            }
        )
        st.dataframe(show, use_container_width=True, hide_index=True)

    with t2:
        st.subheader("Asset Allocation vs Target")
        alloc_show = bucket_alloc.copy()
        alloc_show["bucket"] = alloc_show["index_bucket"].map(display_bucket_name)
        st.dataframe(
            alloc_show[["bucket", "current_weight", "target_weight", "drift", "market_value"]].rename(
                columns={
                    "bucket": "Bucket",
                    "current_weight": "Current Weight",
                    "target_weight": "Target Weight",
                    "drift": "Drift",
                    "market_value": "Market Value",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.subheader("Diversification")
        d1, d2, d3 = st.columns(3)
        d1.metric("HHI", f"{overview['hhi']:.0f}")
        d2.metric("Top-5 Concentration", format_pct(overview["top5_share"]))
        d3.metric("Defensive Share", format_pct(overview["defensive_share"]))

    with t3:
        st.subheader("Risk Assessment")
        r1, r2, r3 = st.columns(3)
        r1.metric("Estimated Portfolio Beta", f"{overview['estimated_beta']:.2f}")
        r2.metric("Risk Band", str(overview["volatility_band"]))
        r3.metric("Growth Share", format_pct(overview["growth_share"]))
        st.subheader("Performance Review")
        p1, p2 = st.columns(2)
        p1.metric("Book Value", format_cad(overview["total_book_value"]))
        p2.metric("Unrealized Return", format_pct(safe_divide(overview["total_unrealized_pnl"], overview["total_book_value"])))
        st.caption("Performance values come from current holdings metrics and selected window return columns.")

    with t4:
        st.subheader("Position Recommendations (HOLD / ADD / TRIM / SELL)")
        pos_show = top_holdings.head(15)[
            ["symbol", "index_bucket", "weight", "return_pct", "thesis_status", "action", "confidence", "rationale"]
        ].rename(
            columns={
                "symbol": "Symbol",
                "index_bucket": "Bucket",
                "weight": "Weight",
                "return_pct": "Return %",
                "thesis_status": "Thesis",
                "action": "Action",
                "confidence": "Confidence",
                "rationale": "Rationale",
            }
        )
        st.dataframe(pos_show, use_container_width=True, hide_index=True)

    with t5:
        st.subheader("Rebalancing Recommendations")
        if rebalancing.empty:
            st.info("No material allocation drift above 2%.")
        else:
            rb_show = rebalancing[["priority", "index_bucket", "action", "drift", "trade_value_cad"]].rename(
                columns={
                    "priority": "Priority",
                    "index_bucket": "Bucket",
                    "action": "Action",
                    "drift": "Drift",
                    "trade_value_cad": "Trade Value (CAD)",
                }
            )
            rb_show["Bucket"] = rb_show["Bucket"].map(display_bucket_name)
            st.dataframe(rb_show, use_container_width=True, hide_index=True)
        st.subheader("Action Items")
        st.dataframe(action_items.rename(columns={"priority": "Priority", "item": "Item", "detail": "Detail"}), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="WealthSimpler", layout="wide")
    render_sticky_brand("WealthSimpler")
    st.title("WealthSimpler")
    st.caption("This tool provides portfolio structure insights only and does not constitute investment advice.")

    toolbar_left, toolbar_right = st.columns([3, 1])
    with toolbar_left:
        offline_mode = st.checkbox(
            "Offline mode (use local data only)",
            value=False,
            help="Skip live market and benchmark fetches.",
        )
        if offline_mode:
            st.caption("Data source: local artifacts/pipeline only (online market fetch disabled)")
        else:
            st.caption("Data source: skill artifacts (`outputs/`) with pipeline fallback")
    with toolbar_right:
        if st.button("Refresh Outputs", use_container_width=True):
            ok, message = _refresh_outputs_with_skill(_project_root(), force=True)
            if ok:
                build_dataset.clear()
                st.success("Outputs refreshed from portfolio-manager skill.")
                st.rerun()
            st.error(f"Refresh failed: {message}")

    data = build_dataset(
        "raw.json",
        prefer_outputs=True,
        cache_key=_data_cache_key(_project_root()),
        allow_online_fetch=not offline_mode,
    )
    positions = data["positions"]
    controls_left, controls_right = st.columns(2)
    with controls_left:
        st.caption("Return Window")
        window_key = st.radio(
            "Return Window",
            list(RETURN_WINDOWS.keys()),
            horizontal=True,
            index=0,
            label_visibility="collapsed",
        )
    return_col = RETURN_WINDOWS[window_key]
    period_label = window_key
    if return_col not in positions.columns:
        st.warning(
            f"Missing column `{return_col}` in cached dataset. "
            "Please clear Streamlit cache and rerun to refresh market data."
        )

    with controls_right:
        st.caption("Account Scope")
        account_options = ["ALL"] + sorted(
            {str(v).strip() for v in positions["account_type"].dropna().tolist() if str(v).strip()}
        )
        account_tab = st.radio(
            "Account Scope",
            account_options,
            horizontal=True,
            label_visibility="collapsed",
        )
    scoped = positions if account_tab == "ALL" else positions[positions["account_type"] == account_tab]
    scoped = build_window_view(scoped, return_col)

    render_kpis(scoped, period_label)
    render_data_quality(data["market"])

    st.subheader(f"Return Treemap ({period_label})")
    render_treemap(scoped, period_label, window_key)
    st.subheader("Portfolio Allocation")
    render_weight_sunburst(scoped, period_label)

    render_top_movers_strip(scoped, period_label, data["benchmarks"], return_col)
    render_period_panel(scoped, period_label)
    render_holdings_table(scoped, period_label)
    render_account_compare(positions)

    # Portfolio Manager Report - Full narrative analysis
    st.markdown("---")
    views = build_portfolio_manager_views(
        scoped_df=scoped,
        full_df=build_window_view(positions, return_col),
        account_summary=data["account_summary"],
        market_df=data["market"],
        period_label=period_label,
    )
    render_full_report(views, positions, data.get("overlap", pd.DataFrame()))


if __name__ == "__main__":
    main()
