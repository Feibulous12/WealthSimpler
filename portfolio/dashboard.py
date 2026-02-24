from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from portfolio.data_pipeline.aggregate import aggregate_by_account
from portfolio.data_pipeline.compute import compute_position_metrics
from portfolio.data_pipeline.exposure import (
    build_index_buckets,
)
from portfolio.data_pipeline.market_fetch import fetch_market_data
from portfolio.data_pipeline.parser import parse_portfolio_json
from portfolio.data_pipeline.ticker_builder import build_yahoo_tickers


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


@st.cache_data(show_spinner=False)
def build_dataset(path: str = "raw.json") -> dict[str, pd.DataFrame]:
    raw_json = load_json(path)
    positions_df = build_yahoo_tickers(parse_portfolio_json(raw_json))
    tickers = positions_df["ticker"].dropna().unique().tolist()
    market_df = fetch_market_data(tickers)
    benchmark_raw = fetch_market_data(list(BENCHMARK_TICKERS.values()))
    benchmark_df = pd.DataFrame(
        {
            "name": list(BENCHMARK_TICKERS.keys()),
            "ticker": list(BENCHMARK_TICKERS.values()),
        }
    ).merge(benchmark_raw, on="ticker", how="left")
    positions_enriched = compute_position_metrics(positions_df, market_df)

    bucket_map_df = build_index_buckets(positions_df[["ticker", "symbol"]].drop_duplicates())
    account_summary = aggregate_by_account(positions_enriched)

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
    }


def render_kpis(df: pd.DataFrame, period_label: str) -> None:
    total_mv = df["market_value"].sum(min_count=1)
    total_book = df["book_value"].sum(min_count=1)
    total_pnl = df["unrealized_pnl"].sum(min_count=1)
    pnl_pct = (total_pnl / total_book) if pd.notna(total_book) and total_book != 0 else float("nan")
    period_pnl = df["selected_pnl_amount"].sum(min_count=1)
    period_base = (df["market_value"] / (1 + df["selected_return_pct"])).sum(min_count=1)
    period_pct = (period_pnl / period_base) if pd.notna(period_base) and period_base != 0 else float("nan")

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
    leaf_df["period_return_pct"] = leaf_df["period_pnl"] / leaf_df["total_market_value"]
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
    bucket_df["period_return_pct"] = bucket_df["period_pnl"] / bucket_df["total_market_value"]
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
    leaf_df["period_return"] = leaf_df["selected_pnl_amount"] / leaf_df["market_value"]
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
    bucket_df["period_return"] = bucket_df["selected_pnl_amount"] / bucket_df["market_value"]
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
        members["bucket_weight"] = members["market_value"] / bucket.market_value if bucket.market_value else 0.0

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
            others_ret = (others_pnl / others_mv) if pd.notna(others_mv) and others_mv else float("nan")
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
    view["portfolio_weight"] = view["market_value"] / total_mv if pd.notna(total_mv) and total_mv else pd.NA

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


def main() -> None:
    st.set_page_config(page_title="WealthSimpler", layout="wide")
    render_sticky_brand("WealthSimpler")
    st.title("WealthSimpler")
    st.caption("This tool provides portfolio structure insights only and does not constitute investment advice.")

    data = build_dataset("raw.json")
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
        account_tab = st.radio(
            "Account Scope",
            ["ALL", "TFSA", "FHSA"],
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


if __name__ == "__main__":
    main()
