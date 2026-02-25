"""
Portfolio Manager Report - Narrative analysis for Dashboard
Generates skill-style written analysis with insights and recommendations
"""
from __future__ import annotations

import ast
import pandas as pd
import streamlit as st


# Utility functions (copied here to avoid circular import)
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


def display_bucket_name(bucket: str) -> str:
    return BUCKET_DISPLAY_MAP.get(bucket, bucket.replace("_", " ").title())


REPORT_CSS = """
<style>
.report-section {
    background: #ffffff;
    border-radius: 12px;
    padding: 24px;
    margin: 20px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.report-section h2 {
    color: #1f2937;
    border-bottom: 2px solid #3b82f6;
    padding-bottom: 12px;
    margin-bottom: 20px;
    font-size: 24px;
}

.report-section h3 {
    color: #374151;
    margin-top: 20px;
    margin-bottom: 12px;
    font-size: 18px;
}

.report-section h4 {
    color: #4b5563;
    margin-top: 16px;
    margin-bottom: 8px;
    font-size: 16px;
}

.insight-box {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    border-left: 4px solid #3b82f6;
    padding: 16px 20px;
    margin: 16px 0;
    border-radius: 0 8px 8px 0;
}

.warning-box {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border-left: 4px solid #f59e0b;
    padding: 16px 20px;
    margin: 16px 0;
    border-radius: 0 8px 8px 0;
}

.danger-box {
    background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    border-left: 4px solid #ef4444;
    padding: 16px 20px;
    margin: 16px 0;
    border-radius: 0 8px 8px 0;
}

.success-box {
    background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
    border-left: 4px solid #10b981;
    padding: 16px 20px;
    margin: 16px 0;
    border-radius: 0 8px 8px 0;
}

.position-card {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 16px;
    margin: 12px 0;
}

.position-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
    flex-wrap: wrap;
}

.position-header .symbol {
    font-size: 20px;
    font-weight: 700;
    color: #1f2937;
}

.position-header .bucket {
    font-size: 14px;
    color: #6b7280;
    background: #f3f4f6;
    padding: 4px 12px;
    border-radius: 12px;
}

.position-header .action {
    font-size: 12px;
    font-weight: 600;
    color: white;
    padding: 4px 12px;
    border-radius: 12px;
    text-transform: uppercase;
}

.position-header .action-hold { background-color: #6b7280; }
.position-header .action-add { background-color: #10b981; }
.position-header .action-trim { background-color: #f59e0b; }
.position-header .action-sell { background-color: #ef4444; }

.position-body {
    color: #4b5563;
}

.position-body .metrics {
    font-size: 14px;
    margin-bottom: 8px;
}

.position-body .rationale {
    font-style: italic;
    color: #6b7280;
    margin: 8px 0;
    font-size: 14px;
}

.metric-row {
    display: flex;
    gap: 24px;
    margin: 12px 0;
    flex-wrap: wrap;
}

.metric-item {
    text-align: center;
}

.metric-item .label {
    font-size: 12px;
    color: #6b7280;
    text-transform: uppercase;
}

.metric-item .value {
    font-size: 24px;
    font-weight: 700;
    color: #1f2937;
}

.metric-item .value.positive { color: #10b981; }
.metric-item .value.negative { color: #ef4444; }

.findings-list {
    list-style: none;
    padding: 0;
    margin: 12px 0;
}

.findings-list li {
    padding: 8px 0 8px 28px;
    position: relative;
    border-bottom: 1px solid #f3f4f6;
}

.findings-list li:last-child {
    border-bottom: none;
}

.findings-list li::before {
    position: absolute;
    left: 0;
    font-size: 16px;
}

.findings-list li.positive::before { content: "✅"; }
.findings-list li.warning::before { content: "⚠️"; }
.findings-list li.danger::before { content: "🔴"; }
.findings-list li.info::before { content: "ℹ️"; }
</style>
"""


def _safe_format(value, formatter, default="N/A"):
    """Safely format a value, returning default if None or NaN"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    return formatter(value)


def _safe_ratio(numerator, denominator, default: float = 0.0) -> float:
    n = pd.to_numeric(pd.Series([numerator]), errors="coerce").iloc[0]
    d = pd.to_numeric(pd.Series([denominator]), errors="coerce").iloc[0]
    if pd.isna(n) or pd.isna(d) or d == 0:
        return default
    return float(n / d)


def _coerce_member_tickers(value) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except (SyntaxError, ValueError):
            return [value]
    return []


def generate_findings_list(positions: pd.DataFrame, overview: dict) -> list[dict]:
    """Generate list of key findings about the portfolio"""
    findings = []
    if positions.empty:
        return findings

    # Performance finding
    total_return = _safe_ratio(overview.get("total_unrealized_pnl"), overview.get("total_book_value"), default=float("nan"))
    if total_return > 0.15:
        findings.append({
            'type': 'positive',
            'text': f"Portfolio total return is +{total_return*100:.1f}%, showing excellent performance and significantly outperforming the benchmark"
        })
    elif total_return > 0:
        findings.append({
            'type': 'positive',
            'text': f"Portfolio total return is +{total_return*100:.1f}%, maintaining positive returns"
        })

    # Account performance
    tfsa_data = positions[positions['account_type'] == 'TFSA']
    if not tfsa_data.empty:
        tfsa_return = _safe_ratio(tfsa_data['unrealized_pnl'].sum(), tfsa_data['book_value'].sum(), default=float("nan"))
        if tfsa_return > 0.20:
            findings.append({
                'type': 'positive',
                'text': f"TFSA account showing strong performance with returns of +{tfsa_return*100:.1f}%, effectively utilizing tax-free allowance"
            })

    # Cash position
    cash_weight = _safe_ratio(
        positions[positions['index_bucket'] == 'CASH_DEFENSIVE']['market_value'].sum(),
        overview.get("total_market_value"),
    )
    if cash_weight > 0.20:
        findings.append({
            'type': 'warning',
            'text': f"Cash defensive assets account for {cash_weight*100:.1f}% of portfolio, which may drag on long-term returns. Consider increasing equity allocation"
        })

    # Growth allocation
    growth_buckets = ['NASDAQ_100', 'S&P_500', 'US_TOTAL_MARKET', 'GLOBAL_EQUITY']
    growth_weight = _safe_ratio(
        positions[positions['index_bucket'].isin(growth_buckets)]['market_value'].sum(),
        overview.get("total_market_value"),
    )
    if growth_weight > 0.50:
        findings.append({
            'type': 'info',
            'text': f"Growth assets are appropriately allocated at {growth_weight*100:.1f}%, supporting long-term capital appreciation"
        })

    return findings


def generate_concerns_list(positions: pd.DataFrame, overlap: pd.DataFrame) -> list[dict]:
    """Generate list of concerns and risks"""
    concerns = []

    # Check for high overlap (only if overlap data is valid)
    if not overlap.empty and 'overlap_flag' in overlap.columns:
        high_overlap = overlap[overlap['overlap_flag'] == 'High']
        for _, row in high_overlap.iterrows():
            member_count = len(row.get('member_tickers', [])) if isinstance(row.get('member_tickers'), list) else row.get('member_count', 1)
            total_weight_pct = pd.to_numeric(pd.Series([row.get("total_weight_pct")]), errors="coerce").iloc[0]
            if pd.isna(total_weight_pct):
                total_weight_pct = 0.0
            concerns.append({
                'type': 'warning',
                'text': f"<b>{display_bucket_name(row['index_bucket'])}</b> has overlapping positions held through {member_count} products with weight of {total_weight_pct*100:.1f}%. Consider consolidating"
            })

    # Check for concentrated positions
    if positions.empty:
        return concerns

    total_mv = positions['market_value'].sum()
    top_series = positions.groupby('symbol')['market_value'].sum().sort_values(ascending=False)
    top_weight = _safe_ratio(top_series.iloc[0], total_mv, default=float("nan")) if not top_series.empty else float("nan")
    if top_weight > 0.10:
        concerns.append({
            'type': 'warning',
            'text': f"Largest single position accounts for {top_weight*100:.1f}% of portfolio. Monitor concentration risk"
        })

    # Check for international exposure
    intl_weight = _safe_ratio(
        positions[positions['index_bucket'] == 'INTL_EQUITY']['market_value'].sum(),
        total_mv,
        default=float("nan"),
    )
    if intl_weight < 0.05:
        concerns.append({
            'type': 'info',
            'text': f"International equity allocation is low (only {intl_weight*100:.1f}%). Consider increasing to diversify geographic risk"
        })

    return concerns


def render_executive_summary(views: dict, positions: pd.DataFrame, overlap: pd.DataFrame = None) -> None:
    """Render Executive Summary section with narrative analysis"""
    overview = views['overview']

    if overlap is None:
        overlap = pd.DataFrame()
    account_summary = views['account_summary']

    total_mv = overview['total_market_value']
    total_pnl = overview['total_unrealized_pnl']
    total_book = overview['total_book_value']
    total_return = total_pnl / total_book if total_book > 0 else 0

    # Determine performance description
    if total_return > 0.20:
        performance_desc = "performing <strong>very strongly</strong>"
        benchmark_comparison = "significantly outperforming market benchmarks"
    elif total_return > 0.10:
        performance_desc = "performing <strong>well</strong>"
        benchmark_comparison = "outperforming market benchmarks"
    elif total_return > 0:
        performance_desc = "performing <strong>steadily</strong>"
        benchmark_comparison = "tracking in line with market benchmarks"
    else:
        performance_desc = "under <strong>pressure</strong>"
        benchmark_comparison = "underperforming market benchmarks"

    # Determine risk description
    beta = overview['estimated_beta']
    if beta < 0.8:
        risk_desc = "Low"
        risk_style = "Defensive"
    elif beta < 1.0:
        risk_desc = "Moderately Low"
        risk_style = "Conservative"
    elif beta < 1.2:
        risk_desc = "Moderate"
        risk_style = "Balanced"
    else:
        risk_desc = "High"
        risk_style = "Aggressive"

    # Get findings and concerns
    findings = generate_findings_list(positions, overview)
    concerns = generate_concerns_list(positions, overlap)

    st.markdown(REPORT_CSS, unsafe_allow_html=True)

    st.markdown("""
    <div class='report-section'>
    <h2>📋 Executive Summary</h2>

    <div class='insight-box'>
    <p style='font-size: 16px; line-height: 1.6;'>
    Your investment portfolio has a total value of <b style='font-size: 18px;'>{}</b>, {}.
    Unrealized P&L stands at <b style='color: {}'>{} ({:+.2f}%)</b>, {}.
    </p>
    <p style='font-size: 16px; line-height: 1.6; margin-top: 12px;'>
    From a risk perspective, the portfolio has an estimated Beta of <b>{:.2f}</b>, indicating a <b>{}</b> risk level with an overall <b>{}</b> style.
    Current defensive assets (cash + bonds) account for {:.1f}%, indicating a {}.
    </p>
    </div>
    """.format(
        format_cad(total_mv),
        performance_desc,
        "#10b981" if total_return > 0 else "#ef4444",
        format_cad(total_pnl),
        total_return * 100,
        benchmark_comparison,
        beta,
        risk_desc,
        risk_style,
        overview['defensive_share'] * 100,
        "conservative allocation" if overview['defensive_share'] > 0.25 else "balanced allocation"
    ), unsafe_allow_html=True)

    # Key metrics row
    st.markdown("""
    <div class='metric-row'>
        <div class='metric-item'>
            <div class='label'>Total Value</div>
            <div class='value'>{}</div>
        </div>
        <div class='metric-item'>
            <div class='label'>Total P&L</div>
            <div class='value {}'>{}</div>
        </div>
        <div class='metric-item'>
            <div class='label'>Return</div>
            <div class='value {}'>{}</div>
        </div>
        <div class='metric-item'>
            <div class='label'>Positions</div>
            <div class='value'>{}</div>
        </div>
        <div class='metric-item'>
            <div class='label'>Portfolio Beta</div>
            <div class='value'>{:.2f}</div>
        </div>
    </div>
    """.format(
        format_cad(total_mv),
        'positive' if total_pnl > 0 else 'negative',
        format_cad(total_pnl),
        'positive' if total_return > 0 else 'negative',
        format_pct_signed(total_return),
        len(positions),
        beta
    ), unsafe_allow_html=True)

    # Findings section
    if findings:
        st.markdown("<h4>🎯 Key Findings</h4>", unsafe_allow_html=True)
        findings_html = "<ul class='findings-list'>"
        for f in findings:
            findings_html += f"<li class='{f['type']}'>{f['text']}</li>"
        findings_html += "</ul>"
        st.markdown(findings_html, unsafe_allow_html=True)

    # Concerns section
    if concerns:
        st.markdown("<h4>⚠️ Areas of Concern</h4>", unsafe_allow_html=True)
        concerns_html = "<ul class='findings-list'>"
        for c in concerns:
            concerns_html += f"<li class='{c['type']}'>{c['text']}</li>"
        concerns_html += "</ul>"
        st.markdown(concerns_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_holdings_overview(views: dict, positions: pd.DataFrame) -> None:
    """Render Holdings Overview section"""
    overview = views['overview']
    top_holdings = views['top_holdings']

    total_positions = len(positions)
    accounts = positions['account_type'].nunique()
    top5_share = overview['top5_share']

    # Analyze position structure
    etf_count = len(positions[~positions['index_bucket'].isin(['INDIVIDUAL_STOCK', 'CASH_DEFENSIVE'])])
    stock_count = len(positions[positions['index_bucket'] == 'INDIVIDUAL_STOCK'])

    largest_symbol = top_holdings.iloc[0]['symbol'] if not top_holdings.empty else "N/A"
    largest_weight = top_holdings.iloc[0]['weight'] if not top_holdings.empty else 0
    largest_bucket = display_bucket_name(top_holdings.iloc[0]['index_bucket']) if not top_holdings.empty else "N/A"

    st.markdown("""
    <div class='report-section'>
    <h2>📊 Holdings Overview</h2>

    <p style='font-size: 15px; line-height: 1.7;'>
    Your portfolio currently holds <b>{}</b> positions across {} account(s).
    The structure is primarily <b>{}</b>, comprising {} ETFs/funds and {} individual stocks.
    </p>

    <p style='font-size: 15px; line-height: 1.7;'>
    Top 10 holdings represent <b>{:.1%}</b> of total assets, indicating {} concentration.
    The largest single position is <b>{}</b> ({}) at {:.1f}%, which is {} the concentration risk threshold.
    </p>
    """.format(
        total_positions,
        accounts,
        "index funds" if etf_count > stock_count else "individual stocks",
        etf_count,
        stock_count,
        top5_share,
        "moderate" if top5_share < 0.4 else "elevated" if top5_share < 0.5 else "high",
        largest_symbol,
        largest_bucket,
        largest_weight * 100,
        "below" if largest_weight < 0.10 else "approaching" if largest_weight < 0.15 else "exceeds"
    ), unsafe_allow_html=True)

    # Show top holdings table
    st.markdown("<h4>Top 10 Holdings</h4>", unsafe_allow_html=True)
    display_cols = ['symbol', 'index_bucket', 'market_value', 'weight', 'unrealized_pnl', 'return_pct']
    display_df = top_holdings.head(10)[display_cols].copy()
    display_df['index_bucket'] = display_df['index_bucket'].apply(display_bucket_name)
    display_df['weight'] = display_df['weight'].apply(lambda x: f"{x*100:.1f}%")
    display_df['return_pct'] = display_df['return_pct'].apply(lambda x: f"{x*100:+.1f}%")

    st.dataframe(
        display_df.rename(columns={
            'symbol': 'Symbol',
            'index_bucket': 'Category',
            'market_value': 'Market Value',
            'weight': 'Weight',
            'unrealized_pnl': 'P&L',
            'return_pct': 'Return'
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("</div>", unsafe_allow_html=True)


def render_asset_allocation(views: dict) -> None:
    """Render Asset Allocation section"""
    bucket_alloc = views['bucket_alloc']
    rebalancing = views['rebalancing']

    if bucket_alloc.empty:
        st.info("No allocation data available")
        return

    # Find significant drifts
    significant_drifts = bucket_alloc[bucket_alloc['drift'].abs() >= 0.05].sort_values('drift', key=abs, ascending=False)

    # Determine allocation style
    equity_weight = bucket_alloc[bucket_alloc['index_bucket'].isin(['S&P_500', 'NASDAQ_100', 'US_TOTAL_MARKET', 'GLOBAL_EQUITY', 'INDIVIDUAL_STOCK'])]['current_weight'].sum()

    if equity_weight > 0.80:
        style_desc = "Growth-Oriented"
        style_detail = "Higher equity allocation suitable for investors seeking capital appreciation"
    elif equity_weight > 0.60:
        style_desc = "Balanced"
        style_detail = "Balanced equity-bond allocation providing both growth and stability"
    else:
        style_desc = "Defensive"
        style_detail = "Higher defensive asset allocation suitable for risk-averse investors"

    st.markdown("""
    <div class='report-section'>
    <h2>📈 Asset Allocation Analysis</h2>

    <p style='font-size: 15px; line-height: 1.7;'>
    Current allocation shows <b>{}</b> characteristics. {}
    Equity assets (including individual stocks) represent approximately {:.1f}%, with defensive assets at {:.1f}%.
    </p>
    """.format(
        style_desc,
        style_detail,
        equity_weight * 100,
        bucket_alloc[bucket_alloc['index_bucket'].isin(['CASH_DEFENSIVE', 'CAD_BONDS'])]['current_weight'].sum() * 100
    ), unsafe_allow_html=True)

    # Show allocation vs target
    st.markdown("<h4>Current vs Target Allocation</h4>", unsafe_allow_html=True)
    alloc_display = bucket_alloc[['index_bucket', 'current_weight', 'target_weight', 'drift']].copy()
    alloc_display['index_bucket'] = alloc_display['index_bucket'].apply(display_bucket_name)
    alloc_display['current_weight'] = alloc_display['current_weight'].apply(lambda x: f"{x*100:.1f}%")
    alloc_display['target_weight'] = alloc_display['target_weight'].apply(lambda x: f"{x*100:.1f}%")
    alloc_display['drift'] = alloc_display['drift'].apply(lambda x: f"{x*100:+.1f}%")

    st.dataframe(
        alloc_display.rename(columns={
            'index_bucket': 'Asset Class',
            'current_weight': 'Current',
            'target_weight': 'Target',
            'drift': 'Drift'
        }),
        use_container_width=True,
        hide_index=True
    )

    # Drift analysis
    if not significant_drifts.empty:
        st.markdown("<h4>⚠️ Significant Allocation Drifts</h4>", unsafe_allow_html=True)
        for _, row in significant_drifts.head(3).iterrows():
            direction = "Overweight" if row['drift'] > 0 else "Underweight"
            st.markdown(f"""
            <div class='warning-box' style='margin: 8px 0;'>
                <b>{display_bucket_name(row['index_bucket'])}</b>: {direction} {abs(row['drift'])*100:.1f}%
                <br><span style='font-size: 13px;'>Current {row['current_weight']*100:.1f}% vs Target {row['target_weight']*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

    # Rebalancing recommendations
    if not rebalancing.empty:
        st.markdown("<h4>🎯 Rebalancing Recommendations</h4>", unsafe_allow_html=True)
        st.markdown("<p>Based on current drift levels, consider the following actions:</p>", unsafe_allow_html=True)

        for _, row in rebalancing.head(5).iterrows():
            priority_emoji = {"Immediate": "🚨", "High": "⚠️", "Medium": "📌", "Low": "💡"}.get(row['priority'], "💡")
            st.markdown(f"""
            <div style='padding: 8px 0; border-bottom: 1px solid #f3f4f6;'>
                {priority_emoji} <b>{row['action']}</b> {display_bucket_name(row['index_bucket'])}:
                Drift of {row['drift']:+.1%}, recommend trading approximately {format_cad(row['trade_value_cad'])}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_diversification_analysis(views: dict, overlap: pd.DataFrame) -> None:
    """Render Diversification Analysis section"""
    overview = views['overview']
    hhi = overview['hhi']
    top5_share = overview['top5_share']

    # HHI interpretation
    if hhi < 1000:
        hhi_status = "Well Diversified"
        hhi_desc = "Position distribution is relatively uniform with limited single-position impact. This indicates good risk diversification characteristics in your portfolio."
        hhi_class = "success-box"
    elif hhi < 1800:
        hhi_status = "Moderately Concentrated"
        hhi_desc = "Some concentration exists with higher weights in top positions, but within acceptable range. Monitor changes in top holdings."
        hhi_class = "insight-box"
    else:
        hhi_status = "Highly Concentrated"
        hhi_desc = "Portfolio concentration is high with few positions having significant impact. Consider diversifying to reduce single-position risk."
        hhi_class = "warning-box"

    st.markdown("""
    <div class='report-section'>
    <h2>🌐 Diversification Analysis</h2>

    <h4>Concentration Metrics</h4>

    <div class='{}'>
        <p style='margin: 0;'><b>Herfindahl-Hirschman Index (HHI): {:.0f}</b> — {}</p>
        <p style='margin: 8px 0 0 0; font-size: 14px;'>{}</p>
    </div>

    <p style='font-size: 15px; line-height: 1.7; margin-top: 16px;'>
    Top 5 holdings concentration is <b>{:.1%}</b>, indicating {}.
    According to modern portfolio theory, holding 15-30 stocks can eliminate approximately 90% of unsystematic risk.
    Your current position count of {} {} diversification requirements.
    </p>
    """.format(
        hhi_class,
        hhi,
        hhi_status,
        hhi_desc,
        top5_share,
        "good diversification" if top5_share < 0.35 else "moderate concentration" if top5_share < 0.45 else "elevated concentration",
        len(views['top_holdings']),
        "meets" if len(views['top_holdings']) >= 15 else "does not meet"
    ), unsafe_allow_html=True)

    # Overlap analysis
    if not overlap.empty:
        st.markdown("<h4>🔍 Holdings Overlap Analysis</h4>", unsafe_allow_html=True)

        high_overlap = overlap[overlap['overlap_flag'] == 'High']
        medium_overlap = overlap[overlap['overlap_flag'] == 'Medium']

        if not high_overlap.empty:
            st.markdown("<p><b>The following categories have high overlap risk - consider consolidating:</b></p>", unsafe_allow_html=True)
            for _, row in high_overlap.iterrows():
                member_tickers = _coerce_member_tickers(row.get("member_tickers"))
                tickers = ", ".join(member_tickers[:5]) if member_tickers else "N/A"
                if len(member_tickers) > 5:
                    tickers += f" and {len(member_tickers)-5} more"
                st.markdown(f"""
                <div class='danger-box' style='margin: 8px 0;'>
                    <b>{display_bucket_name(row['index_bucket'])}</b>: {row['total_weight_pct']*100:.1f}% weight
                    <br>Holdings: {tickers}
                    <br><span style='font-size: 13px;'>💡 Recommendation: Keep the one with lowest expense ratio, sell duplicate positions</span>
                </div>
                """, unsafe_allow_html=True)

        if not medium_overlap.empty:
            st.markdown("<p><b>The following categories have moderate overlap - monitor:</b></p>", unsafe_allow_html=True)
            for _, row in medium_overlap.iterrows():
                member_tickers = _coerce_member_tickers(row.get("member_tickers"))
                st.markdown(f"""
                <div class='insight-box' style='margin: 8px 0;'>
                    {display_bucket_name(row['index_bucket'])}: {row['total_weight_pct']*100:.1f}% weight, {len(member_tickers)} positions
                </div>
                """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_risk_assessment(views: dict) -> None:
    """Render Risk Assessment section"""
    overview = views['overview']

    beta = overview['estimated_beta']
    volatility_band = overview['volatility_band']
    growth_share = overview['growth_share']

    # Beta interpretation
    if beta < 0.8:
        beta_desc = "Portfolio volatility is lower than the market. During market downturns, declines may be smaller, but gains during rallies may also be limited."
    elif beta < 1.0:
        beta_desc = "Portfolio volatility is in line with the market, tracking market movements closely."
    elif beta < 1.2:
        beta_desc = "Portfolio volatility is slightly higher than the market. Gains may be larger during rallies, but losses during downturns may also be greater."
    else:
        beta_desc = "Portfolio volatility is significantly higher than the market, indicating an aggressive allocation that requires tolerance for substantial fluctuations."

    st.markdown("""
    <div class='report-section'>
    <h2>⚠️ Risk Assessment</h2>

    <div class='metric-row'>
        <div class='metric-item'>
            <div class='label'>Estimated Beta</div>
            <div class='value'>{:.2f}</div>
        </div>
        <div class='metric-item'>
            <div class='label'>Risk Level</div>
            <div class='value'>{}</div>
        </div>
        <div class='metric-item'>
            <div class='label'>Growth Assets</div>
            <div class='value'>{}</div>
        </div>
    </div>

    <div class='insight-box' style='margin-top: 20px;'>
        <p style='margin: 0;'><b>Beta Coefficient Interpretation</b></p>
        <p style='margin: 8px 0 0 0;'>{}</p>
    </div>

    <h4>Risk Composition Analysis</h4>
    <p style='font-size: 15px; line-height: 1.7;'>
    Your portfolio risk primarily comes from the following sources:
    </p>
    <ul style='line-height: 1.8;'>
        <li><b>Systematic Risk (Market Risk):</b>{}</li>
        <li><b>Concentration Risk:</b>{}</li>
        <li><b>Style Risk:</b>{}</li>
    </ul>
    """.format(
        beta,
        volatility_band,
        f"{growth_share*100:.0f}%",
        beta_desc,
        f"Beta of {beta:.2f}, portfolio expected to move {beta*10:.1f}% when market moves 10%",
        "Top 5 holdings have elevated weights" if overview['top5_share'] > 0.4 else "Positions are well diversified with low concentration risk",
        "Growth-oriented" if growth_share > 0.5 else "Balanced style" if growth_share > 0.3 else "Defensive style"
    ), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_position_analysis(views: dict) -> None:
    """Render Position Analysis section with detailed commentary"""
    top_holdings = views['top_holdings']

    st.markdown("""
    <div class='report-section'>
    <h2>🔍 Position Analysis & Recommendations</h2>

    <p style='font-size: 15px; line-height: 1.7;'>
    Below is detailed analysis of your top 15 holdings. Each position is evaluated across:
    Investment thesis status, position weight appropriateness, and recommended action (HOLD/ADD/TRIM/SELL).
    </p>
    """, unsafe_allow_html=True)

    # Action summary
    action_counts = top_holdings.head(15)['action'].value_counts()
    st.markdown("<h4>📊 Action Summary</h4>", unsafe_allow_html=True)
    action_html = "<div style='display: flex; gap: 16px; margin: 12px 0;'>"
    for action, count in action_counts.items():
        color = {"HOLD": "#6b7280", "ADD": "#10b981", "TRIM": "#f59e0b", "SELL": "#ef4444"}.get(action, "#6b7280")
        action_html += f"<span style='background: {color}; color: white; padding: 4px 12px; border-radius: 12px; font-size: 13px; font-weight: 600;'>{action}: {count}</span>"
    action_html += "</div>"
    st.markdown(action_html, unsafe_allow_html=True)

    # Individual position cards
    st.markdown("<h4>📋 Position Details</h4>", unsafe_allow_html=True)

    for _, row in top_holdings.head(15).iterrows():
        action_class = f"action-{row['action'].lower()}"

        st.markdown(f"""
        <div class='position-card'>
            <div class='position-header'>
                <span class='symbol'>{row['symbol']}</span>
                <span class='bucket'>{display_bucket_name(row['index_bucket'])}</span>
                <span class='action {action_class}'>{row['action']}</span>
            </div>
            <div class='position-body'>
                <div class='metrics'>
                    <b>Weight:</b> {row['weight']*100:.1f}% |
                    <b>Value:</b> {format_cad(row['market_value'])} |
                    <b>P&L:</b> <span style='color: {"#10b981" if row['unrealized_pnl'] > 0 else "#ef4444"};'>{format_cad(row['unrealized_pnl'])} ({row['return_pct']*100:+.1f}%)</span> |
                    <b>Thesis:</b> {row['thesis_status']}
                </div>
                <div class='rationale'>
                    <i>Rationale: {row['rationale']}</i>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_rebalancing_plan(views: dict) -> None:
    """Render Rebalancing and Action Items section"""
    action_items = views['action_items']

    st.markdown("""
    <div class='report-section'>
    <h2>🎯 Rebalancing Plan & Action Items</h2>

    <p style='font-size: 15px; line-height: 1.7;'>
    Based on the above analysis, we have developed the following rebalancing plan. We recommend executing these actions in order of priority:
    </p>
    """, unsafe_allow_html=True)

    if action_items.empty or (len(action_items) == 1 and action_items.iloc[0]['item'] == 'No major drift detected'):
        st.markdown("""
        <div class='success-box'>
            <p style='margin: 0;'><b>✅ Portfolio in Good Standing</b></p>
            <p style='margin: 8px 0 0 0;'>No significant drifts detected. Recommend maintaining current allocation with periodic reviews.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Group by priority
        priorities = ['Immediate', 'High', 'Medium', 'Low']
        priority_emojis = {'Immediate': '🚨', 'High': '⚠️', 'Medium': '📌', 'Low': '💡'}
        priority_titles = {'Immediate': 'Immediate Action', 'High': 'High Priority', 'Medium': 'Medium Priority', 'Low': 'Low Priority'}

        for priority in priorities:
            items = action_items[action_items['priority'] == priority]
            if not items.empty:
                st.markdown(f"<h4>{priority_emojis.get(priority, '💡')} {priority_titles.get(priority, priority)}</h4>", unsafe_allow_html=True)

                for _, row in items.iterrows():
                    st.markdown(f"""
                    <div style='background: #f9fafb; border-left: 3px solid {"#ef4444" if priority == "Immediate" else "#f59e0b" if priority == "High" else "#3b82f6"}; padding: 12px 16px; margin: 8px 0; border-radius: 0 8px 8px 0;'>
                        <b>{row['item']}</b>
                        <p style='margin: 4px 0 0 0; font-size: 14px; color: #6b7280;'>{row['detail']}</p>
                    </div>
                    """, unsafe_allow_html=True)

    # Summary table
    st.markdown("<h4>📋 Complete Action List</h4>", unsafe_allow_html=True)
    st.dataframe(
        action_items.rename(columns={
            'priority': 'Priority',
            'item': 'Action Item',
            'detail': 'Details'
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("</div>", unsafe_allow_html=True)


def render_full_report(views: dict, positions: pd.DataFrame, overlap: pd.DataFrame) -> None:
    """Render the complete narrative report"""
    # Add CSS
    st.markdown(REPORT_CSS, unsafe_allow_html=True)

    # Render all sections
    render_executive_summary(views, positions, overlap)
    render_holdings_overview(views, positions)
    render_asset_allocation(views)
    render_diversification_analysis(views, overlap)
    render_risk_assessment(views)
    render_position_analysis(views)
    render_rebalancing_plan(views)
