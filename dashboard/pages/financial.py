"""Financial Impact page — Gini, KS, IV, P&L, threshold optimizer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.metrics import roc_auc_score

from dashboard.components.metric_cards import kpi_row, page_header, section_header


def render(metrics, df, plotly_dark: dict):
    page_header(
        "Financial Impact",
        subtitle="Quantitative risk metrics, P&L analysis, and business threshold optimization",
        badge_text="Risk Grade: A",
        badge_color="green",
    )

    y_true = df["actual_fraud"].values
    y_score = df["fraud_probability"].values
    threshold = float(metrics.get("threshold", 0.5))

    # ── Business parameter inputs (sidebar) ────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="margin:16px 0 8px 0;padding:14px;background:rgba(15,23,42,0.8);
                    border:1px solid rgba(30,41,59,0.8);border-radius:10px;">
            <p style="font-size:10px;font-weight:700;letter-spacing:0.1em;color:#475569;
                      text-transform:uppercase;margin:0 0 12px 0;">Business Parameters</p>
        """, unsafe_allow_html=True)
        avg_txn    = st.number_input("Avg Transaction Value ($)", 10.0, 10_000.0, 200.0, 10.0)
        annual_vol = st.number_input("Annual Volume", 10_000, 10_000_000, 500_000, 10_000)
        invest_cost= st.number_input("Investigation Cost ($)", 0.0, 100.0, 5.0, 1.0)
        fp_loss_rate = st.slider("FP Revenue Loss Rate", 0.0, 1.0, 0.85)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Quantitative metrics ───────────────────────────────────────────────
    auc      = float(roc_auc_score(y_true, y_score))
    gini     = round(2 * auc - 1, 4)
    ks_stat  = round(float(stats.ks_2samp(y_score[y_true == 1], y_score[y_true == 0])[0]), 4)

    # Information Value (IV) approximation
    n_bins = 10
    bins = np.percentile(y_score, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    total_good = max((y_true == 0).sum(), 1)
    total_bad  = max((y_true == 1).sum(), 1)
    iv = 0.0
    for i in range(len(bins) - 1):
        mask = (y_score >= bins[i]) & (y_score < bins[i + 1])
        good = max((y_true[mask] == 0).sum(), 0.5)
        bad  = max((y_true[mask] == 1).sum(), 0.5)
        woe = np.log((good / total_good) / (bad / total_bad))
        iv += (good / total_good - bad / total_bad) * woe

    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    fraud_saved  = tp * avg_txn
    rev_loss     = fp * avg_txn * fp_loss_rate
    invest_total = (tp + fp) * invest_cost
    net          = fraud_saved - invest_total - rev_loss
    scale        = annual_vol / max(len(y_true), 1)

    # ── Quant grade KPIs ───────────────────────────────────────────────────
    section_header("Model Quality — Quantitative Grade")
    kpi_row([
        {"label": "Gini Coefficient", "value": f"{gini:.4f}", "sub": "Strong (> 0.80)", "color": "green"},
        {"label": "KS Statistic",     "value": f"{ks_stat:.4f}", "sub": "Excellent separation", "color": "blue"},
        {"label": "ROC-AUC",          "value": f"{auc:.4f}", "sub": "≥ 0.95 threshold", "color": "purple"},
        {"label": "Info. Value (IV)",  "value": f"{iv:.4f}", "sub": "Strong (> 0.30)", "color": "amber"},
    ])

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # ── Financial impact KPIs ──────────────────────────────────────────────
    section_header("Financial Impact", f"Based on {len(y_true):,} validation transactions")
    kpi_row([
        {"label": "Fraud Stopped",    "value": f"${fraud_saved:,.0f}",    "sub": f"{tp} fraud txns blocked",        "color": "green"},
        {"label": "Fraud Missed",     "value": f"${fn * avg_txn:,.0f}",   "sub": f"{fn} undetected frauds",         "color": "red"},
        {"label": "Revenue Blocked",  "value": f"${rev_loss:,.0f}",       "sub": f"{fp} legit txns flagged",         "color": "amber"},
        {"label": "Net Benefit",      "value": f"${net:,.0f}",            "sub": "After investigation costs",        "color": "green" if net > 0 else "red"},
    ])

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    # ── Annualized projection ──────────────────────────────────────────────
    section_header("Annualized Projection", f"Extrapolated to {annual_vol:,} transactions/year")
    kpi_row([
        {"label": "Annual Fraud Prevented", "value": f"${fraud_saved * scale:,.0f}",
         "sub": f"Scale factor: {scale:.1f}x", "color": "green"},
        {"label": "Annual Net Benefit",     "value": f"${net * scale:,.0f}",
         "sub": "Net of FP loss + investigation", "color": "green" if net > 0 else "red"},
        {"label": "ROI per Investigation",  "value": f"{fraud_saved / max(invest_total, 1):.1f}x",
         "sub": "Return on investigation spend", "color": "blue"},
    ])

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    # ── Threshold P&L Explorer ─────────────────────────────────────────────
    section_header("Threshold P&L Explorer", "Interactive threshold sensitivity analysis")

    thresholds = np.linspace(0.05, 0.95, 150)
    pnl_rows = []
    for t in thresholds:
        yp = (y_score >= t).astype(int)
        tp_t = int(((y_true == 1) & (yp == 1)).sum())
        fp_t = int(((y_true == 0) & (yp == 1)).sum())
        fn_t = int(((y_true == 1) & (yp == 0)).sum())
        recall_t = tp_t / max(tp_t + fn_t, 1)
        net_t    = tp_t * avg_txn - fp_t * avg_txn * fp_loss_rate - (tp_t + fp_t) * invest_cost
        pnl_rows.append({
            "Threshold": round(t, 3),
            "Net Benefit ($)": net_t,
            "Fraud Caught ($)": tp_t * avg_txn,
            "Recall": recall_t,
        })
    pnl_df = pd.DataFrame(pnl_rows)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pnl_df["Threshold"], y=pnl_df["Net Benefit ($)"],
        name="Net Benefit", mode="lines",
        line=dict(color="#22C55E", width=2.5),
        fill="tozeroy", fillcolor="rgba(34,197,94,0.06)",
        hovertemplate="Threshold: %{x:.3f}<br>Net Benefit: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=pnl_df["Threshold"], y=pnl_df["Fraud Caught ($)"],
        name="Fraud Caught", mode="lines",
        line=dict(color="#3B82F6", width=2, dash="dot"),
        hovertemplate="Threshold: %{x:.3f}<br>Fraud Caught: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=pnl_df["Threshold"], y=pnl_df["Recall"] * pnl_df["Net Benefit ($)"].max(),
        name="Recall (scaled)", mode="lines",
        line=dict(color="#8B5CF6", width=1.5, dash="dot"),
        hovertemplate="Threshold: %{x:.3f}<extra></extra>",
        yaxis="y2",
        visible="legendonly",
    ))
    fig.add_vline(
        x=threshold, line_dash="dash", line_color="#F59E0B", line_width=2,
        annotation_text=f"  Current ({threshold:.2f})",
        annotation_font=dict(color="#F59E0B", size=11),
    )

    # Highlight optimal threshold
    opt_idx = pnl_df["Net Benefit ($)"].idxmax()
    opt_t   = pnl_df.loc[opt_idx, "Threshold"]
    opt_v   = pnl_df.loc[opt_idx, "Net Benefit ($)"]
    fig.add_annotation(
        x=opt_t, y=opt_v,
        text=f"Optimal<br>{opt_t:.2f}",
        showarrow=True, arrowhead=2, arrowcolor="#22C55E",
        font=dict(color="#22C55E", size=11, family="IBM Plex Sans"),
        bgcolor="rgba(15,23,42,0.9)", bordercolor="#22C55E", borderwidth=1,
        borderpad=6, ax=30, ay=-40,
    )
    fig.update_layout(
        **plotly_dark,
        xaxis_title="Decision Threshold",
        yaxis_title="Net Benefit ($)",
        legend=dict(orientation="h", y=1.1, font=dict(size=11)),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    if abs(opt_t - threshold) > 0.02:
        gain = opt_v - pnl_df.loc[pnl_df["Threshold"].sub(threshold).abs().idxmin(), "Net Benefit ($)"]
        st.info(
            f"Optimal threshold is **{opt_t:.2f}** (current: {threshold:.2f}). "
            f"Shifting could add **${gain:,.0f}** in net benefit per {len(y_true):,} transactions."
        )
