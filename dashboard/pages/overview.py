"""Overview page — KPIs, class distribution, probability histogram."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.metric_cards import kpi_row, page_header, section_header


def render(metrics, df, plotly_dark: dict):
    page_header(
        "Portfolio Overview",
        subtitle="Real-time fraud risk summary across the validation portfolio",
        badge_text="Live",
        badge_color="green",
    )

    # ── KPI row ────────────────────────────────────────────────────────────
    n_total = len(df)
    n_fraud = int(df["actual_fraud"].sum())
    n_legit = n_total - n_fraud
    fraud_rate = 100 * n_fraud / max(n_total, 1)
    auc = metrics.get("auc", 0)
    f1 = metrics.get("f1", 0)
    precision = metrics.get("precision", 0)
    recall = metrics.get("recall", 0)

    kpi_row([
        {"label": "Total Transactions", "value": f"{n_total:,}",
         "sub": f"{n_legit:,} legitimate", "color": "blue"},
        {"label": "Fraud Detected",     "value": f"{n_fraud:,}",
         "sub": f"{fraud_rate:.2f}% fraud rate", "color": "red"},
        {"label": "ROC-AUC",            "value": f"{auc:.4f}",
         "sub": "Excellent (> 0.95)", "color": "green"},
        {"label": "F1 Score",           "value": f"{f1:.4f}",
         "sub": f"Precision {precision:.3f}", "color": "purple"},
        {"label": "Recall",             "value": f"{recall:.4f}",
         "sub": "Fraud capture rate", "color": "amber"},
    ])

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    # ── Charts row ─────────────────────────────────────────────────────────
    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        section_header("Transaction Class Split", "Legitimate vs fraudulent breakdown")
        counts = df["actual_fraud"].value_counts().rename({0: "Legitimate", 1: "Fraud"})
        fig = go.Figure(go.Pie(
            values=counts.values,
            labels=counts.index,
            hole=0.62,
            marker=dict(
                colors=["#22C55E", "#EF4444"],
                line=dict(color="#020617", width=3),
            ),
            textinfo="label+percent",
            textfont=dict(size=13, family="IBM Plex Sans", color="#F8FAFC"),
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
        ))
        fig.update_layout(
            **plotly_dark,
            showlegend=False,
            annotations=[dict(
                text=f"<b>{fraud_rate:.1f}%</b><br><span style='font-size:10px'>Fraud</span>",
                x=0.5, y=0.5, font_size=18, showarrow=False,
                font=dict(color="#F8FAFC", family="IBM Plex Sans"),
            )],
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        section_header("Score Distribution", "Fraud probability by transaction class")
        df_plot = df.copy()
        df_plot["Class"] = df_plot["actual_fraud"].map({0: "Legitimate", 1: "Fraud"})
        fig = px.histogram(
            df_plot, x="fraud_probability", color="Class", nbins=60,
            barmode="overlay", opacity=0.75,
            color_discrete_map={"Legitimate": "#22C55E", "Fraud": "#EF4444"},
        )
        threshold = float(metrics.get("threshold", 0.5))
        fig.add_vline(
            x=threshold, line_dash="dash", line_color="#F59E0B", line_width=2,
            annotation_text=f"  Threshold {threshold:.2f}",
            annotation_font=dict(color="#F59E0B", size=11),
        )
        fig.update_layout(
            **plotly_dark,
            xaxis_title="Fraud Probability Score",
            yaxis_title="Transaction Count",
            legend=dict(orientation="h", yanchor="top", y=1.08, xanchor="right", x=1,
                        font=dict(size=12)),
            height=320,
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    # ── Full metrics table ─────────────────────────────────────────────────
    section_header("Evaluation Metrics", "Complete validation set performance summary")
    metric_df = pd.DataFrame([
        {"Metric": k.upper().replace("_", " "), "Value": round(v, 4) if isinstance(v, float) else v}
        for k, v in metrics.items()
    ])
    st.dataframe(
        metric_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Metric": st.column_config.TextColumn("Metric", width="medium"),
            "Value": st.column_config.NumberColumn("Value", format="%.4f", width="small"),
        },
    )
