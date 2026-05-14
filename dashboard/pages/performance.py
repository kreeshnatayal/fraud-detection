"""Model Performance page — ROC/PR curves, confusion matrix, threshold explorer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from dashboard.components.metric_cards import kpi_row, page_header, section_header


def render(metrics, df, model_metrics, plotly_dark: dict):
    page_header(
        "Model Performance",
        subtitle="ROC/PR curves, confusion matrix, threshold analysis, and per-model comparison",
        badge_text="Stacking Ensemble",
        badge_color="purple",
    )

    y_true = df["actual_fraud"].values
    y_score = df["fraud_probability"].values
    threshold = float(metrics.get("threshold", 0.5))
    y_pred = (y_score >= threshold).astype(int)

    # ── Top KPIs ───────────────────────────────────────────────────────────
    auc = metrics.get("auc", 0)
    ap  = metrics.get("ap", 0)
    f1  = metrics.get("f1", 0)
    cm  = confusion_matrix(y_true, y_pred)
    tn, fp_cnt, fn_cnt, tp_cnt = cm.ravel()

    kpi_row([
        {"label": "ROC-AUC",     "value": f"{auc:.4f}",  "sub": "Stacking ensemble",     "color": "green"},
        {"label": "Avg Precision","value": f"{ap:.4f}",   "sub": "PR curve area",         "color": "blue"},
        {"label": "F1 Score",    "value": f"{f1:.4f}",   "sub": "Harmonic mean P/R",     "color": "purple"},
        {"label": "True Positives","value": f"{tp_cnt:,}","sub": f"@ threshold {threshold:.2f}", "color": "green"},
        {"label": "False Positives","value": f"{fp_cnt:,}","sub": f"{fp_cnt/(max(tn+fp_cnt,1))*100:.1f}% FPR", "color": "amber"},
    ])

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    # ── ROC + PR Curves ────────────────────────────────────────────────────
    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        section_header("ROC Curve", f"AUC = {auc:.4f} · Gini = {2*auc-1:.4f}")
        fpr_arr, tpr_arr, _ = roc_curve(y_true, y_score)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr_arr, y=tpr_arr, mode="lines",
            name=f"Ensemble AUC={auc:.4f}",
            line=dict(color="#22C55E", width=2.5),
            fill="tozeroy", fillcolor="rgba(34,197,94,0.06)",
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random Baseline",
            line=dict(dash="dot", color="#475569", width=1.5),
        ))
        fig.update_layout(
            **plotly_dark,
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            legend=dict(orientation="h", y=1.08, font=dict(size=11)),
            height=340,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        section_header("Precision-Recall Curve", f"Average Precision = {ap:.4f}")
        prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_score)
        baseline = y_true.mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rec_arr, y=prec_arr, mode="lines",
            name=f"Ensemble AP={ap:.4f}",
            line=dict(color="#3B82F6", width=2.5),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.06)",
            hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
        ))
        fig.add_hline(
            y=baseline, line_dash="dot", line_color="#475569", line_width=1.5,
            annotation_text=f"  Baseline ({baseline:.3f})",
            annotation_font=dict(color="#64748B", size=10),
        )
        fig.update_layout(
            **plotly_dark,
            xaxis_title="Recall", yaxis_title="Precision",
            legend=dict(orientation="h", y=1.08, font=dict(size=11)),
            height=340,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Confusion Matrix ───────────────────────────────────────────────────
    section_header("Confusion Matrix", f"Decision threshold = {threshold:.2f}")
    col_cm, col_info = st.columns([1, 1], gap="large")

    with col_cm:
        cm_labels = ["Legitimate", "Fraud"]
        fig = px.imshow(
            cm, text_auto=True,
            x=[f"Predicted {l}" for l in cm_labels],
            y=[f"Actual {l}" for l in cm_labels],
            color_continuous_scale=[[0, "#0F172A"], [0.5, "#1E3A5F"], [1, "#22C55E"]],
            labels=dict(color="Count"),
        )
        fig.update_traces(textfont=dict(size=20, color="#F8FAFC", family="IBM Plex Sans"))
        fig.update_layout(**plotly_dark)
        fig.update_layout(
            coloraxis_showscale=False,
            height=300,
            xaxis=dict(side="top", tickfont=dict(size=12)),
            yaxis=dict(tickfont=dict(size=12)),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        recall_val = tp_cnt / max(tp_cnt + fn_cnt, 1)
        precision_val = tp_cnt / max(tp_cnt + fp_cnt, 1)
        fpr_val = fp_cnt / max(fp_cnt + tn, 1)
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        kpi_row([
            {"label": "Recall",    "value": f"{recall_val:.3f}",    "sub": f"{tp_cnt} / {tp_cnt+fn_cnt} fraud caught", "color": "green"},
            {"label": "Precision", "value": f"{precision_val:.3f}", "sub": f"{fp_cnt} false alarms", "color": "blue"},
        ])
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        kpi_row([
            {"label": "FP Rate",   "value": f"{fpr_val:.3f}", "sub": "Legit flagged as fraud", "color": "amber"},
            {"label": "Specificity","value": f"{1-fpr_val:.3f}", "sub": "Legit correctly passed", "color": "purple"},
        ])

    # ── Threshold Explorer ─────────────────────────────────────────────────
    section_header("Cost-Benefit Threshold Explorer", "Adjust threshold to see business impact")

    exp_cols = st.columns(3, gap="medium")
    with exp_cols[0]:
        avg_val = st.number_input("Avg fraud value ($)", 50.0, 5000.0, 200.0, 50.0, key="tb_avg")
    with exp_cols[1]:
        invest_c = st.number_input("Investigation cost / alert ($)", 0.0, 50.0, 5.0, 1.0, key="tb_inv")
    with exp_cols[2]:
        fp_loss = st.slider("FP revenue loss rate", 0.0, 1.0, 0.85, key="tb_fp")

    new_thresh = st.slider(
        "Decision threshold", 0.01, 0.99, threshold, 0.01,
        help="Move threshold to balance fraud recall vs false positive rate",
    )

    def _calc(t):
        yp = (y_score >= t).astype(int)
        tp_t = int(((y_true == 1) & (yp == 1)).sum())
        fp_t = int(((y_true == 0) & (yp == 1)).sum())
        fn_t = int(((y_true == 1) & (yp == 0)).sum())
        cost = tp_t * avg_val - fp_t * avg_val * fp_loss - (tp_t + fp_t) * invest_c
        return tp_t, fp_t, fn_t, cost

    tp_new, fp_new, fn_new, cost_new = _calc(new_thresh)
    tp_cur, fp_cur, fn_cur, cost_cur = _calc(threshold)
    d_tp = tp_new - tp_cur; d_fp = fp_new - fp_cur
    d_fn = fn_new - fn_cur; d_cost = cost_new - cost_cur

    kpi_row([
        {"label": "Frauds Caught",   "value": f"{tp_new:,}", "sub": f"{d_tp:+d} vs baseline", "color": "green" if d_tp >= 0 else "red"},
        {"label": "Frauds Missed",   "value": f"{fn_new:,}", "sub": f"{d_fn:+d} vs baseline", "color": "red" if d_fn > 0 else "green"},
        {"label": "Legit Blocked",   "value": f"{fp_new:,}", "sub": f"{d_fp:+d} vs baseline", "color": "amber"},
        {"label": "Net Cost Change", "value": f"${d_cost:+,.0f}", "sub": f"Net benefit ${cost_new:,.0f}", "color": "green" if d_cost >= 0 else "red"},
    ])

    if abs(new_thresh - threshold) > 0.001:
        direction = "lower" if new_thresh < threshold else "higher"
        effect = f"{abs(d_tp)} more frauds caught" if d_tp >= 0 else f"{abs(d_tp)} fewer frauds caught"
        st.info(f"Threshold {threshold:.2f} → {new_thresh:.2f} ({direction}): {effect}, {abs(d_fp)} {'more' if d_fp >= 0 else 'fewer'} false alarms")

    # ── Per-model comparison ───────────────────────────────────────────────
    if model_metrics:
        rows = [
            {"Model": k, "AUC": m.get("auc", 0), "F1": m.get("f1", 0),
             "Precision": m.get("precision", 0), "Recall": m.get("recall", 0),
             "AP": m.get("ap", 0)}
            for k, m in model_metrics.items()
            if isinstance(m, dict) and "auc" in m
        ]
        if rows:
            section_header("Per-Model Comparison", "Individual model performance before stacking")
            comp_df = pd.DataFrame(rows).sort_values("AUC", ascending=False)

            col_bar, col_tbl = st.columns([3, 2], gap="large")
            with col_bar:
                fig = go.Figure()
                metrics_to_plot = ["AUC", "F1", "Precision", "Recall"]
                colors = ["#22C55E", "#3B82F6", "#F59E0B", "#8B5CF6"]
                for metric_name, color in zip(metrics_to_plot, colors):
                    fig.add_trace(go.Bar(
                        name=metric_name, x=comp_df["Model"], y=comp_df[metric_name],
                        marker_color=color, opacity=0.85,
                        text=comp_df[metric_name].round(3),
                        textposition="outside",
                        textfont=dict(size=10, color="#94A3B8"),
                    ))
                fig.update_layout(**plotly_dark)
                fig.update_layout(
                    barmode="group", bargap=0.2, bargroupgap=0.05,
                    legend=dict(orientation="h", y=1.12, font=dict(size=11)),
                    yaxis=dict(range=[0, 1.08]),
                    height=340,
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_tbl:
                st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
                st.dataframe(
                    comp_df.set_index("Model").style.format("{:.4f}"),
                    use_container_width=True,
                )
