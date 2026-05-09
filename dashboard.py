"""
Streamlit dashboard for the fraud detection pipeline.

Run locally:
    streamlit run dashboard.py
    streamlit run dashboard.py -- --output ./output   # custom output dir

Demo mode (no pipeline required):
    python generate_sample_data.py   # run once to seed output/
    streamlit run dashboard.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# ── page config must be first ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/krishnatayal/fraud-detection",
        "Report a bug": "https://github.com/krishnatayal/fraud-detection/issues",
        "About": "AI-powered credit card fraud detection using LightGBM, XGBoost, CatBoost and graph-based features.",
    },
)

# ── optional heavy imports ─────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MPL_OK = True
except ImportError:
    MPL_OK = False

try:
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        confusion_matrix,
        precision_recall_curve,
        roc_curve,
    )
    SKL_OK = True
except ImportError:
    SKL_OK = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _output_dir() -> Path:
    """Resolve output directory from CLI arg or default."""
    idx = sys.argv.index("--") + 1 if "--" in sys.argv else None
    if idx:
        p = argparse.ArgumentParser()
        p.add_argument("--output", default="output")
        args, _ = p.parse_known_args(sys.argv[idx:])
        return Path(args.output)
    return Path("output")


OUTPUT = _output_dir()
DEMO_MODE = False


def _ensure_demo_data() -> None:
    """Auto-generate sample data if no pipeline output exists (for hosted demo)."""
    global DEMO_MODE
    required = [OUTPUT / "val_predictions.csv", OUTPUT / "evaluation_metrics.json"]
    if all(p.exists() for p in required):
        return
    OUTPUT.mkdir(exist_ok=True)
    try:
        result = subprocess.run(
            [sys.executable, "generate_sample_data.py"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            DEMO_MODE = True
            st.cache_data.clear()
    except Exception:
        pass


_ensure_demo_data()


@st.cache_data(ttl=60)
def load_predictions() -> Optional[pd.DataFrame]:
    path = OUTPUT / "val_predictions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    for col in ["fraud_probability", "predicted_fraud", "actual_fraud"]:
        if col not in df.columns:
            return None
    return df


@st.cache_data(ttl=60)
def load_metrics() -> Optional[dict]:
    path = OUTPUT / "evaluation_metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=60)
def load_model_metrics() -> Optional[dict]:
    path = OUTPUT / "model_metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=60)
def load_pipeline_summary() -> Optional[dict]:
    path = OUTPUT / "pipeline_summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _metric_card(col, label: str, value, delta=None, fmt=".4f"):
    disp = f"{value:{fmt}}" if isinstance(value, float) else str(value)
    col.metric(label, disp, delta=delta)


def _no_data_warning():
    st.warning(
        "No pipeline output found in `" + str(OUTPUT) + "`. "
        "Run the pipeline first:\n\n"
        "```bash\npython main.py\n```"
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Fraud Detection AI")
    if DEMO_MODE:
        st.info("Running in **Demo Mode** with synthetic data.", icon="🎭")
    else:
        st.success("Live pipeline data loaded.", icon="✅")
    st.caption(f"Output: `{OUTPUT}`")
    st.markdown("---")
    tab_names = ["Overview", "Model Performance", "Predictions", "Feature Analysis", "Pipeline Monitor"]
    page = st.radio("Navigate", tab_names)
    st.markdown("---")
    st.markdown(
        "**Tech Stack**\n"
        "- LightGBM / XGBoost / CatBoost\n"
        "- Stacking + Voting ensembles\n"
        "- NetworkX graph features\n"
        "- Node2Vec embeddings\n"
        "- SMOTE resampling"
    )
    st.markdown("---")
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()
    st.markdown(
        "<small>Built by Krishna Tayal • "
        "[GitHub](https://github.com/krishnatayal/fraud-detection)</small>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------

if page == "Overview":
    st.title("Fraud Detection AI — Overview")
    if DEMO_MODE:
        st.caption("Showing synthetic demo data. Connect your pipeline output to see real results.")

    metrics = load_metrics()
    df = load_predictions()

    if metrics is None or df is None:
        _no_data_warning()
        st.stop()

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    _metric_card(c1, "Total Transactions", len(df), fmt="d")
    n_fraud = int(df["actual_fraud"].sum())
    _metric_card(c2, "Actual Fraud", n_fraud, fmt="d")
    fraud_rate = 100 * n_fraud / max(len(df), 1)
    _metric_card(c3, "Fraud Rate", f"{fraud_rate:.2f}%")
    _metric_card(c4, "ROC-AUC", metrics.get("auc", 0))
    _metric_card(c5, "F1 Score", metrics.get("f1", 0))

    st.markdown("---")

    col_left, col_right = st.columns(2)

    # Fraud vs legit pie
    with col_left:
        st.subheader("Class Distribution")
        counts = df["actual_fraud"].value_counts().rename({0: "Legitimate", 1: "Fraud"})
        if PLOTLY_OK:
            fig = px.pie(values=counts.values, names=counts.index, color_discrete_sequence=["#2ecc71", "#e74c3c"])
            fig.update_layout(margin=dict(t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(counts)

    # Fraud probability distribution
    with col_right:
        st.subheader("Fraud Probability Distribution")
        if PLOTLY_OK:
            fig = px.histogram(
                df, x="fraud_probability", color=df["actual_fraud"].map({0: "Legitimate", 1: "Fraud"}),
                nbins=50, barmode="overlay", opacity=0.7,
                color_discrete_map={"Legitimate": "#2ecc71", "Fraud": "#e74c3c"},
                labels={"color": "Class"},
            )
            fig.update_layout(margin=dict(t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(df["fraud_probability"].round(2).value_counts().sort_index())

    # Summary metrics table
    st.subheader("Full Metrics")
    metric_df = pd.DataFrame([
        {"Metric": k, "Value": round(v, 4) if isinstance(v, float) else v}
        for k, v in metrics.items()
    ])
    st.dataframe(metric_df, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Model Performance
# ---------------------------------------------------------------------------

elif page == "Model Performance":
    st.title("Model Performance")

    metrics = load_metrics()
    df = load_predictions()
    model_metrics = load_model_metrics()

    if metrics is None or df is None:
        _no_data_warning()
        st.stop()

    y_true = df["actual_fraud"].values
    y_score = df["fraud_probability"].values
    threshold = metrics.get("threshold", 0.5)

    col_l, col_r = st.columns(2)

    # ROC curve
    with col_l:
        st.subheader("ROC Curve")
        if SKL_OK:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_val = metrics.get("auc", 0)
            if PLOTLY_OK:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={auc_val:.4f}", line=dict(color="#3498db", width=2)))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash", color="gray")))
                fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", margin=dict(t=10))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Install plotly and scikit-learn for ROC curve visualisation.")

    # PR curve
    with col_r:
        st.subheader("Precision-Recall Curve")
        if SKL_OK:
            prec, rec, _ = precision_recall_curve(y_true, y_score)
            ap_val = metrics.get("ap", 0)
            if PLOTLY_OK:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"AP={ap_val:.4f}", line=dict(color="#e67e22", width=2)))
                fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", margin=dict(t=10))
                st.plotly_chart(fig, use_container_width=True)

    # Confusion matrix
    st.subheader("Confusion Matrix")
    if SKL_OK:
        y_pred = (y_score >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        labels = ["Legitimate", "Fraud"]
        if PLOTLY_OK:
            fig = px.imshow(
                cm, text_auto=True, x=labels, y=labels,
                color_continuous_scale="Blues",
                labels=dict(x="Predicted", y="Actual"),
            )
            fig.update_layout(margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(pd.DataFrame(cm, index=labels, columns=labels))

    # Threshold slider
    st.subheader("Threshold Exploration")
    new_thresh = st.slider("Decision threshold", 0.01, 0.99, float(threshold), 0.01)
    y_pred_new = (y_score >= new_thresh).astype(int)
    from sklearn.metrics import f1_score, precision_score, recall_score
    tc1, tc2, tc3, tc4 = st.columns(4)
    tc1.metric("Precision", f"{precision_score(y_true, y_pred_new, zero_division=0):.4f}")
    tc2.metric("Recall", f"{recall_score(y_true, y_pred_new, zero_division=0):.4f}")
    tc3.metric("F1", f"{f1_score(y_true, y_pred_new, zero_division=0):.4f}")
    tp_new = int(((y_true == 1) & (y_pred_new == 1)).sum())
    fn_new = int(((y_true == 1) & (y_pred_new == 0)).sum())
    tc4.metric("Caught Frauds / Missed", f"{tp_new} / {fn_new}")

    # Per-model comparison
    if model_metrics:
        st.subheader("Per-Model Metrics")
        rows = []
        for model_name, m in model_metrics.items():
            if isinstance(m, dict) and "auc" in m:
                rows.append({"Model": model_name, "AUC": m.get("auc", 0), "F1": m.get("f1", 0),
                              "Precision": m.get("precision", 0), "Recall": m.get("recall", 0)})
        if rows:
            comp_df = pd.DataFrame(rows).sort_values("AUC", ascending=False)
            if PLOTLY_OK:
                fig = px.bar(comp_df, x="Model", y="AUC", color="AUC",
                             color_continuous_scale="Blues", text_auto=".4f")
                fig.update_layout(margin=dict(t=10))
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(comp_df.set_index("Model").style.format("{:.4f}"), use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Predictions
# ---------------------------------------------------------------------------

elif page == "Predictions":
    st.title("Transaction Explorer")

    df = load_predictions()
    if df is None:
        _no_data_warning()
        st.stop()

    # Filters
    with st.expander("Filters", expanded=True):
        fc1, fc2, fc3 = st.columns(3)
        show_fraud = fc1.selectbox("Show", ["All", "Fraud only", "Legitimate only"])
        min_prob = fc2.slider("Min fraud probability", 0.0, 1.0, 0.0, 0.01)
        max_rows = fc3.number_input("Max rows to display", 100, 10_000, 500, 100)

    filtered = df.copy()
    if show_fraud == "Fraud only":
        filtered = filtered[filtered["actual_fraud"] == 1]
    elif show_fraud == "Legitimate only":
        filtered = filtered[filtered["actual_fraud"] == 0]
    filtered = filtered[filtered["fraud_probability"] >= min_prob]

    st.markdown(f"Showing **{min(len(filtered), max_rows):,}** of **{len(filtered):,}** transactions")

    display_cols = ["fraud_probability", "predicted_fraud", "actual_fraud"] + [
        c for c in filtered.columns if c not in ["fraud_probability", "predicted_fraud", "actual_fraud"]
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]

    styled = filtered[display_cols].head(int(max_rows)).style.background_gradient(
        subset=["fraud_probability"], cmap="RdYlGn_r"
    ).format({"fraud_probability": "{:.4f}"})
    st.dataframe(styled, use_container_width=True)

    # Download
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", csv, "fraud_predictions_filtered.csv", "text/csv")


# ---------------------------------------------------------------------------
# Page: Feature Analysis
# ---------------------------------------------------------------------------

elif page == "Feature Analysis":
    st.title("Feature Analysis")

    df = load_predictions()
    if df is None:
        _no_data_warning()
        st.stop()

    # Feature correlation with fraud probability
    numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(
        ["actual_fraud", "predicted_fraud"]
    ).tolist()

    if not numeric_cols:
        st.info("No numeric feature columns found in predictions file.")
        st.stop()

    st.subheader("Feature Correlations with Fraud Probability")
    corr = df[numeric_cols].corrwith(df["fraud_probability"]).sort_values(key=abs, ascending=False)
    top_n = st.slider("Top N features", 5, min(50, len(corr)), 20)
    top_corr = corr.head(top_n)

    if PLOTLY_OK:
        fig = px.bar(
            x=top_corr.values, y=top_corr.index,
            orientation="h",
            color=top_corr.values,
            color_continuous_scale="RdBu",
            labels={"x": "Correlation", "y": "Feature"},
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(top_corr)

    # Feature distribution: fraud vs legit
    st.subheader("Feature Distribution: Fraud vs Legitimate")
    feat = st.selectbox("Select feature", top_corr.index.tolist())
    if feat in df.columns and PLOTLY_OK:
        fig = px.histogram(
            df, x=feat,
            color=df["actual_fraud"].map({0: "Legitimate", 1: "Fraud"}),
            nbins=50, barmode="overlay", opacity=0.7,
            color_discrete_map={"Legitimate": "#2ecc71", "Fraud": "#e74c3c"},
        )
        fig.update_layout(margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Pipeline Monitor
# ---------------------------------------------------------------------------

elif page == "Pipeline Monitor":
    st.title("Pipeline Monitor")

    summary = load_pipeline_summary()

    if summary is None:
        _no_data_warning()
        st.stop()

    # Stage timeline
    st.subheader("Stage Execution Summary")
    rows = []
    for stage, info in summary.items():
        rows.append({
            "Stage": stage,
            "Status": info.get("status", "unknown"),
            "Elapsed (s)": info.get("elapsed_s", 0),
            "Mem Start (MB)": round(info.get("mem_start_mb", 0), 1),
            "Mem End (MB)": round(info.get("mem_end_mb", 0), 1),
        })
    stage_df = pd.DataFrame(rows)

    def _color_status(val):
        if val == "success":
            return "background-color: #d5f5e3"
        if val == "failed":
            return "background-color: #fadbd8"
        return ""

    styled = stage_df.style.applymap(_color_status, subset=["Status"])
    st.dataframe(styled, hide_index=True, use_container_width=True)

    # Elapsed bar chart
    if PLOTLY_OK and not stage_df.empty:
        fig = px.bar(stage_df, x="Stage", y="Elapsed (s)", color="Status",
                     color_discrete_map={"success": "#2ecc71", "failed": "#e74c3c", "running": "#f39c12"},
                     text_auto=".1f")
        fig.update_layout(margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    # Per-stage metrics
    st.subheader("Stage Metrics")
    for stage, info in summary.items():
        metrics = info.get("metrics", {})
        if not metrics:
            continue
        with st.expander(f"{stage} metrics"):
            m_df = pd.DataFrame([
                {"Metric": k, "Value": round(v, 4) if isinstance(v, float) else v}
                for k, v in metrics.items()
            ])
            st.dataframe(m_df, hide_index=True, use_container_width=True)

    # System info
    st.subheader("Output Files")
    out_files = list(OUTPUT.glob("*")) if OUTPUT.exists() else []
    if out_files:
        file_df = pd.DataFrame([
            {"File": f.name, "Size (KB)": round(f.stat().st_size / 1024, 1)}
            for f in sorted(out_files)
        ])
        st.dataframe(file_df, hide_index=True, use_container_width=True)
    else:
        st.info("No output files found yet.")
