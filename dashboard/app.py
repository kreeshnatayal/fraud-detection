"""
FraudShield Pro — Enterprise Fraud Detection Dashboard
Run: streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="FraudShield Pro",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS: OLED Dark + Glassmorphism ──────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

/* ── Base ── */
html, body, [class*="css"], .stApp {
    font-family: 'IBM Plex Sans', sans-serif !important;
}
.stApp {
    background: #020617;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(34,197,94,0.06) 0%, transparent 70%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(59,130,246,0.04) 0%, transparent 60%);
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stDecoration"] { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #080D1A 100%) !important;
    border-right: 1px solid rgba(34,197,94,0.15) !important;
}
[data-testid="stSidebar"] .stRadio > label {
    font-size: 13px;
    font-weight: 500;
    color: #94A3B8;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    font-size: 14px;
    font-weight: 500;
}

/* ── Metric Cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(15,23,42,0.95) 0%, rgba(30,41,59,0.9) 100%);
    border: 1px solid rgba(34,197,94,0.18);
    border-radius: 14px;
    padding: 18px 22px !important;
    box-shadow:
        0 4px 24px rgba(0,0,0,0.35),
        inset 0 1px 0 rgba(255,255,255,0.04);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="metric-container"]:hover {
    border-color: rgba(34,197,94,0.35);
    box-shadow: 0 6px 32px rgba(34,197,94,0.1), 0 4px 24px rgba(0,0,0,0.35);
}
[data-testid="metric-container"] label {
    color: #94A3B8 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] > div {
    color: #F8FAFC !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}
[data-testid="stMetricDelta"] svg { display: none; }
[data-testid="stMetricDelta"] > div {
    font-size: 12px !important;
    font-weight: 500 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #16A34A 0%, #22C55E 100%);
    color: #020617 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 0.03em;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    box-shadow: 0 2px 12px rgba(34,197,94,0.25);
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #22C55E 0%, #4ADE80 100%);
    box-shadow: 0 4px 20px rgba(34,197,94,0.4);
    transform: translateY(-1px);
}
.stButton > button[kind="secondary"] {
    background: rgba(15,23,42,0.8) !important;
    color: #94A3B8 !important;
    border: 1px solid rgba(30,41,59,0.9) !important;
    box-shadow: none;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(30,41,59,0.8);
    border-radius: 12px;
    overflow: hidden;
}

/* ── Selectbox / Number Input / Slider ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background: rgba(15,23,42,0.8) !important;
    border: 1px solid rgba(30,41,59,0.9) !important;
    border-radius: 8px !important;
    color: #F8FAFC !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
.stSlider [data-baseweb="slider"] { cursor: pointer; }

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: rgba(15,23,42,0.6);
    border: 1px solid rgba(30,41,59,0.8) !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary {
    font-weight: 600;
    color: #CBD5E1;
    font-size: 14px;
}

/* ── Info / Warning / Success banners ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 3px !important;
    font-size: 14px;
}

/* ── Divider ── */
hr { border-color: rgba(30,41,59,0.6) !important; }

/* ── Plotly chart container ── */
[data-testid="stPlotlyChart"] {
    border-radius: 12px;
    overflow: hidden;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: rgba(15,23,42,0.8) !important;
    color: #22C55E !important;
    border: 1px solid rgba(34,197,94,0.3) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(34,197,94,0.1) !important;
    border-color: rgba(34,197,94,0.5) !important;
}

/* ── Checkbox ── */
[data-testid="stCheckbox"] label {
    font-size: 14px;
    color: #CBD5E1;
}

/* ── Page headings ── */
h1 {
    font-size: 26px !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em !important;
    color: #F8FAFC !important;
    line-height: 1.2 !important;
}
h2 {
    font-size: 18px !important;
    font-weight: 600 !important;
    color: #CBD5E1 !important;
    letter-spacing: -0.01em !important;
}
h3 {
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #94A3B8 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}

/* ── Caption / small text ── */
[data-testid="stCaptionContainer"] p {
    color: #64748B !important;
    font-size: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Dark Plotly template shared across pages ───────────────────────────────
PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.6)",
    font=dict(family="IBM Plex Sans", color="#CBD5E1", size=12),
    margin=dict(t=32, b=32, l=16, r=16),
    xaxis=dict(gridcolor="rgba(30,41,59,0.6)", linecolor="rgba(30,41,59,0.8)"),
    yaxis=dict(gridcolor="rgba(30,41,59,0.6)", linecolor="rgba(30,41,59,0.8)"),
)


def _find_output() -> Path:
    import sys, argparse
    idx = sys.argv.index("--") + 1 if "--" in sys.argv else None
    if idx:
        p = argparse.ArgumentParser()
        p.add_argument("--output", default="output")
        args, _ = p.parse_known_args(sys.argv[idx:])
        return Path(args.output)
    return Path("output")


OUTPUT = _find_output()


@st.cache_data(ttl=3600)
def load_predictions():
    path = OUTPUT / "val_predictions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    for col in ["fraud_probability", "predicted_fraud", "actual_fraud"]:
        if col not in df.columns:
            return None
    return df


@st.cache_data(ttl=3600)
def load_metrics():
    path = OUTPUT / "evaluation_metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=3600)
def load_model_metrics():
    path = OUTPUT / "model_metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=3600)
def load_pipeline_summary():
    path = OUTPUT / "pipeline_summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 20px 0;">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">
            <div style="width:36px;height:36px;background:linear-gradient(135deg,#16A34A,#22C55E);
                        border-radius:10px;display:flex;align-items:center;justify-content:center;
                        font-size:18px;box-shadow:0 2px 12px rgba(34,197,94,0.3);">🛡</div>
            <div>
                <div style="font-size:16px;font-weight:700;color:#F8FAFC;letter-spacing:-0.02em;">
                    FraudShield Pro
                </div>
                <div style="font-size:11px;color:#22C55E;font-weight:600;letter-spacing:0.06em;">
                    ENTERPRISE AI
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="font-size:10px;font-weight:700;letter-spacing:0.1em;color:#475569;text-transform:uppercase;margin-bottom:8px;">Navigation</p>', unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["Overview", "Model Performance", "Financial Impact",
         "Risk Intelligence", "Predictions", "Pipeline Monitor"],
        label_visibility="collapsed",
    )

    st.markdown("<div style='margin:20px 0;border-top:1px solid rgba(30,41,59,0.8);'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(15,23,42,0.8);border:1px solid rgba(30,41,59,0.8);
                border-radius:10px;padding:14px;">
        <p style="font-size:10px;font-weight:700;letter-spacing:0.1em;color:#475569;
                  text-transform:uppercase;margin:0 0 10px 0;">Model Stack</p>
        <div style="display:flex;flex-direction:column;gap:6px;">
            <div style="display:flex;align-items:center;gap:8px;">
                <div style="width:6px;height:6px;border-radius:50%;background:#22C55E;flex-shrink:0;"></div>
                <span style="font-size:12px;color:#CBD5E1;">LightGBM · XGBoost · CatBoost</span>
            </div>
            <div style="display:flex;align-items:center;gap:8px;">
                <div style="width:6px;height:6px;border-radius:50%;background:#3B82F6;flex-shrink:0;"></div>
                <span style="font-size:12px;color:#CBD5E1;">Stacking Ensemble (LR meta)</span>
            </div>
            <div style="display:flex;align-items:center;gap:8px;">
                <div style="width:6px;height:6px;border-radius:50%;background:#8B5CF6;flex-shrink:0;"></div>
                <span style="font-size:12px;color:#CBD5E1;">SHAP · NetworkX · Node2Vec</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("""
    <div style="margin-top:auto;padding-top:16px;text-align:center;">
        <span style="font-size:11px;color:#334155;">v1.0.0 · AUC 0.9651</span>
    </div>
    """, unsafe_allow_html=True)


# ── Data loading ───────────────────────────────────────────────────────────
metrics = load_metrics()
df = load_predictions()
model_metrics = load_model_metrics()
summary = load_pipeline_summary()

if metrics is None or df is None:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                min-height:60vh;text-align:center;">
        <div style="font-size:48px;margin-bottom:16px;">📊</div>
        <h1 style="color:#F8FAFC;margin-bottom:8px;">No Pipeline Output Found</h1>
        <p style="color:#64748B;font-size:15px;max-width:420px;line-height:1.6;">
            Run the ML pipeline first, or generate demo data to preview the dashboard.
        </p>
        <div style="margin-top:20px;background:rgba(15,23,42,0.9);border:1px solid rgba(30,41,59,0.9);
                    border-radius:10px;padding:16px 24px;text-align:left;">
            <code style="color:#22C55E;font-size:13px;">python scripts/generate_demo_data.py</code>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Page routing ───────────────────────────────────────────────────────────
if page == "Overview":
    from dashboard.pages.overview import render
    render(metrics, df, PLOTLY_DARK)
elif page == "Model Performance":
    from dashboard.pages.performance import render
    render(metrics, df, model_metrics, PLOTLY_DARK)
elif page == "Financial Impact":
    from dashboard.pages.financial import render
    render(metrics, df, PLOTLY_DARK)
elif page == "Risk Intelligence":
    from dashboard.pages.risk_intel import render
    render(metrics, df, PLOTLY_DARK)
elif page == "Predictions":
    from dashboard.pages.predictions import render
    render(metrics, df)
elif page == "Pipeline Monitor":
    from dashboard.pages.monitor import render
    render(metrics, df, summary, PLOTLY_DARK)
