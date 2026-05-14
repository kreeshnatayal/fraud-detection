"""Risk Intelligence page — live transaction scorer with gauge and reason codes."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from dashboard.components.metric_cards import badge, kpi_row, page_header, section_header
from src.fraud_detection.serving.demo_scorer import demo_score_transaction


_TIER_CONFIG = {
    "CRITICAL": {"color": "#EF4444", "bg": "rgba(239,68,68,0.1)",  "border": "rgba(239,68,68,0.3)",  "badge": "red",    "icon": "🔴"},
    "HIGH":     {"color": "#F59E0B", "bg": "rgba(245,158,11,0.1)", "border": "rgba(245,158,11,0.3)", "badge": "amber",  "icon": "🟠"},
    "MEDIUM":   {"color": "#3B82F6", "bg": "rgba(59,130,246,0.1)", "border": "rgba(59,130,246,0.3)", "badge": "blue",   "icon": "🔵"},
    "LOW":      {"color": "#22C55E", "bg": "rgba(34,197,94,0.1)",  "border": "rgba(34,197,94,0.3)",  "badge": "green",  "icon": "🟢"},
}

_DECISION_CONFIG = {
    "BLOCK":  {"color": "#EF4444", "badge": "red"},
    "REVIEW": {"color": "#F59E0B", "badge": "amber"},
    "PASS":   {"color": "#22C55E", "badge": "green"},
}


def render(metrics, df, plotly_dark: dict):
    page_header(
        "Risk Intelligence",
        subtitle="Real-time transaction scoring with SHAP-based adverse action reason codes",
        badge_text="Live Scorer",
        badge_color="blue",
    )

    threshold = float(metrics.get("threshold", 0.5)) if metrics else 0.5

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        section_header("Transaction Details", "Enter transaction attributes for scoring")

        st.markdown("""
        <div style="background:rgba(15,23,42,0.7);border:1px solid rgba(30,41,59,0.8);
                    border-radius:12px;padding:20px;">
        """, unsafe_allow_html=True)

        amount   = st.number_input("Amount ($)", 0.01, 100_000.0, 349.99, key="ri_amount")
        hour     = st.slider("Hour of Day", 0, 23, 14, key="ri_hour",
                             help="0 = midnight, 14 = 2pm, 22+ = high-risk night hours")
        is_wknd  = st.checkbox("Weekend transaction", key="ri_wknd")
        email    = st.selectbox("Email Domain", ["gmail.com", "yahoo.com", "corporate.com", "unknown", "protonmail.com"], key="ri_email")
        device   = st.selectbox("Device", ["Windows PC", "Android Mobile", "iPhone iOS", "Unknown"], key="ri_device")
        is_rapid = st.checkbox("Rapid follow-up (< 5 min)", key="ri_rapid",
                               help="Transaction within 5 minutes of previous one from same card")

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        score_btn = st.button("Score Transaction", type="primary", use_container_width=True)

    with col_out:
        section_header("Risk Assessment", "Real-time fraud probability and decision")

        if score_btn:
            result       = demo_score_transaction(
                amount=amount, hour=hour, email_domain=email,
                device_info=device, rapid_sequence=is_rapid, is_weekend=is_wknd,
            )
            score        = result["fraud_probability"]
            tier         = result["risk_tier"]
            decision     = result["decision"]
            expected_loss= result["expected_loss_usd"]
            reasons      = result["reason_codes"]
            tc           = _TIER_CONFIG.get(tier, _TIER_CONFIG["LOW"])
            dc           = _DECISION_CONFIG.get(decision, _DECISION_CONFIG["PASS"])

            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score * 100,
                number=dict(suffix="%", font=dict(size=36, color="#F8FAFC", family="IBM Plex Sans")),
                title=dict(text="Fraud Risk Score", font=dict(size=14, color="#94A3B8")),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor="#475569", tickfont=dict(color="#94A3B8", size=11)),
                    bar=dict(color=tc["color"], thickness=0.22),
                    bgcolor="rgba(15,23,42,0.5)",
                    borderwidth=0,
                    steps=[
                        {"range": [0, 20],  "color": "rgba(34,197,94,0.12)"},
                        {"range": [20, 50], "color": "rgba(59,130,246,0.10)"},
                        {"range": [50, 80], "color": "rgba(245,158,11,0.10)"},
                        {"range": [80, 100],"color": "rgba(239,68,68,0.15)"},
                    ],
                    threshold=dict(
                        line=dict(color="#F59E0B", width=3),
                        thickness=0.8, value=threshold * 100,
                    ),
                ),
            ))
            fig.update_layout(**plotly_dark)
            fig.update_layout(height=240, margin=dict(t=32, b=16, l=24, r=24))
            st.plotly_chart(fig, use_container_width=True)

            # Decision banner
            st.markdown(f"""
            <div style="background:{tc['bg']};border:1px solid {tc['border']};
                        border-radius:12px;padding:16px 20px;margin:8px 0;
                        display:flex;align-items:center;justify-content:space-between;">
                <div>
                    <div style="font-size:11px;font-weight:700;letter-spacing:0.08em;
                                color:#64748B;text-transform:uppercase;margin-bottom:4px;">
                        Risk Decision
                    </div>
                    <div style="font-size:22px;font-weight:700;color:{tc['color']};">
                        {tc['icon']} {decision}
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:11px;font-weight:700;letter-spacing:0.08em;
                                color:#64748B;text-transform:uppercase;margin-bottom:4px;">
                        Risk Tier
                    </div>
                    {badge(tier, tc['badge'])}
                </div>
            </div>
            """, unsafe_allow_html=True)

            kpi_row([
                {"label": "Fraud Probability", "value": f"{score:.1%}",        "sub": f"Threshold: {threshold:.2f}", "color": "red" if score > threshold else "green"},
                {"label": "Expected Loss",     "value": f"${expected_loss:,.2f}", "sub": "If fraud & not blocked",    "color": "amber"},
            ])

            if reasons:
                st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
                section_header("Adverse Action Reason Codes", "Top risk factors driving this score")
                for i, r in enumerate(reasons, 1):
                    severity = "red" if i == 1 else "amber" if i == 2 else "blue"
                    c = {"red": "#EF4444", "amber": "#F59E0B", "blue": "#3B82F6"}[severity]
                    st.markdown(f"""
                    <div style="display:flex;align-items:flex-start;gap:12px;
                                padding:12px 16px;margin-bottom:6px;
                                background:rgba(15,23,42,0.6);
                                border:1px solid rgba(30,41,59,0.8);
                                border-left:3px solid {c};border-radius:8px;">
                        <div style="font-size:12px;font-weight:700;color:{c};
                                    min-width:20px;margin-top:1px;">#{i}</div>
                        <div style="font-size:13px;color:#CBD5E1;line-height:1.5;">{r}</div>
                    </div>
                    """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                        min-height:360px;text-align:center;
                        background:rgba(15,23,42,0.4);border:1px solid rgba(30,41,59,0.6);
                        border-radius:12px;">
                <div style="font-size:40px;margin-bottom:12px;opacity:0.5;">🛡</div>
                <p style="color:#475569;font-size:14px;margin:0;line-height:1.6;">
                    Fill in transaction details<br>and click <strong style="color:#94A3B8;">Score Transaction</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

            if df is not None:
                st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
                section_header("Sample Portfolio Transactions")
                sample = df.sample(min(5, len(df)), random_state=42)[["fraud_probability", "actual_fraud"]]
                sample.columns = ["Fraud Probability", "Actual Fraud"]
                sample["Fraud Probability"] = sample["Fraud Probability"].round(4)
                st.dataframe(
                    sample,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Fraud Probability": st.column_config.ProgressColumn(
                            "Fraud Probability", min_value=0, max_value=1, format="%.4f"
                        ),
                        "Actual Fraud": st.column_config.CheckboxColumn("Actual Fraud"),
                    },
                )
