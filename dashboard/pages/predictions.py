"""Transaction Explorer page — filterable table with CSV export."""
from __future__ import annotations

import streamlit as st

from dashboard.components.metric_cards import kpi_row, page_header, section_header


def render(metrics, df):
    page_header(
        "Transaction Explorer",
        subtitle="Browse, filter, and export the validation set predictions",
        badge_text=f"{len(df):,} Records",
        badge_color="blue",
    )

    # ── Filter controls ────────────────────────────────────────────────────
    with st.expander("Filters", expanded=True):
        fc1, fc2, fc3 = st.columns(3, gap="medium")
        with fc1:
            show_fraud = st.selectbox("Transaction Type", ["All", "Fraud only", "Legitimate only"])
        with fc2:
            min_prob = st.slider("Min fraud probability", 0.0, 1.0, 0.0, 0.01)
        with fc3:
            max_rows = st.number_input("Max rows to display", 100, 10_000, 500, 100)

    # ── Filter logic ───────────────────────────────────────────────────────
    filtered = df.copy()
    if show_fraud == "Fraud only":
        filtered = filtered[filtered["actual_fraud"] == 1]
    elif show_fraud == "Legitimate only":
        filtered = filtered[filtered["actual_fraud"] == 0]
    filtered = filtered[filtered["fraud_probability"] >= min_prob]
    display_df = filtered.head(int(max_rows)).copy()
    display_df["fraud_probability"] = display_df["fraud_probability"].round(4)

    # ── Filter summary KPIs ────────────────────────────────────────────────
    n_shown   = min(len(filtered), int(max_rows))
    n_fraud   = int(display_df["actual_fraud"].sum())
    n_pred    = int(display_df["predicted_fraud"].sum())
    avg_score = display_df["fraud_probability"].mean()

    kpi_row([
        {"label": "Showing",          "value": f"{n_shown:,}",    "sub": f"of {len(filtered):,} filtered", "color": "blue"},
        {"label": "Actual Frauds",    "value": f"{n_fraud:,}",    "sub": f"{100*n_fraud/max(n_shown,1):.1f}% of shown",   "color": "red"},
        {"label": "Predicted Frauds", "value": f"{n_pred:,}",     "sub": f"@ threshold {metrics.get('threshold',0.5):.2f}", "color": "amber"},
        {"label": "Avg Score",        "value": f"{avg_score:.4f}","sub": "Mean fraud probability",          "color": "purple"},
    ])

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    section_header("Prediction Table", f"Sorted by fraud probability (highest first)")

    # ── Reorder columns: key ones first ───────────────────────────────────
    key_cols = ["fraud_probability", "predicted_fraud", "actual_fraud"]
    extra_cols = [c for c in display_df.columns if c not in key_cols]
    display_df = display_df[key_cols + extra_cols].sort_values(
        "fraud_probability", ascending=False
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "fraud_probability": st.column_config.ProgressColumn(
                "Fraud Probability", min_value=0, max_value=1, format="%.4f", width="medium",
            ),
            "actual_fraud": st.column_config.CheckboxColumn("Actual Fraud", width="small"),
            "predicted_fraud": st.column_config.CheckboxColumn("Predicted", width="small"),
        },
    )

    # ── Export ─────────────────────────────────────────────────────────────
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    col_dl, col_info = st.columns([1, 3])
    with col_dl:
        st.download_button(
            label="Download CSV",
            data=filtered.to_csv(index=False).encode(),
            file_name="fraud_predictions_export.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_info:
        st.caption(f"Exporting {len(filtered):,} transactions matching current filters.")
