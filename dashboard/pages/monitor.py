"""Pipeline Monitor page — stage timeline, memory usage, feature correlations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.metric_cards import kpi_row, page_header, section_header


def render(metrics, df, summary, plotly_dark: dict):
    page_header(
        "Pipeline Monitor",
        subtitle="Stage execution timeline, memory usage, and feature importance analysis",
        badge_text="6 Stages",
        badge_color="purple",
    )

    if summary:
        rows = []
        for stage_name, info in summary.items():
            rows.append({
                "Stage": stage_name,
                "Status": info.get("status", "unknown"),
                "Elapsed (s)": round(info.get("elapsed_s", 0), 2),
                "Mem Start (MB)": round(info.get("mem_start_mb", 0), 1),
                "Mem End (MB)": round(info.get("mem_end_mb", 0), 1),
                "Delta Mem (MB)": round(info.get("mem_end_mb", 0) - info.get("mem_start_mb", 0), 1),
            })
        stage_df = pd.DataFrame(rows)

        # ── Pipeline KPIs ──────────────────────────────────────────────────
        total_time = stage_df["Elapsed (s)"].sum()
        n_success  = (stage_df["Status"] == "success").sum()
        max_mem    = stage_df["Mem End (MB)"].max()
        mem_growth = stage_df["Delta Mem (MB)"].sum()

        kpi_row([
            {"label": "Total Runtime",    "value": f"{total_time:.1f}s",  "sub": f"{n_success}/{len(stage_df)} stages OK", "color": "green" if n_success == len(stage_df) else "amber"},
            {"label": "Peak Memory",      "value": f"{max_mem:.0f} MB",   "sub": f"+{mem_growth:.0f} MB during pipeline", "color": "blue"},
            {"label": "Stages Complete",  "value": f"{n_success}/{len(stage_df)}", "sub": "Pipeline health",               "color": "green" if n_success == len(stage_df) else "red"},
        ])

        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

        # ── Stage Timeline chart ───────────────────────────────────────────
        section_header("Stage Execution Timeline", "Duration per pipeline stage")
        color_map = {"success": "#22C55E", "failed": "#EF4444", "unknown": "#F59E0B"}
        fig = go.Figure()
        for status, color in color_map.items():
            sub_df = stage_df[stage_df["Status"] == status]
            if sub_df.empty:
                continue
            fig.add_trace(go.Bar(
                x=sub_df["Stage"], y=sub_df["Elapsed (s)"],
                name=status.title(), marker_color=color,
                text=sub_df["Elapsed (s)"].apply(lambda v: f"{v:.1f}s"),
                textposition="outside",
                textfont=dict(size=11, color="#94A3B8"),
                hovertemplate="<b>%{x}</b><br>Duration: %{y:.2f}s<extra></extra>",
            ))
        fig.update_layout(
            **plotly_dark,
            barmode="stack",
            xaxis_tickangle=-20,
            yaxis_title="Duration (seconds)",
            legend=dict(orientation="h", y=1.1, font=dict(size=11)),
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Memory usage chart ─────────────────────────────────────────────
        section_header("Memory Usage Per Stage", "Memory at start vs end of each stage")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=stage_df["Stage"], y=stage_df["Mem Start (MB)"],
            mode="lines+markers", name="Start",
            line=dict(color="#3B82F6", width=2),
            marker=dict(size=8, symbol="circle"),
            hovertemplate="<b>%{x}</b><br>Mem Start: %{y:.0f} MB<extra></extra>",
        ))
        fig2.add_trace(go.Scatter(
            x=stage_df["Stage"], y=stage_df["Mem End (MB)"],
            mode="lines+markers", name="End",
            line=dict(color="#22C55E", width=2),
            marker=dict(size=8, symbol="diamond"),
            fill="tonexty", fillcolor="rgba(34,197,94,0.05)",
            hovertemplate="<b>%{x}</b><br>Mem End: %{y:.0f} MB<extra></extra>",
        ))
        fig2.update_layout(
            **plotly_dark,
            xaxis_tickangle=-20,
            yaxis_title="Memory (MB)",
            legend=dict(orientation="h", y=1.1, font=dict(size=11)),
            height=280,
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── Stage details table ────────────────────────────────────────────
        section_header("Stage Details")
        st.dataframe(
            stage_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Stage":         st.column_config.TextColumn("Stage", width="medium"),
                "Status":        st.column_config.TextColumn("Status", width="small"),
                "Elapsed (s)":   st.column_config.NumberColumn("Duration (s)", format="%.2f", width="small"),
                "Mem Start (MB)":st.column_config.NumberColumn("Mem Start", format="%.0f MB", width="small"),
                "Mem End (MB)":  st.column_config.NumberColumn("Mem End",   format="%.0f MB", width="small"),
                "Delta Mem (MB)":st.column_config.NumberColumn("Delta Mem", format="%.0f MB", width="small"),
            },
        )
    else:
        st.info("No pipeline summary found. Run the pipeline to see stage timing and memory data.")

    # ── Feature Importance (correlation proxy) ─────────────────────────────
    if df is not None:
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        section_header("Feature Importance Analysis", "Correlation with fraud probability score")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(
            ["actual_fraud", "predicted_fraud"]
        ).tolist()

        if numeric_cols:
            corr = df[numeric_cols].corrwith(df["fraud_probability"]).sort_values(key=abs, ascending=False)
            top_n = st.slider("Top N features", 5, min(50, len(corr)), 20, key="mon_topn")
            top_corr = corr.head(top_n)

            colors = ["#EF4444" if v > 0 else "#3B82F6" for v in top_corr.values]
            fig3 = go.Figure(go.Bar(
                x=top_corr.values, y=top_corr.index,
                orientation="h", marker_color=colors,
                text=[f"{v:+.3f}" for v in top_corr.values],
                textposition="outside",
                textfont=dict(size=10, color="#94A3B8"),
                hovertemplate="<b>%{y}</b><br>Correlation: %{x:.4f}<extra></extra>",
            ))
            fig3.add_vline(x=0, line_color="#475569", line_width=1)
            fig3.update_layout(**plotly_dark)
            fig3.update_layout(
                xaxis_title="Pearson Correlation with Fraud Probability",
                yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
                height=max(400, top_n * 22),
                margin=dict(t=32, b=32, l=180, r=80),
            )
            st.plotly_chart(fig3, use_container_width=True)

            # Legend
            st.markdown("""
            <div style="display:flex;gap:20px;padding:8px 0;font-size:12px;color:#64748B;">
                <span><span style="color:#EF4444;font-weight:700;">■</span> Positive correlation (higher value → more fraud)</span>
                <span><span style="color:#3B82F6;font-weight:700;">■</span> Negative correlation (higher value → less fraud)</span>
            </div>
            """, unsafe_allow_html=True)
