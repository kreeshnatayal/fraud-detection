"""Shared premium metric card components."""
from __future__ import annotations

import streamlit as st


def metric_card(col, label: str, value, fmt=".4f", delta=None, delta_good: bool = True):
    disp = f"{value:{fmt}}" if isinstance(value, float) else str(value)
    col.metric(label, disp, delta=delta, delta_color="normal" if delta_good else "inverse")


def kpi_row(items: list[dict]):
    """
    Render a row of premium HTML KPI cards.

    Each dict: {label, value, sub?, color?}
      color: 'green' | 'red' | 'blue' | 'amber' | 'purple' (default 'green')
    """
    color_map = {
        "green":  {"bg": "rgba(34,197,94,0.08)",  "border": "rgba(34,197,94,0.25)",  "accent": "#22C55E"},
        "red":    {"bg": "rgba(239,68,68,0.08)",   "border": "rgba(239,68,68,0.25)",   "accent": "#EF4444"},
        "blue":   {"bg": "rgba(59,130,246,0.08)",  "border": "rgba(59,130,246,0.25)",  "accent": "#3B82F6"},
        "amber":  {"bg": "rgba(245,158,11,0.08)",  "border": "rgba(245,158,11,0.25)",  "accent": "#F59E0B"},
        "purple": {"bg": "rgba(139,92,246,0.08)",  "border": "rgba(139,92,246,0.25)",  "accent": "#8B5CF6"},
        "cyan":   {"bg": "rgba(6,182,212,0.08)",   "border": "rgba(6,182,212,0.25)",   "accent": "#06B6D4"},
    }

    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        c = color_map.get(item.get("color", "green"), color_map["green"])
        sub_html = (
            f'<div style="font-size:12px;color:{c["accent"]};font-weight:500;margin-top:4px;">'
            f'{item["sub"]}</div>'
        ) if item.get("sub") else ""

        col.markdown(f"""
        <div style="
            background:{c['bg']};
            border:1px solid {c['border']};
            border-radius:14px;
            padding:20px 22px;
            min-height:100px;
            position:relative;
            box-shadow:0 4px 24px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.03);
        ">
            <div style="font-size:10px;font-weight:700;letter-spacing:0.1em;
                        color:#64748B;text-transform:uppercase;margin-bottom:8px;">
                {item['label']}
            </div>
            <div style="font-size:26px;font-weight:700;color:#F8FAFC;letter-spacing:-0.03em;
                        line-height:1.1;">
                {item['value']}
            </div>
            {sub_html}
            <div style="position:absolute;top:16px;right:16px;width:8px;height:8px;
                        border-radius:50%;background:{c['accent']};
                        box-shadow:0 0 8px {c['accent']};"></div>
        </div>
        """, unsafe_allow_html=True)


def section_header(title: str, subtitle: str = ""):
    """Render a styled section header with optional subtitle."""
    sub_html = f'<p style="color:#64748B;font-size:13px;margin:4px 0 0 0;">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div style="margin:24px 0 16px 0;">
        <h2 style="font-size:17px!important;font-weight:700!important;color:#CBD5E1!important;
                   letter-spacing:-0.01em!important;margin:0!important;">{title}</h2>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def badge(text: str, color: str = "green") -> str:
    """Return an inline HTML badge string."""
    color_map = {
        "green":  ("#22C55E", "rgba(34,197,94,0.15)"),
        "red":    ("#EF4444", "rgba(239,68,68,0.15)"),
        "amber":  ("#F59E0B", "rgba(245,158,11,0.15)"),
        "blue":   ("#3B82F6", "rgba(59,130,246,0.15)"),
        "purple": ("#8B5CF6", "rgba(139,92,246,0.15)"),
    }
    fg, bg = color_map.get(color, color_map["green"])
    return (
        f'<span style="display:inline-block;padding:3px 10px;border-radius:20px;'
        f'background:{bg};color:{fg};font-size:11px;font-weight:700;'
        f'letter-spacing:0.04em;text-transform:uppercase;">{text}</span>'
    )


def page_header(title: str, subtitle: str = "", badge_text: str = "", badge_color: str = "green"):
    """Render a professional page header."""
    badge_html = f"&nbsp;&nbsp;{badge(badge_text, badge_color)}" if badge_text else ""
    sub_html = f'<p style="color:#64748B;font-size:14px;margin:6px 0 0 0;font-weight:400;">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div style="margin-bottom:28px;padding-bottom:20px;border-bottom:1px solid rgba(30,41,59,0.6);">
        <div style="display:flex;align-items:center;flex-wrap:wrap;gap:8px;">
            <h1 style="font-size:24px!important;font-weight:700!important;color:#F8FAFC!important;
                       letter-spacing:-0.03em!important;margin:0!important;">{title}</h1>
            {badge_html}
        </div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)
