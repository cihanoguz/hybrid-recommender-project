"""UI components for Streamlit application."""

import base64
from pathlib import Path
from typing import Optional

import streamlit as st

from logging_config import get_logger

logger = get_logger(__name__)


def img_to_base64(path: Path) -> Optional[str]:
    """Convert image file to base64 string."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        logger.warning(f"Logo file not found: {path}")
        return None
    except Exception as e:
        logger.error(f"Error encoding logo: {e}")
        return None


def render_header(logo_b64: Optional[str] = None) -> None:
    """Render the DataHub header banner."""
    if logo_b64:
        logo_html = (
            f"<img src='data:image/png;base64,{logo_b64}' "
            "style=\"height:40px; border-radius:.5rem; "
            "background:rgba(0,0,0,.15); padding:4px;"
            "box-shadow:0 10px 20px rgba(0,0,0,0.4);\"/>"
        )
    else:
        logo_html = (
            "<div style=\"height:40px; width:40px; border-radius:.5rem; "
            "background:rgba(0,0,0,.15); display:flex; align-items:center; "
            "justify-content:center; font-size:.6rem; font-weight:600; "
            "box-shadow:0 10px 20px rgba(0,0,0,0.4); color:white;\">DH</div>"
        )

    datahub_banner_html = (
        "<div style=\""
        "background: linear-gradient(90deg, rgba(37,99,235,1) 0%, rgba(16,185,129,1) 100%);"
        "padding: .75rem 1rem;"
        "border-radius: .5rem;"
        "color: white;"
        "font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Inter', Roboto, 'Segoe UI', sans-serif;"
        "font-size: .9rem;"
        "font-weight: 500;"
        "display: flex;"
        "align-items: center;"
        "gap: .75rem;"
        "margin-bottom: 1rem;"
        "border: 1px solid rgba(255,255,255,0.3);"
        "box-shadow: 0 20px 40px -10px rgba(0,0,0,0.4);"
        "\">"

        # logo
        + logo_html +

        # badge
        "<div style=\""
        "background: rgba(255,255,255,0.15);"
        "border-radius: .5rem;"
        "padding: .5rem .75rem;"
        "font-size: .8rem;"
        "font-weight: 600;"
        "line-height: 1;"
        "display: flex;"
        "align-items: center;"
        "\">"
        "DataHub"
        "</div>"

        "<div style=\"flex:1; font-size:.9rem; font-weight:500;\">"
        "In the real world, hybrid approach combines these three ideas: "
        "community taste (user-based), product similarity (item-based), "
        "content similarity (content-based)."
        "</div>"

        "</div>"
    )

    st.markdown(datahub_banner_html, unsafe_allow_html=True)


def render_styles() -> None:
    """Render custom CSS styles for the application."""
    st.markdown(
        """
        <style>
        .metric-card {
            background-color: #ffffff;
            border-radius: 0.75rem;
            padding: 0.9rem 1rem;
            border: 1px solid rgba(0,0,0,0.07);
            box-shadow: 0 12px 24px -12px rgba(0,0,0,0.20);
            margin-bottom: 1rem;
            min-height: 5.5rem;
        }
        .metric-title {
            font-weight: 500;
            font-size: .8rem;
            color: #6b7280;
            margin-bottom: .25rem;
        }
        .metric-value {
            font-size: 1.15rem;
            font-weight: 600;
            color: #111827;
            line-height: 1.4rem;
            word-break: break-word;
        }

        .header-badge-wrap {
            display: flex;
            flex-wrap: wrap;
            gap: .5rem 1rem;
            margin-bottom: .75rem;
        }
        .header-badge {
            background: #fdf8c7;
            color: #111827;
            display: inline-block;
            padding: .4rem .6rem;
            border-radius: .25rem;
            font-size: .8rem;
            font-weight: 500;
            border: 1px solid #e2e0a8;
        }

        table.var-table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 2rem;
            font-size: 0.9rem;
            background: #ffffff;
            color: #111827;
        }
        table.var-table th {
            text-align: left;
            background: #fdf8c7;
            color: #111827;
            font-weight: 600;
            padding: .5rem .6rem;
            border: 1px solid #d4d4d4;
            white-space: nowrap;
        }
        table.var-table td {
            padding: .5rem .6rem;
            border: 1px solid #d4d4d4;
            vertical-align: top;
            background: #ffffff;
            color: #111827;
        }

        table.stage-table {
            border-collapse: collapse;
            width: 100%;
            margin-top: 1rem;
            font-size: .85rem;
            background: #ffffff;
            color: #111827;
        }
        table.stage-table th {
            text-align: left;
            background: #eef2ff;
            color: #111827;
            font-weight: 600;
            padding: .5rem .6rem;
            border: 1px solid #c7c9df;
            white-space: nowrap;
        }
        table.stage-table td {
            padding: .5rem .6rem;
            border: 1px solid #c7c9df;
            vertical-align: top;
            background: #ffffff;
            color: #111827;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

