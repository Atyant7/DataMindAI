"""
Industrial Light & Dark Theme CSS Engine for DataMind AI.

Provides consistent styling across backgrounds, cards, typography,
inputs, tables, metrics, buttons, sidebar, and chat bubbles.
"""

from __future__ import annotations

import streamlit as st


def get_theme_css(theme: str = "dark") -> str:
    """Generate CSS rules based on selected theme ('dark' or 'light')."""
    # Define base color palette
    if theme == "light":
        bg_primary = "#F8FAFC"
        bg_secondary = "#FFFFFF"
        bg_card = "#FFFFFF"
        bg_card_hover = "#F1F5F9"
        border_color = "#E2E8F0"
        text_primary = "#0F172A"
        text_secondary = "#475569"
        text_muted = "#94A3B8"
        accent = "#2563EB"
        accent_hover = "#1D4ED8"
        accent_light = "#EFF6FF"
        chat_user_bg = "#E0E7FF"
        chat_assistant_bg = "#F8FAFC"
        sidebar_bg = "#F1F5F9"
    else:
        # Dark Theme (default)
        bg_primary = "#0B0F19"
        bg_secondary = "#111827"
        bg_card = "#1E293B"
        bg_card_hover = "#283548"
        border_color = "#334155"
        text_primary = "#F8FAFC"
        text_secondary = "#CBD5E1"
        text_muted = "#64748B"
        accent = "#3B82F6"
        accent_hover = "#60A5FA"
        accent_light = "#1E3A8A"
        chat_user_bg = "#1E293B"
        chat_assistant_bg = "#0F172A"
        sidebar_bg = "#0F172A"

    # Typography scale custom properties
    type_scale = {
        "h1": "2.5rem",
        "h2": "2rem",
        "h3": "1.75rem",
        "body": "1rem",
        "caption": "0.85rem",
    }
    # Spacing scale (rem)
    spacing = {
        "xs": "0.25rem",
        "sm": "0.5rem",
        "md": "1rem",
        "lg": "1.5rem",
        "xl": "2rem",
    }
    # Elevation shadows
    elevation = {
        "card": "0 2px 4px rgba(0,0,0,0.05)",
        "modal": "0 4px 12px rgba(0,0,0,0.1)",
    }

    return f"""
    <style>
    /* Base Body & Container */
    .stApp {{
        background-color: {bg_primary};
        color: {text_primary};
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        /* Typography custom properties */
        --font-h1: {type_scale["h1"]};
        --font-h2: {type_scale["h2"]};
        --font-h3: {type_scale["h3"]};
        --font-body: {type_scale["body"]};
        --font-caption: {type_scale["caption"]};
        /* Spacing custom properties */
        --spacing-xs: {spacing["xs"]};
        --spacing-sm: {spacing["sm"]};
        --spacing-md: {spacing["md"]};
        --spacing-lg: {spacing["lg"]};
        --spacing-xl: {spacing["xl"]};
        /* Elevation custom properties */
        --elev-card: {elevation["card"]};
        --elev-modal: {elevation["modal"]};
    }}

    /* Focus-visible outline for interactive elements */
    button:focus-visible, a:focus-visible, input:focus-visible, select:focus-visible, textarea:focus-visible {{
        outline: 2px solid {accent};
        outline-offset: 2px;
    }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        border-right: 1px solid {border_color};
    }}
    section[data-testid="stSidebar"] .block-container {{
        padding-top: 1.5rem;
    }}

    /* Card & Container Blocks */
    div.dm-card {{
        background-color: {bg_card};
        border: 1px solid {border_color};
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: all 0.2s ease-in-out;
    }}
    div.dm-card:hover {{
        border-color: {accent};
        background-color: {bg_card_hover};
    }}

    /* Metric Containers */
    div[data-testid="stMetric"] {{
        background-color: {bg_card};
        border: 1px solid {border_color};
        border-radius: 8px;
        padding: 0.75rem 1rem;
    }}
    div[data-testid="stMetricValue"] {{
        color: {text_primary} !important;
        font-weight: 700;
    }}
    div[data-testid="stMetricLabel"] {{
        color: {text_muted} !important;
        font-size: 0.85rem;
        font-weight: 500;
    }}

    /* Typography & Headers */
    h1, h2, h3, h4, h5, h6 {{
        color: {text_primary} !important;
        font-weight: 600;
        letter-spacing: -0.02em;
    }}
    p, span, label {{
        color: {text_secondary};
    }}

    /* Buttons */
    button[kind="primary"] {{
        background-color: {accent} !important;
        border: 1px solid {accent} !important;
        color: #FFFFFF !important;
        font-weight: 600;
        border-radius: 6px;
        transition: background-color 0.15s ease;
    }}
    button[kind="primary"]:hover {{
        background-color: {accent_hover} !important;
    }}
    button[kind="secondary"] {{
        background-color: {bg_card} !important;
        border: 1px solid {border_color} !important;
        color: {text_primary} !important;
        border-radius: 6px;
    }}
    button[kind="secondary"]:hover {{
        border-color: {accent} !important;
        color: {accent} !important;
    }}

    /* Input Fields */
    input, textarea, select {{
        background-color: {bg_secondary} !important;
        color: {text_primary} !important;
        border: 1px solid {border_color} !important;
        border-radius: 6px !important;
    }}

    /* Chat Messages */
    [data-testid="stChatMessage"] {{
        padding: 0.85rem 1rem;
        border-radius: 10px;
        margin-bottom: 0.75rem;
        border: 1px solid {border_color};
    }}
    [data-testid="stChatMessage"][data-avatar="assistant"] {{
        background-color: {chat_assistant_bg};
    }}
    [data-testid="stChatMessage"][data-avatar="user"] {{
        background-color: {chat_user_bg};
    }}

    /* Chat Input Bar */
    [data-testid="stChatInput"] {{
        border-color: {border_color} !important;
        border-radius: 8px !important;
    }}

    /* DataFrames & Tables */
    div[data-testid="stDataFrame"] {{
        border: 1px solid {border_color};
        border-radius: 8px;
        overflow: hidden;
    }}

    /* Popovers & Modals */
    div[data-testid="stPopoverBody"] {{
        background-color: {bg_card} !important;
        border: 1px solid {border_color} !important;
        border-radius: 8px !important;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2) !important;
    }}

    /* Tabs Styling */
    button[data-baseweb="tab"] {{
        font-weight: 500 !important;
        color: {text_secondary} !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {accent} !important;
        font-weight: 600 !important;
    }}

    /* Expanders */
    div[data-testid="stExpander"] {{
        background-color: {bg_card};
        border: 1px solid {border_color};
        border-radius: 8px;
    }}

    /* Alerts and Dividers */
    hr {{
        border-color: {border_color} !important;
        margin: 1.25rem 0;
    }}
    </style>
    """


def apply_theme(theme: str | None = None) -> None:
    """Inject CSS stylesheet for current theme into Streamlit page."""
    current_theme = theme or st.session_state.get("theme", "dark")
    st.markdown(get_theme_css(current_theme), unsafe_allow_html=True)
