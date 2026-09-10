"""Reusable UI components for DataMind AI frontend.

This module provides:
- `render_icon(name, size=24, color=None)`: Returns an SVG icon string
  from a minimal built‑in Lucide icon set. The SVG is injected via
  `st.markdown(..., unsafe_allow_html=True)` so it can be used inside any
  Streamlit container.
- `render_card(title, body_fn, actions=None)`: Helper to render a consistent
  card UI (`.dm-card` CSS class defined in `styles.py`).  The `body_fn` is a
  callable that receives a `st.container` in which the caller can place any
  Streamlit widgets.  Optional `actions` is a list of Streamlit ``Button``
  objects or markdown links that will be rendered in a right‑aligned action
  bar.

The implementation is intentionally lightweight – the icon set includes only
the icons required for the current UI.  Adding more icons later only requires
extending the ``_ICON_SVGS`` dictionary.
"""

from __future__ import annotations

import streamlit as st
from typing import Callable, List, Optional

# Minimal Lucide‑style SVG icons (MIT‑licensed).  The SVG strings are kept
# inline to avoid external file dependencies.  Additional icons can be added
# by extending this dictionary.
_ICON_SVGS: dict[str, str] = {
    "dashboard": """<svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M3 13h8V3H3v10zM13 21h8V11h-8v10zM3 21h8v-6H3v6zM13 3v6h8V3h-8z'></path></svg>""",
    "project": """<svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M3 7h18'></path><path d='M3 12h18'></path><path d='M3 17h18'></path></svg>""",
    "chat": """<svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2z'></path></svg>""",
    "workspace": """<svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><rect x='3' y='3' width='7' height='7'></rect><rect x='14' y='3' width='7' height='7'></rect><rect x='14' y='14' width='7' height='7'></rect><rect x='3' y='14' width='7' height='7'></rect></svg>""",
    "prediction": """<svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M12 2l7 7-7 7-7-7 7-7z'></path></svg>""",
    "settings": """<svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><circle cx='12' cy='12' r='3'></circle><path d='M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33h.09a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09c0 .7.4 1.32 1 1.51a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82v.09c0 .7.4 1.32 1 1.51A1.65 1.65 0 0 0 21 12v.09a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z'></path></svg>""",
}


def render_icon(name: str, size: int = 24, color: Optional[str] = None) -> None:
    """Render an SVG icon inline.

    Parameters
    ----------
    name: str
        Key of the icon in ``_ICON_SVGS``.
    size: int, optional
        Width and height in pixels (default 24).
    color: str | None, optional
        CSS colour to apply via ``stroke``. If ``None`` the current text colour
        is used.
    """
    svg = _ICON_SVGS.get(name)
    if not svg:
        # Gracefully degrade – render a placeholder box.
        svg = f"<svg width='{size}' height='{size}'><rect width='100%' height='100%' fill='%23ccc'></rect></svg>"
    # Adjust size and colour if supplied.
    if size != 24:
        svg = svg.replace("width='24'", f"width='{size}'").replace("height='24'", f"height='{size}'")
    if color:
        svg = svg.replace("stroke='currentColor'", f"stroke='{color}'")
    st.markdown(svg, unsafe_allow_html=True)


def render_section_header(title: str, icon: str | None = None, action: Optional[object] = None) -> None:
    """Render a section header with optional icon and optional action button.

    Parameters
    ----------
    title: str
        Header text.
    icon: str | None
        Icon name from SECTION_ICONS; rendered via ``render_icon``.
    action: st.Button | None
        Optional Streamlit button displayed on the right.
    """
    # Build HTML for header
    icon_html = f"{render_icon(icon, size=20)} " if icon else ""
    # Use a container to layout flex
    st.markdown(
        f"""
        <div style='display:flex; align-items:center; justify-content:space-between; margin-bottom:0.75rem;'>
            <div style='display:flex; align-items:center; gap:0.4rem;'>
                {icon_html}<h3 style='margin:0;'>{title}</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if action:
        # Place action button after header (simple implementation)
        st.button(action.label, key=action.key)

def render_card(
    title: str,
    body_fn: Callable[[st.container], None],
    actions: Optional[List[st.Button]] = None,
) -> None:
    """Render a consistent card UI.

    The card uses the `.dm-card` CSS class defined in ``styles.py``.  The caller
    supplies a ``body_fn`` that receives a Streamlit container where any widgets
    can be placed.  ``actions`` can contain pre‑created Streamlit button objects
    which will be rendered in a right‑aligned action bar.
    """
    # Card wrapper with title.
    st.markdown(
        f"""
        <div class='dm-card'>
            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;'>
                <h3 style='margin:0;'>{title}</h3>
                <div class='dm-card-actions'>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )
    # Body content.
    container = st.container()
    body_fn(container)
    # Optional action buttons.
    if actions:
        for btn in actions:
            btn
    # Close card.
    st.markdown("""
        </div>
    """, unsafe_allow_html=True)

__all__ = ["render_icon", "render_card", "render_avatar", "render_section_header"]


def render_avatar(name: str, size: int = 32) -> None:
    """Render a circular avatar with initials derived from a name or email.

    Parameters
    ----------
    name: str
        Full name or email address. Initials are taken from the first two
        alphabetical characters (e.g., "John Doe" -> "JD", "user@example.com" -> "U").
    size: int, optional
        Diameter of the avatar in pixels (default 32).
    """
    # Derive initials
    import re
    initials = "".join([c for c in re.findall(r"[A-Za-z]", name)[:2]]).upper()
    if not initials:
        initials = "U"
    # Simple style – circle with background accent color
    html = f"""
    <div style='
        width:{size}px; height:{size}px; border-radius:50%;
        background:var(--accent); color:white; display:flex;
        align-items:center; justify-content:center; font-weight:600; font-size:{int(size/2)}px;'>
        {initials}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
