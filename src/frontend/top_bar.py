"""
Top bar component for DataMind AI.

Provides consistent global branding, active project pill, theme toggle,
and user profile / logout dropdown.
"""

from __future__ import annotations

import streamlit as st

from src.core.app_state import app_state
from src.services.auth_service import AuthService
from src.services.notification_service import NotificationService


def render_top_bar(user: dict, current_project: dict | None = None) -> None:
    """Render the industrial SaaS top bar."""
    is_chat = st.session_state.get("active_section") == "Chat"

    if is_chat:
        # Clean, minimal top bar for Chat: Branding on left, Theme & User on right
        top_col1, top_col2, top_col3 = st.columns([6, 1, 2])
    else:
        top_col1, top_col2, top_col3, top_col4 = st.columns([4, 2, 1, 2])

    # 1. Left: Branding and Active Project Pill
    with top_col1:
        proj_badge = ""
        if current_project:
            proj_badge = f"""
            <span style="
                background: rgba(59, 130, 246, 0.15);
                border: 1px solid rgba(59, 130, 246, 0.35);
                padding: 3px 10px;
                border-radius: 12px;
                font-size: 0.8rem;
                font-weight: 500;
                margin-left: 10px;
                vertical-align: middle;
            ">📁 {current_project['name']}</span>
            """
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; height: 100%;">
                <span style="font-size: 1.25rem; font-weight: 700; letter-spacing: -0.02em;">DataMind AI</span>
                {proj_badge}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Notifications Popover (omitted on Chat page to keep header clean and minimal)
    if not is_chat:
        with top_col2:
            unread_notifs = NotificationService.get_unread_notifications(user["id"], limit=5)
            bell_icon = f"🔔 ({len(unread_notifs)})" if unread_notifs else "🔔"

            with st.popover(bell_icon, use_container_width=True):
                st.markdown("#### 🔔 Notifications")
                if unread_notifs:
                    for notif in unread_notifs:
                        lvl = notif.get("level", "info")
                        prefix = "✓" if lvl == "success" else ("✕" if lvl == "error" else "ℹ")
                        st.markdown(f"**{prefix} {notif['title']}**")
                        st.caption(notif["message"])
                        st.divider()

                    if st.button("Mark all as read", key="mark_notifs_read_btn", use_container_width=True):
                        NotificationService.mark_all_as_read(user["id"])
                        st.rerun()
                else:
                    st.caption("No new notifications.")

    # 3. Theme Quick Toggle
    theme_col = top_col2 if is_chat else top_col3
    with theme_col:
        current_theme = st.session_state.get("theme", user.get("theme_preference", "dark"))
        target_theme = "light" if current_theme == "dark" else "dark"
        theme_icon = "☀️" if current_theme == "dark" else "🌙"

        if st.button(theme_icon, help=f"Switch to {target_theme.title()} Mode", key="top_theme_toggle_btn", use_container_width=True):
            AuthService.update_theme(user["id"], target_theme)
            st.session_state["theme"] = target_theme
            st.rerun()

    # 4. User Profile Dropdown Menu
    user_col = top_col3 if is_chat else top_col4
    with user_col:
        with st.popover(f"👤 {user['username']}", use_container_width=True):
            st.markdown(f"**{user['username']}**")
            st.caption(user.get("email", ""))
            st.divider()

            if st.button("📊 Dashboard", key="top_nav_dash", use_container_width=True):
                st.session_state["active_section"] = "Dashboard"
                st.rerun()

            if st.button("📁 Projects", key="top_nav_projects", use_container_width=True):
                st.session_state["active_section"] = "Projects"
                st.rerun()

            if st.button("⚙️ Settings & Profile", key="top_nav_settings", use_container_width=True):
                st.session_state["active_section"] = "Settings"
                st.rerun()

            st.divider()

            if st.button("🚪 Sign Out", key="top_nav_logout", type="secondary", use_container_width=True):
                st.session_state.clear()
                app_state.reset()
                st.rerun()

    st.markdown("<hr style='margin: 0.35rem 0 0.85rem 0;'>", unsafe_allow_html=True)
