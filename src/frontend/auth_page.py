"""
Authentication view for DataMind AI.

Provides Login and Registration with secure credential verification.
"""

from __future__ import annotations

import streamlit as st

from src.services.auth_service import AuthService


def show_auth_page() -> None:
    """Render the login and registration screen."""
    st.markdown(
        """
        <div style="text-align: center; margin-top: 2rem; margin-bottom: 2rem;">
            <h1 style="font-size: 2.75rem; margin-bottom: 0.25rem;">🧠 DataMind AI</h1>
            <p style="font-size: 1.15rem; opacity: 0.8;">Autonomous Enterprise AI Data Scientist</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        tab_login, tab_register = st.tabs(["🔐 Sign In", "📝 Create Account"])

        # ----------------------------------------------------
        # LOGIN TAB
        # ----------------------------------------------------
        with tab_login:
            st.write("")
            with st.form("login_form"):
                username_or_email = st.text_input(
                    "Username or Email",
                    placeholder="e.g. atyant or user@example.com",
                )
                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter your password",
                )
                submit_login = st.form_submit_button(
                    "Sign In",
                    type="primary",
                    use_container_width=True,
                )

                if submit_login:
                    if not username_or_email or not password:
                        st.error("Please provide both username/email and password.")
                    else:
                        user = AuthService.authenticate(username_or_email, password)
                        if user:
                            st.session_state["user"] = user
                            st.session_state["theme"] = user.get("theme_preference", "dark")
                            st.success(f"Welcome back, {user['username']}!")
                            st.rerun()
                        else:
                            st.error("Invalid username/email or password.")

        # ----------------------------------------------------
        # REGISTER TAB
        # ----------------------------------------------------
        with tab_register:
            st.write("")
            with st.form("register_form"):
                new_username = st.text_input(
                    "Choose Username",
                    placeholder="e.g. data_analyst",
                )
                new_email = st.text_input(
                    "Email Address",
                    placeholder="name@company.com",
                )
                new_password = st.text_input(
                    "Password (min 6 characters)",
                    type="password",
                    placeholder="••••••••",
                )
                confirm_password = st.text_input(
                    "Confirm Password",
                    type="password",
                    placeholder="••••••••",
                )
                theme_pref = st.selectbox(
                    "Preferred Theme",
                    options=["dark", "light"],
                    index=0,
                    format_func=lambda x: "🌙 Dark Mode" if x == "dark" else "☀️ Light Mode",
                )
                submit_register = st.form_submit_button(
                    "Create Account",
                    type="primary",
                    use_container_width=True,
                )

                if submit_register:
                    if not new_username or not new_email or not new_password:
                        st.error("Please fill in all required fields.")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match.")
                    else:
                        try:
                            user = AuthService.register_user(
                                username=new_username,
                                email=new_email,
                                password=new_password,
                                theme_preference=theme_pref,
                            )
                            st.session_state["user"] = user
                            st.session_state["theme"] = user.get("theme_preference", "dark")
                            st.success("Account created successfully! Logging you in...")
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))
