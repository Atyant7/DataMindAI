"""
Comprehensive Settings, User Profile, Security, and Data Privacy for DataMind AI.

Supports:
- User profile viewing and updating
- Visual statistics summary
- Appearance / Theme preference
- Password change with strict verification
- User data export (JSON download)
- Account deletion with strict confirmation
- Project settings (rename, delete)
"""

from __future__ import annotations

import json
import streamlit as st

from src.core.app_state import app_state
from src.services.auth_service import AuthService
from src.services.project_service import ProjectService


def show_settings_page(
    user_data: dict,
    current_project: dict | None,
) -> None:
    """Render the comprehensive Account & Settings workspace."""
    st.markdown("## ⚙️ Account & Project Settings")

    tab_profile, tab_appearance, tab_security, tab_data, tab_project = st.tabs(
        [
            "👤 Profile",
            "🎨 Appearance",
            "🔒 Security",
            "💾 Data & Privacy",
            "📁 Project Settings",
        ]
    )

    user_id = user_data["id"]

    # --------------------------------------------------------
    # 1. PROFILE TAB
    # --------------------------------------------------------
    with tab_profile:
        st.markdown("### User Profile")

        # Visual Stats
        stats = ProjectService.get_user_dashboard_stats(user_id)
        stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
        with stat_c1:
            st.metric("Projects", stats["projects"])
        with stat_c2:
            st.metric("Datasets", stats["datasets"])
        with stat_c3:
            st.metric("Models", stats["models"])
        with stat_c4:
            st.metric("Predictions", stats["predictions"])

        st.divider()

        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.markdown(f"**Username:** `{user_data['username']}`")
            st.markdown(f"**Email:** `{user_data['email']}`")
        with info_col2:
            st.markdown(f"**Account ID:** `{user_data['id']}`")
            created_str = user_data.get("created_at", "")[:10]
            st.markdown(f"**Member Since:** `{created_str or 'Recent'}`")

        st.divider()

        with st.expander("✏️ **Edit Profile Information**"):
            with st.form("edit_profile_form"):
                new_username = st.text_input("Username", value=user_data["username"])
                new_email = st.text_input("Email", value=user_data["email"])
                save_profile_btn = st.form_submit_button("Save Profile", type="primary")

                if save_profile_btn:
                    try:
                        updated_user = AuthService.update_profile(
                            user_id=user_id,
                            username=new_username,
                            email=new_email,
                        )
                        st.session_state["user"] = updated_user
                        st.success("Profile updated successfully!")
                        st.rerun()
                    except Exception as err:
                        st.error(str(err))

    # --------------------------------------------------------
    # 2. APPEARANCE TAB
    # --------------------------------------------------------
    with tab_appearance:
        st.markdown("### Appearance & Theme")
        st.caption("Customize your visual interface theme. Preferences are saved to your account.")

        current_theme = st.session_state.get("theme", user_data.get("theme_preference", "dark"))
        selected_theme = st.radio(
            "Color Theme",
            options=["dark", "light"],
            index=0 if current_theme == "dark" else 1,
            format_func=lambda x: "🌙 Dark Mode" if x == "dark" else "☀️ Light Mode",
            horizontal=True,
        )

        if selected_theme != current_theme:
            AuthService.update_theme(user_id, selected_theme)
            st.session_state["theme"] = selected_theme
            st.success(f"Switched theme to {selected_theme.title()} Mode.")
            st.rerun()

    # --------------------------------------------------------
    # 3. SECURITY TAB
    # --------------------------------------------------------
    with tab_security:
        st.markdown("### Security & Password")
        st.caption("Update your login password. Passwords are protected using NIST-compliant PBKDF2 hashing.")

        with st.form("change_password_form"):
            curr_pass = st.text_input("Current Password", type="password")
            new_pass = st.text_input("New Password (min 6 characters)", type="password")
            confirm_new_pass = st.text_input("Confirm New Password", type="password")
            update_pass_btn = st.form_submit_button("Update Password", type="primary")

            if update_pass_btn:
                if not curr_pass or not new_pass:
                    st.error("Please fill in all password fields.")
                elif new_pass != confirm_new_pass:
                    st.error("New password and confirmation do not match.")
                else:
                    try:
                        AuthService.change_password(user_id, curr_pass, new_pass)
                        st.success("Password updated successfully!")
                    except Exception as pass_err:
                        st.error(str(pass_err))

    # --------------------------------------------------------
    # 4. DATA & PRIVACY TAB
    # --------------------------------------------------------
    with tab_data:
        st.markdown("### Data Management & Export")
        st.caption("Download a copy of your account data or permanently delete your account.")

        # Data Export
        st.markdown("#### 📥 Export My Data")
        st.write("Export your complete profile, projects metadata, chat history, models, and predictions as JSON.")

        try:
            export_payload = AuthService.export_user_data(user_id)
            json_str = json.dumps(export_payload, indent=2)
            st.download_button(
                label="📥 Download User Data (JSON)",
                data=json_str,
                file_name=f"datamind_export_{user_data['username']}.json",
                mime="application/json",
            )
        except Exception as exp_err:
            st.warning(f"Could not prepare data export: {exp_err}")

        st.divider()

        # Account Deletion
        st.markdown("#### ⚠️ Delete Account")
        st.error("Permanently delete your account and all associated projects, datasets, models, and chats. This action is irreversible.")

        with st.expander("🚨 **Permanently Delete Account**"):
            with st.form("delete_account_form"):
                st.write("Type **DELETE** to confirm:")
                confirm_word = st.text_input("Confirmation", placeholder="DELETE")
                del_pass = st.text_input("Enter Your Password", type="password")
                del_btn = st.form_submit_button("Permanently Delete Account", type="primary")

                if del_btn:
                    if confirm_word.strip() != "DELETE":
                        st.error("Please type DELETE to confirm.")
                    elif not del_pass:
                        st.error("Please enter your password.")
                    else:
                        try:
                            AuthService.delete_account(user_id, del_pass)
                            st.session_state.clear()
                            app_state.reset()
                            st.success("Account deleted. Redirecting...")
                            st.rerun()
                        except Exception as del_err:
                            st.error(str(del_err))

    # --------------------------------------------------------
    # 5. PROJECT SETTINGS TAB
    # --------------------------------------------------------
    with tab_project:
        if current_project is None:
            st.info("No active project currently open. Open a project from the Dashboard or Projects page.")
        else:
            st.markdown(f"### Active Project: **{current_project['name']}**")
            st.caption(f"Project ID: `{current_project['id']}`")

            with st.form("settings_edit_project_form"):
                proj_name = st.text_input("Project Name", value=current_project["name"])
                proj_desc = st.text_area("Description", value=current_project.get("description") or "")
                update_btn = st.form_submit_button("Save Changes", type="primary")

                if update_btn:
                    if not proj_name.strip():
                        st.error("Project name cannot be empty.")
                    else:
                        updated = ProjectService.update_project(
                            user_id=user_id,
                            project_id=current_project["id"],
                            name=proj_name,
                            description=proj_desc,
                        )
                        if updated:
                            st.session_state["current_project"] = updated
                            st.success("Project updated successfully!")
                            st.rerun()

            st.divider()
            st.markdown("#### Danger Zone")
            st.warning("Deleting this project will permanently remove its datasets, models, chats, and predictions.")

            if st.button("🗑 Delete Project", type="secondary", key="settings_del_proj_btn"):
                st.session_state["settings_confirm_delete_project"] = True

            if st.session_state.get("settings_confirm_delete_project"):
                st.error(f"Are you sure you want to delete '{current_project['name']}'?")
                sc1, sc2 = st.columns(2)
                with sc1:
                    if st.button("Confirm Delete", key="settings_proj_del_yes", type="primary", use_container_width=True):
                        ProjectService.delete_project(
                            user_id=user_id,
                            project_id=current_project["id"],
                        )
                        st.session_state["current_project"] = None
                        app_state.active_project_id = None
                        st.session_state["active_section"] = "Dashboard"
                        del st.session_state["settings_confirm_delete_project"]
                        st.success("Project deleted.")
                        st.rerun()
                with sc2:
                    if st.button("Cancel", key="settings_proj_del_no", use_container_width=True):
                        del st.session_state["settings_confirm_delete_project"]
                        st.rerun()
