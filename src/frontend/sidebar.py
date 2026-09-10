"""
DataMindAI sidebar navigation, user profile, project switcher,
and session controls.
"""

from __future__ import annotations

import streamlit as st

from src.core.app_state import app_state
from src.core.config import SUPPORTED_FILE_TYPES
from src.services.chat_service import ChatService
from src.services.project_service import ProjectService

from src.frontend.constants import (
    PRIMARY_SECTIONS,
    WORKSPACE_SUBSECTIONS,
)

from src.frontend.components import render_icon, render_avatar


# ============================================================
# SIDEBAR STATE
# ============================================================

def init_sidebar_state() -> None:
    """Initialize sidebar-related Streamlit session state."""

    if "uploader_version" not in st.session_state:
        st.session_state["uploader_version"] = 0

    if "active_section" not in st.session_state:
        st.session_state["active_section"] = "Dashboard"

    if "active_chat_id" not in st.session_state:
        st.session_state["active_chat_id"] = None


# ============================================================
# PROJECT SWITCHING
# ============================================================

def open_project_workspace(user_id: str, project: dict) -> None:
    """
    Switch the active project.

    Parameters
    ----------
    user_id : str
        ID of the logged-in user.

    project : dict
        Project selected by the user.
    """

    # Store selected project
    st.session_state["current_project"] = project

    # Clear active chat when switching projects
    st.session_state["active_chat_id"] = None

    # Reset current chat state
    app_state.reset_chat()

    # Update app_state if the method/property exists
    try:

        if hasattr(app_state, "set_project"):
            app_state.set_project(project)

        elif hasattr(app_state, "current_project"):
            app_state.current_project = project

    except Exception:
        # Project switching should not crash the sidebar
        pass


# ============================================================
# SIDEBAR
# ============================================================

def show_sidebar():
    """
    Render the Streamlit sidebar.

    Returns
    -------
    tuple
        (uploaded_file, selected_section)
    """

    # --------------------------------------------------------
    # Initialize state
    # --------------------------------------------------------

    init_sidebar_state()

    render_icon("dashboard")

    user = st.session_state.get("user")

    user_id = user["id"] if user else None

    # ========================================================
    # SIDEBAR
    # ========================================================

    with st.sidebar:

        # ====================================================
        # BRANDING
        # ====================================================

        st.html(
            """
            <div style="
                padding: 0.25rem 0 0.5rem 0;
                text-align: center;
            ">

                <div style="
                    font-size: 1.35rem;
                    font-weight: 700;
                ">
                    🧠 DataMind AI
                </div>

                <div style="
                    opacity: 0.7;
                    font-size: 0.78rem;
                    margin-top: 0.15rem;
                ">
                    Autonomous AI Data Scientist
                </div>

            </div>
            """
        )

        # ====================================================
        # PROJECT SELECTOR
        # ====================================================

        if user_id:

            user_projects = ProjectService.list_user_projects(
                user_id,
                include_archived=False,
            )

            current_project = st.session_state.get(
                "current_project"
            )

            if user_projects:

                project_names = [
                    project["name"]
                    for project in user_projects
                ]

                # Default selected project
                current_index = 0

                if (
                    current_project
                    and current_project.get("name")
                    in project_names
                ):

                    current_index = project_names.index(
                        current_project["name"]
                    )

                selected_project_name = st.selectbox(
                    "Active Project",
                    options=project_names,
                    index=current_index,
                    key="sidebar_project_selector",
                )

                # ------------------------------------------------
                # Detect project change
                # ------------------------------------------------

                if (
                    not current_project
                    or current_project.get("name")
                    != selected_project_name
                ):

                    new_active_project = user_projects[
                        project_names.index(
                            selected_project_name
                        )
                    ]

                    open_project_workspace(
                        user_id,
                        new_active_project,
                    )

        # ====================================================
        # DATASET UPLOADER
        # ====================================================

        with st.container():

            st.html(
                """
                <div class="dm-card" style="
                    margin-bottom: 1rem;
                    border: 1px solid var(--border-color);
                    padding: 10px;
                    border-radius: 8px;
                ">

                    <h3 style="
                        margin-top: 0;
                        font-size: 0.9rem;
                    ">
                        Upload Dataset
                    </h3>

                </div>
                """
            )

            uploaded_file = st.file_uploader(
                "",
                type=SUPPORTED_FILE_TYPES,
                key=(
                    "dataset_uploader_"
                    f"{st.session_state['uploader_version']}"
                ),
                label_visibility="collapsed",
            )

        # ====================================================
        # DATASET INFORMATION
        # ====================================================

        if app_state.has_dataset():

            st.success(
                f"**{app_state.dataset_name}**\n\n"
                f"{len(app_state.dataset):,} rows × "
                f"{len(app_state.dataset.columns)} columns"
            )

        # ====================================================
        # WORKSPACE NAVIGATION
        # ====================================================

        st.divider()

        st.caption("WORKSPACE")

        # Current active section
        primary_active = st.session_state.get(
            "active_section",
            "Dashboard",
        )

        selected_primary = primary_active

        # ====================================================
        # PRIMARY NAVIGATION
        # ====================================================

        for section in PRIMARY_SECTIONS:

            is_active = (
                section == primary_active
            )

            # -----------------------------------------------
            # IMPORTANT:
            #
            # Do NOT pass HTML into st.button().
            #
            # st.button() expects normal text.
            # -----------------------------------------------

            button_label = section

            if st.button(
                button_label,
                key=f"nav_{section}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):

                st.session_state[
                    "active_section"
                ] = section

                selected_primary = section

                st.rerun()

        # ====================================================
        # WORKSPACE SUB-SECTIONS
        # ====================================================

        if selected_primary == "Workspace":

            sub_active = st.session_state.get(
                "workspace_subsection",
                "Dataset",
            )

            if sub_active not in WORKSPACE_SUBSECTIONS:

                sub_active = "Dataset"

            selected_sub = st.radio(
                "Workspace",
                options=WORKSPACE_SUBSECTIONS,
                index=WORKSPACE_SUBSECTIONS.index(
                    sub_active
                ),
                key="workspace_subsection",
                label_visibility="collapsed",
            )

            st.session_state[
                "workspace_subsection"
            ] = selected_sub

            st.session_state[
                "active_section"
            ] = selected_sub

            selected_section = selected_sub

        else:

            st.session_state[
                "active_section"
            ] = selected_primary

            selected_section = selected_primary

        # ====================================================
        # NEW CHAT
        # ====================================================

        st.divider()

        if st.button(
            "💬 New Chat",
            use_container_width=True,
        ):

            # Reset current chat
            app_state.reset_chat()

            # Create a new chat for current project
            current_project = st.session_state.get(
                "current_project"
            )

            if user_id and current_project:

                project_id = current_project["id"]

                new_chat = ChatService.create_chat(
                    user_id,
                    project_id,
                    "New Chat",
                )

                st.session_state[
                    "active_chat_id"
                ] = new_chat["id"]

            # Navigate to Chat
            st.session_state[
                "active_section"
            ] = "Chat"

            st.rerun()

        # ====================================================
        # ACCOUNT SECTION
        # ====================================================

        st.divider()

        with st.container():

            col_a, col_b, col_c = st.columns(
                [1, 4, 2]
            )

            # -----------------------------------------------
            # Avatar
            # -----------------------------------------------

            with col_a:

                if user:

                    render_avatar(
                        user.get(
                            "email",
                            "U",
                        )
                    )

                else:

                    render_avatar("U")

            # -----------------------------------------------
            # User name
            # -----------------------------------------------

            with col_b:

                if user:

                    st.write(
                        f"**{user.get('name', 'User')}**"
                    )

                else:

                    st.write("**User**")

            # -----------------------------------------------
            # Sign out
            # -----------------------------------------------

            with col_c:

                if st.button(
                    "Sign Out",
                    key="sidebar_signout",
                ):

                    app_state.reset()

                    st.session_state.clear()

                    st.rerun()

        # ====================================================
        # CLEAR SESSION
        # ====================================================

        if st.button(
            "🗑 Clear Session",
            use_container_width=True,
        ):

            app_state.reset()

            # Force a fresh uploader
            st.session_state[
                "uploader_version"
            ] += 1

            # Go back to dashboard
            st.session_state[
                "active_section"
            ] = "Dashboard"

            st.rerun()

    # ========================================================
    # RETURN VALUES
    # ========================================================

    return uploaded_file, selected_section