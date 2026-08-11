"""DataMindAI sidebar navigation and session controls."""

from __future__ import annotations

import streamlit as st

from src.core.app_state import app_state
from src.core.config import SUPPORTED_FILE_TYPES


WORKSPACE_SECTIONS = ("Chat", "Model", "Prediction")


def _initialize_sidebar_state() -> None:
    """Initialize sidebar state without binding it to a widget key."""
    if "uploader_version" not in st.session_state:
        st.session_state["uploader_version"] = 0

    if "active_section" not in st.session_state:
        st.session_state["active_section"] = "Chat"


def show_sidebar():
    """
    Render the collapsible Streamlit sidebar.

    Returns
    -------
    tuple
        (uploaded_file, active_section)

    Notes
    -----
    ``active_section`` is intentionally NOT used as the key of the
    radio widget. Streamlit does not allow a widget's keyed session
    state value to be modified after that widget has been created.
    """

    _initialize_sidebar_state()

    with st.sidebar:
        st.markdown(
            """
            <div style="
                padding: 0.35rem 0 0.75rem 0;
                text-align: center;
            ">
                <div style="
                    font-size: 1.35rem;
                    font-weight: 700;
                ">
                    🧠 DataMindAI
                </div>
                <div style="
                    opacity: 0.6;
                    font-size: 0.78rem;
                    margin-top: 0.15rem;
                ">
                    AI Data Scientist
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        uploaded_file = st.file_uploader(
            "Upload Dataset",
            type=SUPPORTED_FILE_TYPES,
            key=f"dataset_uploader_{st.session_state['uploader_version']}",
            help="Supported formats: CSV and XLSX.",
        )

        if app_state.has_dataset():
            st.success(
                f"**{app_state.dataset_name}**\n\n"
                f"{len(app_state.dataset):,} rows × "
                f"{len(app_state.dataset.columns)} columns"
            )

        st.divider()
        st.caption("WORKSPACE")

        current_section = st.session_state.get(
            "active_section",
            "Chat",
        )

        if current_section not in WORKSPACE_SECTIONS:
            current_section = "Chat"

        selected_index = WORKSPACE_SECTIONS.index(
            current_section
        )

        selected_section = st.radio(
            "Workspace",
            options=WORKSPACE_SECTIONS,
            index=selected_index,
            label_visibility="collapsed",
        )

        # This is safe because ``active_section`` is NOT the widget key.
        st.session_state["active_section"] = selected_section

        st.divider()

        if st.button(
            "💬 New Chat",
            use_container_width=True,
        ):
            app_state.reset_chat()
            st.session_state["active_section"] = "Chat"
            st.rerun()

        if st.button(
            "🗑 Clear Session",
            use_container_width=True,
        ):
            app_state.reset()
            st.session_state["uploader_version"] += 1
            st.session_state["active_section"] = "Chat"
            st.rerun()

        st.divider()

        st.caption(
            "Use Streamlit's sidebar control to collapse or reopen "
            "the sidebar."
        )

    return uploaded_file, selected_section