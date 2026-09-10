"""
Dedicated Projects page for DataMind AI.

Supports:
- Active vs Archived tabs
- Search by project name
- Task-based filtering (Classification, Regression, Computer Vision, Geospatial)
- Sorting by updated, created, or name
- Server-side database pagination
- Project card action menu (Open, Rename, Duplicate, Archive/Restore, Delete)
"""

from __future__ import annotations

import streamlit as st

from src.core.logger import get_logger
from src.frontend.home import open_project_workspace
from src.services.project_service import ProjectService

logger = get_logger(__name__)

PAGE_SIZE = 6


def show_projects_page(user_data: dict | None = None) -> None:
    """Render the full Projects library workspace."""
    user = user_data or st.session_state.get("user")
    if not user:
        st.warning("Please sign in to view your projects.")
        return

    user_id = user["id"]

    # Header
    head_c1, head_c2 = st.columns([4, 1])
    with head_c1:
        st.markdown("## 📁 Projects Library")
        st.caption("Manage all your active and archived DataMind AI workspaces.")
    with head_c2:
        st.write("")
        if st.button("➕ New Project", type="primary", use_container_width=True):
            st.session_state["show_projects_new_modal"] = True

    # New Project Expander / Modal
    if st.session_state.get("show_projects_new_modal"):
        with st.container():
            st.markdown("### Create New Project")
            with st.form("projects_page_new_project_form"):
                p_name = st.text_input("Project Name")
                p_desc = st.text_area("Description (optional)")
                fc1, fc2 = st.columns([1, 4])
                with fc1:
                    submit_create = st.form_submit_button("Create", type="primary")
                with fc2:
                    cancel_create = st.form_submit_button("Cancel")

                if submit_create:
                    if not p_name.strip():
                        st.error("Please provide a project name.")
                    else:
                        new_p = ProjectService.create_project(user_id, p_name, p_desc)
                        st.session_state["show_projects_new_modal"] = False
                        open_project_workspace(user_id, new_p)

                if cancel_create:
                    st.session_state["show_projects_new_modal"] = False
                    st.rerun()

        st.divider()

    # Tabs: Active vs Archived
    tab_active, tab_archived = st.tabs(["Active Projects", "📦 Archived Projects"])

    with tab_active:
        _render_project_list(user_id, archived_only=False)

    with tab_archived:
        _render_project_list(user_id, archived_only=True)


def _render_project_list(user_id: str, archived_only: bool = False) -> None:
    """Render filtered and paginated list of projects."""
    key_prefix = "arch" if archived_only else "act"

    # Search, Filter, Sort Controls
    filter_c1, filter_c2, filter_c3 = st.columns([3, 2, 2])

    with filter_c1:
        search_query = st.text_input(
            "Search projects",
            placeholder="🔍 Search by project name...",
            key=f"{key_prefix}_search_input",
            label_visibility="collapsed",
        )

    with filter_c2:
        filter_options = ["All Tasks", "Classification", "Regression", "Computer Vision", "Geospatial"]
        selected_task = st.selectbox(
            "Filter Task",
            options=filter_options,
            key=f"{key_prefix}_task_filter",
            label_visibility="collapsed",
        )

    with filter_c3:
        sort_options = {
            "Recently Updated": "updated_at",
            "Recently Created": "created_at",
            "Name (A-Z)": "name_asc",
            "Name (Z-A)": "name_desc",
        }
        selected_sort_label = st.selectbox(
            "Sort by",
            options=list(sort_options.keys()),
            key=f"{key_prefix}_sort_select",
            label_visibility="collapsed",
        )
        sort_by = sort_options[selected_sort_label]

    # Current Page Session State
    page_key = f"{key_prefix}_current_page"
    if page_key not in st.session_state:
        st.session_state[page_key] = 1

    current_page = st.session_state[page_key]

    # Query Projects with Pagination
    result = ProjectService.list_user_projects(
        user_id=user_id,
        include_archived=archived_only,
        archived_only=archived_only,
        search=search_query,
        task_filter=selected_task if selected_task != "All Tasks" else None,
        sort_by=sort_by,
        page=current_page,
        page_size=PAGE_SIZE,
    )

    projects = result.get("projects", [])
    total = result.get("total", 0)
    pages = result.get("pages", 1)

    if total == 0:
        if archived_only:
            st.info("No archived projects found.")
        else:
            st.info("No projects match your search criteria.")
        return

    # Pagination Indicator
    start_idx = (current_page - 1) * PAGE_SIZE + 1
    end_idx = min(start_idx + len(projects) - 1, total)
    st.caption(f"Showing {start_idx}–{end_idx} of {total} projects")

    # Render Project Cards
    for proj in projects:
        with st.container():
            col_info, col_actions = st.columns([4, 1])

            with col_info:
                st.markdown(f"### {proj['name']}")
                ds_name = proj.get("dataset_name") or "No dataset uploaded"
                rows_info = f" • {proj['dataset_rows']:,} rows" if proj.get("dataset_rows") else ""
                cols_info = f" × {proj['dataset_columns']} cols" if proj.get("dataset_columns") else ""
                st.caption(f"📊 `{ds_name}`{rows_info}{cols_info}")

                task_str = f" • {proj['task'].title()}" if proj.get("task") else ""
                model_str = f" • Best Model: **{proj['best_model']}**" if proj.get("best_model") else ""
                metric_str = f" ({proj['best_metric']})" if proj.get("best_metric") else ""
                status_str = "Archived" if proj.get("is_archived") else "Active"
                st.markdown(
                    f"Status: **{status_str}**{task_str}{model_str}{metric_str} · Updated: {proj.get('updated_at', '')[:10]}"
                )

            with col_actions:
                st.write("")
                if st.button("Open →", key=f"open_proj_btn_{proj['id']}", type="primary", use_container_width=True):
                    open_project_workspace(user_id, proj)

                with st.popover("⋮", use_container_width=True):
                    st.markdown(f"**{proj['name']}**")

                    # Rename
                    with st.form(f"lib_rename_{proj['id']}"):
                        new_n = st.text_input("Rename", value=proj["name"])
                        if st.form_submit_button("Save Name"):
                            ProjectService.update_project(user_id, proj["id"], name=new_n)
                            st.success("Project renamed!")
                            st.rerun()

                    # Duplicate
                    if st.button("📋 Duplicate", key=f"lib_dup_{proj['id']}", use_container_width=True):
                        ProjectService.duplicate_project(user_id, proj["id"])
                        st.success("Duplicated!")
                        st.rerun()

                    # Archive / Restore
                    if archived_only:
                        if st.button("♻️ Restore Project", key=f"lib_rest_{proj['id']}", use_container_width=True):
                            ProjectService.archive_project(user_id, proj["id"], archive=False)
                            st.success("Project restored to Active!")
                            st.rerun()
                    else:
                        if st.button("📦 Archive Project", key=f"lib_arch_{proj['id']}", use_container_width=True):
                            ProjectService.archive_project(user_id, proj["id"], archive=True)
                            st.success("Project archived!")
                            st.rerun()

                    # Delete
                    if st.button("🗑 Delete Project", key=f"lib_del_{proj['id']}", type="secondary", use_container_width=True):
                        st.session_state[f"lib_confirm_delete_{proj['id']}"] = True

            # Delete Confirmation Dialog
            if st.session_state.get(f"lib_confirm_delete_{proj['id']}"):
                st.error(f"Permanently delete '{proj['name']}'? All models and datasets will be removed.")
                dc1, dc2 = st.columns(2)
                with dc1:
                    if st.button("Confirm Delete", key=f"lib_del_yes_{proj['id']}", type="primary", use_container_width=True):
                        ProjectService.delete_project(user_id, proj["id"])
                        del st.session_state[f"lib_confirm_delete_{proj['id']}"]
                        st.rerun()
                with dc2:
                    if st.button("Cancel", key=f"lib_del_no_{proj['id']}", use_container_width=True):
                        del st.session_state[f"lib_confirm_delete_{proj['id']}"]
                        st.rerun()

            st.markdown("<hr style='margin: 0.75rem 0;'>", unsafe_allow_html=True)

    # Pagination Buttons
    if pages > 1:
        p_c1, p_c2, p_c3 = st.columns([1, 2, 1])
        with p_c1:
            if st.button("← Previous", key=f"{key_prefix}_prev_btn", disabled=(current_page <= 1)):
                st.session_state[page_key] = max(1, current_page - 1)
                st.rerun()
        with p_c2:
            st.markdown(
                f"<div style='text-align: center; padding-top: 0.4rem;'>Page <strong>{current_page}</strong> of <strong>{pages}</strong></div>",
                unsafe_allow_html=True,
            )
        with p_c3:
            if st.button("Next →", key=f"{key_prefix}_next_btn", disabled=(current_page >= pages)):
                st.session_state[page_key] = min(pages, current_page + 1)
                st.rerun()
