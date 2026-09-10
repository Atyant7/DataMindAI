"""
Production-quality User Dashboard for DataMind AI.

Displays:
- Real-time database aggregated overview metrics
- Recent projects with enriched metadata (dataset, task, best model, performance)
- Project action menu (Open, Rename, Duplicate, Archive, Delete with confirmation)
- Chronological audit activity feed
- Quick Actions
- Modern empty state for new users
"""

from __future__ import annotations

import datetime
from typing import Any

import streamlit as st

from src.core.app_state import app_state
from src.core.logger import get_logger
from src.ml.model_persistence import load_artifact
from src.services.activity_service import ActivityService
from src.services.chat_service import ChatService
from src.services.dataset_service import DatasetService
from src.services.model_registry_service import ModelRegistryService
from src.services.project_service import ProjectService

logger = get_logger(__name__)


def _get_greeting() -> str:
    """Return time-appropriate greeting."""
    hour = datetime.datetime.now().hour
    if hour < 12:
        return "Good morning"
    elif hour < 17:
        return "Good afternoon"
    else:
        return "Good evening"


def open_project_workspace(user_id: str, project: dict[str, Any]) -> None:
    """
    Restore complete project state into memory and route user to Chat.
    Restores:
    - Active project context
    - Dataset and profile
    - Chat history and session chat ID
    - Model artifact
    """
    st.session_state["current_project"] = project
    app_state.active_user_id = user_id
    app_state.active_project_id = project["id"]

    # 1. Restore Dataset
    ds_list = DatasetService.list_project_datasets(user_id, project["id"])
    if ds_list:
        try:
            df, meta = DatasetService.load_dataframe(user_id, ds_list[0]["id"])
            app_state.dataset = df
            app_state.dataset_name = meta["name"]
            app_state.dataset_signature = meta.get("signature")
            from src.backend.dataset_intelligence import DatasetIntelligence
            intel = DatasetIntelligence(df, meta["name"])
            app_state.dataset_profile = intel.generate_profile()
        except Exception as exc:
            logger.warning("Could not auto-restore project dataset: %s", exc)
    else:
        app_state.dataset = None
        app_state.dataset_name = None
        app_state.dataset_profile = None

    # 2. Restore Chat History
    chat_rec = ChatService.get_or_create_default_chat(user_id, project["id"])
    st.session_state["active_chat_id"] = chat_rec["id"]
    msgs = ChatService.get_chat_messages(chat_rec["id"])
    app_state.chat_history = [{"role": m["role"], "content": m["content"]} for m in msgs]

    # 3. Restore Model Artifact
    best_model_rec = ModelRegistryService.get_best_model(user_id, project["id"])
    if best_model_rec:
        try:
            art = load_artifact(best_model_rec["artifact_path"], best_model_rec["metadata_path"])
            app_state.model_artifact = art
            app_state.trained_model = art.model
            app_state.model_metadata = art.to_dict()
            app_state.current_task = art.task
        except Exception as art_err:
            logger.warning("Could not load project model artifact: %s", art_err)
    else:
        app_state.model_artifact = None
        app_state.trained_model = None

    st.session_state["active_section"] = "Chat"
    st.rerun()


def show_home(user_data: dict | None = None) -> None:
    """Render the User Dashboard."""
    user = user_data or st.session_state.get("user")
    if not user:
        st.warning("Please sign in to view your dashboard.")
        return

    user_id = user["id"]
    user_name = user["username"]
    greeting = _get_greeting()

    # --------------------------------------------------------
    # 1. WELCOME BANNER & PRIMARY ACTIONS
    # --------------------------------------------------------
    banner_col, cta_col = st.columns([3, 2])
    with banner_col:
        st.markdown(
            f"""
            <div style="margin-bottom: 0.5rem;">
                <h1 style="font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2rem; letter-spacing: -0.02em;">
                    {greeting}, {user_name} 👋
                </h1>
                <p style="font-size: 1.05rem; opacity: 0.8; margin-top: 0;">
                    Your Data Science Workspace — Analyze data, train models, and make predictions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with cta_col:
        st.write("")
        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            show_new_proj_modal = st.button("➕ New Project", type="primary", use_container_width=True)
        with btn_c2:
            if st.button("📤 Upload Dataset", type="secondary", use_container_width=True):
                st.session_state["active_section"] = "Dataset"
                st.rerun()

    # --------------------------------------------------------
    # MODAL / EXPANDER: CREATE NEW PROJECT
    # --------------------------------------------------------
    if show_new_proj_modal or st.session_state.get("show_new_project_dialog"):
        st.session_state["show_new_project_dialog"] = True
        with st.container():
            st.markdown("### 📁 Create New Project")
            with st.form("dashboard_new_project_form"):
                p_name = st.text_input("Project Name", placeholder="e.g. Customer Churn Prediction")
                p_desc = st.text_area("Description", placeholder="Predicting user churn probabilities...")
                f_c1, f_c2 = st.columns([1, 4])
                with f_c1:
                    create_submit = st.form_submit_button("Create Project", type="primary")
                with f_c2:
                    cancel_create = st.form_submit_button("Cancel")

                if create_submit:
                    if not p_name.strip():
                        st.error("Please provide a project name.")
                    else:
                        new_p = ProjectService.create_project(user_id, p_name, p_desc)
                        st.session_state["show_new_project_dialog"] = False
                        open_project_workspace(user_id, new_p)

                if cancel_create:
                    st.session_state["show_new_project_dialog"] = False
                    st.rerun()

    st.markdown("<hr style='margin: 1.25rem 0;'>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # 2. OVERVIEW STATISTICS (DATABASE COUNTS)
    # --------------------------------------------------------
    st.markdown("### 📊 Overview")
    stats = ProjectService.get_user_dashboard_stats(user_id)

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.metric("Projects", f"{stats['projects']}")
    with col_s2:
        st.metric("Datasets", f"{stats['datasets']}")
    with col_s3:
        st.metric("Models", f"{stats['models']}")
    with col_s4:
        st.metric("Predictions", f"{stats['predictions']}")

    st.markdown("<hr style='margin: 1.25rem 0;'>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # 3. RECENT PROJECTS & RECENT ACTIVITY (TWO-COLUMN LAYOUT)
    # --------------------------------------------------------
    left_col, right_col = st.columns([3, 2])

    with left_col:
        # Header with [View All Projects] link
        proj_head_c1, proj_head_c2 = st.columns([3, 1])
        with proj_head_c1:
            st.markdown("### 📁 Recent Projects")
        with proj_head_c2:
            if st.button("View All →", key="view_all_projects_top", help="Open complete projects library"):
                st.session_state["active_section"] = "Projects"
                st.rerun()

        # Fetch recent 5 projects
        recent_projects = ProjectService.list_user_projects(
            user_id=user_id,
            include_archived=False,
            sort_by="updated_at",
        )

        if not recent_projects:
            # EMPTY STATE
            st.markdown(
                """
                <div style="
                    border: 1px dashed rgba(100, 116, 139, 0.4);
                    border-radius: 10px;
                    padding: 2.5rem 1.5rem;
                    text-align: center;
                    margin: 1rem 0;
                ">
                    <h3 style="margin-bottom: 0.5rem;">Welcome to DataMind AI 👋</h3>
                    <p style="opacity: 0.8; max-width: 480px; margin: 0 auto 1.25rem auto;">
                        You don't have any projects yet. Create your first project to start analyzing datasets, training ML models, and generating predictions.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("➕ Create Your First Project", type="primary", use_container_width=True, key="empty_state_create_btn"):
                st.session_state["show_new_project_dialog"] = True
                st.rerun()
        else:
            # Render Project Cards
            for proj in recent_projects[:5]:
                with st.container():
                    p_col1, p_col2 = st.columns([4, 1])

                    with p_col1:
                        st.markdown(f"#### {proj['name']}")
                        ds_label = proj.get("dataset_name") or "No dataset uploaded"
                        rows_label = f" • {proj['dataset_rows']:,} rows" if proj.get("dataset_rows") else ""
                        cols_label = f" × {proj['dataset_columns']} cols" if proj.get("dataset_columns") else ""
                        st.caption(f"📊 `{ds_label}`{rows_label}{cols_label}")

                        task_str = f" • {proj['task'].title()}" if proj.get("task") else ""
                        model_str = f" • Best: **{proj['best_model']}**" if proj.get("best_model") else ""
                        metric_str = f" ({proj['best_metric']})" if proj.get("best_metric") else ""
                        updated_at_str = proj.get("updated_at", "")[:10]
                        st.markdown(f"Status: **Active**{task_str}{model_str}{metric_str} · Updated {updated_at_str}")

                    with p_col2:
                        st.write("")
                        if st.button("Open →", key=f"open_proj_{proj['id']}", type="primary", use_container_width=True):
                            open_project_workspace(user_id, proj)

                        # Card Action Menu Popover (Rename, Duplicate, Archive, Delete)
                        with st.popover("⋮", use_container_width=True):
                            st.markdown(f"**Manage '{proj['name']}'**")
                            # Rename
                            with st.form(f"rename_form_{proj['id']}"):
                                ren_val = st.text_input("New Name", value=proj["name"])
                                if st.form_submit_button("Rename"):
                                    ProjectService.update_project(user_id, proj["id"], name=ren_val)
                                    st.success("Renamed!")
                                    st.rerun()

                            # Duplicate
                            if st.button("📋 Duplicate", key=f"dup_{proj['id']}", use_container_width=True):
                                ProjectService.duplicate_project(user_id, proj["id"])
                                st.success("Duplicated!")
                                st.rerun()

                            # Archive
                            if st.button("📦 Archive", key=f"arch_{proj['id']}", use_container_width=True):
                                ProjectService.archive_project(user_id, proj["id"], archive=True)
                                st.success("Archived!")
                                st.rerun()

                            # Delete
                            if st.button("🗑 Delete", key=f"del_{proj['id']}", type="secondary", use_container_width=True):
                                st.session_state[f"confirm_delete_{proj['id']}"] = True

                    # Confirmation alert for delete if triggered
                    if st.session_state.get(f"confirm_delete_{proj['id']}"):
                        st.error(f"Delete '{proj['name']}'? This will permanently remove its datasets, models, and chats.")
                        del_c1, del_c2 = st.columns(2)
                        with del_c1:
                            if st.button("Yes, Delete", key=f"del_confirm_{proj['id']}", type="primary", use_container_width=True):
                                ProjectService.delete_project(user_id, proj["id"])
                                del st.session_state[f"confirm_delete_{proj['id']}"]
                                st.rerun()
                        with del_c2:
                            if st.button("Cancel", key=f"del_cancel_{proj['id']}", use_container_width=True):
                                del st.session_state[f"confirm_delete_{proj['id']}"]
                                st.rerun()

                    st.markdown("<hr style='margin: 0.75rem 0;'>", unsafe_allow_html=True)

    with right_col:
        # ----------------------------------------------------
        # 4. RECENT ACTIVITY AUDIT TIMELINE
        # ----------------------------------------------------
        st.markdown("### 🕒 Recent Activity")
        activities = ActivityService.get_recent_activities(user_id, limit=6)

        if activities:
            for act in activities:
                icon = act.get("icon", "⚡")
                created_date = act.get("created_at", "")[:16].replace("T", " ")
                st.markdown(
                    f"""
                    <div style="
                        padding: 0.65rem 0.85rem;
                        border-left: 2px solid #3B82F6;
                        margin-bottom: 0.75rem;
                        background: rgba(255,255,255,0.02);
                    ">
                        <div style="font-weight: 600; font-size: 0.95rem;">{icon} {act['title']}</div>
                        <div style="font-size: 0.8rem; opacity: 0.7;">{created_date}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No recorded activity yet. Actions you take in projects will appear here.")

        st.markdown("<hr style='margin: 1.25rem 0;'>", unsafe_allow_html=True)

        # ----------------------------------------------------
        # 5. QUICK ACTIONS
        # ----------------------------------------------------
        st.markdown("### ⚡ Quick Actions")
        qa_col1, qa_col2 = st.columns(2)
        with qa_col1:
            if st.button("💬 Open Chat", use_container_width=True):
                st.session_state["active_section"] = "Chat"
                st.rerun()
            if st.button("🤖 View Models", use_container_width=True):
                st.session_state["active_section"] = "Models"
                st.rerun()

        with qa_col2:
            if st.button("🔮 Make Prediction", use_container_width=True):
                st.session_state["active_section"] = "Prediction"
                st.rerun()
            if st.button("📈 Visualizations", use_container_width=True):
                st.session_state["active_section"] = "Visualization"
                st.rerun()
