"""
Enterprise conversational workspace for DataMind AI.

Modern ChatGPT-style AI chat interface:
- Chronological, continuous conversation flow
- User messages distinctly styled with user avatar
- DataMind AI responses styled with AI avatar and full markdown/plots
- Transient loading lifecycle ('DataMind AI is thinking...') cleanly replaced with final response
- Strict project-scoped conversation persistence in the database
- Safe 'New Chat' creation preserving previous histories
- Zero duplicate messages and protection against rapid multiple submissions
"""

from __future__ import annotations
import uuid

import streamlit as st

from src.agent.agent import DataMindAgent
from src.core.app_state import app_state
from src.core.logger import get_logger
from src.services.chat_service import ChatService

logger = get_logger(__name__)


def _apply_chat_style() -> None:
    """Apply layout styling for comfortable, modern conversational flow."""
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 920px;
            padding-top: 0.5rem;
            padding-bottom: 6.5rem;
        }
        [data-testid="stChatMessage"] {
            padding: 0.85rem 1.15rem;
            border-radius: 12px;
            margin-bottom: 0.85rem;
            border: 1px solid rgba(120, 140, 160, 0.15);
            line-height: 1.65;
        }
        [data-testid="stChatMessageContent"] {
            max-width: 820px;
        }
        [data-testid="stChatInput"] {
            width: min(920px, calc(100vw - 2.5rem));
            margin-left: auto;
            margin-right: auto;
        }
        /* Markdown code and table formatting */
        div[data-testid="stChatMessage"] pre {
            background: rgba(0, 0, 0, 0.25) !important;
            border: 1px solid rgba(120, 140, 160, 0.2) !important;
            border-radius: 8px !important;
            padding: 0.75rem !important;
        }
        div[data-testid="stChatMessage"] table {
            border-collapse: collapse;
            width: 100%;
            margin: 0.75rem 0;
        }
        div[data-testid="stChatMessage"] th, div[data-testid="stChatMessage"] td {
            border: 1px solid rgba(120, 140, 160, 0.2);
            padding: 6px 12px;
            text-align: left;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_empty_state() -> None:
    """Render helpful initial chat prompt suggestions."""
    st.markdown(
        """
        <div style="padding: 1.5rem 0 1rem 0; text-align: center;">
            <h2 style="font-weight: 700; margin-bottom: 0.4rem;">How can I assist your analysis today?</h2>
            <p style="opacity: 0.75; font-size: 0.95rem; max-width: 580px; margin: 0 auto 1.5rem auto;">
                Ask questions about your data, handle missing values, visualize trends, train machine learning models, or generate predictions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            **Explore & Clean Data:**
            - 📊 *"Show dataset overview"*
            - ❓ *"What columns and data types are present?"*
            - 🧹 *"How many missing values are present?"*
            - 🛠 *"Handle missing values"*
            """
        )

    with col2:
        st.markdown(
            """
            **Model & Predict:**
            - 📈 *"Show the correlation between columns"*
            - 🤖 *"Train a model to predict [target]"*
            - 🔮 *"Predict for these values: ... "*
            - 🗺 *"Show locations on a map"*
            """
        )


def show_chat_page() -> None:
    """Render the primary AI Data Scientist chat workspace."""
    if not app_state.has_dataset():
        st.info("⬅ Please upload a dataset from the sidebar or select a project to begin chatting.")
        return

    _apply_chat_style()
    # Ensure chat input is enabled when returning to this page
    if st.session_state.get("chat_is_processing"):
        st.session_state["chat_is_processing"] = False

    user = st.session_state.get("user")
    current_project = st.session_state.get("current_project")
    user_id = user["id"] if user else None
    project_id = current_project["id"] if current_project else None

    # --------------------------------------------------------
    # 1. SYNCHRONIZE PROJECT CHAT SESSION & DB MESSAGES
    # --------------------------------------------------------
    current_chat_id = st.session_state.get("active_chat_id")
    loaded_project_id = st.session_state.get("loaded_chat_project_id")

    if user_id and project_id:
        # Project switched or initial chat session loading
        if not current_chat_id or loaded_project_id != project_id:
            chat_record = ChatService.get_or_create_default_chat(user_id, project_id)
            current_chat_id = chat_record["id"]
            st.session_state["active_chat_id"] = current_chat_id
            st.session_state["loaded_chat_project_id"] = project_id
            db_messages = ChatService.get_chat_messages(current_chat_id, user_id=user_id)
            st.session_state["chat_messages"] = db_messages
            app_state.chat_history = [
                {"role": m["role"], "content": m["content"]}
                for m in db_messages
            ]
        elif "chat_messages" not in st.session_state:
            db_messages = ChatService.get_chat_messages(current_chat_id, user_id=user_id)
            st.session_state["chat_messages"] = db_messages
            app_state.chat_history = [
                {"role": m["role"], "content": m["content"]}
                for m in db_messages
            ]
    else:
        # Non-project fallback
        if "chat_messages" not in st.session_state:
            st.session_state["chat_messages"] = [
                {"role": m["role"], "content": m["content"]}
                for m in app_state.chat_history
            ]

    # Initialize agent
    if app_state.agent is None:
        app_state.agent = DataMindAgent()

    if user_id and project_id:
        app_state.active_user_id = user_id
        app_state.active_project_id = project_id

    # --------------------------------------------------------
    # 2. SLEEK CHAT SUBHEADER (NO UNNECESSARY DROPDOWNS)
    # --------------------------------------------------------
    sub_col1, sub_col2 = st.columns([5, 1])
    with sub_col1:
        ds_name = app_state.dataset_name or "Active Dataset"
        rows_str = f"{len(app_state.dataset):,} rows" if app_state.dataset is not None else ""
        cols_str = f"{len(app_state.dataset.columns)} cols" if app_state.dataset is not None else ""
        proj_label = f"📁 **{current_project['name']}** · " if current_project else ""
        st.markdown(
            f"<div style='padding: 0.15rem 0 0.5rem 0; font-size: 0.92rem; opacity: 0.85;'>"
            f"{proj_label}Dataset: <code>{ds_name}</code> ({rows_str} × {cols_str})"
            f"</div>",
            unsafe_allow_html=True,
        )

    with sub_col2:
        if st.button("➕ New Chat", key="chat_page_new_chat_btn", use_container_width=True, help="Start a new conversation in this project"):
            if user_id and project_id:
                proj_chats = ChatService.list_project_chats(user_id, project_id)
                new_chat = ChatService.create_chat(user_id, project_id, f"Chat {len(proj_chats) + 1}")
                st.session_state["active_chat_id"] = new_chat["id"]
            st.session_state["chat_messages"] = []
            app_state.reset_chat()
            st.rerun()

    # --------------------------------------------------------
    # 3. RENDER CHRONOLOGICAL MESSAGE HISTORY
    # --------------------------------------------------------
    messages = st.session_state.get("chat_messages", [])

    if not messages:
        _render_empty_state()
    else:
        for msg in messages:
            role = msg.get("role", "assistant")
            content = msg.get("content", "")
            if not content:
                continue

            with st.chat_message(role, avatar="👤" if role == "user" else "🧠"):
                # Render Plotly chart if present in message
                if msg.get("figure") is not None:
                    st.plotly_chart(msg["figure"], use_container_width=True, key=f"chart_{msg.get('id', '')}")
                st.markdown(content)

    # --------------------------------------------------------
    # 4. CHAT INPUT & EXECUTION LIFECYCLE
    # --------------------------------------------------------
    is_processing = st.session_state.get("chat_is_processing", False)

    prompt = st.chat_input(
        "Ask a question about your data, request cleaning, or say 'Train a model'...",
        disabled=is_processing,
    )

    if prompt and not is_processing:
        clean_prompt = prompt.strip()
        if clean_prompt:
            st.session_state["chat_is_processing"] = True

            # 1. Save user message to database immediately
            user_record = None
            if current_chat_id:
                try:
                    user_record = ChatService.add_message(
                        current_chat_id,
                        "user",
                        clean_prompt,
                        user_id=user_id,
                    )
                except Exception as save_err:
                    logger.warning("Could not persist user message: %s", save_err)

            if not user_record:
                user_record = {"role": "user", "content": clean_prompt}

            st.session_state["chat_messages"].append(user_record)

            # 2. Immediately render user message in UI
            with st.chat_message("user", avatar="👤"):
                st.markdown(clean_prompt)

            # 3. Render Assistant response with transient status container
            with st.chat_message("assistant", avatar="🧠"):
                status_placeholder = st.empty()
                with status_placeholder.container():
                    with st.spinner("DataMind AI is thinking..."):
                        # Build history for agent
                        agent_history = [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state["chat_messages"][:-1]
                        ]
                        try:
                            response = app_state.agent.chat(clean_prompt, agent_history)
                        except Exception as exc:
                            logger.exception("Agent chat execution error: %s", exc)
                            response = {
                                "type": "text",
                                "content": f"I couldn't complete that request.\n\n`{exc}`",
                            }

                # Clear loading indicator completely
                status_placeholder.empty()

                # Render actual response
                figure = response.get("figure")
                if figure is not None:
                    st.plotly_chart(figure, use_container_width=True, key=f"chart_{str(uuid.uuid4())}")

                content = response.get("content") or "I couldn't generate a response for that question."
                st.markdown(content)

            # 4. Save AI response to DB
            ai_record = None
            if current_chat_id:
                try:
                    ai_record = ChatService.add_message(
                        current_chat_id,
                        "assistant",
                        content,
                        operation=response.get("operation"),
                        evidence=response.get("evidence"),
                        user_id=user_id,
                    )
                except Exception as save_err:
                    logger.warning("Could not persist assistant message: %s", save_err)

            if not ai_record:
                ai_record = {"role": "assistant", "content": content}

            if figure is not None:
                ai_record["figure"] = figure

            st.session_state["chat_messages"].append(ai_record)
            app_state.chat_history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state["chat_messages"]
            ]

            st.session_state["chat_is_processing"] = False
            st.rerun()