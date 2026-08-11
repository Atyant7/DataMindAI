"""Clean ChatGPT-style dataset Q&A interface for DataMindAI."""

from __future__ import annotations

import streamlit as st

from src.agent.agent import DataMindAgent
from src.core.app_state import app_state


def _apply_chat_style() -> None:
    """Apply only layout styling; all visible UI uses native Streamlit widgets."""
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 900px;
            padding-top: 1rem;
            padding-bottom: 6rem;
        }

        [data-testid="stChatMessage"] {
            padding-top: 0.65rem;
            padding-bottom: 0.65rem;
        }

        [data-testid="stChatMessageContent"] {
            max-width: 760px;
            line-height: 1.65;
        }

        [data-testid="stChatInput"] {
            width: min(900px, calc(100vw - 2rem));
            margin-left: auto;
            margin-right: auto;
        }

        @media (max-width: 768px) {
            .block-container {
                padding-left: 0.75rem;
                padding-right: 0.75rem;
                padding-bottom: 6rem;
            }

            [data-testid="stChatInput"] {
                width: calc(100vw - 1rem);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_empty_state() -> None:
    """Render the initial chat state using native Streamlit elements."""
    st.markdown(
        "## How can I help?",
    )

    st.caption(
        "Ask questions about your dataset, explore patterns, "
        "calculate statistics, compare columns, or create "
        "visualizations."
    )

    st.write("**Try asking:**")

    suggestions = [
        "Show me the top 5 companies by employees.",
        "What is the average salary?",
        "Which columns contain missing values?",
        "Show the relationship between revenue and employees.",
    ]

    for suggestion in suggestions:
        st.markdown(f"• {suggestion}")


def _render_history() -> None:
    """Render saved conversation messages in chronological order."""
    for message in app_state.chat_history:
        role = message.get("role", "assistant")
        content = message.get("content", "")

        if not content:
            continue

        with st.chat_message(role):
            st.markdown(content)


def _render_response(response: dict) -> str:
    """Render an agent response and return text for chat history."""
    if response.get("type") == "visualization":
        figure = response.get("figure")

        if figure is not None:
            st.plotly_chart(
                figure,
                use_container_width=True,
            )

    content = response.get("content")

    if not content:
        content = "I couldn't generate a response for that question."

    st.markdown(content)
    return content


def show_chat_page() -> None:
    """Render the dataset Q&A workspace."""
    if not app_state.has_dataset():
        st.info(
            "Upload a dataset from the sidebar to start chatting."
        )
        return

    _apply_chat_style()

    st.title("🧠 DataMindAI")
    st.caption("AI Data Analyst")

    if app_state.agent is None:
        app_state.agent = DataMindAgent()

    if app_state.chat_history:
        _render_history()
    else:
        _render_empty_state()

    prompt = st.chat_input(
        "Message DataMindAI..."
    )

    if prompt is None:
        return

    prompt = prompt.strip()

    if not prompt:
        return

    previous_history = list(
        app_state.chat_history
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your data..."):
            try:
                response = app_state.agent.chat(
                    prompt,
                    previous_history,
                )
            except Exception as exc:
                response = {
                    "type": "text",
                    "content": (
                        "I couldn't complete that request.\n\n"
                        f"`{exc}`"
                    ),
                }

        assistant_message = _render_response(
            response
        )

    app_state.chat_history.extend(
        [
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": assistant_message,
            },
        ]
    )