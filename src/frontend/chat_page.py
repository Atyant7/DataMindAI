import streamlit as st

from src.core.app_state import app_state
from src.agent.agent import DataMindAgent


def show_chat_page():
    """
    Main conversational interface of DataMindAI.
    """

    profile = app_state.dataset_profile

    # ---------------------------------------------
    # Header
    # ---------------------------------------------
    st.title("🧠 DataMindAI")

    st.caption("Your Autonomous AI Data Scientist")

    st.divider()

    # ---------------------------------------------
    # Dataset Summary
    # ---------------------------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Rows",
            profile.rows
        )

    with col2:
        st.metric(
            "Columns",
            profile.columns
        )

    with col3:
        st.metric(
            "Health Score",
            f"{profile.health_score}/100"
        )

    st.success(
        f"Dataset Loaded Successfully : {profile.dataset_name}"
    )

    st.divider()

    # ---------------------------------------------
    # Initialize Agent
    # ---------------------------------------------
    if app_state.agent is None:
        app_state.agent = DataMindAgent()

    agent = app_state.agent

    # ---------------------------------------------
    # Initialize Chat History
    # ---------------------------------------------
    if len(app_state.chat_history) == 0:

        app_state.chat_history.append({
            "role": "assistant",
            "content": """
Hello! I'm **DataMindAI** 👋

I've already analyzed your dataset and prepared a preprocessing plan.

You can now ask me things like:

• Analyze my dataset

• Show the relationship between Age and Salary

• Train the best model

• Explain the preprocessing plan

• Predict using the trained model

How can I help you today?
"""
        })

    # ---------------------------------------------
    # Display Chat History
    # ---------------------------------------------
    for message in app_state.chat_history:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    # ---------------------------------------------
    # Chat Input
    # ---------------------------------------------
    prompt = st.chat_input(
        "Ask anything about your dataset..."
    )

    if prompt:

        # -----------------------------------------
        # Display User Message
        # -----------------------------------------
        with st.chat_message("user"):
            st.markdown(prompt)

        # -----------------------------------------
        # Get Agent Response
        # -----------------------------------------
        with st.spinner("Thinking..."):

            response = agent.chat(
                prompt,
                app_state.chat_history
            )

        # -----------------------------------------
        # Display Response
        # -----------------------------------------
        with st.chat_message("assistant"):

            if response["type"] == "visualization":

                st.plotly_chart(
                    response["figure"],
                    use_container_width=True
                )

                assistant_message = (
                    f"Generated a visualization for your request: "
                    f"**{prompt}**"
                )

                st.markdown(assistant_message)

            else:

                assistant_message = response["content"]

                st.markdown(
                    assistant_message
                )

        # -----------------------------------------
        # Store Conversation History
        # -----------------------------------------
        app_state.chat_history.append({
            "role": "user",
            "content": prompt
        })

        app_state.chat_history.append({
            "role": "assistant",
            "content": assistant_message
        })