import streamlit as st


def show_home():

    st.title("🧠 DataMindAI")

    st.subheader("Your Autonomous AI Data Scientist")

    st.divider()

    st.markdown(
        """
### 👋 Welcome!

Upload your dataset and ask questions in natural language.

### Example Prompts

- Analyze my dataset
- Show every graph between Age and Salary
- Remove duplicate rows
- Fill missing values using median
- Train the best regression model
- Predict salary for this employee
- Why did sales increase in February 2017?
"""
    )

    st.divider()

    st.subheader("✨ What I Can Do")

    st.markdown(
        """
- 📊 Dataset Analysis

- 📈 Data Visualization

- 🧹 Data Cleaning

- 🤖 Machine Learning

- 📉 Prediction

- 🧠 Explain AI Models

- 📄 Generate Reports

- 🌍 Business Reasoning using External Knowledge
"""
    )

    st.divider()

    st.info("⬅ Upload a dataset from the sidebar to get started.")