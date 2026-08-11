import streamlit as st


def show_home():
    st.title("🧠 DataMindAI")
    st.subheader("Your Autonomous AI Data Scientist")
    st.divider()

    st.markdown(
        """
### 👋 Welcome!

Upload a dataset and interact with it using natural language.

### Currently available

- Understand the dataset automatically
- Inspect missing values and duplicates
- Analyze column types and statistics
- Detect outliers and feature-quality issues
- Analyze correlations
- Review preprocessing recommendations
- Ask questions about the dataset
- Generate visualizations using natural-language requests

### Coming next

- Automated ML model competition
- XGBoost / LightGBM training
- Best-model selection
- Model explainability
- Prediction workspace
- Experiment tracking
- What-if analysis
- Evidence-based reports
"""
    )

    st.divider()

    st.info(
        "⬅ Upload a CSV or XLSX dataset from the sidebar to begin."
    )
