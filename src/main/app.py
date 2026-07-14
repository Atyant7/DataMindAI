import streamlit as st

from src.backend.data_loader import load_data
from src.backend.dataset_summary import get_dataset_summary
from src.backend.dataset_analyzer import analyze_dataset


def main():

    st.set_page_config(
        page_title="DataMindAI",
        page_icon="🧠",
        layout="wide"
    )

    st.title("🧠 DataMindAI")

    st.subheader("Your AI-Powered Data Analytics Assistant")

    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=["csv", "xlsx"]
    )

    if uploaded_file:

        st.success("Dataset uploaded successfully!")

        df = load_data(uploaded_file)

        summary = get_dataset_summary(df)

        analysis = analyze_dataset(df)

        st.subheader("📊 Dataset Overview")

        st.write(f"Rows : {summary['Rows']}")

        st.write(f"Columns : {summary['Columns']}")

        st.write(f"Missing Values : {summary['Missing Values']}")

        st.write(f"Duplicate Rows : {summary['Duplicate Rows']}")

        st.divider()

        st.subheader("🧠 Dataset Intelligence")

        st.write(f"Numerical Columns : {analysis['Numerical Columns']}")

        st.write(f"Categorical Columns : {analysis['Categorical Columns']}")

        st.write(f"Missing Values : {analysis['Missing Values']}")

        st.write(f"Duplicate Rows : {analysis['Duplicate Rows']}")

        st.divider()

        st.subheader("📄 Dataset Preview")

        st.dataframe(df.head())