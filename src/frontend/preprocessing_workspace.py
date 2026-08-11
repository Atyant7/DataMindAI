"""Preprocessing recommendation workspace."""

import pandas as pd
import streamlit as st

from src.core.app_state import app_state


def show_processing_workspace():
    """Display preprocessing recommendations for the current dataset."""
    plan = app_state.preprocessing_plan

    if plan is None:
        st.info("Upload a dataset to generate preprocessing recommendations.")
        return

    st.title("🛠 Preprocessing Intelligence")
    st.divider()

    st.subheader("📋 Pipeline Summary")
    if plan.pipeline_summary:
        for step in plan.pipeline_summary:
            st.success(step)
    else:
        st.info("No preprocessing actions were recommended.")

    if plan.notes:
        st.subheader("ℹ️ Notes")
        for note in plan.notes:
            st.info(note)

    st.divider()

    sections = [
        ("🧹 Missing Value Strategy", plan.missing_value_plan),
        ("🏷 Encoding Strategy", plan.encoding_plan),
        ("📏 Scaling Strategy", plan.scaling_plan),
        ("🎯 Feature Selection", plan.feature_selection_plan),
        ("📈 Outlier Treatment", plan.outlier_treatment_plan),
    ]

    for title, rows in sections:
        st.subheader(title)
        if rows:
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No recommendations available.")
        st.divider()

    st.subheader("🚀 Train/Test Recommendation")

    if plan.train_test_plan:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Train Size",
                f"{plan.train_test_plan['train_size']}%",
            )

        with col2:
            st.metric(
                "Test Size",
                f"{plan.train_test_plan['test_size']}%",
            )

        with col3:
            folds = plan.train_test_plan.get("cv_folds", 0)
            st.metric(
                "Cross Validation",
                f"{folds} folds" if folds else "Not recommended",
            )

        st.info(plan.train_test_plan["reason"])
