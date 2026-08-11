"""Dataset workspace UI.

The main application currently opens the AI Analyst by default. This workspace
is kept as a reusable page and will become the primary Dataset Workspace in the
next UI phase.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.core.app_state import app_state


def show_workspace():
    """Display the current dataset intelligence workspace."""
    profile = app_state.dataset_profile

    if profile is None:
        st.info("Upload a dataset to open the dataset workspace.")
        return

    st.title("📊 Dataset Workspace")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", profile.rows)
    with col2:
        st.metric("Columns", profile.columns)
    with col3:
        st.metric("Health Score", f"{profile.health_score}/100")
    with col4:
        st.metric("Missing Values", profile.missing_values)

    st.divider()

    st.subheader("🩺 Dataset Health")
    st.write(f"**Status:** {profile.health_status}")

    if profile.quality_issues:
        for issue in profile.quality_issues:
            if issue["severity"] == "High":
                st.error(f"{issue['type']}: {issue['message']}")
            elif issue["severity"] == "Moderate":
                st.warning(f"{issue['type']}: {issue['message']}")
            else:
                st.info(f"{issue['type']}: {issue['message']}")
    else:
        st.success("No basic data-quality issues were detected.")

    st.divider()

    st.subheader("📂 Column Information")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Numerical Columns**")
        st.write(profile.numerical_columns)

        st.write("**Datetime Columns**")
        st.write(profile.datetime_columns)

    with col2:
        st.write("**Categorical Columns**")
        st.write(profile.categorical_columns)

        st.write("**Boolean Columns**")
        st.write(profile.boolean_columns)

    st.divider()

    st.subheader("🎯 Target Candidates")
    if profile.possible_target_columns:
        st.dataframe(
            pd.DataFrame(profile.possible_target_columns),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No target candidates were identified.")

    st.divider()

    st.subheader("📈 Numerical Statistics")
    if profile.numerical_statistics:
        st.dataframe(
            pd.DataFrame(profile.numerical_statistics).T,
            use_container_width=True,
        )
    else:
        st.info("No numerical columns are available.")

    st.divider()

    st.subheader("🔗 Correlation Intelligence")
    if profile.correlation_matrix is not None:
        figure = px.imshow(
            profile.correlation_matrix,
            text_auto=".2f",
            zmin=-1,
            zmax=1,
            aspect="auto",
            title="Correlation Heatmap",
        )
        st.plotly_chart(figure, use_container_width=True)

        if profile.correlation_insights:
            st.write("**Strongest relationships:**")
            st.dataframe(
                pd.DataFrame(profile.correlation_insights),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No meaningful correlations were detected.")
    else:
        st.info("Correlation analysis requires at least two numerical columns.")

    st.divider()

    st.subheader("🚨 Outlier Detection")
    if profile.outlier_summary:
        st.dataframe(
            pd.DataFrame(profile.outlier_summary),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No numerical columns are available for outlier analysis.")