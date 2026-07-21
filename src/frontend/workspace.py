import streamlit as st
from src.core.app_state import app_state

def show_workspace():
    """Display the main workspace after the dataset is uploaded"""
    
    profile = app_state.dataset_profile
    
    st.title("Dataset Overview")
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Rows",
            value = profile.rows
        )
    with col2:
        st.metric(
            label="Columns",
            value = profile.columns
        )
    with col3:
        st.metric(
            label="Memory Usage",
            value = profile.memory_usage
        )
        
    st.divider()
    
    st.subheader("🩺 Dataset Health")
    
    # Health Score
    score_col1, score_col2 = st.columns(2)
    with score_col1:
        st.metric(
            label="Heath Score",
            value=f"{profile.health_score}/100"
        )
        
    with score_col2:
        st.metric(
            label="Status",
            value=f"{profile.health_status}"
        )
        
    st.write("")
    
    # MISSING AND DUPLICATE VALUES
    col1 , col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Missing Values",
            value = profile.missing_values,
            delta=f"{profile.missing_percentage}%"
        )
        
    with col2:
        st.metric(
            label="Duplicate Values",
            value = profile.duplicate_rows,
            delta = f"{profile.duplicate_percentage}%"
        )
        
    st.divider()
    
    st.subheader("Dataset Name")
    st.write(profile.dataset_name)
    
    st.divider()

    st.subheader("🚀 Coming Soon")
    
    st.subheader("📂 Column Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Numerical Columns",
            len(profile.numerical_columns)
        )

        st.write(profile.numerical_columns)

    with col2:
        st.metric(
            "Categorical Columns",
            len(profile.categorical_columns)
        )

        st.write(profile.categorical_columns)

    col3, col4 = st.columns(2)

    with col3:
        st.metric(
            "Boolean Columns",
            len(profile.boolean_columns)
        )

        st.write(profile.boolean_columns)

    with col4:
        st.metric(
            "Datetime Columns",
            len(profile.datetime_columns)
        )

        st.write(profile.datetime_columns)
    
    st.divider()
    
    st.subheader("Column Statistics")
    
    for column, stats in profile.numerical_statistics.items():
        with st.expander(f"📊 {column}"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Minimum", stats["min"])
                st.metric("Mean", stats["mean"])
                st.metric("Standard Deviation", stats["std"])
            with col2:
                st.metric("Maximum", stats["max"])
                st.metric("Median", stats["median"])
    
    for column, stats in profile.categorical_statistics.items():
        with st.expander(f"📝 {column}"):
            st.write(f"**Unique Values:** {stats['unique-values']}")
            st.write(f"**Most Frequent:** {stats['most-frequent']}")
            st.write(f"**Frequency:** {stats['frequency']}")
        
    st.divider()
    
    st.subheader("📊 Visualization Recommendations")

    for recommendation in profile.recommended_visualizations:
        with st.expander(recommendation["chart"]):
            st.write(
                f"**Columns:** {', '.join(recommendation['columns'])}"
            )
            st.write(
                f"**Reason:** {recommendation['reason']}"
            )
            
    st.divider()
    
    st.subheader("💡 Intelligent Recommendations")
    
    if profile.recommendations:
        for recommendation in profile.recommendations:
            priority = recommendation["priority"]
            
            if priority == "High":
                st.error(recommendation["message"])
            elif priority == "Medium":
                st.warning(recommendation["message"])
            else:
                st.info(recommendation["message"])
    else:
        st.success("No major issue detected. Your dataset looks good")
        
    st.divider()
    
    st.info(
        """
        Upcoming features:

        • Dataset Health

        • Column Analysis

        • Statistics

        • Visualizations

        • AI Insights
        """
    )