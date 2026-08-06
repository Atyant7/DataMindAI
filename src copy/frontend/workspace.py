# import streamlit as st
# from src.core.app_state import app_state
# import plotly.express as px 
# import pandas as pd 

# def show_workspace():
#     """Display the main workspace after the dataset is uploaded"""
    
#     profile = app_state.dataset_profile
    
#     st.title("Dataset Overview")
#     st.divider()
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.metric(
#             label="Rows",
#             value = profile.rows
#         )
#     with col2:
#         st.metric(
#             label="Columns",
#             value = profile.columns
#         )
#     with col3:
#         st.metric(
#             label="Memory Usage",
#             value = profile.memory_usage
#         )
        
#     st.divider()
    
#     st.subheader("🩺 Dataset Health")
    
#     # Health Score
#     score_col1, score_col2 = st.columns(2)
#     with score_col1:
#         st.metric(
#             label="Heath Score",
#             value=f"{profile.health_score}/100"
#         )
        
#     with score_col2:
#         st.metric(
#             label="Status",
#             value=f"{profile.health_status}"
#         )
        
#     st.write("")
    
#     # MISSING AND DUPLICATE VALUES
#     col1 , col2 = st.columns(2)
    
#     with col1:
#         st.metric(
#             label="Missing Values",
#             value = profile.missing_values,
#             delta=f"{profile.missing_percentage}%"
#         )
        
#     with col2:
#         st.metric(
#             label="Duplicate Values",
#             value = profile.duplicate_rows,
#             delta = f"{profile.duplicate_percentage}%"
#         )
        
#     st.divider()
    
#     st.subheader("Dataset Name")
#     st.write(profile.dataset_name)
    
#     st.divider()
    
#     st.subheader("📂 Column Information")

#     col1, col2 = st.columns(2)

#     with col1:
#         st.metric(
#             "Numerical Columns",
#             len(profile.numerical_columns)
#         )

#         st.write(profile.numerical_columns)

#     with col2:
#         st.metric(
#             "Categorical Columns",
#             len(profile.categorical_columns)
#         )

#         st.write(profile.categorical_columns)

#     col3, col4 = st.columns(2)

#     with col3:
#         st.metric(
#             "Boolean Columns",
#             len(profile.boolean_columns)
#         )

#         st.write(profile.boolean_columns)

#     with col4:
#         st.metric(
#             "Datetime Columns",
#             len(profile.datetime_columns)
#         )

#         st.write(profile.datetime_columns)
    
#     st.divider()
    
#     st.subheader("Column Statistics")
    
#     for column, stats in profile.numerical_statistics.items():
#         with st.expander(f"📊 {column}"):
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.metric("Minimum", stats["min"])
#                 st.metric("Mean", stats["mean"])
#                 st.metric("Standard Deviation", stats["std"])
#             with col2:
#                 st.metric("Maximum", stats["max"])
#                 st.metric("Median", stats["median"])
    
#     for column, stats in profile.categorical_statistics.items():
#         with st.expander(f"📝 {column}"):
#             st.write(f"**Unique Values:** {stats['unique-values']}")
#             st.write(f"**Most Frequent:** {stats['most-frequent']}")
#             st.write(f"**Frequency:** {stats['frequency']}")
        
#     st.divider()
    
#     st.subheader("📊 Visualization Recommendations")

#     for recommendation in profile.recommended_visualizations:
#         with st.expander(recommendation["chart"]):
#             st.write(
#                 f"**Columns:** {', '.join(recommendation['columns'])}"
#             )
#             st.write(
#                 f"**Reason:** {recommendation['reason']}"
#             )
            
#     st.divider()
        
    
#     st.subheader("📊 Correlation Intelligence")
    
#     if profile.correlation_metrix is None:
#         st.write("Correlation analysis is not available for this dataset.")
#         return
    
#     fig = px.imshow(
#         profile.correlation_metrix,
#         text_auto=".2f",
#         color_continuous_scale='RdBu',
#         zmin=-1,
#         zmax=1,
#         aspect='auto'
#     )
#     fig.update_layout(
#         title="Correlation Heatmap",
#         height=700
#     )
#     st.plotly_chart(fig, use_container_width=True)
        
#     st.markdown("### 📈 Correlation Insights")

#     st.write(
#         f"**Meaningful Correlations Found:** "
#         f"{len(profile.correlation_insights)}"
#     )

#     if len(profile.correlation_insights) == 0:
#         st.info("No moderate, strong, or very strong correlations were found.")

#     else:
#         for index, insight in enumerate(profile.correlation_insights, start=1):

#             st.markdown("---")

#             icon = "🟢" if insight["type"] == "Positive" else "🔴"

#             st.markdown(
#                 f"""
#                 ### #{index} {insight['column_1']} ↔ {insight['column_2']}

#                 **Correlation:** {insight['correlation']}

#                 **Relationship:** {icon} {insight['strength']}

#                 **Observation:** {insight['observation']}
#                 """
#             )
            
#     st.divider()
    
#     st.subheader("🚨 Outlier Detection")    
    
#     if not profile.outlier_summary:
#         st.info('No outlier analysis available.')
        
#     else :
        
#         outlier_df = pd.DataFrame(profile.outlier_summary)
#         st.dataframe(outlier_df[['column' , 'outliers' , 'percentage', 'status']] , use_container_width=True, hide_index=True)
#         st.write(f"**Columns Analyzed:** {len(profile.outlier_summary)}")
#         for item in profile.outlier_summary:
#             with st.expander(f"📌 {item['column']}"):
#                 st.write(f"**Lower Bound:** {item['lower_bound']}")
#                 st.write(f"**Upper Bound:** {item['upper_bound']}")
#                 st.write(f"**Outliers:** {item['outliers']}")
#                 st.write(f"**Percentage:** {item['percentage']}%")
#                 st.write(f"**Observation:** {item['observation']}")

#                 if item["status"] == "High":
#                     st.error("🔴 High")

#                 elif item["status"] == "Moderate":
#                     st.warning("🟡 Moderate")

#                 elif item["status"] == "Low":
#                     st.success("🟢 Low")

#                 else:
#                     st.info("✅ None")
            
    
    
    
    
#     st.info(
#         """
#         Upcoming features:

#         • Dataset Health

#         • Column Analysis

#         • Statistics

#         • Visualizations

#         • AI Insights
#         """
#     )