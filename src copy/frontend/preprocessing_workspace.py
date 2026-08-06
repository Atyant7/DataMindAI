# import streamlit as st 
# import pandas as pd

# from src.core.app_state import app_state

# def show_processing_workspace():
#     plan = app_state.preprocessing_plan

#     st.title("🛠 Preprocessing Intelligence")

#     st.divider()
    
#     st.subheader("📋 Pipeline Summary")

#     if plan.pipeline_summary:
#         for step in plan.pipeline_summary:
#             st.success(step)
#     else:
#         st.info("No preprocessing actions recommended.")
        
#     st.divider()

#     st.subheader("🧹 Missing Value Strategy")

#     missing_df = pd.DataFrame(plan.missing_value_plan)

#     st.dataframe(
#         missing_df,
#         use_container_width=True,
#         hide_index=True
#     )
    
#     st.divider()

#     st.subheader("🏷 Encoding Strategy")

#     encoding_df = pd.DataFrame(plan.encoding_plan)

#     st.dataframe(
#         encoding_df,
#         use_container_width=True,
#         hide_index=True
#     )
    
#     st.divider()

#     st.subheader("📏 Scaling Strategy")

#     scaling_df = pd.DataFrame(plan.scaling_plan)

#     st.dataframe(
#         scaling_df,
#         use_container_width=True,
#         hide_index=True
#     )
    
#     st.divider()

#     st.subheader("🎯 Feature Selection")

#     feature_df = pd.DataFrame(plan.feature_selection_plan)

#     st.dataframe(
#         feature_df,
#         use_container_width=True,
#         hide_index=True
#     )
    
#     st.divider()

#     st.subheader("📈 Outlier Treatment")

#     outlier_df = pd.DataFrame(plan.outlier_treatment_plan)

#     st.dataframe(
#         outlier_df,
#         use_container_width=True,
#         hide_index=True
#     )
    
#     st.divider()

#     st.subheader("🚀 Train/Test Recommendation")

#     col1, col2, col3 = st.columns(3)

#     with col1:
#         st.metric(
#             "Train Size",
#             f"{plan.train_test_plan['train_size']}%"
#         )

#     with col2:
#         st.metric(
#             "Test Size",
#             f"{plan.train_test_plan['test_size']}%"
#         )

#     with col3:
#         st.metric(
#             "Cross Validation",
#             "Yes" if plan.train_test_plan["cross_validation"] else "No"
#         )

#     st.info(plan.train_test_plan["reason"])