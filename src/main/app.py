import pandas as pd
import streamlit as st

from src.backend.data_loader import (
    get_file_signature,
    load_data,
)
from src.backend.dataset_intelligence import (
    DatasetIntelligence,
)
from src.backend.preprocessing_intelligence import (
    PreprocessingIntelligence,
)

from src.core.app_state import app_state
from src.core.config import (
    ensure_project_directories,
)
from src.core.exceptions import (
    DataMindAIError,
)
from src.core.logger import get_logger

from src.database.init_db import init_db
from src.frontend.auth_page import show_auth_page
from src.frontend.chat_page import (
    show_chat_page,
)
from src.frontend.home import (
    show_home,
)
from src.frontend.prediction_page import (
    show_prediction_page,
)
from src.frontend.preprocessing_workspace import (
    show_processing_workspace,
)
from src.frontend.projects_page import (
    show_projects_page,
)
from src.frontend.settings_page import (
    show_settings_page,
)
from src.frontend.sidebar import (
    show_sidebar,
)
from src.frontend.styles import (
    apply_theme,
)
from src.frontend.top_bar import (
    render_top_bar,
)
from src.frontend.visualization_page import (
    show_visualization_page,
)
from src.frontend.workspace import (
    show_workspace,
)

from src.services.activity_service import ActivityService
from src.services.dataset_service import DatasetService
from src.services.model_registry_service import ModelRegistryService
from src.services.notification_service import NotificationService

import plotly.express as px

from src.ml import (
    run_ml_pipeline,
)
from src.ml.explainability import (
    explain_training_result,
)
from src.ml.model_persistence import (
    save_best_model,
)
from src.ml.prediction_engine import (
    predict_from_artifact,
)
from src.ml.task_detector import (
    MLTask,
    detect_task,
)


logger = get_logger(__name__)


# ============================================================
# APPLICATION INITIALIZATION
# ============================================================

def initialize_application():
    """
    Initialize the DataMindAI application.
    """

    ensure_project_directories()
    init_db()

    st.set_page_config(
        page_title="DataMind AI",
        page_icon="🧠",
        layout="wide",
    )



# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset(uploaded_file):
    """
    Load, profile and prepare a new dataset
    for the current session.
    """

    filename = uploaded_file.name

    signature = get_file_signature(
        uploaded_file
    )

    logger.info(
        "Loading dataset: %s",
        filename,
    )

    dataframe = load_data(
        uploaded_file
    )

    # --------------------------------------------------------
    # Reset previous project state
    # --------------------------------------------------------

    app_state.reset()

    app_state.dataset = dataframe
    app_state.dataset_name = filename
    app_state.dataset_signature = signature

    logger.info(
        "Dataset loaded successfully: %s rows x %s columns",
        dataframe.shape[0],
        dataframe.shape[1],
    )

    # --------------------------------------------------------
    # Dataset intelligence
    # --------------------------------------------------------

    # Dataset loading must NOT depend on profiling succeeding.
    # A dataset is usable for chat/ML even if an optional
    # profiling recommendation fails on an unusual column type.
    try:
        intelligence = DatasetIntelligence(
            dataframe,
            filename,
        )

        profile = intelligence.generate_profile()
        app_state.dataset_profile = profile

        logger.info(
            "Dataset profile generated. Health score: %s/100",
            profile.health_score,
        )

    except Exception as error:
        app_state.dataset_profile = None

        logger.exception(
            "Dataset profiling failed for %s",
            filename,
        )

        st.warning(
            "The dataset was loaded successfully, but automatic "
            "profiling could not be completed. Chat and ML features "
            "can still be used."
        )

    # --------------------------------------------------------
    # Preprocessing intelligence
    # --------------------------------------------------------

    # Preprocessing recommendations are also optional and must
    # never prevent the dataset from becoming available.
    try:
        profile = app_state.dataset_profile

        if profile is not None:
            preprocessing = PreprocessingIntelligence(
                dataframe,
                profile,
            )

            app_state.preprocessing_plan = (
                preprocessing.generate_plan()
            )
        else:
            app_state.preprocessing_plan = None

    except Exception as error:
        app_state.preprocessing_plan = None

        logger.exception(
            "Preprocessing intelligence failed for %s",
            filename,
        )

        st.warning(
            "The dataset was loaded successfully, but automatic "
            "preprocessing recommendations could not be generated. "
            "The dataset is still available."
        )


# ============================================================
# AUTOML WORKSPACE
# ============================================================

def render_automl_workspace(
    df: pd.DataFrame,
) -> None:
    """
    Render the DataMindAI AutoML workspace.

    This frontend communicates with the high-level
    ML pipeline and does not contain model-training logic.
    """

    st.subheader(
        "🤖 AutoML Workspace"
    )

    st.write(
        "Automatically understand the prediction task, "
        "train multiple machine-learning models, evaluate "
        "their performance and select the best model."
    )

    # --------------------------------------------------------
    # Dataset validation
    # --------------------------------------------------------

    if df is None or df.empty:

        st.info(
            "Upload a dataset to start model training."
        )

        return

    # --------------------------------------------------------
    # Dataset summary
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Rows",
            len(df),
        )

    with col2:

        st.metric(
            "Columns",
            len(df.columns),
        )

    with col3:

        st.metric(
            "Missing Values",
            int(
                df.isna()
                .sum()
                .sum()
            ),
        )

    st.divider()

    # --------------------------------------------------------
    # Target selection
    # --------------------------------------------------------

    st.markdown(
        "### 🎯 Select Prediction Target"
    )

    target_column = st.selectbox(
        "Target variable",
        options=df.columns.tolist(),
        key="automl_target",
    )

    # --------------------------------------------------------
    # Detect ML task
    # --------------------------------------------------------

    try:

        task_detection = detect_task(
            df,
            target_column,
        )

    except Exception as exc:

        st.error(
            f"Unable to determine the ML task: {exc}"
        )

        return

    task = task_detection.task

    # --------------------------------------------------------
    # Display detected task
    # --------------------------------------------------------

    task_col1, task_col2 = st.columns(2)

    with task_col1:

        st.metric(
            "Detected Task",
            task.value.title(),
        )

    with task_col2:

        st.metric(
            "Target",
            target_column,
        )

    # --------------------------------------------------------
    # Select appropriate candidate models
    # --------------------------------------------------------

    if task == MLTask.CLASSIFICATION:

        model_options = [
            "logistic_regression",
            "random_forest",
            "xgboost",
            "lightgbm",
        ]

        default_models = [
            "random_forest",
            "xgboost",
            "lightgbm",
        ]

    elif task == MLTask.REGRESSION:

        model_options = [
            "linear_regression",
            "random_forest",
            "xgboost",
            "lightgbm",
        ]

        default_models = [
            "random_forest",
            "xgboost",
            "lightgbm",
        ]

    else:

        st.error(
            f"Unsupported ML task: {task.value}"
        )

        return

    # --------------------------------------------------------
    # Model selection
    # --------------------------------------------------------

    st.markdown(
        "### 🧠 Candidate Models"
    )

    selected_models = st.multiselect(
        "Models to train",
        options=model_options,
        default=default_models,
        key="automl_models",
    )

    if not selected_models:

        st.warning(
            "Select at least one model."
        )

        return

    # --------------------------------------------------------
    # Run AutoML
    # --------------------------------------------------------

    if st.button(
        "🚀 Run AutoML",
        type="primary",
        use_container_width=True,
        key="run_automl",
    ):

        try:

            with st.spinner(
                "DataMindAI is training and evaluating models..."
            ):

                # --------------------------------------------
                # Run complete ML pipeline
                # --------------------------------------------

                result = run_ml_pipeline(
                    df=df,
                    target_column=target_column,
                    model_names=selected_models,
                )

                # --------------------------------------------
                # Save best model
                # --------------------------------------------

                artifact = save_best_model(
                    result,
                    model_directory="artifacts/models",
                )

                # Register in DB Model Registry if project is active
                user = st.session_state.get("user")
                curr_proj = st.session_state.get("current_project")
                if user and curr_proj:
                    try:
                        ModelRegistryService.register_model(
                            user_id=user["id"],
                            project_id=curr_proj["id"],
                            model_name=artifact.model_name,
                            display_name=artifact.display_name,
                            task=artifact.task,
                            target_column=artifact.target_column,
                            artifact_path=artifact.model_path,
                            metadata_path=artifact.metadata_path,
                            feature_names=artifact.feature_names,
                            metrics={result.model_selection.primary_metric: result.model_selection.primary_score},
                            training_config={"models_tested": list(result.training_results.keys())},
                            is_best=True,
                        )
                        ActivityService.log_activity(
                            user_id=user["id"],
                            project_id=curr_proj["id"],
                            action="model_trained",
                            title=f"Trained {result.best_model_display_name} model",
                            description=f"Target: {artifact.target_column} ({artifact.task})",
                            icon="🧠",
                        )
                        NotificationService.create_notification(
                            user_id=user["id"],
                            title="AutoML Training Complete",
                            message=f"Best model '{result.best_model_display_name}' saved.",
                            level="success",
                        )
                    except Exception as reg_err:
                        logger.warning("AutoML DB registration error: %s", reg_err)

            # ------------------------------------------------
            # Store result
            # ------------------------------------------------

            st.session_state[
                "automl_result"
            ] = result

            # ------------------------------------------------
            # Synchronize with global application state
            # ------------------------------------------------

            app_state.automl_result = result
            app_state.model_artifact = artifact
            app_state.trained_model = result.best_training_result.pipeline
            app_state.model_metadata = artifact.to_dict()
            app_state.current_task = result.task.value

            # ------------------------------------------------
            # Store target associated with result
            # ------------------------------------------------

            st.session_state[
                "automl_result_target"
            ] = result.target_column

            # ------------------------------------------------
            # Store persisted model artifact
            # ------------------------------------------------

            st.session_state[
                "automl_artifact"
            ] = artifact

            # ------------------------------------------------
            # Clear previous prediction
            # ------------------------------------------------

            st.session_state.pop(
                "prediction_result",
                None,
            )

            st.success(
                "AutoML workflow completed and "
                "the best model has been saved."
            )

        except Exception as exc:

            logger.exception(
                "AutoML execution failed"
            )

            st.error(
                f"AutoML failed: {exc}"
            )

            return

    # --------------------------------------------------------
    # Retrieve current/previous AutoML result
    # --------------------------------------------------------

    result = st.session_state.get(
        "automl_result"
    )

    if result is None:

        st.info(
            "Select a target and models, then click "
            "'Run AutoML' to begin."
        )

        return

    # --------------------------------------------------------
    # Prevent displaying stale result
    # --------------------------------------------------------

    stored_target = st.session_state.get(
        "automl_result_target"
    )

    if stored_target is not None:

        if stored_target != target_column:

            st.info(
                "The selected target has changed. "
                "Run AutoML again to generate new results."
            )

            return

    # --------------------------------------------------------
    # Store target associated with result
    # --------------------------------------------------------

    st.session_state[
        "automl_result_target"
    ] = result.target_column

    st.divider()

    # ========================================================
    # AUTOML RESULTS
    # ========================================================

    st.markdown(
        "## 📊 AutoML Results"
    )

    # --------------------------------------------------------
    # Best Model
    # --------------------------------------------------------

    st.markdown(
        "### 🏆 Best Model"
    )

    best_col1, best_col2, best_col3 = (
        st.columns(3)
    )

    with best_col1:

        st.metric(
            "Model",
            result.best_model_display_name,
        )

    with best_col2:

        st.metric(
            result.model_selection
            .primary_metric
            .upper(),
            f"{result.model_selection.primary_score:.4f}",
        )

    with best_col3:

        st.metric(
            "Models Evaluated",
            len(
                result.evaluation_results
            ),
        )

    st.info(
        result.model_selection
        .selection_reason
    )

    # --------------------------------------------------------
    # Saved Model
    # --------------------------------------------------------

    artifact = st.session_state.get(
        "automl_artifact"
    )

    if artifact is not None:

        st.markdown(
            "### 💾 Saved Model"
        )

        artifact_col1, artifact_col2 = (
            st.columns(2)
        )

        with artifact_col1:

            st.success(
                "Best model saved successfully."
            )

        with artifact_col2:

            st.code(
                artifact.model_path
            )

    # --------------------------------------------------------
    # Model Leaderboard
    # --------------------------------------------------------

    st.markdown(
        "### 🥇 Model Leaderboard"
    )

    st.dataframe(
        result.leaderboard,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # Detailed comparison
    # --------------------------------------------------------

    st.markdown(
        "### 📈 Detailed Model Comparison"
    )

    st.dataframe(
        result.comparison_table,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # Best model metrics
    # --------------------------------------------------------

    st.markdown(
        "### 📌 Best Model Metrics"
    )

    best_evaluation = (
        result.best_evaluation_result
    )

    metric_items = list(
        best_evaluation
        .metrics
        .items()
    )

    if metric_items:

        metric_columns = st.columns(
            min(
                len(metric_items),
                4,
            )
        )

        for index, (
            metric_name,
            metric_value,
        ) in enumerate(
            metric_items
        ):

            with metric_columns[
                index % len(metric_columns)
            ]:

                st.metric(
                    metric_name.upper(),
                    f"{metric_value:.4f}",
                )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    if (
        result.task == MLTask.CLASSIFICATION
        and best_evaluation.confusion_matrix
    ):

        st.markdown(
            "### 🔲 Confusion Matrix"
        )

        confusion_df = pd.DataFrame(
            best_evaluation
            .confusion_matrix
        )

        st.dataframe(
            confusion_df,
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    if (
        result.task == MLTask.CLASSIFICATION
        and best_evaluation.classification_report
    ):

        st.markdown(
            "### 📋 Classification Report"
        )

        report_df = pd.DataFrame(
            best_evaluation
            .classification_report
        ).transpose()

        st.dataframe(
            report_df,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # Feature Importance & Explainability
    # --------------------------------------------------------

    st.markdown(
        "### 🧠 Feature Importance & Explainability"
    )

    try:
        best_training = result.best_training_result
        X_explain = df.drop(columns=[result.target_column])
        y_explain = df[result.target_column]

        explanation = explain_training_result(
            best_training,
            X=X_explain,
            y=y_explain,
        )
        app_state.last_explanation = explanation

        st.info(
            f"Global feature importance calculated using **{explanation.method}** "
            f"for **{result.best_model_display_name}**."
        )

        top_df = explanation.to_dataframe()

        if not top_df.empty:
            importance_col1, importance_col2 = st.columns([3, 2])

            with importance_col1:
                fig_importance = px.bar(
                    top_df.sort_values("importance", ascending=True).tail(10),
                    x="importance",
                    y="feature",
                    orientation="h",
                    title=f"Top Features ({result.best_model_display_name})",
                    labels={"importance": "Importance Score", "feature": "Feature"},
                    color="importance",
                    color_continuous_scale="Blues",
                )
                fig_importance.update_layout(
                    showlegend=False,
                    margin=dict(l=10, r=10, t=40, b=20),
                    height=350,
                )
                st.plotly_chart(
                    fig_importance,
                    use_container_width=True,
                )

            with importance_col2:
                st.write("**Feature Importance Ranking**")
                display_cols = [c for c in ["rank", "feature", "importance", "direction"] if c in top_df.columns]
                st.dataframe(
                    top_df[display_cols],
                    use_container_width=True,
                    hide_index=True,
                )
    except Exception as exp_err:
        logger.warning("AutoML explainability rendering error: %s", exp_err)
        st.caption("Feature importance could not be computed for this model.")

    # --------------------------------------------------------
    # Pipeline warnings
    # --------------------------------------------------------

    if result.warnings:

        st.markdown(
            "### ⚠️ Pipeline Warnings"
        )

        for warning in result.warnings:

            st.warning(
                warning
            )


# ============================================================
# PREDICTION WORKSPACE
# ============================================================

def render_prediction_workspace(
    df: pd.DataFrame,
) -> None:
    """
    Render the dynamic DataMindAI prediction workspace.

    The input fields are generated from the saved model's
    feature schema.

    The persisted model pipeline is responsible for applying
    the same preprocessing that was used during training.
    """

    st.markdown(
        "## 🔮 Prediction Workspace"
    )

    st.write(
        "Use the selected best model to generate "
        "predictions on new data."
    )

    # --------------------------------------------------------
    # Retrieve saved model
    # --------------------------------------------------------

    artifact = st.session_state.get(
        "automl_artifact"
    )

    if artifact is None:

        st.info(
            "Run AutoML first to create a prediction model."
        )

        return

    # --------------------------------------------------------
    # Validate dataset
    # --------------------------------------------------------

    if df is None or df.empty:

        st.info(
            "A dataset is required for prediction."
        )

        return

    # --------------------------------------------------------
    # Model information
    # --------------------------------------------------------

    info_col1, info_col2, info_col3 = (
        st.columns(3)
    )

    with info_col1:

        st.metric(
            "Model",
            artifact.display_name,
        )

    with info_col2:

        st.metric(
            "Task",
            artifact.task.title(),
        )

    with info_col3:

        st.metric(
            "Target",
            artifact.target_column,
        )

    st.divider()

    # --------------------------------------------------------
    # Feature columns
    # --------------------------------------------------------

    feature_columns = (
        artifact.feature_names
    )

    if not feature_columns:

        st.error(
            "The saved model does not contain "
            "feature information."
        )

        return

    # --------------------------------------------------------
    # Verify required features exist
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in df.columns
    ]

    if missing_features:

        st.error(
            "The current dataset is missing feature(s) "
            "required by the saved model: "
            + ", ".join(missing_features)
        )

        return

    # --------------------------------------------------------
    # Dynamic input form
    # --------------------------------------------------------

    st.markdown(
        "### 📝 Enter Prediction Values"
    )

    prediction_values = {}

    with st.form(
        "prediction_form"
    ):

        input_columns = st.columns(2)

        for index, feature in enumerate(
            feature_columns
        ):

            column = df[feature]

            non_null = (
                column.dropna()
            )

            with input_columns[
                index % 2
            ]:

                # ------------------------------------------------
                # Boolean feature
                # ------------------------------------------------

                if pd.api.types.is_bool_dtype(
                    column
                ):

                    prediction_values[
                        feature
                    ] = st.checkbox(
                        feature,
                        value=False,
                    )

                # ------------------------------------------------
                # Numeric feature
                # ------------------------------------------------

                elif pd.api.types.is_numeric_dtype(
                    column
                ):

                    if non_null.empty:

                        default_value = 0.0

                    else:

                        default_value = float(
                            non_null.median()
                        )

                    prediction_values[
                        feature
                    ] = st.number_input(
                        feature,
                        value=default_value,
                    )

                # ------------------------------------------------
                # Categorical / text feature
                # ------------------------------------------------

                else:

                    unique_values = (
                        column
                        .dropna()
                        .unique()
                        .tolist()
                    )

                    unique_values = sorted(
                        unique_values,
                        key=lambda value: str(
                            value
                        ),
                    )

                    if unique_values:

                        prediction_values[
                            feature
                        ] = st.selectbox(
                            feature,
                            options=unique_values,
                        )

                    else:

                        prediction_values[
                            feature
                        ] = st.text_input(
                            feature
                        )

        # ----------------------------------------------------
        # Predict button
        # ----------------------------------------------------

        predict_clicked = (
            st.form_submit_button(
                "🔮 Predict",
                type="primary",
                use_container_width=True,
            )
        )

    if not predict_clicked:

        return

    # --------------------------------------------------------
    # Build prediction DataFrame
    # --------------------------------------------------------

    prediction_df = pd.DataFrame(
        [prediction_values]
    )

    prediction_df = prediction_df[
        feature_columns
    ]

    # --------------------------------------------------------
    # Generate prediction
    # --------------------------------------------------------

    try:

        with st.spinner(
            "Generating prediction..."
        ):

            prediction_result = (
                predict_from_artifact(
                    artifact,
                    prediction_df,
                )
            )

        st.session_state[
            "prediction_result"
        ] = prediction_result

    except Exception as exc:

        logger.exception(
            "Prediction failed"
        )

        st.error(
            f"Prediction failed: {exc}"
        )

        return

    # --------------------------------------------------------
    # Retrieve prediction result
    # --------------------------------------------------------

    result = st.session_state.get(
        "prediction_result"
    )

    if result is None:

        return

    st.divider()

    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    st.markdown(
        "### 🎯 Prediction Result"
    )

    prediction_col1, prediction_col2 = (
        st.columns(2)
    )

    with prediction_col1:

        st.markdown(
            "#### Prediction"
        )

        st.success(
            str(
                result.prediction
            )
        )

    with prediction_col2:

        st.markdown(
            "#### Model"
        )

        st.info(
            result.display_name
        )

    # --------------------------------------------------------
    # Classification confidence
    # --------------------------------------------------------

    if result.confidence is not None:

        st.markdown(
            "### 📊 Prediction Confidence"
        )

        confidence_percentage = (
            result.confidence * 100
        )

        st.progress(
            result.confidence
        )

        st.metric(
            "Confidence",
            f"{confidence_percentage:.2f}%",
        )

    # --------------------------------------------------------
    # Class probabilities
    # --------------------------------------------------------

    if result.probabilities:

        st.markdown(
            "### 📈 Class Probabilities"
        )

        probability_columns = st.columns(
            len(
                result.probabilities
            )
        )

        for index, (
            class_name,
            probability,
        ) in enumerate(
            result.probabilities.items()
        ):

            with probability_columns[
                index
            ]:

                st.metric(
                    str(class_name),
                    f"{probability * 100:.2f}%",
                )

    # --------------------------------------------------------
    # Input summary
    # --------------------------------------------------------

    with st.expander(
        "🔎 View Prediction Input"
    ):

        input_df = pd.DataFrame(
            [
                result.input_data
            ]
        )

        st.dataframe(
            input_df,
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # Prediction warnings
    # --------------------------------------------------------

    if result.warnings:

        st.markdown(
            "### ⚠️ Prediction Warnings"
        )

        for warning in result.warnings:

            st.warning(
                warning
            )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    initialize_application()

    # --------------------------------------------------------
    # User Authentication Gate
    # --------------------------------------------------------
    user = st.session_state.get("user")
    if user is None:
        apply_theme("dark")
        show_auth_page()
        return

    # Apply user's active theme
    current_theme = st.session_state.get("theme", user.get("theme_preference", "dark"))
    apply_theme(current_theme)

    # Render global SaaS top bar
    render_top_bar(user, st.session_state.get("current_project"))

    uploaded_file, active_section = show_sidebar()

    # --------------------------------------------------------
    # Dataset loading
    # --------------------------------------------------------
    if uploaded_file is not None:
        try:
            signature = get_file_signature(uploaded_file)

            if not app_state.is_same_dataset(
                uploaded_file.name,
                signature,
            ):
                load_dataset(uploaded_file)

                # Persist dataset to project storage if active
                current_project = st.session_state.get("current_project")
                if current_project:
                    try:
                        DatasetService.save_dataset(
                            user_id=user["id"],
                            project_id=current_project["id"],
                            file_obj=uploaded_file,
                            filename=uploaded_file.name,
                        )
                        ActivityService.log_activity(
                            user_id=user["id"],
                            project_id=current_project["id"],
                            action="dataset_uploaded",
                            title=f"Uploaded dataset '{uploaded_file.name}'",
                            icon="📊",
                        )
                        NotificationService.create_notification(
                            user_id=user["id"],
                            title="Dataset Uploaded",
                            message=f"'{uploaded_file.name}' uploaded successfully.",
                            level="success",
                        )
                    except Exception as save_err:
                        logger.warning("Could not persist dataset to storage: %s", save_err)

        except DataMindAIError as error:
            st.error(str(error))
            logger.error(
                "DataMindAI dataset error: %s",
                error,
            )
            return

        except Exception as error:
            logger.exception(
                "Unexpected dataset-loading error"
            )
            st.error(
                f"Dataset could not be loaded: {error}"
            )
            return

    # --------------------------------------------------------
    # Navigation and Workspaces
    # --------------------------------------------------------
    if active_section in ("Dashboard", "Home"):
        show_home(user)

    elif active_section == "Projects":
        show_projects_page(user)

    elif active_section == "Chat":
        show_chat_page()

    elif active_section == "Dataset":
        show_workspace()

    elif active_section == "Visualization":
        show_visualization_page()

    elif active_section == "Preprocessing":
        show_processing_workspace()

    elif active_section in ("Model", "Models"):
        render_automl_workspace(
            app_state.dataset
        )

    elif active_section == "Prediction":
        proj_id = (
            st.session_state["current_project"]["id"]
            if st.session_state.get("current_project")
            else None
        )
        show_prediction_page(
            user_id=user["id"],
            project_id=proj_id,
        )

    elif active_section == "Settings":
        show_settings_page(
            user_data=user,
            current_project=st.session_state.get("current_project"),
        )

    else:
        st.session_state["active_section"] = "Dashboard"
        show_home(user)


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()