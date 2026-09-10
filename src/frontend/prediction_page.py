"""
Enterprise Dynamic Prediction Workspace for DataMind AI.

Supports:
- Model Registry detection (tabular, image, and geospatial models)
- Dynamic tabular feature inputs with schema and type validation
- Image upload and preview prediction for computer vision models
- Prediction history logging and retrieval
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
import streamlit as st

from src.core.app_state import app_state
from src.core.logger import get_logger
from src.ml.model_persistence import load_artifact
from src.ml.predictor_base import ImagePredictor, TabularPredictor
from src.services.model_registry_service import ModelRegistryService
from src.services.prediction_service import PredictionService

logger = get_logger(__name__)


def show_prediction_page(
    user_id: str | None = None,
    project_id: str | None = None,
) -> None:
    """Render the dynamic prediction workspace."""
    st.markdown("## 🔮 Prediction Workspace")
    st.caption("Generate real-time predictions using your registered models.")

    # 1. Fetch available models from Model Registry (and fallback to session artifact)
    registered_models: list[dict[str, Any]] = []
    if user_id and project_id:
        try:
            registered_models = ModelRegistryService.list_project_models(user_id, project_id)
        except Exception as exc:
            logger.warning("Could not fetch models from registry: %s", exc)

    # If no DB models, fallback to app_state.model_artifact if present
    if not registered_models and app_state.model_artifact is not None:
        art = app_state.model_artifact
        registered_models.append({
            "id": "session_model",
            "model_name": art.model_name,
            "display_name": art.display_name,
            "version": 1,
            "task": art.task,
            "target_column": art.target_column,
            "model_type": "tabular",
            "is_image": False,
            "feature_names": art.feature_names,
            "metrics": {art.primary_metric: art.primary_score},
            "artifact_path": art.model_path,
            "metadata_path": art.metadata_path,
            "created_at": "Current Session",
        })

    if not registered_models:
        st.info(
            "No trained models found for this project. "
            "Go to the **Chat** tab and say *'Train a model'* or open **Models** to run AutoML."
        )
        return

    # 2. Model Selection Dropdown
    model_labels = [
        f"{m['display_name']} (v{m.get('version', 1)}) — Target: {m['target_column']} ({m['task'].title()})"
        for m in registered_models
    ]

    selected_index = 0
    selected_label = st.selectbox(
        "Select Model",
        options=model_labels,
        index=selected_index,
        key="prediction_model_selector",
    )
    selected_model_idx = model_labels.index(selected_label)
    selected_model = registered_models[selected_model_idx]

    # Model Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model", selected_model["display_name"])
    with col2:
        st.metric("Task", selected_model["task"].title())
    with col3:
        st.metric("Target", selected_model["target_column"])
    with col4:
        metric_val = "N/A"
        if selected_model.get("metrics"):
            first_key = list(selected_model["metrics"].keys())[0]
            metric_val = f"{selected_model['metrics'][first_key]:.4f}" if isinstance(selected_model['metrics'][first_key], (int, float)) else str(selected_model['metrics'][first_key])
            st.metric(first_key.upper(), metric_val)
        else:
            st.metric("Version", f"v{selected_model.get('version', 1)}")

    st.divider()

    # 3. Dynamic UI Dispatch: Image vs Tabular
    is_image = selected_model.get("is_image", False) or selected_model.get("model_type") == "image"

    if is_image:
        _render_image_prediction_ui(user_id, project_id, selected_model)
    else:
        _render_tabular_prediction_ui(user_id, project_id, selected_model)

    # 4. Prediction History Section
    st.divider()
    _render_prediction_history(user_id, project_id, selected_model.get("id"))


def _render_tabular_prediction_ui(
    user_id: str | None,
    project_id: str | None,
    model_info: dict[str, Any],
) -> None:
    """Render dynamic input form strictly for tabular features."""
    st.markdown("### 📝 Enter Feature Values")
    feature_names = model_info.get("feature_names", [])

    if not feature_names:
        st.error("This model does not contain feature schema metadata.")
        return

    # Load artifact
    try:
        artifact = load_artifact(model_info["artifact_path"], model_info["metadata_path"])
    except Exception:
        artifact = app_state.model_artifact
        if artifact is None:
            st.error("Could not load model artifact file from disk.")
            return

    # Reference dataset for defaults and options
    df = app_state.dataset

    prediction_values: dict[str, Any] = {}

    with st.form("tabular_prediction_form"):
        input_cols = st.columns(2)

        for index, feature in enumerate(feature_names):
            with input_cols[index % 2]:
                # Inspect column if present in active dataset
                if df is not None and feature in df.columns:
                    col_series = df[feature].dropna()

                    if pd.api.types.is_bool_dtype(col_series):
                        prediction_values[feature] = st.checkbox(feature, value=False)
                    elif pd.api.types.is_numeric_dtype(col_series):
                        default_val = float(col_series.median()) if not col_series.empty else 0.0
                        prediction_values[feature] = st.number_input(feature, value=default_val)
                    else:
                        uniques = sorted([str(u) for u in col_series.unique().tolist()])
                        if uniques:
                            prediction_values[feature] = st.selectbox(feature, options=uniques)
                        else:
                            prediction_values[feature] = st.text_input(feature)
                else:
                    # Generic input if dataset not currently loaded in memory
                    prediction_values[feature] = st.text_input(feature, value="")

        submit_predict = st.form_submit_button(
            "🔮 Generate Prediction",
            type="primary",
            use_container_width=True,
        )

    if submit_predict:
        # Validate inputs
        validated_row: dict[str, Any] = {}
        for feature in feature_names:
            val = prediction_values.get(feature)
            if val is None or val == "":
                # Try fallback median or default
                if df is not None and feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
                    val = float(df[feature].median())
                else:
                    st.error(f"Please provide a valid value for '{feature}'.")
                    return
            # Coerce numeric strings
            if isinstance(val, str) and val.replace(".", "", 1).isdigit():
                val = float(val) if "." in val else int(val)
            validated_row[feature] = val

        try:
            predictor = TabularPredictor(artifact)
            result = predictor.predict(validated_row)

            # Record in prediction history
            if user_id and project_id and model_info.get("id"):
                try:
                    PredictionService.record_prediction(
                        user_id=user_id,
                        project_id=project_id,
                        model_id=model_info["id"],
                        input_data=validated_row,
                        prediction=result.prediction,
                        confidence=result.confidence,
                        probabilities=result.probabilities,
                    )
                except Exception as rec_exc:
                    logger.warning("Could not record prediction: %s", rec_exc)

            # Display Result
            _display_prediction_result(result)

        except Exception as exc:
            logger.exception("Prediction failed")
            st.error(f"Prediction failed: {exc}")


def _render_image_prediction_ui(
    user_id: str | None,
    project_id: str | None,
    model_info: dict[str, Any],
) -> None:
    """Render image uploader strictly for computer vision models."""
    st.markdown("### 🖼 Upload Image for Prediction")
    st.caption("This model was trained on image data. Tabular feature inputs are not required.")

    uploaded_image = st.file_uploader(
        "Choose an image file",
        type=["png", "jpg", "jpeg"],
        key="image_prediction_uploader",
    )

    if uploaded_image is not None:
        try:
            pil_img = Image.open(uploaded_image)
            st.image(pil_img, caption="Image Preview", width=300)

            if st.button("🔮 Predict Image Class", type="primary", use_container_width=True):
                # Load or mock image predictor
                import joblib
                model_obj = joblib.load(model_info["artifact_path"])
                classes = model_info.get("training_config", {}).get("classes", ["Class A", "Class B"])
                predictor = ImagePredictor(
                    model=model_obj,
                    classes=classes,
                    display_name=model_info["display_name"],
                    target_column=model_info["target_column"],
                )
                result = predictor.predict(pil_img)

                # Record prediction
                if user_id and project_id and model_info.get("id"):
                    try:
                        PredictionService.record_prediction(
                            user_id=user_id,
                            project_id=project_id,
                            model_id=model_info["id"],
                            input_data={"filename": uploaded_image.name, "format": pil_img.format},
                            prediction=result.prediction,
                            confidence=result.confidence,
                            probabilities=result.probabilities,
                        )
                    except Exception as rec_exc:
                        logger.warning("Could not record image prediction: %s", rec_exc)

                _display_prediction_result(result)

        except Exception as exc:
            st.error(f"Failed to process image: {exc}")


def _display_prediction_result(result: Any) -> None:
    """Render formatted prediction result cards and probabilities."""
    st.markdown("### 🎯 Prediction Output")
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.markdown("#### Predicted Value / Class")
        st.success(f"**{result.prediction}**")

    with res_col2:
        st.markdown("#### Model")
        st.info(result.display_name)

    if result.confidence is not None:
        st.markdown("#### Prediction Confidence")
        st.progress(result.confidence)
        st.metric("Confidence", f"{result.confidence * 100:.1f}%")

    if result.probabilities:
        st.markdown("#### Class Probabilities")
        prob_cols = st.columns(len(result.probabilities))
        for idx, (cls_name, prob) in enumerate(result.probabilities.items()):
            with prob_cols[idx % len(prob_cols)]:
                st.metric(str(cls_name), f"{prob * 100:.1f}%")


def _render_prediction_history(
    user_id: str | None,
    project_id: str | None,
    model_id: str | None,
) -> None:
    """Render table of previous predictions for this project."""
    st.markdown("### 📜 Prediction History")

    if not user_id or not project_id:
        st.caption("Log in and work inside a project to save prediction history.")
        return

    try:
        history = PredictionService.get_project_predictions(
            user_id=user_id,
            project_id=project_id,
            model_id=model_id,
            limit=20,
        )

        if not history:
            st.info("No predictions recorded yet for this model.")
            return

        rows = []
        for item in history:
            rows.append({
                "Timestamp": item.get("created_at", "")[:19].replace("T", " "),
                "Prediction": item.get("prediction_result"),
                "Confidence": f"{item['confidence'] * 100:.1f}%" if item.get("confidence") is not None else "N/A",
                "Input Summary": str(item.get("input_data"))[:80] + ("..." if len(str(item.get("input_data"))) > 80 else ""),
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    except Exception as exc:
        logger.warning("Could not render prediction history: %s", exc)
