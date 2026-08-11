"""
Prediction engine for DataMindAI.

Responsibilities
----------------
- Load a persisted trained model.
- Validate prediction input.
- Apply the exact preprocessing stored inside the trained pipeline.
- Generate predictions for classification and regression.
- Generate probabilities when supported.
- Return a standardized prediction result.

This module does NOT:
- train models,
- evaluate models,
- select models,
- render Streamlit UI,
- call an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .model_persistence import (
    load_model,
    load_model_metadata,
)


# ============================================================
# RESULT OBJECT
# ============================================================

@dataclass
class PredictionResult:
    """
    Standardized result returned by the prediction engine.
    """

    prediction: Any

    task: str

    model_name: str

    display_name: str

    target_column: str

    confidence: float | None = None

    probabilities: dict[str, float] | None = None

    input_data: dict[str, Any] = field(
        default_factory=dict
    )

    warnings: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert prediction result into a serializable dictionary.
        """

        return {
            "prediction": self.prediction,
            "task": self.task,
            "model_name": self.model_name,
            "display_name": self.display_name,
            "target_column": self.target_column,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "input_data": self.input_data,
            "warnings": self.warnings,
        }


# ============================================================
# VALIDATION
# ============================================================

def _validate_input_dataframe(
    input_data: pd.DataFrame,
) -> None:
    """
    Validate prediction input.
    """

    if not isinstance(
        input_data,
        pd.DataFrame,
    ):
        raise TypeError(
            "input_data must be a pandas DataFrame."
        )

    if input_data.empty:
        raise ValueError(
            "Prediction input cannot be empty."
        )


def _validate_feature_columns(
    input_data: pd.DataFrame,
    expected_features: list[str],
) -> None:
    """
    Ensure prediction input contains exactly the
    features expected by the trained model.
    """

    if not expected_features:
        return

    missing_features = [
        feature
        for feature in expected_features
        if feature not in input_data.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing prediction feature(s): "
            + ", ".join(missing_features)
        )

    # Extra columns are allowed because the prediction
    # engine will select only the features used during training.


def _clean_prediction_value(
    value: Any,
) -> Any:
    """
    Convert numpy values into normal Python values.
    """

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    return value


# ============================================================
# PROBABILITY HANDLING
# ============================================================

def _get_class_labels(
    model: Any,
) -> list[Any] | None:
    """
    Try to retrieve class labels from a fitted model/pipeline.
    """

    # Pipeline may expose classes_ directly.
    if hasattr(
        model,
        "classes_",
    ):

        return list(
            model.classes_
        )

    # Some pipelines expose the final estimator.
    if hasattr(
        model,
        "named_steps",
    ):

        for estimator in reversed(
            list(
                model.named_steps.values()
            )
        ):

            if hasattr(
                estimator,
                "classes_",
            ):

                return list(
                    estimator.classes_
                )

    return None


def _generate_probabilities(
    model: Any,
    input_data: pd.DataFrame,
) -> tuple[
    dict[str, float] | None,
    float | None,
    list[str],
]:
    """
    Generate class probabilities when supported.

    Returns
    -------
    probabilities
    confidence
    warnings
    """

    warnings: list[str] = []

    if not hasattr(
        model,
        "predict_proba",
    ):

        warnings.append(
            "The selected model does not support "
            "probability predictions."
        )

        return (
            None,
            None,
            warnings,
        )

    try:

        probabilities = model.predict_proba(
            input_data
        )

    except (
        AttributeError,
        ValueError,
    ) as exc:

        warnings.append(
            "Probability prediction failed: "
            f"{exc}"
        )

        return (
            None,
            None,
            warnings,
        )

    if probabilities is None:

        return (
            None,
            None,
            warnings,
        )

    probabilities = np.asarray(
        probabilities
    )

    if probabilities.ndim != 2:

        warnings.append(
            "Unexpected probability output shape."
        )

        return (
            None,
            None,
            warnings,
        )

    classes = _get_class_labels(
        model
    )

    # ---------------------------------------------------------
    # Single prediction
    # ---------------------------------------------------------

    row = probabilities[0]

    if classes is None:

        probability_dict = {
            str(index): float(probability)
            for index, probability
            in enumerate(row)
        }

    else:

        probability_dict = {
            str(
                _clean_prediction_value(
                    label
                )
            ): float(probability)
            for label, probability
            in zip(
                classes,
                row,
            )
        }

    confidence = float(
        np.max(row)
    )

    return (
        probability_dict,
        confidence,
        warnings,
    )


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def predict(
    model_path: str | Path,
    metadata_path: str | Path,
    input_data: pd.DataFrame,
) -> PredictionResult:
    """
    Generate a prediction using a persisted DataMindAI model.

    Parameters
    ----------
    model_path:
        Path to the saved .joblib model.

    metadata_path:
        Path to the saved model metadata.

    input_data:
        DataFrame containing one or more rows of prediction data.

    Returns
    -------
    PredictionResult
    """

    _validate_input_dataframe(
        input_data
    )

    # ---------------------------------------------------------
    # Load model + metadata
    # ---------------------------------------------------------

    model = load_model(
        model_path
    )

    metadata = load_model_metadata(
        metadata_path
    )

    expected_features = metadata.get(
        "feature_names",
        [],
    )

    _validate_feature_columns(
        input_data,
        expected_features,
    )

    # ---------------------------------------------------------
    # Select only training features
    # ---------------------------------------------------------

    if expected_features:

        prediction_input = (
            input_data[
                expected_features
            ]
            .copy()
        )

    else:

        prediction_input = (
            input_data.copy()
        )

    # ---------------------------------------------------------
    # Generate prediction
    # ---------------------------------------------------------

    try:

        predictions = model.predict(
            prediction_input
        )

    except Exception as exc:

        raise RuntimeError(
            "Prediction failed: "
            f"{exc}"
        ) from exc

    if len(predictions) == 0:

        raise RuntimeError(
            "The model returned no predictions."
        )

    # ---------------------------------------------------------
    # Use first prediction for the workspace
    # ---------------------------------------------------------

    prediction = _clean_prediction_value(
        predictions[0]
    )

    task = metadata.get(
        "task",
        "",
    )

    model_name = metadata.get(
        "model_name",
        Path(model_path).stem,
    )

    display_name = metadata.get(
        "display_name",
        model_name,
    )

    target_column = metadata.get(
        "target_column",
        "",
    )

    warnings: list[str] = []

    probabilities = None
    confidence = None

    # ---------------------------------------------------------
    # Classification probabilities
    # ---------------------------------------------------------

    if task == "classification":

        (
            probabilities,
            confidence,
            probability_warnings,
        ) = _generate_probabilities(
            model,
            prediction_input,
        )

        warnings.extend(
            probability_warnings
        )

    # ---------------------------------------------------------
    # Build result
    # ---------------------------------------------------------

    input_dict = {
        column: _clean_prediction_value(
            value
        )
        for column, value
        in input_data.iloc[0].to_dict().items()
    }

    return PredictionResult(
        prediction=prediction,
        task=task,
        model_name=model_name,
        display_name=display_name,
        target_column=target_column,
        confidence=confidence,
        probabilities=probabilities,
        input_data=input_dict,
        warnings=warnings,
    )


def predict_from_artifact(
    artifact: Any,
    input_data: pd.DataFrame,
) -> PredictionResult:
    """
    Generate a prediction using a ModelArtifact.

    This is a convenience wrapper for the frontend.
    """

    if artifact is None:

        raise ValueError(
            "A valid model artifact is required."
        )

    return predict(
        model_path=artifact.model_path,
        metadata_path=artifact.metadata_path,
        input_data=input_data,
    )