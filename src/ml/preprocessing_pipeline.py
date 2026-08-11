"""
Model-aware preprocessing pipeline for DataMindAI.

Responsibilities
----------------
- Detect numeric and categorical feature columns.
- Build preprocessing pipelines.
- Handle missing values.
- Encode categorical features.
- Scale features when required by the model.
- Keep preprocessing fitted only on training data.
- Return a complete sklearn Pipeline that can be reused for
  validation, testing, and future predictions.

This module does NOT:
- train models,
- evaluate models,
- select models,
- call an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .model_factory import ModelFamily
from .task_detector import MLTask


@dataclass
class FeatureColumns:
    """
    Stores the feature-column groups used by preprocessing.

    Attributes
    ----------
    numeric:
        Numerical feature columns.

    categorical:
        Categorical feature columns.

    datetime:
        Datetime feature columns that are not directly passed into
        the standard preprocessing pipeline.
    """

    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    datetime: list[str] = field(default_factory=list)

    @property
    def all_features(self) -> list[str]:
        """Return all detected feature columns."""

        return (
            self.numeric
            + self.categorical
            + self.datetime
        )


@dataclass
class PreprocessingResult:
    """
    Result returned after creating a preprocessing pipeline.

    Attributes
    ----------
    pipeline:
        sklearn ColumnTransformer used to preprocess the features.

    feature_columns:
        Detected feature-column groups.

    transformed_feature_names:
        Feature names generated after preprocessing.

    scaling_applied:
        Whether numerical scaling is applied.

    dropped_columns:
        Columns excluded from the training pipeline.

    warnings:
        Warnings generated during preprocessing construction.
    """

    pipeline: ColumnTransformer

    feature_columns: FeatureColumns

    transformed_feature_names: list[str] = field(
        default_factory=list
    )

    scaling_applied: bool = False

    dropped_columns: list[str] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert the result into a serializable dictionary."""

        return {
            "numeric_columns": self.feature_columns.numeric,
            "categorical_columns": self.feature_columns.categorical,
            "datetime_columns": self.feature_columns.datetime,
            "transformed_feature_names": (
                self.transformed_feature_names
            ),
            "scaling_applied": self.scaling_applied,
            "dropped_columns": self.dropped_columns,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------
# Model preprocessing policy
# ---------------------------------------------------------------------

SCALING_REQUIRED_MODELS = {
    ModelFamily.LOGISTIC_REGRESSION.value,
    ModelFamily.LINEAR_REGRESSION.value,
}


def model_requires_scaling(model_name: str) -> bool:
    """
    Determine whether the model benefits from feature scaling.

    Linear and logistic regression use scaled numerical features.

    Tree-based models such as Random Forest, XGBoost, and LightGBM
    do not require numerical feature scaling.
    """

    return model_name.strip().lower() in SCALING_REQUIRED_MODELS


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------

def _validate_feature_data(
    X: pd.DataFrame,
) -> None:
    """Validate the input feature DataFrame."""

    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "X must be a pandas DataFrame."
        )

    if X.empty:
        raise ValueError(
            "Cannot build a preprocessing pipeline from "
            "an empty DataFrame."
        )


def _validate_task(task: MLTask) -> None:
    """Validate the ML task."""

    if not isinstance(task, MLTask):
        raise ValueError(
            "task must be an MLTask enum value."
        )


def _validate_model_name(model_name: str) -> None:
    """Validate the model name."""

    if not isinstance(model_name, str):
        raise TypeError(
            "model_name must be a string."
        )

    if not model_name.strip():
        raise ValueError(
            "model_name cannot be empty."
        )


# ---------------------------------------------------------------------
# Feature detection
# ---------------------------------------------------------------------

def detect_feature_columns(
    X: pd.DataFrame,
) -> FeatureColumns:
    """
    Detect numerical, categorical, and datetime columns.

    Parameters
    ----------
    X:
        Feature DataFrame without the target column.

    Returns
    -------
    FeatureColumns
        Grouped feature columns.
    """

    _validate_feature_data(X)

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=[
            "object",
            "category",
            "string",
            "bool",
        ]
    ).columns.tolist()

    datetime_columns = X.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()

    return FeatureColumns(
        numeric=numeric_columns,
        categorical=categorical_columns,
        datetime=datetime_columns,
    )


# ---------------------------------------------------------------------
# Pipeline components
# ---------------------------------------------------------------------

def _build_numeric_pipeline(
    scaling_required: bool,
) -> Pipeline:
    """
    Build the numerical preprocessing pipeline.

    Missing numerical values are replaced using the median.

    Scaling is added only for models that benefit from it.
    """

    steps: list[tuple[str, Any]] = [
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        )
    ]

    if scaling_required:
        steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )

    return Pipeline(steps=steps)


def _build_categorical_pipeline() -> Pipeline:
    """
    Build the categorical preprocessing pipeline.

    Missing categorical values are replaced with the most
    frequent category, then one-hot encoded.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------
# Main pipeline builder
# ---------------------------------------------------------------------

def build_preprocessing_pipeline(
    X: pd.DataFrame,
    task: MLTask,
    model_name: str,
) -> PreprocessingResult:
    """
    Build a model-aware preprocessing pipeline.

    Parameters
    ----------
    X:
        Feature DataFrame.

    task:
        Detected ML task.

    model_name:
        Model that will consume the transformed features.

    Returns
    -------
    PreprocessingResult
        Complete preprocessing configuration.

    Notes
    -----
    The returned ColumnTransformer is NOT fitted here.

    It must be fitted only on training data.

    Example
    -------
    pipeline.fit(X_train)

    X_train_processed = pipeline.transform(X_train)
    X_test_processed = pipeline.transform(X_test)
    """

    _validate_feature_data(X)
    _validate_task(task)
    _validate_model_name(model_name)

    feature_columns = detect_feature_columns(X)

    scaling_required = model_requires_scaling(
        model_name
    )

    transformers: list[tuple[str, Any, list[str]]] = []

    # ---------------------------------------------------------
    # Numerical features
    # ---------------------------------------------------------

    if feature_columns.numeric:

        numeric_pipeline = _build_numeric_pipeline(
            scaling_required=scaling_required
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                feature_columns.numeric,
            )
        )

    # ---------------------------------------------------------
    # Categorical features
    # ---------------------------------------------------------

    if feature_columns.categorical:

        categorical_pipeline = (
            _build_categorical_pipeline()
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                feature_columns.categorical,
            )
        )

    # ---------------------------------------------------------
    # Datetime handling
    # ---------------------------------------------------------

    dropped_columns: list[str] = []
    warnings: list[str] = []

    if feature_columns.datetime:

        dropped_columns.extend(
            feature_columns.datetime
        )

        warnings.append(
            "Datetime columns were excluded from the current "
            "training pipeline. Datetime feature engineering "
            "will be added in a later version."
        )

    # ---------------------------------------------------------
    # No usable features
    # ---------------------------------------------------------

    if not transformers:

        raise ValueError(
            "No supported feature columns were found. "
            "The dataset must contain at least one numerical "
            "or categorical feature."
        )

    # ---------------------------------------------------------
    # ColumnTransformer
    # ---------------------------------------------------------

    pipeline = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return PreprocessingResult(
        pipeline=pipeline,
        feature_columns=feature_columns,
        transformed_feature_names=[],
        scaling_applied=scaling_required,
        dropped_columns=dropped_columns,
        warnings=warnings,
    )


# ---------------------------------------------------------------------
# Fitting and transformation
# ---------------------------------------------------------------------

def fit_preprocessing_pipeline(
    preprocessing_result: PreprocessingResult,
    X_train: pd.DataFrame,
) -> PreprocessingResult:
    """
    Fit preprocessing using training data only.

    This function is deliberately separate from
    build_preprocessing_pipeline() so that the architecture
    explicitly follows:

        build
          ↓
        fit on X_train
          ↓
        transform X_train / X_test / prediction data
    """

    if not isinstance(
        preprocessing_result,
        PreprocessingResult,
    ):
        raise TypeError(
            "preprocessing_result must be a "
            "PreprocessingResult."
        )

    _validate_feature_data(X_train)

    preprocessing_result.pipeline.fit(
        X_train
    )

    try:
        feature_names = (
            preprocessing_result
            .pipeline
            .get_feature_names_out()
            .tolist()
        )

        preprocessing_result.transformed_feature_names = (
            feature_names
        )

    except (AttributeError, ValueError):
        preprocessing_result.transformed_feature_names = []

    return preprocessing_result


def transform_features(
    preprocessing_result: PreprocessingResult,
    X: pd.DataFrame,
):
    """
    Transform features using an already-fitted pipeline.

    Parameters
    ----------
    preprocessing_result:
        Fitted preprocessing result.

    X:
        Data to transform.

    Returns
    -------
    Transformed feature matrix.
    """

    if not isinstance(
        preprocessing_result,
        PreprocessingResult,
    ):
        raise TypeError(
            "preprocessing_result must be a "
            "PreprocessingResult."
        )

    _validate_feature_data(X)

    return preprocessing_result.pipeline.transform(X)


def fit_transform_training_data(
    preprocessing_result: PreprocessingResult,
    X_train: pd.DataFrame,
):
    """
    Fit the preprocessing pipeline and transform training data.

    This is a convenience method for the training stage.
    """

    if not isinstance(
        preprocessing_result,
        PreprocessingResult,
    ):
        raise TypeError(
            "preprocessing_result must be a "
            "PreprocessingResult."
        )

    _validate_feature_data(X_train)

    transformed = (
        preprocessing_result
        .pipeline
        .fit_transform(X_train)
    )

    try:
        preprocessing_result.transformed_feature_names = (
            preprocessing_result
            .pipeline
            .get_feature_names_out()
            .tolist()
        )

    except (AttributeError, ValueError):
        preprocessing_result.transformed_feature_names = []

    return transformed


def get_transformed_feature_names(
    preprocessing_result: PreprocessingResult,
) -> list[str]:
    """
    Return feature names after preprocessing.

    The pipeline must have been fitted before calling this function.
    """

    if not isinstance(
        preprocessing_result,
        PreprocessingResult,
    ):
        raise TypeError(
            "preprocessing_result must be a "
            "PreprocessingResult."
        )

    if preprocessing_result.transformed_feature_names:
        return list(
            preprocessing_result.transformed_feature_names
        )

    try:
        return (
            preprocessing_result
            .pipeline
            .get_feature_names_out()
            .tolist()
        )

    except (AttributeError, ValueError) as exc:
        raise RuntimeError(
            "The preprocessing pipeline has not been fitted yet."
        ) from exc