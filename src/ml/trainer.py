"""
Model training engine for DataMindAI.

Responsibilities
----------------
- Validate training inputs.
- Split data into train/test sets.
- Build model-aware preprocessing.
- Fit preprocessing using training data only.
- Train the selected model.
- Keep preprocessing and model together in one sklearn Pipeline.
- Return structured training metadata.

This module does NOT:
- compare models,
- select the best model,
- perform hyperparameter optimization,
- call an LLM,
- generate reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .model_factory import create_model, get_model_spec
from .preprocessing_pipeline import (
    build_preprocessing_pipeline,
)
from .task_detector import MLTask


@dataclass
class TrainingConfig:
    """
    Configuration used for model training.

    Attributes
    ----------
    test_size:
        Fraction of the dataset reserved for testing.

    random_state:
        Random seed used for reproducibility.

    stratify:
        Whether classification targets should be stratified.

    n_jobs:
        Number of CPU workers used by supported models.
    """

    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    n_jobs: int = -1


@dataclass
class TrainingResult:
    """
    Structured result returned after model training.

    Attributes
    ----------
    model_name:
        Internal name of the trained model.

    display_name:
        Human-readable model name.

    task:
        ML task.

    target_column:
        Target column used for training.

    pipeline:
        Complete fitted preprocessing + model pipeline.

    X_train:
        Original training features.

    X_test:
        Original testing features.

    y_train:
        Training target.

    y_test:
        Testing target.

    train_rows:
        Number of training rows.

    test_rows:
        Number of testing rows.

    train_size:
        Fraction of data used for training.

    test_size:
        Fraction of data used for testing.

    feature_names:
        Original feature names.

    transformed_feature_names:
        Feature names after preprocessing.

    warnings:
        Training warnings.
    """

    model_name: str
    display_name: str

    task: MLTask
    target_column: str

    pipeline: Pipeline

    X_train: pd.DataFrame
    X_test: pd.DataFrame

    y_train: pd.Series
    y_test: pd.Series

    train_rows: int
    test_rows: int

    train_size: float
    test_size: float

    feature_names: list[str] = field(
        default_factory=list
    )

    transformed_feature_names: list[str] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    @property
    def model(self) -> Any:
        """
        Return the trained estimator from the pipeline.
        """

        return self.pipeline.named_steps["model"]

    def to_dict(self) -> dict[str, Any]:
        """
        Convert training metadata into a serializable dictionary.

        The actual DataFrames and trained model are intentionally
        excluded.
        """

        return {
            "model_name": self.model_name,
            "display_name": self.display_name,
            "task": self.task.value,
            "target_column": self.target_column,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "feature_names": self.feature_names,
            "transformed_feature_names": (
                self.transformed_feature_names
            ),
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def _validate_training_inputs(
    X: pd.DataFrame,
    y: pd.Series,
    task: MLTask,
    target_column: str,
) -> None:
    """Validate training inputs."""

    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "X must be a pandas DataFrame."
        )

    if not isinstance(y, pd.Series):
        raise TypeError(
            "y must be a pandas Series."
        )

    if X.empty:
        raise ValueError(
            "Cannot train a model with an empty feature DataFrame."
        )

    if y.empty:
        raise ValueError(
            "Cannot train a model with an empty target Series."
        )

    if len(X) != len(y):
        raise ValueError(
            "X and y must contain the same number of rows."
        )

    if not isinstance(task, MLTask):
        raise ValueError(
            "task must be an MLTask enum value."
        )

    if not isinstance(target_column, str):
        raise TypeError(
            "target_column must be a string."
        )

    if not target_column.strip():
        raise ValueError(
            "target_column cannot be empty."
        )


def _validate_config(
    config: TrainingConfig,
) -> None:
    """Validate training configuration."""

    if not 0 < config.test_size < 1:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    if config.n_jobs == 0:
        raise ValueError(
            "n_jobs cannot be 0."
        )


# ---------------------------------------------------------------------
# Target validation
# ---------------------------------------------------------------------

def _validate_target(
    y: pd.Series,
    task: MLTask,
) -> None:
    """Validate target values before training."""

    if y.isna().all():
        raise ValueError(
            "The target contains only missing values."
        )

    if y.isna().any():
        raise ValueError(
            "The target contains missing values. "
            "Target missing values must be handled before "
            "model training."
        )

    if y.nunique() < 2:
        raise ValueError(
            "The target must contain at least two unique values."
        )

    if task == MLTask.CLASSIFICATION:
        if y.nunique() < 2:
            raise ValueError(
                "Classification requires at least two classes."
            )


# ---------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------

def _split_data(
    X: pd.DataFrame,
    y: pd.Series,
    task: MLTask,
    config: TrainingConfig,
):
    """
    Split data into training and testing sets.

    Classification uses stratification when possible.
    Regression uses a normal random split.
    """

    stratify_target = None

    if (
        task == MLTask.CLASSIFICATION
        and config.stratify
    ):
        stratify_target = y

    try:
        return train_test_split(
            X,
            y,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=stratify_target,
        )

    except ValueError as exc:

        # Some small or highly imbalanced datasets cannot be
        # stratified. Instead of silently changing the split,
        # provide a clear fallback warning at a higher level.
        if (
            task == MLTask.CLASSIFICATION
            and config.stratify
        ):
            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=config.test_size,
                    random_state=config.random_state,
                    stratify=None,
                )
            )

            return (
                X_train,
                X_test,
                y_train,
                y_test,
                str(exc),
            )

        raise


# ---------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------

def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_name: str,
    task: MLTask,
    target_column: str,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """
    Train one model using a complete preprocessing + model pipeline.

    Parameters
    ----------
    X:
        Feature DataFrame.

    y:
        Target Series.

    model_name:
        Model to train.

    task:
        Classification or regression.

    target_column:
        Name of the prediction target.

    config:
        Optional TrainingConfig.

    Returns
    -------
    TrainingResult
        Structured information containing the fitted pipeline,
        train/test data, and training metadata.

    Important
    ---------
    The preprocessing pipeline is fitted only on X_train.

    The resulting sklearn Pipeline contains:

        preprocessing
             ↓
           model

    This same pipeline can later be used directly for prediction.
    """

    if config is None:
        config = TrainingConfig()

    _validate_config(config)

    _validate_training_inputs(
        X=X,
        y=y,
        task=task,
        target_column=target_column,
    )

    _validate_target(
        y=y,
        task=task,
    )

    # ---------------------------------------------------------
    # Validate model
    # ---------------------------------------------------------

    model_spec = get_model_spec(
        model_name=model_name,
        task=task,
    )

    # ---------------------------------------------------------
    # Split dataset
    # ---------------------------------------------------------

    split_result = _split_data(
        X=X,
        y=y,
        task=task,
        config=config,
    )

    warnings: list[str] = []

    if len(split_result) == 5:
        (
            X_train,
            X_test,
            y_train,
            y_test,
            split_warning,
        ) = split_result

        warnings.append(
            "Stratified splitting could not be used for this "
            f"dataset. Falling back to a random split. "
            f"Original reason: {split_warning}"
        )

    else:
        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = split_result

    # ---------------------------------------------------------
    # Build preprocessing
    # ---------------------------------------------------------

    preprocessing_result = (
        build_preprocessing_pipeline(
            X=X_train,
            task=task,
            model_name=model_name,
        )
    )

    warnings.extend(
        preprocessing_result.warnings
    )

    # ---------------------------------------------------------
    # Create model
    # ---------------------------------------------------------

    model = create_model(
        model_name=model_name,
        task=task,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )

    # ---------------------------------------------------------
    # Complete pipeline
    # ---------------------------------------------------------

    pipeline = Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessing_result.pipeline,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    pipeline.fit(
        X_train,
        y_train,
    )

    # ---------------------------------------------------------
    # Get transformed feature names
    # ---------------------------------------------------------

    transformed_feature_names: list[str] = []

    try:
        transformed_feature_names = (
            pipeline
            .named_steps["preprocessing"]
            .get_feature_names_out()
            .tolist()
        )

    except (AttributeError, ValueError):
        transformed_feature_names = []

    return TrainingResult(
        model_name=model_spec.name,
        display_name=model_spec.display_name,
        task=task,
        target_column=target_column,
        pipeline=pipeline,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_rows=len(X_train),
        test_rows=len(X_test),
        train_size=len(X_train) / len(X),
        test_size=len(X_test) / len(X),
        feature_names=X.columns.tolist(),
        transformed_feature_names=transformed_feature_names,
        warnings=warnings,
    )


def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    task: MLTask,
    target_column: str,
    model_names: list[str] | None = None,
    config: TrainingConfig | None = None,
) -> dict[str, TrainingResult]:
    """
    Train multiple models using the same train/test split.

    Parameters
    ----------
    X:
        Feature DataFrame.

    y:
        Target Series.

    task:
        Classification or regression.

    target_column:
        Target column name.

    model_names:
        Optional list of models. If None, all supported models
        for the task are trained.

    config:
        Optional TrainingConfig.

    Returns
    -------
    dict[str, TrainingResult]
        Mapping of model name to training result.

    Important
    ---------
    Each model currently receives the same random train/test split
    because the split is controlled by the same random_state.

    This makes model comparison fair and reproducible.
    """

    if config is None:
        config = TrainingConfig()

    if model_names is None:

        from .model_factory import get_model_names

        model_names = get_model_names(
            task
        )

    if not model_names:
        raise ValueError(
            "model_names cannot be empty."
        )

    results: dict[str, TrainingResult] = {}

    for model_name in model_names:

        result = train_model(
            X=X,
            y=y,
            model_name=model_name,
            task=task,
            target_column=target_column,
            config=config,
        )

        results[model_name] = result

    return results


def predict(
    training_result: TrainingResult,
    X: pd.DataFrame,
):
    """
    Generate predictions using a trained pipeline.

    This is a small convenience wrapper.

    The same preprocessing used during training is automatically
    applied before prediction.
    """

    if not isinstance(
        training_result,
        TrainingResult,
    ):
        raise TypeError(
            "training_result must be a TrainingResult."
        )

    if not isinstance(
        X,
        pd.DataFrame,
    ):
        raise TypeError(
            "X must be a pandas DataFrame."
        )

    if X.empty:
        raise ValueError(
            "Cannot generate predictions from an empty DataFrame."
        )

    return training_result.pipeline.predict(X)


def predict_proba(
    training_result: TrainingResult,
    X: pd.DataFrame,
):
    """
    Generate class probabilities for a trained classification model.
    """

    if not isinstance(
        training_result,
        TrainingResult,
    ):
        raise TypeError(
            "training_result must be a TrainingResult."
        )

    if training_result.task != MLTask.CLASSIFICATION:
        raise ValueError(
            "predict_proba is only available for classification."
        )

    if not hasattr(
        training_result.pipeline,
        "predict_proba",
    ):
        raise ValueError(
            f"Model '{training_result.model_name}' "
            "does not support probability prediction."
        )

    return training_result.pipeline.predict_proba(X)