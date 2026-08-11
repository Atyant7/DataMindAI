"""
Model evaluation engine for DataMindAI.

Responsibilities
----------------
- Evaluate trained classification models.
- Evaluate trained regression models.
- Calculate task-appropriate metrics.
- Generate confusion matrices for classification.
- Generate classification reports.
- Return standardized evaluation results.

This module does NOT:
- train models,
- select the best model,
- perform hyperparameter optimization,
- call an LLM,
- generate final reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from .task_detector import MLTask
from .trainer import TrainingResult


@dataclass
class EvaluationResult:
    """
    Standardized result produced after evaluating one model.

    Attributes
    ----------
    model_name:
        Internal model name.

    display_name:
        Human-readable model name.

    task:
        Classification or regression.

    metrics:
        Dictionary containing calculated metrics.

    primary_metric:
        Metric that should be used for model comparison.

    primary_score:
        Value of the primary metric.

    confusion_matrix:
        Confusion matrix for classification models.

    classification_report:
        Detailed classification report.

    predictions:
        Predictions generated on the test set.

    prediction_probabilities:
        Probability predictions for classification models
        when supported.

    warnings:
        Evaluation warnings.
    """

    model_name: str
    display_name: str

    task: MLTask

    metrics: dict[str, float] = field(
        default_factory=dict
    )

    primary_metric: str = ""

    primary_score: float = 0.0

    confusion_matrix: list[list[int]] | None = None

    classification_report: dict[str, Any] | None = None

    predictions: list[Any] = field(
        default_factory=list
    )

    prediction_probabilities: list[Any] | None = None

    warnings: list[str] = field(
        default_factory=list
    )

    @property
    def is_classification(self) -> bool:
        """Return True when evaluating classification."""

        return self.task == MLTask.CLASSIFICATION

    @property
    def is_regression(self) -> bool:
        """Return True when evaluating regression."""

        return self.task == MLTask.REGRESSION

    def to_dict(
        self,
        include_predictions: bool = False,
    ) -> dict[str, Any]:
        """
        Convert evaluation results into a serializable dictionary.

        Predictions are excluded by default because they can be large.
        """

        result = {
            "model_name": self.model_name,
            "display_name": self.display_name,
            "task": self.task.value,
            "metrics": self.metrics,
            "primary_metric": self.primary_metric,
            "primary_score": self.primary_score,
            "confusion_matrix": self.confusion_matrix,
            "classification_report": (
                self.classification_report
            ),
            "warnings": self.warnings,
        }

        if include_predictions:
            result["predictions"] = self.predictions
            result["prediction_probabilities"] = (
                self.prediction_probabilities
            )

        return result


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def _safe_float(
    value: Any,
) -> float:
    """
    Convert a metric value to a normal Python float.

    Handles numpy numeric types and NaN/inf values.
    """

    value = float(value)

    if not np.isfinite(value):
        return 0.0

    return value


def _validate_training_result(
    training_result: TrainingResult,
) -> None:
    """Validate a TrainingResult before evaluation."""

    if not isinstance(
        training_result,
        TrainingResult,
    ):
        raise TypeError(
            "training_result must be a TrainingResult."
        )

    if training_result.X_test.empty:
        raise ValueError(
            "Cannot evaluate a model without test data."
        )

    if training_result.y_test.empty:
        raise ValueError(
            "Cannot evaluate a model without test targets."
        )


# ---------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------

def _evaluate_classification(
    training_result: TrainingResult,
) -> EvaluationResult:
    """Evaluate a trained classification model."""

    pipeline = training_result.pipeline

    X_test = training_result.X_test
    y_test = training_result.y_test

    warnings: list[str] = []

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------

    predictions = pipeline.predict(
        X_test
    )

    # ---------------------------------------------------------
    # Basic metrics
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    metrics: dict[str, float] = {
        "accuracy": _safe_float(accuracy),
        "precision": _safe_float(precision),
        "recall": _safe_float(recall),
        "f1": _safe_float(f1),
    }

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------

    labels = sorted(
        pd.Series(y_test).unique().tolist()
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    confusion_matrix_result = (
        cm.astype(int).tolist()
    )

    # ---------------------------------------------------------
    # Classification report
    # ---------------------------------------------------------

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    # Convert numpy values to Python floats.
    classification_report_result = {}

    for label, values in report.items():

        if isinstance(values, dict):

            classification_report_result[
                str(label)
            ] = {
                metric: _safe_float(score)
                for metric, score in values.items()
            }

        else:
            classification_report_result[
                str(label)
            ] = _safe_float(values)

    # ---------------------------------------------------------
    # Probability-based metrics
    # ---------------------------------------------------------

    prediction_probabilities = None

    if hasattr(
        pipeline,
        "predict_proba",
    ):

        try:

            prediction_probabilities = (
                pipeline.predict_proba(X_test)
            )

            classes = (
                pipeline.classes_
                if hasattr(
                    pipeline,
                    "classes_",
                )
                else None
            )

            # -------------------------------------------------
            # ROC-AUC
            # -------------------------------------------------

            if classes is not None:

                try:

                    if len(classes) == 2:

                        positive_probabilities = (
                            prediction_probabilities[:, 1]
                        )

                        y_binary = (
                            pd.Series(y_test)
                            == classes[1]
                        ).astype(int)
                        
                        roc_auc = roc_auc_score(
                            y_binary,
                            positive_probabilities,
                        )

                    else:

                        roc_auc = roc_auc_score(
                            y_test,
                            prediction_probabilities,
                            multi_class="ovr",
                            average="weighted",
                        )

                    metrics["roc_auc"] = (
                        _safe_float(roc_auc)
                    )

                except ValueError as exc:

                    warnings.append(
                        "ROC-AUC could not be calculated: "
                        f"{exc}"
                    )

                # -------------------------------------------------
                # Average Precision / PR-AUC
                # -------------------------------------------------

                try:

                    if len(classes) == 2:

                        y_binary = (
                            pd.Series(y_test)
                            == classes[1]
                        ).astype(int)

                        average_precision = (
                            average_precision_score(
                                y_binary,
                                prediction_probabilities[:, 1],
                            )
                        )

                        metrics[
                            "average_precision"
                        ] = _safe_float(
                            average_precision
                        )

                except ValueError as exc:

                    warnings.append(
                        "Average precision could not be "
                        f"calculated: {exc}"
                    )

        except (AttributeError, ValueError) as exc:

            warnings.append(
                "Probability predictions could not be "
                f"generated: {exc}"
            )

    else:

        warnings.append(
            "The selected model does not support "
            "probability prediction."
        )

    # ---------------------------------------------------------
    # Primary metric
    # ---------------------------------------------------------

    if "f1" in metrics:
        primary_metric = "f1"
        primary_score = metrics["f1"]

    else:
        primary_metric = "accuracy"
        primary_score = metrics["accuracy"]

    return EvaluationResult(
        model_name=training_result.model_name,
        display_name=training_result.display_name,
        task=MLTask.CLASSIFICATION,
        metrics=metrics,
        primary_metric=primary_metric,
        primary_score=primary_score,
        confusion_matrix=confusion_matrix_result,
        classification_report=(
            classification_report_result
        ),
        predictions=predictions.tolist(),
        prediction_probabilities=(
            prediction_probabilities.tolist()
            if prediction_probabilities is not None
            else None
        ),
        warnings=warnings,
    )


# ---------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------

def _evaluate_regression(
    training_result: TrainingResult,
) -> EvaluationResult:
    """Evaluate a trained regression model."""

    pipeline = training_result.pipeline

    X_test = training_result.X_test
    y_test = training_result.y_test

    warnings: list[str] = []

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------

    predictions = pipeline.predict(
        X_test
    )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    mse = mean_squared_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions,
    )

    metrics: dict[str, float] = {
        "mae": _safe_float(mae),
        "mse": _safe_float(mse),
        "rmse": _safe_float(rmse),
        "r2": _safe_float(r2),
    }

    # ---------------------------------------------------------
    # Primary metric
    # ---------------------------------------------------------

    primary_metric = "rmse"
    primary_score = metrics["rmse"]

    return EvaluationResult(
        model_name=training_result.model_name,
        display_name=training_result.display_name,
        task=MLTask.REGRESSION,
        metrics=metrics,
        primary_metric=primary_metric,
        primary_score=primary_score,
        predictions=predictions.tolist(),
        warnings=warnings,
    )


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def evaluate_model(
    training_result: TrainingResult,
) -> EvaluationResult:
    """
    Evaluate one trained model.

    Classification uses:

        accuracy
        precision
        recall
        F1
        ROC-AUC when possible
        Average Precision when possible
        confusion matrix
        classification report

    Regression uses:

        MAE
        MSE
        RMSE
        R²
    """

    _validate_training_result(
        training_result
    )

    if training_result.task == MLTask.CLASSIFICATION:

        return _evaluate_classification(
            training_result
        )

    if training_result.task == MLTask.REGRESSION:

        return _evaluate_regression(
            training_result
        )

    raise ValueError(
        f"Unsupported ML task: "
        f"{training_result.task}"
    )


def evaluate_models(
    training_results: dict[str, TrainingResult],
) -> dict[str, EvaluationResult]:
    """
    Evaluate multiple trained models.

    Parameters
    ----------
    training_results:
        Dictionary returned by train_models().

    Returns
    -------
    dict[str, EvaluationResult]
        Evaluation results keyed by model name.
    """

    if not isinstance(
        training_results,
        dict,
    ):
        raise TypeError(
            "training_results must be a dictionary."
        )

    if not training_results:
        raise ValueError(
            "training_results cannot be empty."
        )

    evaluation_results: dict[
        str,
        EvaluationResult,
    ] = {}

    for model_name, training_result in (
        training_results.items()
    ):

        evaluation_results[
            model_name
        ] = evaluate_model(
            training_result
        )

    return evaluation_results


def create_comparison_table(
    evaluation_results: dict[str, EvaluationResult],
) -> pd.DataFrame:
    """
    Create a DataFrame suitable for model comparison.

    Each row represents one model.

    Metrics that do not exist for a particular task
    are represented as NaN.
    """

    if not evaluation_results:
        raise ValueError(
            "evaluation_results cannot be empty."
        )

    rows: list[dict[str, Any]] = []

    for result in evaluation_results.values():

        row = {
            "model": result.display_name,
            "model_name": result.model_name,
            "task": result.task.value,
            "primary_metric": result.primary_metric,
            "primary_score": result.primary_score,
        }

        row.update(
            result.metrics
        )

        rows.append(row)

    return pd.DataFrame(rows)