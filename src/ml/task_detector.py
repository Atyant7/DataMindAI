"""
Machine-learning task detection for DataMindAI.

Responsibilities
----------------
- Validate the target column.
- Inspect the target variable.
- Determine whether the problem is classification or regression.
- Determine binary vs multiclass classification.
- Recommend suitable evaluation metrics.
- Provide confidence and warnings.

This module does NOT:
- train models,
- preprocess features,
- call an LLM,
- modify the original DataFrame.

All decisions are deterministic and based on the supplied data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class MLTask(str, Enum):
    """Supported machine-learning task types."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class TargetType(str, Enum):
    """More specific target categories."""

    BINARY = "binary_classification"
    MULTICLASS = "multiclass_classification"
    REGRESSION = "regression"


@dataclass
class TaskDetectionResult:
    """Structured result returned by the task detector."""

    target_column: str
    task: MLTask
    target_type: TargetType

    n_rows: int
    n_unique: int
    missing_count: int

    class_distribution: dict[str, float] = field(default_factory=dict)

    recommended_metrics: list[str] = field(default_factory=list)

    confidence: float = 0.0

    warnings: list[str] = field(default_factory=list)

    reasoning: str = ""

    @property
    def is_classification(self) -> bool:
        """Return True when the detected task is classification."""

        return self.task == MLTask.CLASSIFICATION

    @property
    def is_regression(self) -> bool:
        """Return True when the detected task is regression."""

        return self.task == MLTask.REGRESSION

    @property
    def is_binary(self) -> bool:
        """Return True for binary classification."""

        return self.target_type == TargetType.BINARY

    @property
    def is_multiclass(self) -> bool:
        """Return True for multiclass classification."""

        return self.target_type == TargetType.MULTICLASS

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a serializable dictionary."""

        return {
            "target_column": self.target_column,
            "task": self.task.value,
            "target_type": self.target_type.value,
            "n_rows": self.n_rows,
            "n_unique": self.n_unique,
            "missing_count": self.missing_count,
            "class_distribution": self.class_distribution,
            "recommended_metrics": self.recommended_metrics,
            "confidence": self.confidence,
            "warnings": self.warnings,
            "reasoning": self.reasoning,
        }


def _is_numeric_target(target: pd.Series) -> bool:
    """
    Determine whether the target is numeric.

    Boolean targets are excluded because they represent
    binary classification.
    """

    if pd.api.types.is_bool_dtype(target):
        return False

    return pd.api.types.is_numeric_dtype(target)


def _is_boolean_target(target: pd.Series) -> bool:
    """Return True if the target is boolean."""

    return pd.api.types.is_bool_dtype(target)


def _is_datetime_target(target: pd.Series) -> bool:
    """Return True if the target is datetime-like."""

    return pd.api.types.is_datetime64_any_dtype(target)


def _is_categorical_target(target: pd.Series) -> bool:
    """Return True if the target uses pandas categorical dtype."""

    return isinstance(target.dtype, pd.CategoricalDtype)


def _calculate_class_distribution(
    target: pd.Series,
) -> dict[str, float]:
    """Calculate the percentage distribution of classes."""

    proportions = target.value_counts(
        normalize=True,
        dropna=True,
    )

    distribution: dict[str, float] = {}

    for value, proportion in proportions.items():
        distribution[str(value)] = round(float(proportion), 6)

    return distribution


def _calculate_imbalance_ratio(target: pd.Series) -> float:
    """
    Calculate minority/majority class ratio.

    Examples
    --------
    50/50 -> 1.0
    10/90 -> 0.111...
    """

    counts = target.value_counts(dropna=True)

    if counts.empty:
        return 0.0

    if len(counts) == 1:
        return 1.0

    majority = float(counts.max())
    minority = float(counts.min())

    if majority == 0:
        return 0.0

    return minority / majority


def _detect_classification_metrics(
    target: pd.Series,
) -> tuple[list[str], list[str]]:
    """
    Select appropriate classification metrics.

    Accuracy remains available as a metric, but additional
    imbalance-aware metrics are recommended when necessary.
    """

    warnings: list[str] = []

    imbalance_ratio = _calculate_imbalance_ratio(target)

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]

    if imbalance_ratio < 0.5:
        warnings.append(
            "The target classes are imbalanced. "
            "Accuracy should not be used as the only "
            "model-selection metric."
        )

        metrics.extend(
            [
                "roc_auc",
                "average_precision",
            ]
        )

    else:
        metrics.append("roc_auc")

    return metrics, warnings


def _detect_regression_metrics() -> list[str]:
    """Return the default regression evaluation metrics."""

    return [
        "mae",
        "mse",
        "rmse",
        "r2",
    ]


def _validate_inputs(
    df: pd.DataFrame,
    target_column: str,
) -> None:
    """Validate task-detector inputs."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError(
            "Cannot detect ML task from an empty DataFrame."
        )

    if not isinstance(target_column, str) or not target_column.strip():
        raise ValueError(
            "target_column must be a non-empty string."
        )

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            "does not exist in the dataset."
        )


def _build_classification_result(
    *,
    target_column: str,
    target_type: TargetType,
    n_rows: int,
    n_unique: int,
    missing_count: int,
    target: pd.Series,
    confidence: float,
    reasoning: str,
    warnings: list[str],
) -> TaskDetectionResult:
    """Build a classification detection result."""

    class_distribution = _calculate_class_distribution(target)

    metrics, metric_warnings = _detect_classification_metrics(target)

    warnings.extend(metric_warnings)

    if missing_count > 0:
        warnings.append(
            f"Target contains {missing_count} missing values. "
            "These rows must be handled before training."
        )

    return TaskDetectionResult(
        target_column=target_column,
        task=MLTask.CLASSIFICATION,
        target_type=target_type,
        n_rows=n_rows,
        n_unique=n_unique,
        missing_count=missing_count,
        class_distribution=class_distribution,
        recommended_metrics=metrics,
        confidence=confidence,
        warnings=warnings,
        reasoning=reasoning,
    )


def detect_task(
    df: pd.DataFrame,
    target_column: str,
) -> TaskDetectionResult:
    """
    Detect the ML task represented by a target column.

    Parameters
    ----------
    df:
        Input dataset.

    target_column:
        Column selected as the prediction target.

    Returns
    -------
    TaskDetectionResult
        Structured information about the detected ML problem.

    Detection rules
    ---------------
    Boolean target
        -> Binary classification.

    Object/string/category target
        -> Binary or multiclass classification.

    Numeric target
        -> Binary classification when there are exactly two
           integer-like values.

        -> Potential multiclass classification when there are
           3-5 integer-like unique values.

        -> Regression when there are more than 5 unique values.

    Notes
    -----
    Numeric low-cardinality targets are inherently ambiguous.
    Therefore classification decisions carry warnings and lower
    confidence where appropriate.
    """

    _validate_inputs(df, target_column)

    target = df[target_column]

    n_rows = int(len(target))
    missing_count = int(target.isna().sum())

    if missing_count == n_rows:
        raise ValueError(
            f"Target column '{target_column}' contains only "
            "missing values."
        )

    clean_target = target.dropna()

    n_unique = int(clean_target.nunique())

    if n_unique < 2:
        raise ValueError(
            f"Target column '{target_column}' must contain at least "
            "two unique non-null values."
        )

    warnings: list[str] = []

    # =========================================================
    # 1. Boolean target
    # =========================================================

    if _is_boolean_target(clean_target):

        return _build_classification_result(
            target_column=target_column,
            target_type=TargetType.BINARY,
            n_rows=n_rows,
            n_unique=n_unique,
            missing_count=missing_count,
            target=clean_target,
            confidence=1.0,
            reasoning=(
                "The target is boolean, therefore the problem "
                "is binary classification."
            ),
            warnings=warnings,
        )

    # =========================================================
    # 2. Datetime target
    # =========================================================

    if _is_datetime_target(clean_target):

        raise ValueError(
            f"Target column '{target_column}' is datetime-like. "
            "Datetime targets are not supported by the current "
            "ML engine. Convert the target into a meaningful "
            "numeric or categorical prediction target first."
        )

    # =========================================================
    # 3. Categorical / string target
    # =========================================================

    if (
        pd.api.types.is_object_dtype(clean_target)
        or pd.api.types.is_string_dtype(clean_target)
        or _is_categorical_target(clean_target)
    ):

        if n_unique == 2:

            return _build_classification_result(
                target_column=target_column,
                target_type=TargetType.BINARY,
                n_rows=n_rows,
                n_unique=n_unique,
                missing_count=missing_count,
                target=clean_target,
                confidence=1.0,
                reasoning=(
                    "The target is categorical and contains "
                    "exactly two classes, therefore the problem "
                    "is binary classification."
                ),
                warnings=warnings,
            )

        return _build_classification_result(
            target_column=target_column,
            target_type=TargetType.MULTICLASS,
            n_rows=n_rows,
            n_unique=n_unique,
            missing_count=missing_count,
            target=clean_target,
            confidence=0.98,
            reasoning=(
                "The target is categorical and contains more "
                "than two classes, therefore the problem is "
                "multiclass classification."
            ),
            warnings=warnings,
        )

    # =========================================================
    # 4. Numeric target
    # =========================================================

    if _is_numeric_target(clean_target):

        numeric_values = clean_target.to_numpy(dtype=float)

        integer_like = bool(
            np.all(
                np.isfinite(numeric_values)
                & (
                    numeric_values
                    == np.floor(numeric_values)
                )
            )
        )

        # -----------------------------------------------------
        # Binary numeric target
        # -----------------------------------------------------

        if integer_like and n_unique == 2:

            return _build_classification_result(
                target_column=target_column,
                target_type=TargetType.BINARY,
                n_rows=n_rows,
                n_unique=n_unique,
                missing_count=missing_count,
                target=clean_target,
                confidence=0.95,
                reasoning=(
                    "The target is numeric, integer-like, and "
                    "contains exactly two unique values. "
                    "It is likely a binary classification target."
                ),
                warnings=warnings,
            )

        # -----------------------------------------------------
        # Potential multiclass numeric target
        # -----------------------------------------------------

        if integer_like and 3 <= n_unique <= 5:

            warnings.append(
                "The numeric target has low cardinality and may "
                "represent class labels. Confirm that classification "
                "is intended."
            )

            return _build_classification_result(
                target_column=target_column,
                target_type=TargetType.MULTICLASS,
                n_rows=n_rows,
                n_unique=n_unique,
                missing_count=missing_count,
                target=clean_target,
                confidence=0.80,
                reasoning=(
                    "The target is numeric, integer-like, and has "
                    "3-5 unique values. It may represent multiclass "
                    "labels, so classification is suggested with "
                    "user confirmation."
                ),
                warnings=warnings,
            )

        # -----------------------------------------------------
        # Regression
        # -----------------------------------------------------

        confidence = 0.95

        if integer_like and n_unique <= 10:
            warnings.append(
                "The numeric target is integer-like with relatively "
                "low cardinality. Regression is selected because the "
                "target has more than five unique values, but verify "
                "that the target is continuous rather than ordinal."
            )

            confidence = 0.75

        if missing_count > 0:
            warnings.append(
                f"Target contains {missing_count} missing values. "
                "These rows must be handled before training."
            )

        return TaskDetectionResult(
            target_column=target_column,
            task=MLTask.REGRESSION,
            target_type=TargetType.REGRESSION,
            n_rows=n_rows,
            n_unique=n_unique,
            missing_count=missing_count,
            class_distribution={},
            recommended_metrics=_detect_regression_metrics(),
            confidence=confidence,
            warnings=warnings,
            reasoning=(
                "The target is numeric with sufficient cardinality "
                "to represent a continuous outcome, therefore the "
                "problem is treated as regression."
            ),
        )

    # =========================================================
    # 5. Unsupported target
    # =========================================================

    raise ValueError(
        f"Unsupported target datatype for '{target_column}': "
        f"{target.dtype}"
    )


def detect_task_from_profile(
    df: pd.DataFrame,
    target_column: str | None,
) -> TaskDetectionResult | None:
    """
    Convenience wrapper for DataMindAI.

    Returns None when no target has been selected.
    """

    if target_column is None:
        return None

    return detect_task(df, target_column)