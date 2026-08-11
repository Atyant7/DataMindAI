"""
Model explainability engine for DataMindAI.

Responsibilities
----------------
- Calculate global feature importance.
- Support tree-based models through feature_importances_.
- Support linear models through coefficients.
- Fall back to permutation importance when required.
- Map transformed features back to original dataset features.
- Return standardized explainability results.

This module does NOT:
- train models,
- select models,
- generate predictions,
- render Streamlit UI,
- call an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


# ============================================================
# RESULT OBJECTS
# ============================================================

@dataclass
class FeatureImportance:
    """
    Represents the importance of one original dataset feature.
    """

    feature: str

    importance: float

    rank: int

    direction: str = "positive"

    method: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "importance": self.importance,
            "rank": self.rank,
            "direction": self.direction,
            "method": self.method,
        }


@dataclass
class ExplainabilityResult:
    """
    Complete model explainability result.
    """

    model_name: str

    display_name: str

    task: str

    method: str

    feature_importance: list[
        FeatureImportance
    ] = field(
        default_factory=list
    )

    transformed_feature_importance: pd.DataFrame = (
        field(
            default_factory=pd.DataFrame
        )
    )

    warnings: list[str] = field(
        default_factory=list
    )

    @property
    def top_features(
        self,
    ) -> list[FeatureImportance]:
        """
        Return the top features.
        """

        return self.feature_importance

    def to_dataframe(
        self,
    ) -> pd.DataFrame:
        """
        Convert original-feature importance into
        a DataFrame suitable for Streamlit or reports.
        """

        return pd.DataFrame(
            [
                item.to_dict()
                for item in self.feature_importance
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "display_name": self.display_name,
            "task": self.task,
            "method": self.method,
            "feature_importance": [
                item.to_dict()
                for item in self.feature_importance
            ],
            "warnings": self.warnings,
        }


# ============================================================
# PIPELINE / ESTIMATOR HELPERS
# ============================================================

def _get_final_estimator(
    model: Any,
) -> Any:
    """
    Retrieve the final estimator from a fitted pipeline.

    If the supplied object is already an estimator,
    return it directly.
    """

    if hasattr(
        model,
        "named_steps",
    ):

        steps = list(
            model.named_steps.values()
        )

        for step in reversed(steps):

            if hasattr(
                step,
                "feature_importances_",
            ) or hasattr(
                step,
                "coef_",
            ):

                return step

        if steps:

            return steps[-1]

    return model


def _get_preprocessor(
    model: Any,
) -> Any | None:
    """
    Try to find the preprocessing transformer
    inside a fitted pipeline.
    """

    if not hasattr(
        model,
        "named_steps",
    ):
        return None

    for name, step in (
        model.named_steps.items()
    ):

        name_lower = str(
            name
        ).lower()

        if (
            "preprocess" in name_lower
            or "transform" in name_lower
        ):

            return step

    return None


# ============================================================
# FEATURE NAME EXTRACTION
# ============================================================

def _get_transformed_feature_names(
    model: Any,
    original_features: list[str],
) -> list[str]:
    """
    Retrieve feature names after preprocessing.

    Falls back to original feature names when the
    preprocessing pipeline cannot expose names.
    """

    preprocessor = _get_preprocessor(
        model
    )

    if preprocessor is None:

        return list(
            original_features
        )

    if hasattr(
        preprocessor,
        "get_feature_names_out",
    ):

        try:

            names = (
                preprocessor
                .get_feature_names_out(
                    original_features
                )
            )

            return [
                str(name)
                for name in names
            ]

        except Exception:
            pass

    return list(
        original_features
    )


# ============================================================
# FEATURE MAPPING
# ============================================================

def _match_original_feature(
    transformed_name: str,
    original_features: list[str],
) -> str:
    """
    Map a transformed feature back to its original
    dataset feature.

    Examples
    --------
    num__age
        -> age

    cat__city_Delhi
        -> city

    onehot__gender_Male
        -> gender
    """

    name = str(
        transformed_name
    )

    # Remove common sklearn transformer prefixes.
    if "__" in name:

        name = name.split(
            "__",
            1,
        )[1]

    # Exact match.
    if name in original_features:

        return name

    # Longest feature name first prevents
    # partial collisions.
    sorted_features = sorted(
        original_features,
        key=len,
        reverse=True,
    )

    for feature in sorted_features:

        if (
            name == feature
            or name.startswith(
                f"{feature}_"
            )
        ):

            return feature

    return name


def _aggregate_importance(
    transformed_names: list[str],
    importances: np.ndarray,
    original_features: list[str],
) -> pd.DataFrame:
    """
    Aggregate transformed feature importance back to
    original dataset columns.
    """

    rows = []

    for name, importance in zip(
        transformed_names,
        importances,
    ):

        original_feature = (
            _match_original_feature(
                name,
                original_features,
            )
        )

        rows.append(
            {
                "feature": original_feature,
                "transformed_feature": name,
                "importance": float(
                    abs(importance)
                ),
            }
        )

    transformed_df = pd.DataFrame(
        rows
    )

    if transformed_df.empty:

        return transformed_df

    transformed_df = (
        transformed_df
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    return transformed_df


# ============================================================
# BUILT-IN TREE IMPORTANCE
# ============================================================

def _tree_feature_importance(
    model: Any,
    original_features: list[str],
) -> tuple[
    pd.DataFrame | None,
    str | None,
]:
    """
    Extract feature importance from tree-based models.
    """

    estimator = _get_final_estimator(
        model
    )

    if not hasattr(
        estimator,
        "feature_importances_",
    ):

        return (
            None,
            None,
        )

    importances = np.asarray(
        estimator.feature_importances_,
        dtype=float,
    )

    transformed_names = (
        _get_transformed_feature_names(
            model,
            original_features,
        )
    )

    if len(importances) != len(
        transformed_names
    ):

        return (
            None,
            None,
        )

    transformed_df = (
        _aggregate_importance(
            transformed_names,
            importances,
            original_features,
        )
    )

    return (
        transformed_df,
        "built_in_feature_importance",
    )


# ============================================================
# LINEAR MODEL IMPORTANCE
# ============================================================

def _linear_feature_importance(
    model: Any,
    original_features: list[str],
) -> tuple[
    pd.DataFrame | None,
    str | None,
]:
    """
    Extract coefficient-based importance from linear models.
    """

    estimator = _get_final_estimator(
        model
    )

    if not hasattr(
        estimator,
        "coef_",
    ):

        return (
            None,
            None,
        )

    coefficients = np.asarray(
        estimator.coef_,
        dtype=float,
    )

    # Binary / multiclass handling.
    if coefficients.ndim == 2:

        coefficients = np.mean(
            np.abs(
                coefficients
            ),
            axis=0,
        )

    else:

        coefficients = np.abs(
            coefficients
        )

    transformed_names = (
        _get_transformed_feature_names(
            model,
            original_features,
        )
    )

    if len(coefficients) != len(
        transformed_names
    ):

        return (
            None,
            None,
        )

    transformed_df = (
        _aggregate_importance(
            transformed_names,
            coefficients,
            original_features,
        )
    )

    return (
        transformed_df,
        "coefficient_importance",
    )


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

def _permutation_feature_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    original_features: list[str],
    task: str,
) -> tuple[
    pd.DataFrame | None,
    str | None,
    list[str],
]:
    """
    Calculate model-agnostic permutation importance.

    This is used as a fallback when the estimator does not
    expose native feature importance.
    """

    warnings: list[str] = []

    if X.empty:

        warnings.append(
            "Permutation importance cannot be calculated "
            "because the feature dataset is empty."
        )

        return (
            None,
            None,
            warnings,
        )

    try:

        if task == "classification":

            scoring = "f1_weighted"

        elif task == "regression":

            scoring = "neg_root_mean_squared_error"

        else:

            scoring = None

        result = permutation_importance(
            model,
            X,
            y,
            scoring=scoring,
            n_repeats=5,
            random_state=42,
            n_jobs=-1,
        )

    except Exception as exc:

        warnings.append(
            "Permutation importance failed: "
            f"{exc}"
        )

        return (
            None,
            None,
            warnings,
        )

    importances = np.asarray(
        result.importances_mean,
        dtype=float,
    )

    if len(importances) != len(
        original_features
    ):

        warnings.append(
            "Permutation importance output does not "
            "match the number of original features."
        )

        return (
            None,
            None,
            warnings,
        )

    transformed_df = pd.DataFrame(
        {
            "feature": original_features,
            "transformed_feature": original_features,
            "importance": np.abs(
                importances
            ),
        }
    )

    transformed_df = (
        transformed_df
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    return (
        transformed_df,
        "permutation_importance",
        warnings,
    )


# ============================================================
# RESULT CONSTRUCTION
# ============================================================

def _build_feature_importance(
    transformed_df: pd.DataFrame,
    method: str,
) -> list[FeatureImportance]:
    """
    Convert aggregated feature importance into
    FeatureImportance objects.
    """

    if transformed_df.empty:

        return []

    grouped = (
        transformed_df
        .groupby(
            "feature",
            as_index=False,
        )["importance"]
        .sum()
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    total = float(
        grouped["importance"]
        .sum()
    )

    if total > 0:

        grouped["importance"] = (
            grouped["importance"]
            / total
        )

    results = []

    for index, row in (
        grouped.iterrows()
    ):

        results.append(
            FeatureImportance(
                feature=str(
                    row["feature"]
                ),
                importance=float(
                    row["importance"]
                ),
                rank=index + 1,
                method=method,
            )
        )

    return results


# ============================================================
# PUBLIC API
# ============================================================

def explain_model(
    model: Any,
    *,
    model_name: str,
    display_name: str,
    task: str,
    feature_names: list[str],
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
) -> ExplainabilityResult:
    """
    Explain a fitted model.

    Strategy
    --------
    1. Tree models:
       native feature_importances_

    2. Linear models:
       absolute coefficients

    3. Other models:
       permutation importance

    Parameters
    ----------
    model:
        Fitted model or fitted sklearn Pipeline.

    model_name:
        Internal model name.

    display_name:
        Human-readable model name.

    task:
        "classification" or "regression".

    feature_names:
        Original dataset feature names.

    X, y:
        Required only when permutation importance is needed.
    """

    if model is None:

        raise ValueError(
            "A fitted model is required."
        )

    if not feature_names:

        raise ValueError(
            "feature_names cannot be empty."
        )

    warnings: list[str] = []

    # ---------------------------------------------------------
    # Tree models
    # ---------------------------------------------------------

    (
        transformed_df,
        method,
    ) = _tree_feature_importance(
        model,
        feature_names,
    )

    # ---------------------------------------------------------
    # Linear models
    # ---------------------------------------------------------

    if transformed_df is None:

        (
            transformed_df,
            method,
        ) = _linear_feature_importance(
            model,
            feature_names,
        )

    # ---------------------------------------------------------
    # Permutation fallback
    # ---------------------------------------------------------

    if transformed_df is None:

        if X is None or y is None:

            warnings.append(
                "The selected model does not expose "
                "native feature importance. "
                "Provide X and y to calculate permutation importance."
            )

            return ExplainabilityResult(
                model_name=model_name,
                display_name=display_name,
                task=task,
                method="unavailable",
                warnings=warnings,
            )

        (
            transformed_df,
            method,
            permutation_warnings,
        ) = _permutation_feature_importance(
            model,
            X,
            y,
            feature_names,
            task,
        )

        warnings.extend(
            permutation_warnings
        )

    if transformed_df is None:

        return ExplainabilityResult(
            model_name=model_name,
            display_name=display_name,
            task=task,
            method="unavailable",
            warnings=warnings,
        )

    # ---------------------------------------------------------
    # Build final result
    # ---------------------------------------------------------

    feature_importance = (
        _build_feature_importance(
            transformed_df,
            method,
        )
    )

    return ExplainabilityResult(
        model_name=model_name,
        display_name=display_name,
        task=task,
        method=method,
        feature_importance=feature_importance,
        transformed_feature_importance=(
            transformed_df
        ),
        warnings=warnings,
    )


def explain_training_result(
    training_result: Any,
    *,
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
) -> ExplainabilityResult:
    """
    Explain a TrainingResult directly.

    This convenience function keeps the frontend from
    needing to understand TrainingResult internals.
    """

    if training_result is None:

        raise ValueError(
            "training_result is required."
        )

    return explain_model(
        model=training_result.pipeline,
        model_name=training_result.model_name,
        display_name=training_result.display_name,
        task=training_result.task.value,
        feature_names=training_result.feature_names,
        X=X,
        y=y,
    )