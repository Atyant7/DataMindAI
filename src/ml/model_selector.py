"""
Model selection engine for DataMindAI.

Responsibilities
----------------
- Compare evaluated models.
- Select the best model using task-aware metrics.
- Rank all candidate models.
- Handle higher-is-better and lower-is-better metrics.
- Provide a model leaderboard.
- Provide deterministic selection reasoning.

This module does NOT:
- train models,
- preprocess data,
- call an LLM,
- perform hyperparameter optimization,
- generate reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .evaluator import EvaluationResult
from .task_detector import MLTask


# ---------------------------------------------------------------------
# Metric configuration
# ---------------------------------------------------------------------

HIGHER_IS_BETTER_METRICS = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
    "r2",
}

LOWER_IS_BETTER_METRICS = {
    "mae",
    "mse",
    "rmse",
}


CLASSIFICATION_DEFAULT_METRIC = "f1"

REGRESSION_DEFAULT_METRIC = "rmse"


# ---------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------

@dataclass
class ModelRanking:
    """
    Ranking information for one model.

    Attributes
    ----------
    rank:
        Position of the model in the leaderboard.

    model_name:
        Internal model name.

    display_name:
        Human-readable model name.

    primary_metric:
        Metric used for ranking.

    primary_score:
        Score used for ranking.

    normalized_score:
        Score transformed so higher is always better.

    metrics:
        Complete metric dictionary.

    selection_reason:
        Explanation of why the model received its ranking.
    """

    rank: int
    model_name: str
    display_name: str

    primary_metric: str
    primary_score: float
    normalized_score: float

    metrics: dict[str, float] = field(
        default_factory=dict
    )

    selection_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert ranking information to a dictionary."""

        return {
            "rank": self.rank,
            "model_name": self.model_name,
            "display_name": self.display_name,
            "primary_metric": self.primary_metric,
            "primary_score": self.primary_score,
            "normalized_score": self.normalized_score,
            "metrics": self.metrics,
            "selection_reason": self.selection_reason,
        }


@dataclass
class ModelSelectionResult:
    """
    Final model-selection result.

    Attributes
    ----------
    task:
        ML task.

    selected_model_name:
        Internal name of the winning model.

    selected_display_name:
        Human-readable name of the winning model.

    primary_metric:
        Metric used for model selection.

    primary_score:
        Winning model's primary score.

    leaderboard:
        Ranked list of all evaluated models.

    selection_reason:
        Human-readable explanation of the selection.

    warnings:
        Selection warnings.
    """

    task: MLTask

    selected_model_name: str
    selected_display_name: str

    primary_metric: str
    primary_score: float

    leaderboard: list[ModelRanking] = field(
        default_factory=list
    )

    selection_reason: str = ""

    warnings: list[str] = field(
        default_factory=list
    )

    @property
    def selected_rank(self) -> int:
        """Return the rank of the selected model."""

        return 1

    @property
    def runner_up(self) -> ModelRanking | None:
        """Return the second-ranked model when available."""

        if len(self.leaderboard) < 2:
            return None

        return self.leaderboard[1]

    def to_dict(self) -> dict[str, Any]:
        """Convert the selection result into a dictionary."""

        return {
            "task": self.task.value,
            "selected_model_name": (
                self.selected_model_name
            ),
            "selected_display_name": (
                self.selected_display_name
            ),
            "primary_metric": self.primary_metric,
            "primary_score": self.primary_score,
            "leaderboard": [
                item.to_dict()
                for item in self.leaderboard
            ],
            "selection_reason": (
                self.selection_reason
            ),
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------

def _validate_metric(
    metric: str,
) -> None:
    """Validate that a metric is supported."""

    if not isinstance(metric, str):
        raise TypeError(
            "metric must be a string."
        )

    metric = metric.strip().lower()

    if (
        metric not in HIGHER_IS_BETTER_METRICS
        and metric not in LOWER_IS_BETTER_METRICS
    ):
        raise ValueError(
            f"Unsupported selection metric: '{metric}'."
        )


def metric_direction(
    metric: str,
) -> str:
    """
    Return whether higher or lower values are better.

    Returns
    -------
    str
        Either "higher" or "lower".
    """

    metric = metric.strip().lower()

    _validate_metric(metric)

    if metric in HIGHER_IS_BETTER_METRICS:
        return "higher"

    return "lower"


def normalize_score(
    score: float,
    metric: str,
) -> float:
    """
    Normalize a metric so higher is always better.

    For metrics where higher is better:
        normalized = score

    For metrics where lower is better:
        normalized = -score

    This is primarily useful for internal ranking.
    """

    direction = metric_direction(
        metric
    )

    score = float(score)

    if direction == "higher":
        return score

    return -score


def default_selection_metric(
    task: MLTask,
) -> str:
    """
    Return the default metric for model selection.
    """

    if task == MLTask.CLASSIFICATION:
        return CLASSIFICATION_DEFAULT_METRIC

    if task == MLTask.REGRESSION:
        return REGRESSION_DEFAULT_METRIC

    raise ValueError(
        f"Unsupported ML task: {task}"
    )


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def _validate_evaluation_results(
    evaluation_results: dict[
        str,
        EvaluationResult,
    ],
) -> None:
    """Validate evaluation results."""

    if not isinstance(
        evaluation_results,
        dict,
    ):
        raise TypeError(
            "evaluation_results must be a dictionary."
        )

    if not evaluation_results:
        raise ValueError(
            "evaluation_results cannot be empty."
        )

    for model_name, result in (
        evaluation_results.items()
    ):

        if not isinstance(
            result,
            EvaluationResult,
        ):
            raise TypeError(
                f"Evaluation result for '{model_name}' "
                "must be an EvaluationResult."
            )


def _validate_same_task(
    evaluation_results: dict[
        str,
        EvaluationResult,
    ],
) -> MLTask:
    """
    Ensure all evaluated models belong to the same task.
    """

    tasks = {
        result.task
        for result in evaluation_results.values()
    }

    if len(tasks) != 1:
        raise ValueError(
            "All evaluation results must belong to "
            "the same ML task."
        )

    return next(iter(tasks))


# ---------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------

def _get_metric_score(
    result: EvaluationResult,
    metric: str,
) -> float:
    """Get a specific metric from an evaluation result."""

    metric = metric.strip().lower()

    if metric not in result.metrics:
        raise ValueError(
            f"Metric '{metric}' is not available for "
            f"model '{result.model_name}'."
        )

    return float(
        result.metrics[metric]
    )


def _build_ranking_reason(
    result: EvaluationResult,
    metric: str,
    rank: int,
    total_models: int,
) -> str:
    """Generate deterministic ranking reasoning."""

    score = _get_metric_score(
        result,
        metric,
    )

    direction = metric_direction(
        metric
    )

    if rank == 1:

        if direction == "higher":

            return (
                f"{result.display_name} ranked first because "
                f"it achieved the highest {metric} score of "
                f"{score:.4f} among {total_models} evaluated models."
            )

        return (
            f"{result.display_name} ranked first because "
            f"it achieved the lowest {metric} score of "
            f"{score:.4f} among {total_models} evaluated models."
        )

    if direction == "higher":

        return (
            f"{result.display_name} achieved a {metric} score "
            f"of {score:.4f}, which placed it at rank {rank}."
        )

    return (
        f"{result.display_name} achieved a {metric} score "
        f"of {score:.4f}, which placed it at rank {rank}."
    )


# ---------------------------------------------------------------------
# Main selection
# ---------------------------------------------------------------------

def select_best_model(
    evaluation_results: dict[
        str,
        EvaluationResult,
    ],
    *,
    metric: str | None = None,
) -> ModelSelectionResult:
    """
    Select the best model from evaluated models.

    Parameters
    ----------
    evaluation_results:
        Results produced by evaluate_models().

    metric:
        Optional metric to use for selection.

        If omitted:
            classification -> F1
            regression -> RMSE

    Returns
    -------
    ModelSelectionResult
        Ranked leaderboard and selected model.

    Selection rule
    --------------
    Classification:
        Higher F1 is better.

    Regression:
        Lower RMSE is better.
    """

    _validate_evaluation_results(
        evaluation_results
    )

    task = _validate_same_task(
        evaluation_results
    )

    # ---------------------------------------------------------
    # Select metric
    # ---------------------------------------------------------

    if metric is None:

        metric = default_selection_metric(
            task
        )

    metric = metric.strip().lower()

    _validate_metric(metric)

    # ---------------------------------------------------------
    # Validate metric compatibility
    # ---------------------------------------------------------

    if task == MLTask.CLASSIFICATION:

        if metric in LOWER_IS_BETTER_METRICS:
            raise ValueError(
                f"Metric '{metric}' is not appropriate "
                "as the default primary classification metric."
            )

    elif task == MLTask.REGRESSION:

        if metric in {
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "average_precision",
        }:
            raise ValueError(
                f"Metric '{metric}' is not appropriate "
                "for regression."
            )

    # ---------------------------------------------------------
    # Validate metric availability
    # ---------------------------------------------------------

    missing_models: list[str] = []

    for model_name, result in (
        evaluation_results.items()
    ):

        if metric not in result.metrics:

            missing_models.append(
                model_name
            )

    if missing_models:

        raise ValueError(
            f"Metric '{metric}' is missing for model(s): "
            f"{', '.join(missing_models)}"
        )

    # ---------------------------------------------------------
    # Create sortable ranking
    # ---------------------------------------------------------

    sortable_results = []

    for result in evaluation_results.values():

        score = _get_metric_score(
            result,
            metric,
        )

        normalized = normalize_score(
            score,
            metric,
        )

        sortable_results.append(
            (
                result,
                score,
                normalized,
            )
        )

    sortable_results.sort(
        key=lambda item: item[2],
        reverse=True,
    )

    # ---------------------------------------------------------
    # Build leaderboard
    # ---------------------------------------------------------

    leaderboard: list[ModelRanking] = []

    total_models = len(
        sortable_results
    )

    for index, (
        result,
        score,
        normalized,
    ) in enumerate(
        sortable_results,
        start=1,
    ):

        leaderboard.append(
            ModelRanking(
                rank=index,
                model_name=result.model_name,
                display_name=result.display_name,
                primary_metric=metric,
                primary_score=score,
                normalized_score=normalized,
                metrics=dict(
                    result.metrics
                ),
                selection_reason=(
                    _build_ranking_reason(
                        result=result,
                        metric=metric,
                        rank=index,
                        total_models=total_models,
                    )
                ),
            )
        )

    # ---------------------------------------------------------
    # Winner
    # ---------------------------------------------------------

    winner = leaderboard[0]

    warnings: list[str] = []

    # Check whether the winner is only marginally better.
    if len(leaderboard) > 1:

        second = leaderboard[1]

        difference = abs(
            winner.normalized_score
            - second.normalized_score
        )

        if difference < 0.01:

            warnings.append(
                "The difference between the top two models "
                "is less than 0.01 on the selected metric. "
                "The models may have very similar performance."
            )

    # ---------------------------------------------------------
    # Selection reasoning
    # ---------------------------------------------------------

    direction = metric_direction(
        metric
    )

    if direction == "higher":

        selection_reason = (
            f"{winner.display_name} was selected as the best "
            f"model because it achieved the highest {metric} "
            f"score ({winner.primary_score:.4f}) among "
            f"{total_models} evaluated models."
        )

    else:

        selection_reason = (
            f"{winner.display_name} was selected as the best "
            f"model because it achieved the lowest {metric} "
            f"score ({winner.primary_score:.4f}) among "
            f"{total_models} evaluated models."
        )

    return ModelSelectionResult(
        task=task,
        selected_model_name=winner.model_name,
        selected_display_name=winner.display_name,
        primary_metric=metric,
        primary_score=winner.primary_score,
        leaderboard=leaderboard,
        selection_reason=selection_reason,
        warnings=warnings,
    )


# ---------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------

def create_leaderboard(
    selection_result: ModelSelectionResult,
) -> pd.DataFrame:
    """
    Convert a ModelSelectionResult into a DataFrame.

    This DataFrame can later be directly consumed by:
    - Streamlit
    - report generation
    - experiment tracking
    - LLM explanation layer
    """

    if not isinstance(
        selection_result,
        ModelSelectionResult,
    ):
        raise TypeError(
            "selection_result must be a "
            "ModelSelectionResult."
        )

    rows: list[dict[str, Any]] = []

    for ranking in (
        selection_result.leaderboard
    ):

        row = {
            "rank": ranking.rank,
            "model": ranking.display_name,
            "model_name": ranking.model_name,
            "primary_metric": (
                ranking.primary_metric
            ),
            "primary_score": (
                ranking.primary_score
            ),
        }

        row.update(
            ranking.metrics
        )

        rows.append(row)

    return pd.DataFrame(rows)


def get_selected_model(
    selection_result: ModelSelectionResult,
) -> str:
    """Return the internal name of the selected model."""

    if not isinstance(
        selection_result,
        ModelSelectionResult,
    ):
        raise TypeError(
            "selection_result must be a "
            "ModelSelectionResult."
        )

    return selection_result.selected_model_name