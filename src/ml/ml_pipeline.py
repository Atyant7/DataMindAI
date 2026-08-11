"""
High-level ML pipeline for DataMindAI.

This module connects the deterministic ML components:

    Task Detection
          ↓
    Model Training
          ↓
    Model Evaluation
          ↓
    Model Selection

The frontend and future LangGraph layer should call this
high-level pipeline instead of directly coordinating individual
ML components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .evaluator import (
    EvaluationResult,
    create_comparison_table,
    evaluate_models,
)
from .model_selector import (
    ModelSelectionResult,
    create_leaderboard,
    select_best_model,
)
from .task_detector import (
    MLTask,
    TaskDetectionResult,
    detect_task,
)
from .trainer import (
    TrainingConfig,
    TrainingResult,
    train_models,
)


@dataclass
class MLPipelineResult:
    """
    Complete result of the DataMindAI ML workflow.
    """

    target_column: str

    task_detection: TaskDetectionResult

    training_results: dict[
        str,
        TrainingResult,
    ]

    evaluation_results: dict[
        str,
        EvaluationResult,
    ]

    model_selection: ModelSelectionResult

    comparison_table: pd.DataFrame

    leaderboard: pd.DataFrame

    warnings: list[str] = field(
        default_factory=list
    )

    @property
    def task(self) -> MLTask:
        """Return the detected ML task."""

        return self.task_detection.task

    @property
    def best_model_name(self) -> str:
        """Return the internal name of the selected model."""

        return (
            self.model_selection
            .selected_model_name
        )

    @property
    def best_model_display_name(self) -> str:
        """Return the display name of the selected model."""

        return (
            self.model_selection
            .selected_display_name
        )

    @property
    def best_training_result(self) -> TrainingResult:
        """Return the training result of the selected model."""

        return self.training_results[
            self.best_model_name
        ]

    @property
    def best_evaluation_result(self) -> EvaluationResult:
        """Return the evaluation result of the selected model."""

        return self.evaluation_results[
            self.best_model_name
        ]

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the complete pipeline result into metadata
        suitable for reports or the frontend.
        """

        return {
            "target_column": self.target_column,
            "task": self.task.value,
            "task_detection": (
                self.task_detection.to_dict()
                if hasattr(
                    self.task_detection,
                    "to_dict",
                )
                else {
                    "task": self.task.value,
                }
            ),
            "models": {
                name: result.to_dict()
                for name, result
                in self.training_results.items()
            },
            "evaluations": {
                name: result.to_dict()
                for name, result
                in self.evaluation_results.items()
            },
            "model_selection": (
                self.model_selection.to_dict()
            ),
            "warnings": self.warnings,
        }


def _validate_dataset(
    df: pd.DataFrame,
) -> None:
    """Validate the input dataset."""

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "The dataset is empty."
        )


def _validate_target(
    df: pd.DataFrame,
    target_column: str,
) -> None:
    """Validate the target column."""

    if not isinstance(
        target_column,
        str,
    ):
        raise TypeError(
            "target_column must be a string."
        )

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            "does not exist in the dataset."
        )


def run_ml_pipeline(
    df: pd.DataFrame,
    target_column: str,
    *,
    model_names: list[str] | None = None,
    training_config: TrainingConfig | None = None,
    selection_metric: str | None = None,
) -> MLPipelineResult:
    """
    Execute the complete DataMindAI ML workflow.

    Workflow
    --------
    1. Validate dataset.
    2. Detect ML task.
    3. Separate features and target.
    4. Train candidate models.
    5. Evaluate candidate models.
    6. Select the best model.
    7. Create comparison/leaderboard tables.
    8. Return everything in one structured result.

    Parameters
    ----------
    df:
        Complete dataset.

    target_column:
        Column to predict.

    model_names:
        Optional list of candidate models.

        If None, all models supported for the detected
        task are trained.

    training_config:
        Optional training configuration.

    selection_metric:
        Optional metric used for selecting the best model.

    Returns
    -------
    MLPipelineResult
    """

    _validate_dataset(df)

    _validate_target(
        df,
        target_column,
    )

    # ---------------------------------------------------------
    # 1. Detect task
    # ---------------------------------------------------------

    task_detection = detect_task(
        df,
        target_column,
    )

    task = task_detection.task

    if task not in {
        MLTask.CLASSIFICATION,
        MLTask.REGRESSION,
    }:
        raise ValueError(
            f"Unsupported ML task: {task}"
        )

    # ---------------------------------------------------------
    # 2. Separate features and target
    # ---------------------------------------------------------

    X = df.drop(
        columns=[target_column]
    )

    y = df[target_column]

    # ---------------------------------------------------------
    # 3. Train models
    # ---------------------------------------------------------

    training_results = train_models(
        X=X,
        y=y,
        task=task,
        target_column=target_column,
        model_names=model_names,
        config=training_config,
    )

    # ---------------------------------------------------------
    # 4. Evaluate models
    # ---------------------------------------------------------

    evaluation_results = evaluate_models(
        training_results
    )

    # ---------------------------------------------------------
    # 5. Select best model
    # ---------------------------------------------------------

    model_selection = select_best_model(
        evaluation_results,
        metric=selection_metric,
    )

    # ---------------------------------------------------------
    # 6. Create comparison tables
    # ---------------------------------------------------------

    comparison_table = create_comparison_table(
        evaluation_results
    )

    leaderboard = create_leaderboard(
        model_selection
    )

    # ---------------------------------------------------------
    # 7. Collect warnings
    # ---------------------------------------------------------

    warnings: list[str] = []

    for result in training_results.values():
        warnings.extend(
            result.warnings
        )

    for result in evaluation_results.values():
        warnings.extend(
            result.warnings
        )

    warnings.extend(
        model_selection.warnings
    )

    # Remove duplicate warnings while
    # preserving order.
    warnings = list(
        dict.fromkeys(warnings)
    )

    return MLPipelineResult(
        target_column=target_column,
        task_detection=task_detection,
        training_results=training_results,
        evaluation_results=evaluation_results,
        model_selection=model_selection,
        comparison_table=comparison_table,
        leaderboard=leaderboard,
        warnings=warnings,
    )