"""
Model persistence layer for DataMindAI.

Responsibilities
----------------
- Save the selected trained model.
- Save model metadata.
- Load a previously trained model.
- Keep model artifacts separate from the UI.
- Provide a stable artifact format for the prediction workspace.

This module does NOT:
- train models,
- evaluate models,
- select models,
- generate reports,
- call an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib

from .ml_pipeline import MLPipelineResult
from .trainer import TrainingResult


@dataclass
class ModelArtifact:
    """
    Represents a persisted DataMindAI model.

    Attributes
    ----------
    model_name:
        Internal model name.

    display_name:
        Human-readable model name.

    task:
        Classification or regression.

    target_column:
        Column predicted by the model.

    model_path:
        Location of the saved model.

    metadata_path:
        Location of the saved metadata.

    primary_metric:
        Metric used to select the model.

    primary_score:
        Score achieved by the selected model.

    feature_names:
        Original input feature names.

    transformed_feature_names:
        Features after preprocessing.

    metadata:
        Additional metadata.
    """

    model_name: str
    display_name: str

    task: str
    target_column: str

    model_path: str
    metadata_path: str

    primary_metric: str
    primary_score: float

    feature_names: list[str] = field(
        default_factory=list
    )

    transformed_feature_names: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        """Return serializable artifact metadata."""

        return {
            "model_name": self.model_name,
            "display_name": self.display_name,
            "task": self.task,
            "target_column": self.target_column,
            "model_path": self.model_path,
            "metadata_path": self.metadata_path,
            "primary_metric": self.primary_metric,
            "primary_score": self.primary_score,
            "feature_names": self.feature_names,
            "transformed_feature_names": (
                self.transformed_feature_names
            ),
            "metadata": self.metadata,
        }


def _validate_training_result(
    training_result: TrainingResult,
) -> None:
    """Validate a training result."""

    if not isinstance(
        training_result,
        TrainingResult,
    ):
        raise TypeError(
            "training_result must be a TrainingResult."
        )


def _validate_pipeline_result(
    pipeline_result: MLPipelineResult,
) -> None:
    """Validate a complete ML pipeline result."""

    if not isinstance(
        pipeline_result,
        MLPipelineResult,
    ):
        raise TypeError(
            "pipeline_result must be an MLPipelineResult."
        )


def _ensure_directory(
    directory: str | Path,
) -> Path:
    """Create an artifact directory if necessary."""

    path = Path(directory)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def save_model(
    training_result: TrainingResult,
    *,
    model_directory: str | Path,
    metadata: dict[str, Any] | None = None,
) -> ModelArtifact:
    """
    Save one trained model pipeline.

    The complete sklearn Pipeline is saved, meaning:

        preprocessing
             +
          trained model

    are persisted together.

    This is critical because prediction must use exactly the
    same preprocessing that was fitted during training.
    """

    _validate_training_result(
        training_result
    )

    directory = _ensure_directory(
        model_directory
    )

    model_filename = (
        f"{training_result.model_name}.joblib"
    )

    metadata_filename = (
        f"{training_result.model_name}_metadata.joblib"
    )

    model_path = (
        directory / model_filename
    )

    metadata_path = (
        directory / metadata_filename
    )

    # ---------------------------------------------------------
    # Save complete fitted pipeline
    # ---------------------------------------------------------

    joblib.dump(
        training_result.pipeline,
        model_path,
    )

    # ---------------------------------------------------------
    # Build metadata
    # ---------------------------------------------------------

    artifact_metadata = {
        "model_name": training_result.model_name,
        "display_name": training_result.display_name,
        "task": training_result.task.value,
        "target_column": training_result.target_column,
        "train_rows": training_result.train_rows,
        "test_rows": training_result.test_rows,
        "train_size": training_result.train_size,
        "test_size": training_result.test_size,
        "feature_names": training_result.feature_names,
        "transformed_feature_names": (
            training_result.transformed_feature_names
        ),
        "warnings": training_result.warnings,
    }

    if metadata:
        artifact_metadata.update(
            metadata
        )

    joblib.dump(
        artifact_metadata,
        metadata_path,
    )

    return ModelArtifact(
        model_name=training_result.model_name,
        display_name=training_result.display_name,
        task=training_result.task.value,
        target_column=training_result.target_column,
        model_path=str(model_path),
        metadata_path=str(metadata_path),
        primary_metric=(
            str(
                artifact_metadata.get(
                    "primary_metric",
                    "",
                )
            )
        ),
        primary_score=float(
            artifact_metadata.get(
                "primary_score",
                0.0,
            )
        ),
        feature_names=(
            training_result.feature_names
        ),
        transformed_feature_names=(
            training_result
            .transformed_feature_names
        ),
        metadata=artifact_metadata,
    )


def save_best_model(
    pipeline_result: MLPipelineResult,
    *,
    model_directory: str | Path,
) -> ModelArtifact:
    """
    Save the best model selected by the ML pipeline.

    This is the main persistence function DataMindAI will use.
    """

    _validate_pipeline_result(
        pipeline_result
    )

    best_model_name = (
        pipeline_result
        .best_model_name
    )

    training_result = (
        pipeline_result
        .training_results[
            best_model_name
        ]
    )

    selection = (
        pipeline_result
        .model_selection
    )

    metadata = {
        "primary_metric": (
            selection.primary_metric
        ),
        "primary_score": (
            selection.primary_score
        ),
        "selection_reason": (
            selection.selection_reason
        ),
        "leaderboard": (
            pipeline_result
            .leaderboard
            .to_dict(
                orient="records"
            )
        ),
        "comparison_table": (
            pipeline_result
            .comparison_table
            .to_dict(
                orient="records"
            )
        ),
    }

    return save_model(
        training_result,
        model_directory=model_directory,
        metadata=metadata,
    )


def load_model(
    model_path: str | Path,
):
    """
    Load a persisted trained model pipeline.
    """

    path = Path(
        model_path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Model artifact not found: {path}"
        )

    return joblib.load(
        path
    )


def load_model_metadata(
    metadata_path: str | Path,
) -> dict[str, Any]:
    """
    Load metadata associated with a saved model.
    """

    path = Path(
        metadata_path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Model metadata not found: {path}"
        )

    metadata = joblib.load(
        path
    )

    if not isinstance(
        metadata,
        dict,
    ):

        raise ValueError(
            "Invalid model metadata format."
        )

    return metadata


def load_artifact(
    model_path: str | Path,
    metadata_path: str | Path,
) -> ModelArtifact:
    """
    Load a model and reconstruct its artifact metadata.

    Returns
    -------
    ModelArtifact
    """

    metadata = load_model_metadata(
        metadata_path
    )

    # Make sure the actual model can be loaded.
    load_model(
        model_path
    )

    return ModelArtifact(
        model_name=metadata.get(
            "model_name",
            Path(model_path).stem,
        ),
        display_name=metadata.get(
            "display_name",
            metadata.get(
                "model_name",
                "Unknown Model",
            ),
        ),
        task=metadata.get(
            "task",
            "",
        ),
        target_column=metadata.get(
            "target_column",
            "",
        ),
        model_path=str(
            model_path
        ),
        metadata_path=str(
            metadata_path
        ),
        primary_metric=metadata.get(
            "primary_metric",
            "",
        ),
        primary_score=float(
            metadata.get(
                "primary_score",
                0.0,
            )
        ),
        feature_names=metadata.get(
            "feature_names",
            [],
        ),
        transformed_feature_names=metadata.get(
            "transformed_feature_names",
            [],
        ),
        metadata=metadata,
    )