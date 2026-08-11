"""
Model factory for DataMindAI.

Responsibilities
----------------
- Create supported machine-learning models.
- Keep model construction separate from training.
- Return models configured for the detected ML task.
- Provide model metadata.
- Keep model-specific configuration centralized.

This module does NOT:
- train models,
- evaluate models,
- preprocess data,
- select the best model,
- call an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)
from xgboost import (
    XGBClassifier,
    XGBRegressor,
)
from lightgbm import (
    LGBMClassifier,
    LGBMRegressor,
)

from .task_detector import MLTask, TargetType


class ModelFamily(str, Enum):
    """Supported model families."""

    LOGISTIC_REGRESSION = "logistic_regression"
    LINEAR_REGRESSION = "linear_regression"

    RANDOM_FOREST = "random_forest"

    XGBOOST = "xgboost"

    LIGHTGBM = "lightgbm"


@dataclass(frozen=True)
class ModelSpec:
    """
    Metadata describing an ML model.

    Attributes
    ----------
    name:
        Internal model name.

    display_name:
        Human-readable model name.

    family:
        Model family.

    task:
        Supported ML task.

    description:
        Short description used by the UI/reporting layer.
    """

    name: str
    display_name: str
    family: ModelFamily
    task: MLTask
    description: str


# ---------------------------------------------------------------------
# Model specifications
# ---------------------------------------------------------------------

CLASSIFICATION_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="logistic_regression",
        display_name="Logistic Regression",
        family=ModelFamily.LOGISTIC_REGRESSION,
        task=MLTask.CLASSIFICATION,
        description=(
            "Linear baseline model for classification."
        ),
    ),
    ModelSpec(
        name="random_forest",
        display_name="Random Forest",
        family=ModelFamily.RANDOM_FOREST,
        task=MLTask.CLASSIFICATION,
        description=(
            "Ensemble of decision trees suitable for nonlinear "
            "classification problems."
        ),
    ),
    ModelSpec(
        name="xgboost",
        display_name="XGBoost",
        family=ModelFamily.XGBOOST,
        task=MLTask.CLASSIFICATION,
        description=(
            "Gradient-boosted decision-tree model with strong "
            "performance on structured/tabular data."
        ),
    ),
    ModelSpec(
        name="lightgbm",
        display_name="LightGBM",
        family=ModelFamily.LIGHTGBM,
        task=MLTask.CLASSIFICATION,
        description=(
            "Efficient gradient-boosting model designed for "
            "high-performance tabular learning."
        ),
    ),
)


REGRESSION_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="linear_regression",
        display_name="Linear Regression",
        family=ModelFamily.LINEAR_REGRESSION,
        task=MLTask.REGRESSION,
        description=(
            "Linear baseline model for regression."
        ),
    ),
    ModelSpec(
        name="random_forest",
        display_name="Random Forest",
        family=ModelFamily.RANDOM_FOREST,
        task=MLTask.REGRESSION,
        description=(
            "Ensemble of decision trees suitable for nonlinear "
            "regression problems."
        ),
    ),
    ModelSpec(
        name="xgboost",
        display_name="XGBoost",
        family=ModelFamily.XGBOOST,
        task=MLTask.REGRESSION,
        description=(
            "Gradient-boosted decision-tree model for "
            "structured regression problems."
        ),
    ),
    ModelSpec(
        name="lightgbm",
        display_name="LightGBM",
        family=ModelFamily.LIGHTGBM,
        task=MLTask.REGRESSION,
        description=(
            "Efficient gradient-boosting model for tabular "
            "regression problems."
        ),
    ),
)


def get_model_specs(
    task: MLTask,
) -> list[ModelSpec]:
    """
    Return the candidate model specifications for an ML task.

    Parameters
    ----------
    task:
        Detected ML task.

    Returns
    -------
    list[ModelSpec]
        Candidate models for the task.
    """

    if task == MLTask.CLASSIFICATION:
        return list(CLASSIFICATION_MODELS)

    if task == MLTask.REGRESSION:
        return list(REGRESSION_MODELS)

    raise ValueError(
        f"Unsupported ML task: {task}"
    )


def get_model_spec(
    model_name: str,
    task: MLTask,
) -> ModelSpec:
    """
    Get a specific model specification.

    Parameters
    ----------
    model_name:
        Internal model name.

    task:
        ML task.

    Returns
    -------
    ModelSpec
        Matching model specification.
    """

    model_name = model_name.strip().lower()

    specs = get_model_specs(task)

    for spec in specs:
        if spec.name == model_name:
            return spec

    available = ", ".join(spec.name for spec in specs)

    raise ValueError(
        f"Model '{model_name}' is not available for "
        f"{task.value}. Available models: {available}"
    )


def _classification_model(
    model_name: str,
    random_state: int,
    n_jobs: int,
) -> Any:
    """
    Create a classification model.

    These are intentionally baseline configurations.
    Hyperparameter optimization will be added later.
    """

    if model_name == ModelFamily.LOGISTIC_REGRESSION.value:
        return LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        )

    if model_name == ModelFamily.RANDOM_FOREST.value:
        return RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    if model_name == ModelFamily.XGBOOST.value:
        return XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=n_jobs,
        )

    if model_name == ModelFamily.LIGHTGBM.value:
        return LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=-1,
        )

    raise ValueError(
        f"Unsupported classification model: {model_name}"
    )


def _regression_model(
    model_name: str,
    random_state: int,
    n_jobs: int,
) -> Any:
    """
    Create a regression model.

    These are intentionally baseline configurations.
    Hyperparameter optimization will be added later.
    """

    if model_name == ModelFamily.LINEAR_REGRESSION.value:
        return LinearRegression(
            n_jobs=n_jobs,
        )

    if model_name == ModelFamily.RANDOM_FOREST.value:
        return RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    if model_name == ModelFamily.XGBOOST.value:
        return XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=random_state,
            n_jobs=n_jobs,
        )

    if model_name == ModelFamily.LIGHTGBM.value:
        return LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=-1,
        )

    raise ValueError(
        f"Unsupported regression model: {model_name}"
    )


def create_model(
    model_name: str,
    task: MLTask,
    *,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Any:
    """
    Create a configured ML model.

    Parameters
    ----------
    model_name:
        Name of the model to create.

    task:
        Classification or regression.

    random_state:
        Seed for reproducibility.

    n_jobs:
        Number of CPU workers.

    Returns
    -------
    sklearn-compatible estimator.
    """

    model_name = model_name.strip().lower()

    # Validate that the model exists for this task.
    get_model_spec(model_name, task)

    if task == MLTask.CLASSIFICATION:
        return _classification_model(
            model_name=model_name,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    if task == MLTask.REGRESSION:
        return _regression_model(
            model_name=model_name,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    raise ValueError(
        f"Unsupported ML task: {task}"
    )


def create_models(
    task: MLTask,
    *,
    random_state: int = 42,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """
    Create all candidate models for a task.

    Parameters
    ----------
    task:
        Classification or regression.

    random_state:
        Seed for reproducibility.

    n_jobs:
        Number of CPU workers.

    Returns
    -------
    dict[str, estimator]
        Dictionary mapping model names to model instances.
    """

    models: dict[str, Any] = {}

    for spec in get_model_specs(task):

        models[spec.name] = create_model(
            model_name=spec.name,
            task=task,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    return models


def get_model_names(
    task: MLTask,
) -> list[str]:
    """Return model names available for a task."""

    return [
        spec.name
        for spec in get_model_specs(task)
    ]


def model_supports_probability(
    model: Any,
) -> bool:
    """
    Determine whether a model exposes predict_proba().
    """

    return callable(
        getattr(model, "predict_proba", None)
    )


def model_supports_feature_importance(
    model: Any,
) -> bool:
    """
    Determine whether a model exposes native feature importance.
    """

    return hasattr(model, "feature_importances_") or hasattr(
        model,
        "coef_",
    )