"""
DataMindAI conversational agent.

Architecture
------------
User
    ↓
DataMindAgent
    ↓
    ├── Legacy deterministic DatasetQA
    │       ↓
    │   DatasetQA / Pandas
    │
    └── Local Qwen
            ↓
        Tool selection
            ↓
        Python tools
            ↓
        Verified evidence
            ↓
        Local Qwen
            ↓
        Final answer

The legacy DatasetQA path is intentionally retained for
backward compatibility with the existing project and tests.

The new DataAnalysisTool is the preferred general-purpose
analysis tool for questions that DatasetQA does not handle.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from src.backend.dataset_qa import DatasetQA
from src.backend.preprocessing_intelligence import PreprocessingIntelligence
from src.core.app_state import app_state
from src.core.logger import get_logger
from src.ml.explainability import explain_model, explain_training_result
from src.ml.ml_pipeline import run_ml_pipeline
from src.ml.model_persistence import (
    load_artifact,
    load_model,
    save_best_model,
)
from src.ml.prediction_engine import predict, predict_from_artifact
from src.tools.data_analysis import DataAnalysisTool
from src.visualisation.visualisation_engine import (
    VisualizationEngine,
)


logger = get_logger(__name__)


class DataMindAgent:
    """
    Local-Qwen conversational agent for DataMindAI.

    Qwen:
        - understands natural language
        - selects tools
        - explains verified results

    Python:
        - performs calculations
        - executes visualizations
        - provides deterministic evidence
    """

    def __init__(
        self,
        model: str = "qwen3:8b",
    ) -> None:

        self.model = model

    # ========================================================
    # DATASET CONTEXT
    # ========================================================

    def _dataset_context(self) -> str:
        """
        Build dynamic dataset context for Qwen.
        """

        if not app_state.has_dataset():

            return (
                "No dataset is currently loaded."
            )

        dataframe = app_state.dataset

        columns = [
            str(column)
            for column in dataframe.columns
        ]

        numerical_columns = [
            str(column)
            for column in dataframe.select_dtypes(
                include="number"
            ).columns
        ]

        categorical_columns = [
            str(column)
            for column in dataframe.select_dtypes(
                include=[
                    "object",
                    "category",
                ]
            ).columns
        ]

        context = f"""
DATASET
-------
Name: {app_state.dataset_name}
Rows: {len(dataframe)}
Columns: {len(dataframe.columns)}

Columns:
{columns}

Numerical columns:
{numerical_columns}

Categorical columns:
{categorical_columns}
"""

        profile = app_state.dataset_profile

        if profile is not None:

            context += f"""

DATASET PROFILE
---------------
Health score:
{profile.health_score}

Health status:
{profile.health_status}

Missing values:
{profile.missing_values}

Duplicate rows:
{profile.duplicate_rows}

Possible target columns:
{profile.possible_target_columns}
"""

        return context

    # ========================================================
    # SYSTEM PROMPT
    # ========================================================

    def _system_prompt(self) -> str:
        """
        System prompt for local Qwen.
        """

        return f"""
You are DataMindAI, an AI Data Scientist.

You work with a dataset supplied by the user.

Your responsibility is to understand natural-language
questions and select the correct Python tool.

IMPORTANT:

You are NOT the calculator.

Python/Pandas is the source of truth for numerical
answers.

Use the data_analysis tool whenever the user asks for
information that requires calculation from the dataset.

Examples:

- average
- mean
- median
- sum
- total
- maximum
- minimum
- highest
- lowest
- top N
- bottom N
- ranking
- correlation
- unique values
- counts
- missing values
- duplicate rows
- filtering
- comparison
- grouped analysis
- which company
- which year
- trends requiring numerical analysis

Use the visualize tool when the user asks for:

- chart
- graph
- plot
- visualization
- trend visualization
- relationship visualization
- visual comparison

Use the predict tool when the user asks to predict a value or outcome:
- "Predict for these values: age=30, salary=50000..."
- "Predict salary for an employee with age 45..."
- "Predict churn for this customer..."

Use the train_model tool when the user asks to train or build models:
- "Train the best model"
- "Run AutoML"
- "Train a model for churn"

Use the explain_model tool when the user asks:
- "Explain the model"
- "What are the most important features?"
- "What drives the predictions?"

Use the get_preprocessing_plan tool when the user asks:
- "Explain the preprocessing plan"
- "What preprocessing recommendations do you have?"

The user does NOT need to know the exact dataset
column names.

Map natural language to the actual dataset columns.

Examples:

"workforce"
→ employees-related column

"people laid off"
→ layoffs

"company revenue"
→ revenue-related column

If the requested information does not exist in the
dataset, NEVER invent an answer.

Explain that the required information is not available.

For conceptual questions, you may answer directly.

After a tool returns evidence:

- explain the evidence
- do not invent numbers
- do not modify calculated values
- do not perform a different calculation
- remain faithful to the tool result

{self._dataset_context()}
"""

    # ========================================================
    # DATA ANALYSIS TOOL SCHEMA
    # ========================================================

    @staticmethod
    def _data_analysis_tool_schema() -> dict[str, Any]:
        """
        Native Ollama schema for the data-analysis tool.
        """

        return {
            "type": "function",
            "function": {
                "name": "data_analysis",
                "description": (
                    "Perform deterministic analysis on the "
                    "loaded dataset. Use this for statistics, "
                    "grouped analysis, ranking, filtering, "
                    "correlation, counts, unique values, "
                    "missing values, duplicates, row extremes "
                    "and comparisons."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": [
                                "overview",
                                "statistics",
                                "groupby",
                                "rank",
                                "filter",
                                "correlation",
                                "value_counts",
                                "unique_count",
                                "missing_values",
                                "duplicates",
                                "row_extreme",
                                "compare",
                            ],
                        },
                        "column": {
                            "type": "string",
                            "description": (
                                "Numeric column for statistics, ranking, "
                                "filtering, and similar operations. For "
                                "groupby, use value_column instead."
                            ),
                        },
                        "column1": {
                            "type": "string",
                        },
                        "column2": {
                            "type": "string",
                        },
                        "group_column": {
                            "type": "string",
                            "description": (
                                "Categorical column whose values define "
                                "the groups for a groupby or rank."
                            ),
                        },
                        "value_column": {
                            "type": "string",
                            "description": (
                                "Numeric column to aggregate for a "
                                "groupby operation."
                            ),
                        },
                        "aggregation": {
                            "type": "string",
                            "enum": [
                                "mean",
                                "median",
                                "sum",
                                "min",
                                "max",
                                "std",
                                "count",
                            ],
                        },
                        "order": {
                            "type": "string",
                            "enum": [
                                "ascending",
                                "descending",
                            ],
                        },
                        "sort": {
                            "type": "string",
                            "enum": [
                                "ascending",
                                "descending",
                            ],
                            "description": "Sort direction for groupby results.",
                        },
                        "direction": {
                            "type": "string",
                            "enum": [
                                "min",
                                "max",
                            ],
                        },
                        "operator": {
                            "type": "string",
                            "enum": [
                                "==",
                                "!=",
                                ">",
                                ">=",
                                "<",
                                "<=",
                            ],
                        },
                        "value": {},
                        "limit": {
                            "type": "integer",
                        },
                        "groups": {
                            "type": "array",
                            "items": {},
                        },
                        "method": {
                            "type": "string",
                            "enum": [
                                "pearson",
                                "spearman",
                                "kendall",
                            ],
                        },
                    },
                    "required": [
                        "operation"
                    ],
                },
            },
        }

    # ========================================================
    # VISUALIZATION TOOL SCHEMA
    # ========================================================

    @staticmethod
    def _visualization_tool_schema() -> dict[str, Any]:
        """
        Native Ollama schema for visualization.
        """

        return {
            "type": "function",
            "function": {
                "name": "visualize",
                "description": (
                    "Generate a visualization using two "
                    "actual dataset columns."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "column1": {
                            "type": "string",
                        },
                        "column2": {
                            "type": "string",
                        },
                    },
                    "required": [
                        "column1",
                        "column2",
                    ],
                },
            },
        }

    # ========================================================
    # PREDICTION TOOL SCHEMA
    # ========================================================

    @staticmethod
    def _predict_tool_schema() -> dict[str, Any]:
        """
        Native Ollama schema for model prediction.
        """

        return {
            "type": "function",
            "function": {
                "name": "predict",
                "description": (
                    "Predict a target value or class for user-provided feature values "
                    "using the trained machine-learning pipeline. The trained pipeline "
                    "automatically applies the exact preprocessing used during training."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "values": {
                            "type": "object",
                            "description": (
                                "Key-value mapping of feature names to values for prediction, "
                                "e.g. {'age': 45, 'salary': 50000, 'department': 'Sales'}."
                            ),
                        },
                    },
                    "required": ["values"],
                },
            },
        }

    # ========================================================
    # TRAIN MODEL TOOL SCHEMA
    # ========================================================

    @staticmethod
    def _train_model_tool_schema() -> dict[str, Any]:
        """
        Native Ollama schema for AutoML model training.
        """

        return {
            "type": "function",
            "function": {
                "name": "train_model",
                "description": (
                    "Automatically train, evaluate, and select the best machine-learning model "
                    "for the dataset and persist the best model for future predictions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_column": {
                            "type": "string",
                            "description": "Optional name of the target column to predict.",
                        },
                        "model_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of candidate models to train.",
                        },
                    },
                },
            },
        }

    # ========================================================
    # EXPLAIN MODEL TOOL SCHEMA
    # ========================================================

    @staticmethod
    def _explain_model_tool_schema() -> dict[str, Any]:
        """
        Native Ollama schema for model explainability.
        """

        return {
            "type": "function",
            "function": {
                "name": "explain_model",
                "description": (
                    "Explain the trained machine-learning model by computing global feature "
                    "importance and ranking the features that drive predictions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "top_n": {
                            "type": "integer",
                            "description": "Number of top features to display (default 5).",
                        },
                    },
                },
            },
        }

    # ========================================================
    # PREPROCESSING PLAN TOOL SCHEMA
    # ========================================================

    @staticmethod
    def _preprocessing_plan_tool_schema() -> dict[str, Any]:
        """
        Native Ollama schema for preprocessing plan retrieval.
        """

        return {
            "type": "function",
            "function": {
                "name": "get_preprocessing_plan",
                "description": (
                    "Retrieve the automated preprocessing and cleaning plan recommended for the dataset."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        }

    # ========================================================
    # TOOLS
    # ========================================================

    def _tools(self) -> list[dict[str, Any]]:
        """
        Return tools available to Qwen.
        """

        return [
            self._data_analysis_tool_schema(),
            self._visualization_tool_schema(),
            self._predict_tool_schema(),
            self._train_model_tool_schema(),
            self._explain_model_tool_schema(),
            self._preprocessing_plan_tool_schema(),
        ]

    # ========================================================
    # LEGACY DATASET QA
    # ========================================================

    def _try_legacy_dataset_qa(
        self,
        prompt: str,
    ) -> dict[str, Any] | None:
        """
        Try the existing DatasetQA engine.

        This is retained for backward compatibility.

        If DatasetQA does not understand the question,
        None is returned and the new Qwen tool-calling
        architecture handles the question.
        """

        if not app_state.has_dataset():

            return None

        try:

            qa = DatasetQA(
                app_state.dataset
            )

            result = qa.answer(
                prompt
            )

        except Exception:

            logger.exception(
                "Legacy DatasetQA execution failed"
            )

            return None

        if result is None:
            return None

        if not result.success:
            return None

        # ----------------------------------------------------
        # Normalize the old operation names to the existing
        # test/project contract.
        # ----------------------------------------------------

        operation = result.operation

        operation_aliases = {
            "statistics": "mean",
            "unique_count": "unique_values",
        }

        normalized_operation = (
            operation_aliases.get(
                operation,
                operation,
            )
        )

        evidence = dict(
            result.evidence
            or {}
        )

        # ----------------------------------------------------
        # DatasetQA historically used "value" for mean etc.
        # Keep it untouched.
        # ----------------------------------------------------

        response = {
            "type": "text",
            "content": result.answer,
            "operation": normalized_operation,
            "evidence": evidence,
            "dataframe": result.dataframe,
        }

        app_state.last_qa_result = result

        return response

    # ========================================================
    # DATA ANALYSIS EXECUTION
    # ========================================================

    def _execute_data_analysis(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute DataAnalysisTool.
        """

        if not app_state.has_dataset():

            return {
                "success": False,
                "error": (
                    "No dataset is currently loaded."
                ),
            }

        try:

            tool = DataAnalysisTool(
                app_state.dataset
            )

            operation = arguments.get(
                "operation"
            )

            parameters = {
                key: value
                for key, value in arguments.items()
                if key != "operation"
                and value is not None
            }

            result = tool.execute(
                operation,
                **parameters,
            )

            app_state.last_data_analysis = (
                result
            )

            return result.to_dict()

        except Exception as error:

            logger.exception(
                "DataAnalysisTool failed"
            )

            return {
                "success": False,
                "error": str(error),
            }

    # ========================================================
    # VISUALIZATION EXECUTION
    # ========================================================

    def _execute_visualization(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute VisualizationEngine.
        """

        if not app_state.has_dataset():

            return {
                "success": False,
                "error": (
                    "No dataset is currently loaded."
                ),
            }

        column1 = arguments.get(
            "column1"
        )

        column2 = arguments.get(
            "column2"
        )

        if not column1 or not column2:

            return {
                "success": False,
                "error": (
                    "Both visualization columns are required."
                ),
            }

        columns = set(
            app_state.dataset.columns
        )

        if column1 not in columns:

            return {
                "success": False,
                "error": (
                    f"Column '{column1}' does not exist."
                ),
            }

        if column2 not in columns:

            return {
                "success": False,
                "error": (
                    f"Column '{column2}' does not exist."
                ),
            }

        try:

            engine = VisualizationEngine(
                app_state.dataset
            )

            figure = engine.generate(
                column1,
                column2,
            )

            app_state.current_figure = (
                figure
            )

            app_state.generated_chart.append(
                {
                    "column1": column1,
                    "column2": column2,
                }
            )

            return {
                "success": True,
                "column1": column1,
                "column2": column2,
                "figure": figure,
            }

        except Exception as error:

            logger.exception(
                "Visualization generation failed"
            )

            return {
                "success": False,
                "error": str(error),
            }

    # ========================================================
    # MODEL ARTIFACT LOOKUP
    # ========================================================

    def _find_model_artifact(self) -> Any:
        """
        Find an active or persisted model artifact.
        """
        if app_state.model_artifact is not None:
            return app_state.model_artifact

        model_dir = Path("artifacts/models")
        if model_dir.exists():
            meta_files = list(model_dir.glob("*_metadata.joblib"))
            if meta_files:
                meta_path = meta_files[0]
                model_name = meta_path.name.replace("_metadata.joblib", ".joblib")
                model_path = model_dir / model_name
                if model_path.exists():
                    try:
                        artifact = load_artifact(model_path, meta_path)
                        app_state.model_artifact = artifact
                        return artifact
                    except Exception:
                        pass
        return None

    # ========================================================
    # PREDICTION EXECUTION
    # ========================================================

    def _execute_prediction(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute prediction using the trained machine-learning pipeline.
        """
        if not app_state.has_dataset():
            return {
                "success": False,
                "error": "Please upload a dataset first.",
            }

        values = arguments.get("values")
        if not isinstance(values, dict):
            values = {}

        artifact = self._find_model_artifact()

        # If no saved artifact exists, check if an automl_result can be saved
        if artifact is None and app_state.automl_result is not None:
            artifact = save_best_model(
                app_state.automl_result,
                model_directory="artifacts/models",
            )
            app_state.model_artifact = artifact

        if artifact is None:
            return {
                "success": False,
                "error": (
                    "No trained model is currently available. "
                    "Please run AutoML or say 'Train the best model' first."
                ),
            }

        feature_names = artifact.feature_names
        if not feature_names:
            return {
                "success": False,
                "error": "The trained model does not contain feature schema information.",
            }

        df = app_state.dataset
        row: dict[str, Any] = {}
        imputed_features: list[str] = []

        # Build prediction input matching expected features
        for feature in feature_names:
            matched_val = None
            clean_feat = feature.lower().replace(" ", "_")
            for k, v in values.items():
                if k.lower().replace(" ", "_") == clean_feat:
                    matched_val = v
                    break

            if matched_val is not None:
                # Type coercion
                if isinstance(matched_val, str) and matched_val.replace(".", "", 1).isdigit():
                    try:
                        if "." in matched_val:
                            row[feature] = float(matched_val)
                        else:
                            row[feature] = int(matched_val)
                    except ValueError:
                        row[feature] = matched_val
                else:
                    row[feature] = matched_val
            else:
                # Impute missing feature value from dataset
                if feature in df.columns:
                    col = df[feature].dropna()
                    if pd.api.types.is_numeric_dtype(col) and not col.empty:
                        row[feature] = float(col.median())
                    elif not col.empty:
                        row[feature] = col.mode().iloc[0]
                    else:
                        row[feature] = 0.0
                    imputed_features.append(feature)
                else:
                    row[feature] = 0.0
                    imputed_features.append(feature)

        input_df = pd.DataFrame([row])[feature_names]

        try:
            pred_result = predict_from_artifact(artifact, input_df)
        except Exception as exc:
            logger.exception("Prediction failed")
            return {
                "success": False,
                "error": f"Prediction failed: {exc}",
            }

        app_state.prediction_result = pred_result

        display_name = pred_result.display_name
        target = pred_result.target_column
        prediction_val = pred_result.prediction

        if pred_result.task == "classification":
            confidence_str = (
                f" with **{pred_result.confidence * 100:.1f}%** confidence"
                if pred_result.confidence is not None
                else ""
            )
            prob_str = ""
            if pred_result.probabilities:
                breakdown = ", ".join(
                    f"`{k}`: {v * 100:.1f}%"
                    for k, v in pred_result.probabilities.items()
                )
                prob_str = f"\n\n**Class Probabilities:** {breakdown}"

            answer = (
                f"### 🎯 Prediction Result\n\n"
                f"The trained **{display_name}** pipeline predicts:\n\n"
                f"- **Target ({target})**: `{prediction_val}`{confidence_str}{prob_str}"
            )
        else:
            try:
                formatted_num = f"{float(prediction_val):,.2f}"
            except (ValueError, TypeError):
                formatted_num = str(prediction_val)

            answer = (
                f"### 🎯 Prediction Result\n\n"
                f"The trained **{display_name}** pipeline predicts:\n\n"
                f"- **Target ({target})**: `{formatted_num}`"
            )

        if imputed_features:
            answer += (
                f"\n\n*Note: Default values were imputed from the dataset for unspecified features: "
                f"`{', '.join(imputed_features)}`.*"
            )

        return {
            "success": True,
            "prediction": pred_result.prediction,
            "task": pred_result.task,
            "display_name": pred_result.display_name,
            "target_column": pred_result.target_column,
            "confidence": pred_result.confidence,
            "probabilities": pred_result.probabilities,
            "input_data": pred_result.input_data,
            "answer": answer,
        }

    # ========================================================
    # MODEL TRAINING EXECUTION
    # ========================================================

    def _execute_model_training(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute AutoML pipeline and persist the winning model.
        """
        if not app_state.has_dataset():
            return {
                "success": False,
                "error": "Please upload a dataset first.",
            }

        df = app_state.dataset
        target_column = arguments.get("target_column")

        if not target_column:
            if (
                app_state.dataset_profile is not None
                and app_state.dataset_profile.recommended_target
            ):
                target_column = app_state.dataset_profile.recommended_target
            else:
                target_column = df.columns[-1]

        if target_column not in df.columns:
            return {
                "success": False,
                "error": f"Target column '{target_column}' does not exist in the dataset.",
            }

        model_names = arguments.get("model_names")
        if isinstance(model_names, list) and not model_names:
            model_names = None

        try:
            result = run_ml_pipeline(
                df=df,
                target_column=target_column,
                model_names=model_names,
            )
            artifact = save_best_model(
                result,
                model_directory="artifacts/models",
            )
        except Exception as exc:
            logger.exception("AutoML execution failed")
            return {
                "success": False,
                "error": f"AutoML training failed: {exc}",
            }

        app_state.automl_result = result
        app_state.model_artifact = artifact
        app_state.trained_model = result.best_training_result.pipeline
        app_state.model_metadata = artifact.to_dict()
        app_state.current_task = result.task.value

        metric_name = result.model_selection.primary_metric.upper()
        metric_score = result.model_selection.primary_score
        best_name = result.best_model_display_name

        models_tested_str = ", ".join(result.training_results.keys())
        features_count = len(artifact.feature_names)

        # Register in Model Registry if active session has project
        user_id = getattr(app_state, "active_user_id", None)
        project_id = getattr(app_state, "active_project_id", None)
        if user_id and project_id:
            try:
                from src.services.model_registry_service import ModelRegistryService
                ModelRegistryService.register_model(
                    user_id=user_id,
                    project_id=project_id,
                    model_name=artifact.model_name,
                    display_name=artifact.display_name,
                    task=artifact.task,
                    target_column=artifact.target_column,
                    artifact_path=artifact.model_path,
                    metadata_path=artifact.metadata_path,
                    feature_names=artifact.feature_names,
                    metrics={result.model_selection.primary_metric: metric_score},
                    training_config={"models_tested": list(result.training_results.keys())},
                    is_best=True,
                )
            except Exception as reg_exc:
                logger.warning("Could not auto-register model in DB registry: %s", reg_exc)

        leaderboard_rows = []
        if hasattr(result.leaderboard, "to_dict"):
            leaderboard_rows = result.leaderboard.to_dict(orient="records")

        table_md = ""
        if leaderboard_rows:
            headers = list(leaderboard_rows[0].keys())
            table_md = "\n\n| " + " | ".join(headers) + " |\n"
            table_md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
            for row in leaderboard_rows:
                table_md += "| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n"

        answer = (
            f"### 🏆 AutoML Training Complete\n\n"
            f"Model training completed successfully.\n\n"
            f"- **Target**: `{target_column}`\n"
            f"- **Problem**: {result.task.value.title()}\n"
            f"- **Features**: {features_count} columns\n"
            f"- **Split**: 80% Train / 20% Test\n"
            f"- **Preprocessing**: Missing value imputation, categorical encoding, feature scaling\n"
            f"- **Models Tested**: {models_tested_str}\n"
            f"- **Best Model**: **{best_name}** ({metric_name}: `{metric_score:.4f}`)\n\n"
            f"{result.model_selection.selection_reason}"
            f"{table_md}\n\n"
            f"Your model has been saved successfully.\n\n"
            f"👉 **You can now go to the Prediction section to make predictions.**"
        )

        return {
            "success": True,
            "best_model": best_name,
            "target_column": target_column,
            "task": result.task.value,
            "primary_metric": result.model_selection.primary_metric,
            "primary_score": metric_score,
            "leaderboard": leaderboard_rows,
            "answer": answer,
        }

    # ========================================================
    # MODEL EXPLAINABILITY EXECUTION
    # ========================================================

    def _execute_model_explanation(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Explain the trained model feature importance.
        """
        if not app_state.has_dataset():
            return {
                "success": False,
                "error": "Please upload a dataset first.",
            }

        result = app_state.automl_result
        if result is None:
            artifact = self._find_model_artifact()
            if artifact is None:
                return {
                    "success": False,
                    "error": (
                        "No trained model is currently available to explain. "
                        "Please train a model first using 'Train the best model'."
                    ),
                }

        top_n = arguments.get("top_n", 5)
        if not isinstance(top_n, int) or top_n <= 0:
            top_n = 5

        if result is not None:
            best_training = result.best_training_result
            X = app_state.dataset.drop(columns=[result.target_column])
            y = app_state.dataset[result.target_column]

            try:
                explanation = explain_training_result(best_training, X=X, y=y)
                app_state.last_explanation = explanation
            except Exception as exc:
                logger.exception("Model explanation failed")
                return {
                    "success": False,
                    "error": f"Explainability analysis failed: {exc}",
                }
            target_name = result.target_column
        else:
            artifact = self._find_model_artifact()
            model = load_model(artifact.model_path)
            target_name = artifact.target_column
            X = app_state.dataset.drop(columns=[target_name]) if target_name in app_state.dataset.columns else app_state.dataset
            y = app_state.dataset[target_name] if target_name in app_state.dataset.columns else None

            try:
                explanation = explain_model(
                    model=model,
                    task=artifact.task,
                    feature_names=artifact.feature_names,
                    model_name=artifact.model_name,
                    display_name=artifact.display_name,
                    X=X,
                    y=y,
                )
                app_state.last_explanation = explanation
            except Exception as exc:
                logger.exception("Model explanation failed")
                return {
                    "success": False,
                    "error": f"Explainability analysis failed: {exc}",
                }

        top_features = explanation.top_features[:top_n]
        feat_list = []
        for feat in top_features:
            feat_list.append(
                f"- **{feat.feature}** (Rank #{feat.rank}, Importance: `{feat.importance:.4f}`, Effect: {feat.direction})"
            )

        answer = (
            f"### 🧠 Model Explainability ({explanation.display_name})\n\n"
            f"Here are the top **{len(top_features)}** most important features driving predictions "
            f"for **{target_name}** (using {explanation.method}):\n\n"
            + "\n".join(feat_list)
        )

        return {
            "success": True,
            "display_name": explanation.display_name,
            "method": explanation.method,
            "top_features": [f.to_dict() for f in top_features],
            "answer": answer,
        }

    # ========================================================
    # PREPROCESSING PLAN EXECUTION
    # ========================================================

    def _execute_preprocessing_plan(
        self,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Retrieve and explain the automated preprocessing plan.
        """
        if not app_state.has_dataset():
            return {
                "success": False,
                "error": "Please upload a dataset first.",
            }

        plan = app_state.preprocessing_plan
        if plan is None:
            if app_state.dataset_profile is None:
                try:
                    from src.backend.dataset_intelligence import DatasetIntelligence
                    intel_ds = DatasetIntelligence(
                        app_state.dataset,
                        app_state.dataset_name or "dataset",
                    )
                    app_state.dataset_profile = intel_ds.generate_profile()
                except Exception as exc:
                    logger.warning("Could not auto-generate dataset profile: %s", exc)

            if app_state.dataset_profile is not None:
                try:
                    intel = PreprocessingIntelligence(
                        app_state.dataset,
                        app_state.dataset_profile,
                    )
                    plan = intel.generate_plan()
                    app_state.preprocessing_plan = plan
                except Exception as exc:
                    return {
                        "success": False,
                        "error": f"Could not generate preprocessing plan: {exc}",
                    }
            else:
                return {
                    "success": False,
                    "error": "Dataset profile not available to generate preprocessing recommendations.",
                }

        summary_lines = []
        if plan.pipeline_summary:
            for step in plan.pipeline_summary:
                summary_lines.append(f"• {step}")
        else:
            summary_lines.append("No specific preprocessing actions needed.")

        notes_lines = []
        if plan.notes:
            for note in plan.notes:
                notes_lines.append(f"- *{note}*")

        answer = (
            f"### 🛠 Automated Preprocessing Plan\n\n"
            f"Based on dataset profiling, here is the recommended plan:\n\n"
            + "\n".join(summary_lines)
        )
        if notes_lines:
            answer += "\n\n" + "\n".join(notes_lines)

        return {
            "success": True,
            "pipeline_summary": plan.pipeline_summary,
            "answer": answer,
        }

    # ========================================================
    # DATA CLEANING EXECUTION
    # ========================================================

    def _execute_data_cleaning(
        self,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute in-place data cleaning on the active dataset.
        Handles missing values and optional outlier removal.
        """
        if not app_state.has_dataset():
            return {
                "success": False,
                "error": "Please upload a dataset first.",
            }

        df = app_state.dataset
        clean_type = (arguments or {}).get("clean_type", "missing_values")

        missing_before = int(df.isna().sum().sum())
        from src.backend.preprocessing import handle_missing_values, remove_outliers_iqr

        cleaned_df = handle_missing_values(
            df,
            numeric_strategy="median",
            categorical_strategy="most_frequent",
        )
        if clean_type in ("outliers", "all"):
            cleaned_df = remove_outliers_iqr(cleaned_df)

        app_state.dataset = cleaned_df
        missing_after = int(cleaned_df.isna().sum().sum())

        try:
            from src.backend.dataset_intelligence import DatasetIntelligence
            intelligence = DatasetIntelligence(cleaned_df, app_state.dataset_name or "dataset")
            app_state.dataset_profile = intelligence.generate_profile()
        except Exception:
            pass

        answer = (
            "Done. Numerical missing values were handled using median imputation "
            "and categorical values using the most frequent value."
        )
        if missing_before > 0:
            answer += f" Missing values were reduced from {missing_before} to {missing_after}."

        return {
            "success": True,
            "missing_before": missing_before,
            "missing_after": missing_after,
            "answer": answer,
        }

    # ========================================================
    # GEOSPATIAL MAP EXECUTION
    # ========================================================

    def _execute_geospatial_map(
        self,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Generate a geospatial map visualization.
        """
        if not app_state.has_dataset():
            return {
                "success": False,
                "error": "Please upload a dataset first.",
            }

        df = app_state.dataset
        lat_candidates = [c for c in df.columns if c.lower() in ("latitude", "lat", "y")]
        lon_candidates = [c for c in df.columns if c.lower() in ("longitude", "lon", "lng", "x")]

        if not lat_candidates or not lon_candidates:
            return {
                "success": False,
                "error": "No geographic latitude and longitude columns were found in the dataset.",
            }

        import plotly.express as px
        lat_col = lat_candidates[0]
        lon_col = lon_candidates[0]
        name_cols = [c for c in df.columns if c.lower() in ("name", "city", "location", "place", "title", "country")]
        hover_col = name_cols[0] if name_cols else None

        fig = px.scatter_geo(
            df.dropna(subset=[lat_col, lon_col]),
            lat=lat_col,
            lon=lon_col,
            hover_name=hover_col,
            title=f"Geographical Map ({lat_col}, {lon_col})",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        app_state.current_figure = fig

        return {
            "success": True,
            "figure": fig,
            "column1": lat_col,
            "column2": lon_col,
            "answer": f"Here is the map showing the locations from `{lat_col}` and `{lon_col}`.",
        }

    # ========================================================
    # DETERMINISTIC ML INTENT DETECTION
    # ========================================================

    def _try_deterministic_ml_qa(
        self,
        prompt: str,
    ) -> dict[str, Any] | None:
        """
        Deterministic intent detection for ML tasks:
        - Prediction ("Predict for these values...", "Predict salary for age=30...")
        - Model Training ("Train the best model", "Run AutoML")
        - Explainability ("Explain the model", "Most important features")
        - Preprocessing Plan ("Explain the preprocessing plan")
        """
        if not app_state.has_dataset():
            return None

        normalized = prompt.strip().lower()

        # 1. Prediction requests
        if "predict" in normalized:
            matches = re.findall(
                r"([a-zA-Z_][a-zA-Z0-9_]*)\s*[:=]\s*([a-zA-Z0-9_\.\-]+|\"[^\"]+\"|'[^']+')",
                prompt,
            )
            values: dict[str, Any] = {}
            reserved = {"target", "model", "predict", "for", "these", "values", "the"}
            for k, v in matches:
                if k.lower() not in reserved:
                    v_clean = v.strip("'\"")
                    try:
                        if "." in v_clean:
                            values[k] = float(v_clean)
                        else:
                            values[k] = int(v_clean)
                    except ValueError:
                        values[k] = v_clean

            if values:
                result = self._execute_prediction({"values": values})
                if result.get("success"):
                    return {
                        "type": "text",
                        "content": result["answer"],
                        "operation": "prediction",
                        "evidence": result,
                    }
                else:
                    return {
                        "type": "text",
                        "content": result.get("error", "Prediction could not be completed."),
                    }
            elif any(
                phrase in normalized
                for phrase in [
                    "predict for these values",
                    "predict using the trained model",
                    "predict for values",
                ]
            ):
                artifact = self._find_model_artifact()
                if artifact and artifact.feature_names:
                    features_str = ", ".join(f"`{f}`" for f in artifact.feature_names)
                    return {
                        "type": "text",
                        "content": (
                            f"To make a prediction, please provide values for the required features:\n\n"
                            f"**Required Features:** {features_str}\n\n"
                            f"*Example:* `Predict for these values: {artifact.feature_names[0]}=...`"
                        ),
                    }
                return {
                    "type": "text",
                    "content": (
                        "Please provide feature values for prediction, for example:\n\n"
                        "`Predict for these values: age=30, salary=50000`"
                    ),
                }

        # 2. Model Training requests
        train_phrases = (
            "train best model",
            "train the best model",
            "train a model",
            "train model",
            "run automl",
            "train machine learning model",
            "train classification model",
            "train regression model",
        )
        if any(phrase in normalized for phrase in train_phrases):
            target = None
            target_match = re.search(
                r"(?:for|to predict|target is|target:\s*|target\s*=\s*)\s+([a-zA-Z0-9_\s]+)",
                prompt,
                flags=re.IGNORECASE,
            )
            if target_match:
                candidate = target_match.group(1).strip()
                for col in app_state.dataset.columns:
                    if (
                        col == candidate
                        or col.lower() == candidate.lower()
                        or col.lower().replace(" ", "_") == candidate.lower().replace(" ", "_")
                    ):
                        target = col
                        break

            result = self._execute_model_training({"target_column": target})
            if result.get("success"):
                return {
                    "type": "text",
                    "content": result["answer"],
                    "operation": "model_training",
                    "evidence": result,
                }
            else:
                return {
                    "type": "text",
                    "content": result.get("error", "Model training could not be completed."),
                }

        # 3. Model Explainability requests
        explain_phrases = (
            "explain model",
            "explain the model",
            "feature importance",
            "important features",
            "most important feature",
            "key drivers",
            "explain prediction",
            "model explanation",
        )
        if any(phrase in normalized for phrase in explain_phrases):
            result = self._execute_model_explanation({})
            if result.get("success"):
                return {
                    "type": "text",
                    "content": result["answer"],
                    "operation": "explain_model",
                    "evidence": result,
                }
            else:
                return {
                    "type": "text",
                    "content": result.get("error", "Model explanation could not be completed."),
                }

        # 4. Preprocessing Plan requests
        preprocessing_phrases = (
            "explain the preprocessing plan",
            "explain preprocessing plan",
            "preprocessing plan",
            "show preprocessing plan",
            "what is the preprocessing plan",
            "preprocessing recommendations",
            "data cleaning plan",
        )
        if any(phrase in normalized for phrase in preprocessing_phrases):
            result = self._execute_preprocessing_plan({})
            if result.get("success"):
                return {
                    "type": "text",
                    "content": result["answer"],
                    "operation": "preprocessing_plan",
                    "evidence": result,
                }
            else:
                return {
                    "type": "text",
                    "content": result.get("error", "Preprocessing plan could not be retrieved."),
                    "operation": "preprocessing_plan",
                    "evidence": result,
                }

        # 5. Data Cleaning requests ("handle them", "clean data", "handle missing values")
        cleaning_phrases = (
            "handle them",
            "handle missing values",
            "handle missing",
            "clean data",
            "clean the data",
            "clean dataset",
            "clean the dataset",
            "fix missing values",
            "impute missing values",
            "preprocess data",
            "preprocess the data",
        )
        if any(phrase == normalized or phrase in normalized for phrase in cleaning_phrases):
            result = self._execute_data_cleaning({})
            if result.get("success"):
                return {
                    "type": "text",
                    "content": result["answer"],
                    "operation": "data_cleaning",
                    "evidence": result,
                }
            return {
                "type": "text",
                "content": result.get("error", "Data cleaning could not be completed."),
            }

        # 6. Geospatial / Map requests ("show these locations on a map", "plot map")
        geo_phrases = (
            "show on map",
            "show these locations on a map",
            "show locations on a map",
            "show on a map",
            "plot on map",
            "plot on a map",
            "map visualization",
            "view on map",
            "show map",
            "show me these locations on a map",
        )
        if any(phrase in normalized for phrase in geo_phrases):
            result = self._execute_geospatial_map({})
            if result.get("success"):
                return {
                    "type": "visualization",
                    "figure": result["figure"],
                    "column1": result["column1"],
                    "column2": result["column2"],
                    "content": result["answer"],
                    "operation": "geospatial_map",
                }
            return {
                "type": "text",
                "content": result.get("error", "Geospatial mapping could not be completed."),
            }

        return None

    # ========================================================
    # JSON EXTRACTION
    # ========================================================

    @staticmethod
    def _extract_json(
        content: str,
    ) -> dict[str, Any] | None:
        """
        Extract JSON from Qwen output.
        """

        if not content:

            return None

        cleaned = (
            content
            .strip()
        )

        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

        cleaned = cleaned.strip()

        try:

            parsed = json.loads(
                cleaned
            )

            if isinstance(
                parsed,
                dict,
            ):

                return parsed

        except json.JSONDecodeError:

            pass

        start = cleaned.find(
            "{"
        )

        end = cleaned.rfind(
            "}"
        )

        if (
            start == -1
            or end <= start
        ):

            return None

        try:

            parsed = json.loads(
                cleaned[
                    start:
                    end + 1
                ]
            )

            if isinstance(
                parsed,
                dict,
            ):

                return parsed

        except json.JSONDecodeError:

            return None

        return None

    # ========================================================
    # TOOL RESULT CLEANUP
    # ========================================================

    @staticmethod
    def _tool_result_text(
        result: dict[str, Any],
    ) -> str:
        """
        Convert tool result to compact JSON for Qwen.
        """

        cleaned = dict(
            result
        )

        cleaned.pop(
            "dataframe",
            None,
        )

        cleaned.pop(
            "figure",
            None,
        )

        return json.dumps(
            cleaned,
            default=str,
        )

    # ========================================================
    # FINAL QWEN ANSWER
    # ========================================================

    def _generate_final_answer(
        self,
        prompt: str,
        history: list[dict[str, str]],
        tool_result: dict[str, Any],
    ) -> str:
        """
        Ask Qwen to explain verified evidence.
        """

        evidence = (
            self._tool_result_text(
                tool_result
            )
        )

        messages = [
            {
                "role": "system",
                "content": f"""
You are the final answer generator for DataMindAI.

The user asked:

{prompt}

A Python data-analysis tool has already executed
the requested operation.

VERIFIED EVIDENCE:

{evidence}

Answer the user's question using ONLY the verified
evidence.

Rules:

1. Do not invent numbers.
2. Do not modify numerical values.
3. Do not perform another calculation.
4. Explain the result clearly.
5. Mention relevant column names where useful.
6. If there are multiple results, summarize them clearly.
7. If the tool returned an error, explain the error.
8. Never claim unavailable information exists.
""",
            }
        ]

        for message in history[-6:]:

            messages.append(
                {
                    "role": message["role"],
                    "content": message["content"],
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        try:

            from ollama import chat as ollama_chat

            response = ollama_chat(
                model=self.model,
                messages=messages,
            )

            return (
                response["message"]["content"]
                .strip()
            )

        except Exception as error:

            logger.exception(
                "Final Qwen response failed"
            )

            if tool_result.get(
                "answer"
            ):

                return tool_result[
                    "answer"
                ]

            return (
                "The analysis was completed, but "
                "I could not generate the final explanation."
            )

    # ========================================================
    # NATIVE TOOL CALLING
    # ========================================================

    def _chat_with_tools(
        self,
        prompt: str,
        history: list[dict[str, str]],
    ) -> dict[str, Any]:
        """
        Execute Qwen native tool-calling workflow.
        """

        from ollama import chat as ollama_chat

        messages = [
            {
                "role": "system",
                "content": self._system_prompt(),
            }
        ]

        for message in history:

            messages.append(
                {
                    "role": message["role"],
                    "content": message["content"],
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        try:
            response = ollama_chat(
                model=self.model,
                messages=messages,
                tools=self._tools(),
            )
        except Exception as exc:
            logger.warning(
                "Ollama chat call failed: %s. Using deterministic fallback.",
                exc,
            )
            fallback = self._try_deterministic_ml_qa(prompt)
            if fallback is not None:
                return fallback
            return {
                "type": "text",
                "content": f"I couldn't reach the local language model. {exc}",
            }

        message = response.get(
            "message",
            {},
        )

        tool_calls = (
            message.get(
                "tool_calls"
            )
            or []
        )

        # ----------------------------------------------------
        # No tool call.
        # ----------------------------------------------------

        if not tool_calls:

            content = (
                message.get(
                    "content",
                    "",
                )
                .strip()
            )

            parsed = self._extract_json(
                content
            )

            if parsed:

                return self._execute_legacy_action(
                    parsed
                )

            return {
                "type": "text",
                "content": content,
            }

        # ----------------------------------------------------
        # We intentionally execute one tool call at a time.
        # This keeps the workflow deterministic and easier
        # to test.
        # ----------------------------------------------------

        tool_call = tool_calls[0]

        function = tool_call.get(
            "function",
            {},
        )

        tool_name = function.get(
            "name"
        )

        arguments = function.get(
            "arguments",
            {},
        )

        if isinstance(
            arguments,
            str,
        ):

            try:

                arguments = json.loads(
                    arguments
                )

            except json.JSONDecodeError:

                arguments = {}

        if not isinstance(
            arguments,
            dict,
        ):

            arguments = {}

        # ====================================================
        # DATA ANALYSIS
        # ====================================================

        if tool_name == "data_analysis":

            tool_result = (
                self._execute_data_analysis(
                    arguments
                )
            )

            if not tool_result.get(
                "success",
                False,
            ):

                return {
                    "type": "text",
                    "content": (
                        "I couldn't complete the "
                        "dataset analysis.\n\n"
                        f"{tool_result.get('error', 'Unknown error.')}"
                    ),
                }

            final_answer = (
                self._generate_final_answer(
                    prompt,
                    history,
                    tool_result,
                )
            )

            # ------------------------------------------------
            # Compatibility aliases.
            # ------------------------------------------------

            operation = tool_result.get(
                "operation"
            )

            aliases = {
                "statistics": {
                    "mean": "mean",
                    "median": "median",
                    "sum": "sum",
                    "min": "minimum",
                    "max": "maximum",
                    "std": "std",
                    "count": "count",
                },
                "unique_count": {
                    "unique_values": "unique_values",
                },
            }

            normalized_operation = operation

            if operation == "statistics":

                aggregation = arguments.get(
                    "aggregation",
                    "mean",
                )

                normalized_operation = (
                    aliases["statistics"].get(
                        aggregation,
                        "statistics",
                    )
                )

            elif operation == "unique_count":

                normalized_operation = (
                    "unique_values"
                )

            return {
                "type": "text",
                "content": final_answer,
                "operation": normalized_operation,
                "evidence": tool_result.get(
                    "evidence",
                    {},
                ),
            }

        # ====================================================
        # VISUALIZATION
        # ====================================================

        if tool_name == "visualize":

            visualization_result = (
                self._execute_visualization(
                    arguments
                )
            )

            if not visualization_result.get(
                "success",
                False,
            ):

                return {
                    "type": "text",
                    "content": (
                        "I couldn't generate the "
                        "visualization.\n\n"
                        f"{visualization_result.get('error', 'Unknown error.')}"
                    ),
                }

            return {
                "type": "visualization",
                "figure": visualization_result[
                    "figure"
                ],
                "column1": visualization_result[
                    "column1"
                ],
                "column2": visualization_result[
                    "column2"
                ],
            }

        # ====================================================
        # PREDICTION
        # ====================================================

        if tool_name == "predict":

            pred_result = self._execute_prediction(
                arguments
            )

            if not pred_result.get("success", False):
                return {
                    "type": "text",
                    "content": pred_result.get("error", "Prediction failed."),
                }

            return {
                "type": "text",
                "content": pred_result["answer"],
                "operation": "prediction",
                "evidence": pred_result,
            }

        # ====================================================
        # TRAIN MODEL
        # ====================================================

        if tool_name == "train_model":

            train_result = self._execute_model_training(
                arguments
            )

            if not train_result.get("success", False):
                return {
                    "type": "text",
                    "content": train_result.get("error", "Model training failed."),
                }

            return {
                "type": "text",
                "content": train_result["answer"],
                "operation": "model_training",
                "evidence": train_result,
            }

        # ====================================================
        # EXPLAIN MODEL
        # ====================================================

        if tool_name == "explain_model":

            explain_result = self._execute_model_explanation(
                arguments
            )

            if not explain_result.get("success", False):
                return {
                    "type": "text",
                    "content": explain_result.get("error", "Model explanation failed."),
                }

            return {
                "type": "text",
                "content": explain_result["answer"],
                "operation": "explain_model",
                "evidence": explain_result,
            }

        # ====================================================
        # PREPROCESSING PLAN
        # ====================================================

        if tool_name == "get_preprocessing_plan":

            prep_result = self._execute_preprocessing_plan(
                arguments
            )

            if not prep_result.get("success", False):
                return {
                    "type": "text",
                    "content": prep_result.get("error", "Could not retrieve preprocessing plan."),
                }

            return {
                "type": "text",
                "content": prep_result["answer"],
                "operation": "preprocessing_plan",
                "evidence": prep_result,
            }

        # ====================================================
        # DATA CLEANING
        # ====================================================

        if tool_name == "clean_data":

            clean_res = self._execute_data_cleaning(arguments)
            if not clean_res.get("success", False):
                return {
                    "type": "text",
                    "content": clean_res.get("error", "Data cleaning failed."),
                }
            return {
                "type": "text",
                "content": clean_res["answer"],
                "operation": "data_cleaning",
                "evidence": clean_res,
            }

        # ====================================================
        # GEOSPATIAL MAP
        # ====================================================

        if tool_name == "geospatial_map":

            geo_res = self._execute_geospatial_map(arguments)
            if not geo_res.get("success", False):
                return {
                    "type": "text",
                    "content": geo_res.get("error", "Geospatial mapping failed."),
                }
            return {
                "type": "visualization",
                "figure": geo_res["figure"],
                "column1": geo_res["column1"],
                "column2": geo_res["column2"],
                "content": geo_res["answer"],
                "operation": "geospatial_map",
            }

        return {
            "type": "text",
            "content": (
                f"I don't know how to execute "
                f"the requested tool '{tool_name}'."
            ),
        }

    # ========================================================
    # COMPATIBILITY METHOD
    # ========================================================

    def _chat_with_qwen(
        self,
        prompt: str,
        history: list[dict[str, str]],
    ) -> dict[str, Any]:
        """
        Compatibility wrapper around the new Qwen
        tool-calling architecture.

        Keeping this method is important because the
        existing project tests and older internal code
        reference _chat_with_qwen().
        """

        return self._chat_with_tools(
            prompt,
            history,
        )

    # ========================================================
    # LEGACY ACTION
    # ========================================================

    def _execute_legacy_action(
        self,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Support the previous JSON action format.
        """

        action = result.get(
            "action"
        )

        if action == "visualize":

            return self._execute_visualization(
                {
                    "column1": result.get(
                        "column1"
                    ),
                    "column2": result.get(
                        "column2"
                    ),
                }
            ) | {
                "type": "visualization"
            }

        if action == "chat":

            return {
                "type": "text",
                "content": result.get(
                    "response",
                    "I could not generate a response.",
                ),
            }

        return {
            "type": "text",
            "content": (
                "I couldn't determine what action "
                "to perform."
            ),
        }

    # ========================================================
    # PUBLIC CHAT
    # ========================================================

    def chat(
        self,
        prompt: str,
        history: list[dict[str, str]],
    ) -> dict[str, Any]:
        """
        Main entry point.

        Routing:

        1. Empty question
        2. Dataset validation
        3. Visualization requests → Qwen
        4. Legacy DatasetQA when it recognizes the question
        5. New Qwen tool-calling architecture
        """

        if not prompt or not prompt.strip():

            return {
                "type": "text",
                "content": (
                    "Please enter a question."
                ),
            }

        if not app_state.has_dataset():

            return {
                "type": "text",
                "content": (
                    "Please upload a dataset first."
                ),
            }

        # ----------------------------------------------------
        # Visualization must go directly through Qwen.
        #
        # This also preserves the existing test contract
        # where _chat_with_qwen() is monkey-patched.
        # ----------------------------------------------------

        normalized = (
            prompt
            .strip()
            .lower()
        )

        visualization_words = (
            "visualize",
            "visualise",
            "visualization",
            "visualisation",
            "plot",
            "graph",
            "chart",
            "trend",
            "show the relationship",
            "show relationship",
        )

        is_visualization = any(
            word in normalized
            for word in visualization_words
        )

        if is_visualization:

            return self._chat_with_qwen(
                prompt,
                history,
            )

        # ----------------------------------------------------
        # Legacy DatasetQA compatibility.
        #
        # DatasetQA only handles the operations it already
        # understands. Unsupported questions return None
        # and continue to the new Qwen tool layer.
        # ----------------------------------------------------

        legacy_response = (
            self._try_legacy_dataset_qa(
                prompt
            )
        )

        if legacy_response is not None:

            return legacy_response

        # ----------------------------------------------------
        # Deterministic ML, Prediction, Explainability, and
        # Preprocessing plan intent recognition.
        # ----------------------------------------------------

        deterministic_ml = (
            self._try_deterministic_ml_qa(
                prompt
            )
        )

        if deterministic_ml is not None:

            return deterministic_ml

        # ----------------------------------------------------
        # New AI tool-calling architecture.
        #
        # IMPORTANT:
        # call the compatibility method rather than
        # _chat_with_tools() directly so existing tests
        # can safely mock _chat_with_qwen().
        # ----------------------------------------------------

        return self._chat_with_qwen(
            prompt,
            history,
        )
