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
import re
from typing import Any

from src.backend.dataset_qa import DatasetQA
from src.core.app_state import app_state
from src.core.logger import get_logger
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
                        },
                        "column1": {
                            "type": "string",
                        },
                        "column2": {
                            "type": "string",
                        },
                        "group_column": {
                            "type": "string",
                        },
                        "value_column": {
                            "type": "string",
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
    # TOOLS
    # ========================================================

    def _tools(self) -> list[dict[str, Any]]:
        """
        Return tools available to Qwen.
        """

        return [
            self._data_analysis_tool_schema(),
            self._visualization_tool_schema(),
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

        response = ollama_chat(
            model=self.model,
            messages=messages,
            tools=self._tools(),
        )

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