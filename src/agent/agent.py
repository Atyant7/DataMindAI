from ollama import chat
import json

from src.core.app_state import app_state
from src.visualisation.visualisation_engine import VisualizationEngine


class DataMindAgent:

    def __init__(self):
        self.model = "qwen3:8b"

    def _system_prompt(self):

        profile = app_state.dataset_profile
        plan = app_state.preprocessing_plan
        columns = list(app_state.dataset.columns)

        return f"""
You are DataMindAI, an autonomous AI Data Scientist.

You have access to the user's dataset.

Your job is to understand the user's natural-language request and decide
what DataMindAI should do.

==================================================
AVAILABLE DATASET COLUMNS
==================================================

{columns}

==================================================
DATASET INFORMATION
==================================================

Dataset Name:
{profile.dataset_name}

Rows:
{profile.rows}

Columns:
{profile.columns}

Health Score:
{profile.health_score}

Health Status:
{profile.health_status}

Missing Values:
{profile.missing_values}

Duplicate Rows:
{profile.duplicate_rows}

Numerical Columns:
{profile.numerical_columns}

Categorical Columns:
{profile.categorical_columns}

Boolean Columns:
{profile.boolean_columns}

Datetime Columns:
{profile.datetime_columns}

==================================================
TARGET ANALYSIS
==================================================

{profile.target_analysis}

==================================================
CORRELATION INSIGHTS
==================================================

{profile.correlation_insights}

==================================================
OUTLIER SUMMARY
==================================================

{profile.outlier_summary}

==================================================
FEATURE QUALITY
==================================================

{profile.feature_quality_summary}

==================================================
PREPROCESSING PLAN
==================================================

Missing Values:
{plan.missing_value_plan}

Encoding:
{plan.encoding_plan}

Scaling:
{plan.scaling_plan}

Feature Selection:
{plan.feature_selection_plan}

Outlier Treatment:
{plan.outlier_treatment_plan}

Train/Test Recommendation:
{plan.train_test_plan}

Pipeline Summary:
{plan.pipeline_summary}

==================================================
COLUMN UNDERSTANDING
==================================================

The user does NOT need to type exact column names.

You must map natural-language descriptions to the most appropriate
actual dataset columns.

For example, if the dataset contains:

"hiring_rate_pct"
"gdp_growth_us_pct"

and the user asks:

"Show me the relationship between hiring and gdp"

you must identify:

column1 = "hiring_rate_pct"
column2 = "gdp_growth_us_pct"

Other examples:

"salary" can match "annual_salary"

"age" can match "employee_age"

"sales" can match "monthly_sales"

"profit" can match "net_profit_margin"

Always use the exact actual column name in the JSON response.

==================================================
VISUALIZATION
==================================================

If the user asks to visualize, compare, plot, graph, chart,
show a relationship, show a trend, or asks for a visualization
between two concepts:

1. Identify the two most relevant actual dataset columns.
2. Do NOT require the user to provide exact column names.
3. Return the exact dataset column names.
4. Let Python choose the best visualization type.

Return ONLY:

{{
    "action": "visualize",
    "column1": "actual_dataset_column",
    "column2": "actual_dataset_column"
}}

Do NOT decide the chart type yourself.

Python will automatically select the appropriate chart.

==================================================
NORMAL QUESTIONS
==================================================

For normal questions that do not require executing a tool, return:

{{
    "action": "chat",
    "response": "your answer"
}}

==================================================
IMPORTANT
==================================================

Return ONLY valid JSON.

Do not use markdown.

Do not add explanations outside the JSON.

Never invent column names.
"""

    def _execute_action(self, result):

        action = result.get("action")

        # ==================================================
        # VISUALIZATION
        # ==================================================

        if action == "visualize":

            column1 = result.get("column1")
            column2 = result.get("column2")

            if not column1 or not column2:

                return {
                    "type": "text",
                    "content": (
                        "I couldn't determine the two columns "
                        "you want to visualize."
                    )
                }

            try:

                engine = VisualizationEngine(
                    app_state.dataset
                )

                figure = engine.generate(
                    column1,
                    column2
                )

                return {
                    "type": "visualization",
                    "figure": figure,
                    "column1": column1,
                    "column2": column2
                }

            except Exception as error:

                return {
                    "type": "text",
                    "content": (
                        f"I couldn't generate the visualization: "
                        f"{error}"
                    )
                }

        # ==================================================
        # NORMAL CHAT
        # ==================================================

        if action == "chat":

            return {
                "type": "text",
                "content": result.get(
                    "response",
                    "I could not generate a response."
                )
            }

        return {
            "type": "text",
            "content": (
                "I couldn't determine what action to perform."
            )
        }

    def chat(self, prompt, history):

        messages = [
            {
                "role": "system",
                "content": self._system_prompt()
            }
        ]

        for message in history:

            messages.append({
                "role": message["role"],
                "content": message["content"]
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        response = chat(
            model=self.model,
            messages=messages
        )

        content = response["message"]["content"]

        try:

            result = json.loads(content)

        except json.JSONDecodeError:

            return {
                "type": "text",
                "content": content
            }

        return self._execute_action(result)