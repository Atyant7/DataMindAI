import unittest

import pandas as pd

from src.agent.agent import (
    DataMindAgent,
)
from src.core.app_state import (
    app_state,
)


class TestAgentDatasetQA(
    unittest.TestCase
):

    def setUp(self):

        self.original_dataset = (
            app_state.dataset
        )

        self.original_profile = (
            app_state.dataset_profile
        )

        self.original_plan = (
            app_state.preprocessing_plan
        )

        self.original_agent = (
            app_state.agent
        )

        self.original_qa_result = (
            getattr(
                app_state,
                "last_qa_result",
                None,
            )
        )

        self.df = pd.DataFrame(
            {
                "age": [
                    20,
                    30,
                    40,
                    50,
                    60,
                ],
                "salary": [
                    20000,
                    30000,
                    40000,
                    50000,
                    60000,
                ],
                "department": [
                    "IT",
                    "HR",
                    "IT",
                    "Sales",
                    "HR",
                ],
            }
        )

        app_state.dataset = (
            self.df
        )

    def tearDown(self):

        app_state.dataset = (
            self.original_dataset
        )

        app_state.dataset_profile = (
            self.original_profile
        )

        app_state.preprocessing_plan = (
            self.original_plan
        )

        app_state.agent = (
            self.original_agent
        )

        app_state.last_qa_result = (
            self.original_qa_result
        )

    # ========================================================
    # DATASET QA ROUTING
    # ========================================================

    def test_average_salary_uses_dataset_qa(self):

        agent = DataMindAgent()

        response = agent.chat(
            "What is the average salary?",
            [],
        )

        self.assertEqual(
            response["type"],
            "text",
        )

        self.assertEqual(
            response["operation"],
            "mean",
        )

        self.assertEqual(
            response["evidence"]["column"],
            "salary",
        )

        self.assertEqual(
            response["evidence"]["value"],
            40000.0,
        )

    # ========================================================
    # UNIQUE VALUES
    # ========================================================

    def test_unique_values_use_dataset_qa(self):

        agent = DataMindAgent()

        response = agent.chat(
            "How many unique departments are there?",
            [],
        )

        self.assertEqual(
            response["type"],
            "text",
        )

        self.assertEqual(
            response["operation"],
            "unique_values",
        )

        self.assertEqual(
            response["evidence"]["column"],
            "department",
        )

        self.assertEqual(
            response["evidence"][
                "unique_count"
            ],
            3,
        )

    # ========================================================
    # GROUPBY
    # ========================================================

    def test_groupby_uses_dataset_qa(self):

        agent = DataMindAgent()

        response = agent.chat(
            "What is the average salary for each department?",
            [],
        )

        self.assertEqual(
            response["type"],
            "text",
        )

        self.assertEqual(
            response["operation"],
            "groupby",
        )

        self.assertEqual(
            response["evidence"][
                "group_column"
            ],
            "department",
        )

        self.assertEqual(
            response["evidence"][
                "value_column"
            ],
            "salary",
        )

    # ========================================================
    # VISUALIZATION MUST NOT BE ROUTED TO DATASET QA
    # ========================================================

    def test_visualization_request_is_not_routed_to_dataset_qa(
        self
    ):

        agent = DataMindAgent()

        # We don't want the test to require Ollama.
        # Replace _chat_with_qwen temporarily.
        original_method = (
            agent._chat_with_qwen
        )

        try:

            agent._chat_with_qwen = (
                lambda prompt, history: {
                    "type": "visualization",
                    "figure": None,
                    "column1": "age",
                    "column2": "salary",
                }
            )

            response = agent.chat(
                "Show me a chart of age and salary.",
                [],
            )

        finally:

            agent._chat_with_qwen = (
                original_method
            )

        self.assertEqual(
            response["type"],
            "visualization",
        )

        self.assertEqual(
            response["column1"],
            "age",
        )

        self.assertEqual(
            response["column2"],
            "salary",
        )

    # ========================================================
    # UNSUPPORTED QUESTION FALLBACK
    # ========================================================

    def test_unsupported_question_falls_back_to_qwen(
        self
    ):

        agent = DataMindAgent()

        original_method = (
            agent._chat_with_qwen
        )

        try:

            agent._chat_with_qwen = (
                lambda prompt, history: {
                    "type": "text",
                    "content": "Qwen handled this question.",
                }
            )

            response = agent.chat(
                "Explain why preprocessing is important.",
                [],
            )

        finally:

            agent._chat_with_qwen = (
                original_method
            )

        self.assertEqual(
            response["type"],
            "text",
        )

        self.assertEqual(
            response["content"],
            "Qwen handled this question.",
        )


if __name__ == "__main__":
    unittest.main()