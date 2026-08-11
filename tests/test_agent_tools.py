import unittest
from unittest.mock import patch

import pandas as pd

from src.agent.agent import DataMindAgent
from src.core.app_state import app_state


class TestAgentTools(unittest.TestCase):

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

        self.original_analysis = getattr(
            app_state,
            "last_data_analysis",
            None,
        )

        app_state.dataset = pd.DataFrame(
            {
                "company": [
                    "AMD",
                    "AMD",
                    "Intel",
                    "Intel",
                    "Microsoft",
                ],
                "year": [
                    2020,
                    2021,
                    2020,
                    2021,
                    2021,
                ],
                "revenue": [
                    10.0,
                    15.0,
                    70.0,
                    75.0,
                    180.0,
                ],
                "layoffs": [
                    100,
                    500,
                    1000,
                    2000,
                    500,
                ],
                "employees": [
                    1000,
                    1200,
                    10000,
                    9000,
                    50000,
                ],
            }
        )

        app_state.dataset_name = (
            "test.csv"
        )

        app_state.dataset_profile = None
        app_state.preprocessing_plan = None

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

        app_state.last_data_analysis = (
            self.original_analysis
        )

    # ========================================================
    # DIRECT TOOL EXECUTION
    # ========================================================

    def test_data_analysis_tool_execution(
        self,
    ):

        agent = DataMindAgent()

        result = (
            agent._execute_data_analysis(
                {
                    "operation": "statistics",
                    "column": "revenue",
                    "aggregation": "mean",
                }
            )
        )

        self.assertTrue(
            result["success"]
        )

        self.assertEqual(
            result["operation"],
            "statistics",
        )

        self.assertAlmostEqual(
            result["evidence"]["value"],
            70.0,
        )

    # ========================================================
    # GROUPBY TOOL
    # ========================================================

    def test_groupby_tool_execution(
        self,
    ):

        agent = DataMindAgent()

        result = (
            agent._execute_data_analysis(
                {
                    "operation": "groupby",
                    "group_column": "company",
                    "value_column": "revenue",
                    "aggregation": "mean",
                }
            )
        )

        self.assertTrue(
            result["success"]
        )

        records = (
            result["evidence"]["groups"]
        )

        self.assertEqual(
            records[0]["company"],
            "Microsoft",
        )

    # ========================================================
    # RANK TOOL
    # ========================================================

    def test_rank_tool_execution(
        self,
    ):

        agent = DataMindAgent()

        result = (
            agent._execute_data_analysis(
                {
                    "operation": "rank",
                    "column": "revenue",
                    "group_column": "company",
                    "aggregation": "mean",
                    "order": "descending",
                    "limit": 1,
                }
            )
        )

        self.assertTrue(
            result["success"]
        )

        ranking = (
            result["evidence"]["ranking"]
        )

        self.assertEqual(
            ranking[0]["company"],
            "Microsoft",
        )

    # ========================================================
    # ROW EXTREME
    # ========================================================

    def test_row_extreme_tool_execution(
        self,
    ):

        agent = DataMindAgent()

        result = (
            agent._execute_data_analysis(
                {
                    "operation": "row_extreme",
                    "column": "layoffs",
                    "direction": "max",
                }
            )
        )

        self.assertTrue(
            result["success"]
        )

        self.assertEqual(
            result["evidence"]["value"],
            2000,
        )

        self.assertEqual(
            result["evidence"]["row"]["company"],
            "Intel",
        )

    # ========================================================
    # INVALID COLUMN
    # ========================================================

    def test_invalid_column(
        self,
    ):

        agent = DataMindAgent()

        result = (
            agent._execute_data_analysis(
                {
                    "operation": "statistics",
                    "column": "salary",
                    "aggregation": "mean",
                }
            )
        )

        self.assertFalse(
            result["success"]
        )

    # ========================================================
    # VISUALIZATION VALIDATION
    # ========================================================

    @patch(
        "src.agent.agent.VisualizationEngine"
    )
    def test_visualization_tool_validation(
        self,
        mock_engine,
    ):

        mock_engine.return_value.generate.return_value = (
            "fake_figure"
        )

        agent = DataMindAgent()

        result = (
            agent._execute_visualization(
                {
                    "column1": "year",
                    "column2": "revenue",
                }
            )
        )

        self.assertTrue(
            result["success"]
        )

        self.assertEqual(
            result["column1"],
            "year",
        )

        self.assertEqual(
            result["column2"],
            "revenue",
        )

    # ========================================================
    # INVALID VISUALIZATION COLUMN
    # ========================================================

    def test_invalid_visualization_column(
        self,
    ):

        agent = DataMindAgent()

        result = (
            agent._execute_visualization(
                {
                    "column1": "year",
                    "column2": "salary",
                }
            )
        )

        self.assertFalse(
            result["success"]
        )


if __name__ == "__main__":
    unittest.main()