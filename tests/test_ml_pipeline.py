import unittest

import pandas as pd

from src.ml.ml_pipeline import (
    MLPipelineResult,
    run_ml_pipeline,
)
from src.ml.task_detector import MLTask


class TestMLPipeline(unittest.TestCase):

    def setUp(self):

        self.classification_df = pd.DataFrame({
            "age": [
                20, 22, 25, 27, 30,
                32, 35, 37, 40, 42,
                45, 47, 50, 52, 55,
                57, 60, 62, 65, 67,
            ],
            "income": [
                20, 22, 25, 27, 30,
                32, 35, 37, 40, 42,
                45, 47, 50, 52, 55,
                57, 60, 62, 65, 67,
            ],
            "city": [
                "Delhi",
                "Mumbai",
                "Delhi",
                "Pune",
                "Mumbai",
                "Delhi",
                "Pune",
                "Mumbai",
                "Delhi",
                "Pune",
                "Mumbai",
                "Delhi",
                "Pune",
                "Mumbai",
                "Delhi",
                "Pune",
                "Mumbai",
                "Delhi",
                "Pune",
                "Mumbai",
            ],
            "churn": [
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 1,
            ],
        })

    def test_complete_pipeline(self):

        result = run_ml_pipeline(
            self.classification_df,
            "churn",
        )

        self.assertIsInstance(
            result,
            MLPipelineResult,
        )

        self.assertEqual(
            result.task,
            MLTask.CLASSIFICATION,
        )

        self.assertEqual(
            result.target_column,
            "churn",
        )

        self.assertGreater(
            len(result.training_results),
            0,
        )

        self.assertEqual(
            set(result.training_results.keys()),
            set(result.evaluation_results.keys()),
        )

        self.assertGreater(
            len(result.leaderboard),
            0,
        )

        self.assertIsNotNone(
            result.best_model_name
        )

        self.assertIsNotNone(
            result.best_model_display_name
        )

    def test_selected_model_exists(self):

        result = run_ml_pipeline(
            self.classification_df,
            "churn",
        )

        self.assertIn(
            result.best_model_name,
            result.training_results,
        )

        self.assertIn(
            result.best_model_name,
            result.evaluation_results,
        )

    def test_comparison_table_exists(self):

        result = run_ml_pipeline(
            self.classification_df,
            "churn",
        )

        self.assertFalse(
            result.comparison_table.empty
        )

        self.assertIn(
            "model",
            result.comparison_table.columns,
        )

        self.assertIn(
            "primary_score",
            result.comparison_table.columns,
        )

    def test_invalid_target(self):

        with self.assertRaises(ValueError):

            run_ml_pipeline(
                self.classification_df,
                "does_not_exist",
            )

    def test_empty_dataset(self):

        with self.assertRaises(ValueError):

            run_ml_pipeline(
                pd.DataFrame(),
                "target",
            )

    def test_custom_models(self):

        result = run_ml_pipeline(
            self.classification_df,
            "churn",
            model_names=[
                "random_forest",
                "xgboost",
            ],
        )

        self.assertEqual(
            set(result.training_results.keys()),
            {
                "random_forest",
                "xgboost",
            },
        )


if __name__ == "__main__":
    unittest.main()