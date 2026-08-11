import unittest

import pandas as pd

from src.ml.explainability import (
    ExplainabilityResult,
    explain_training_result,
)
from src.ml.ml_pipeline import (
    run_ml_pipeline,
)


class TestExplainability(unittest.TestCase):

    def setUp(self):

        self.df = pd.DataFrame({
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

    def test_random_forest_explainability(self):

        X = self.df.drop(
            columns=["churn"]
        )

        y = self.df["churn"]

        result = run_ml_pipeline(
            self.df,
            "churn",
            model_names=[
                "random_forest",
            ],
        )

        training_result = (
            result.training_results[
                result.best_model_name
            ]
        )

        explanation = (
            explain_training_result(
                training_result,
                X=X,
                y=y,
            )
        )

        self.assertIsInstance(
            explanation,
            ExplainabilityResult,
        )

        self.assertGreater(
            len(
                explanation.feature_importance
            ),
            0,
        )

        self.assertEqual(
            explanation.model_name,
            training_result.model_name,
        )

    def test_importance_dataframe(self):

        X = self.df.drop(
            columns=["churn"]
        )

        y = self.df["churn"]

        result = run_ml_pipeline(
            self.df,
            "churn",
            model_names=[
                "random_forest",
            ],
        )

        training_result = (
            result.training_results[
                result.best_model_name
            ]
        )

        explanation = (
            explain_training_result(
                training_result,
                X=X,
                y=y,
            )
        )

        importance_df = (
            explanation.to_dataframe()
        )

        self.assertFalse(
            importance_df.empty
        )

        self.assertIn(
            "feature",
            importance_df.columns,
        )

        self.assertIn(
            "importance",
            importance_df.columns,
        )

        self.assertIn(
            "rank",
            importance_df.columns,
        )

    def test_importance_is_normalized(self):

        X = self.df.drop(
            columns=["churn"]
        )

        y = self.df["churn"]

        result = run_ml_pipeline(
            self.df,
            "churn",
            model_names=[
                "random_forest",
            ],
        )

        training_result = (
            result.training_results[
                result.best_model_name
            ]
        )

        explanation = (
            explain_training_result(
                training_result,
                X=X,
                y=y,
            )
        )

        total_importance = sum(
            item.importance
            for item
            in explanation.feature_importance
        )

        self.assertAlmostEqual(
            total_importance,
            1.0,
            places=5,
        )

    def test_top_features_are_ranked(self):

        X = self.df.drop(
            columns=["churn"]
        )

        y = self.df["churn"]

        result = run_ml_pipeline(
            self.df,
            "churn",
            model_names=[
                "random_forest",
            ],
        )

        training_result = (
            result.training_results[
                result.best_model_name
            ]
        )

        explanation = (
            explain_training_result(
                training_result,
                X=X,
                y=y,
            )
        )

        ranks = [
            item.rank
            for item
            in explanation.feature_importance
        ]

        self.assertEqual(
            ranks,
            list(
                range(
                    1,
                    len(ranks) + 1,
                )
            ),
        )


if __name__ == "__main__":
    unittest.main()