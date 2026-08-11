import unittest

import pandas as pd

from src.ml.evaluator import (
    create_comparison_table,
    evaluate_model,
    evaluate_models,
)
from src.ml.task_detector import MLTask
from src.ml.trainer import train_models


class TestEvaluator(unittest.TestCase):

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

        self.regression_df = pd.DataFrame({
            "area": [
                500, 600, 700, 800, 900,
                1000, 1100, 1200, 1300, 1400,
                1500, 1600, 1700, 1800, 1900,
                2000, 2100, 2200, 2300, 2400,
            ],
            "rooms": [
                1, 1, 2, 2, 2,
                3, 3, 3, 3, 4,
                4, 4, 5, 5, 5,
                6, 6, 6, 7, 7,
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
            "price": [
                100000,
                120000,
                140000,
                160000,
                180000,
                200000,
                220000,
                240000,
                260000,
                280000,
                300000,
                320000,
                340000,
                360000,
                380000,
                400000,
                420000,
                440000,
                460000,
                480000,
            ],
        })

    def test_classification_evaluation(self):

        X = self.classification_df.drop(
            columns=["churn"]
        )

        y = self.classification_df["churn"]

        results = train_models(
            X=X,
            y=y,
            task=MLTask.CLASSIFICATION,
            target_column="churn",
        )

        result = evaluate_model(
            results["random_forest"]
        )

        self.assertEqual(
            result.task,
            MLTask.CLASSIFICATION,
        )

        self.assertIn(
            "accuracy",
            result.metrics,
        )

        self.assertIn(
            "precision",
            result.metrics,
        )

        self.assertIn(
            "recall",
            result.metrics,
        )

        self.assertIn(
            "f1",
            result.metrics,
        )

        self.assertIsNotNone(
            result.confusion_matrix
        )

        self.assertIsNotNone(
            result.classification_report
        )

        self.assertGreaterEqual(
            result.primary_score,
            0.0,
        )

        self.assertLessEqual(
            result.primary_score,
            1.0,
        )

    def test_regression_evaluation(self):

        X = self.regression_df.drop(
            columns=["price"]
        )

        y = self.regression_df["price"]

        results = train_models(
            X=X,
            y=y,
            task=MLTask.REGRESSION,
            target_column="price",
        )

        result = evaluate_model(
            results["random_forest"]
        )

        self.assertEqual(
            result.task,
            MLTask.REGRESSION,
        )

        self.assertIn(
            "mae",
            result.metrics,
        )

        self.assertIn(
            "mse",
            result.metrics,
        )

        self.assertIn(
            "rmse",
            result.metrics,
        )

        self.assertIn(
            "r2",
            result.metrics,
        )

        self.assertEqual(
            result.primary_metric,
            "rmse",
        )

        self.assertGreaterEqual(
            result.primary_score,
            0.0,
        )

    def test_evaluate_multiple_models(self):

        X = self.classification_df.drop(
            columns=["churn"]
        )

        y = self.classification_df["churn"]

        training_results = train_models(
            X=X,
            y=y,
            task=MLTask.CLASSIFICATION,
            target_column="churn",
        )

        evaluation_results = evaluate_models(
            training_results
        )

        self.assertEqual(
            set(evaluation_results.keys()),
            set(training_results.keys()),
        )

        for result in evaluation_results.values():

            self.assertIsNotNone(
                result.metrics
            )

    def test_comparison_table(self):

        X = self.classification_df.drop(
            columns=["churn"]
        )

        y = self.classification_df["churn"]

        training_results = train_models(
            X=X,
            y=y,
            task=MLTask.CLASSIFICATION,
            target_column="churn",
        )

        evaluation_results = evaluate_models(
            training_results
        )

        table = create_comparison_table(
            evaluation_results
        )

        self.assertEqual(
            len(table),
            len(evaluation_results),
        )

        self.assertIn(
            "model",
            table.columns,
        )

        self.assertIn(
            "primary_metric",
            table.columns,
        )

        self.assertIn(
            "primary_score",
            table.columns,
        )


if __name__ == "__main__":
    unittest.main()