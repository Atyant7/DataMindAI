import unittest

import pandas as pd

from src.ml.evaluator import evaluate_models
from src.ml.model_selector import (
    create_leaderboard,
    default_selection_metric,
    get_selected_model,
    metric_direction,
    normalize_score,
    select_best_model,
)
from src.ml.task_detector import MLTask
from src.ml.trainer import train_models


class TestModelSelector(unittest.TestCase):

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

    def test_default_classification_metric(self):

        self.assertEqual(
            default_selection_metric(
                MLTask.CLASSIFICATION
            ),
            "f1",
        )

    def test_default_regression_metric(self):

        self.assertEqual(
            default_selection_metric(
                MLTask.REGRESSION
            ),
            "rmse",
        )

    def test_metric_direction(self):

        self.assertEqual(
            metric_direction("f1"),
            "higher",
        )

        self.assertEqual(
            metric_direction("accuracy"),
            "higher",
        )

        self.assertEqual(
            metric_direction("rmse"),
            "lower",
        )

        self.assertEqual(
            metric_direction("mae"),
            "lower",
        )

    def test_normalize_score(self):

        self.assertEqual(
            normalize_score(
                0.90,
                "f1",
            ),
            0.90,
        )

        self.assertEqual(
            normalize_score(
                10.0,
                "rmse",
            ),
            -10.0,
        )

    def test_classification_selection(self):

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

        selection = select_best_model(
            evaluation_results
        )

        self.assertEqual(
            selection.task,
            MLTask.CLASSIFICATION,
        )

        self.assertEqual(
            selection.primary_metric,
            "f1",
        )

        self.assertEqual(
            selection.selected_rank,
            1,
        )

        self.assertGreater(
            len(selection.leaderboard),
            0,
        )

        self.assertEqual(
            selection.leaderboard[0].rank,
            1,
        )

        self.assertEqual(
            selection.selected_model_name,
            selection.leaderboard[0].model_name,
        )

    def test_regression_selection(self):

        X = self.regression_df.drop(
            columns=["price"]
        )

        y = self.regression_df["price"]

        training_results = train_models(
            X=X,
            y=y,
            task=MLTask.REGRESSION,
            target_column="price",
        )

        evaluation_results = evaluate_models(
            training_results
        )

        selection = select_best_model(
            evaluation_results
        )

        self.assertEqual(
            selection.task,
            MLTask.REGRESSION,
        )

        self.assertEqual(
            selection.primary_metric,
            "rmse",
        )

        self.assertEqual(
            selection.selected_rank,
            1,
        )

    def test_custom_metric(self):

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

        selection = select_best_model(
            evaluation_results,
            metric="accuracy",
        )

        self.assertEqual(
            selection.primary_metric,
            "accuracy",
        )

    def test_leaderboard(self):

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

        selection = select_best_model(
            evaluation_results
        )

        leaderboard = create_leaderboard(
            selection
        )

        self.assertEqual(
            len(leaderboard),
            len(selection.leaderboard),
        )

        self.assertIn(
            "rank",
            leaderboard.columns,
        )

        self.assertIn(
            "model",
            leaderboard.columns,
        )

        self.assertIn(
            "primary_score",
            leaderboard.columns,
        )

    def test_get_selected_model(self):

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

        selection = select_best_model(
            evaluation_results
        )

        selected = get_selected_model(
            selection
        )

        self.assertEqual(
            selected,
            selection.selected_model_name,
        )

    def test_empty_results(self):

        with self.assertRaises(ValueError):

            select_best_model({})

    def test_incompatible_metric(self):

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

        with self.assertRaises(ValueError):

            select_best_model(
                evaluation_results,
                metric="rmse",
            )


if __name__ == "__main__":
    unittest.main()