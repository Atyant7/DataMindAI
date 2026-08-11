import unittest

import numpy as np
import pandas as pd

from src.ml.task_detector import MLTask
from src.ml.trainer import (
    TrainingConfig,
    predict,
    predict_proba,
    train_model,
    train_models,
)


class TestTrainer(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

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

    def test_train_logistic_regression(self):

        X = self.classification_df.drop(
            columns=["churn"]
        )

        y = self.classification_df["churn"]

        result = train_model(
            X=X,
            y=y,
            model_name="logistic_regression",
            task=MLTask.CLASSIFICATION,
            target_column="churn",
        )

        self.assertEqual(
            result.model_name,
            "logistic_regression",
        )

        self.assertEqual(
            result.train_rows + result.test_rows,
            len(X),
        )

        predictions = predict(
            result,
            result.X_test,
        )

        self.assertEqual(
            len(predictions),
            len(result.X_test),
        )

    def test_train_random_forest(self):

        X = self.classification_df.drop(
            columns=["churn"]
        )

        y = self.classification_df["churn"]

        result = train_model(
            X=X,
            y=y,
            model_name="random_forest",
            task=MLTask.CLASSIFICATION,
            target_column="churn",
        )

        predictions = predict(
            result,
            result.X_test,
        )

        self.assertEqual(
            len(predictions),
            len(result.X_test),
        )

    def test_train_xgboost(self):

        X = self.classification_df.drop(
            columns=["churn"]
        )

        y = self.classification_df["churn"]

        result = train_model(
            X=X,
            y=y,
            model_name="xgboost",
            task=MLTask.CLASSIFICATION,
            target_column="churn",
        )

        predictions = predict(
            result,
            result.X_test,
        )

        self.assertEqual(
            len(predictions),
            len(result.X_test),
        )

    def test_train_lightgbm(self):

        X = self.classification_df.drop(
            columns=["churn"]
        )

        y = self.classification_df["churn"]

        result = train_model(
            X=X,
            y=y,
            model_name="lightgbm",
            task=MLTask.CLASSIFICATION,
            target_column="churn",
        )

        predictions = predict(
            result,
            result.X_test,
        )

        self.assertEqual(
            len(predictions),
            len(result.X_test),
        )

    def test_classification_probability(self):

        X = self.classification_df.drop(
            columns=["churn"]
        )

        y = self.classification_df["churn"]

        result = train_model(
            X=X,
            y=y,
            model_name="logistic_regression",
            task=MLTask.CLASSIFICATION,
            target_column="churn",
        )

        probabilities = predict_proba(
            result,
            result.X_test,
        )

        self.assertEqual(
            probabilities.shape[0],
            len(result.X_test),
        )

        self.assertEqual(
            probabilities.shape[1],
            2,
        )

    def test_train_linear_regression(self):

        X = self.regression_df.drop(
            columns=["price"]
        )

        y = self.regression_df["price"]

        result = train_model(
            X=X,
            y=y,
            model_name="linear_regression",
            task=MLTask.REGRESSION,
            target_column="price",
        )

        predictions = predict(
            result,
            result.X_test,
        )

        self.assertEqual(
            len(predictions),
            len(result.X_test),
        )

    def test_train_regression_xgboost(self):

        X = self.regression_df.drop(
            columns=["price"]
        )

        y = self.regression_df["price"]

        result = train_model(
            X=X,
            y=y,
            model_name="xgboost",
            task=MLTask.REGRESSION,
            target_column="price",
        )

        predictions = predict(
            result,
            result.X_test,
        )

        self.assertEqual(
            len(predictions),
            len(result.X_test),
        )

    def test_train_multiple_models(self):

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

        self.assertEqual(
            set(results.keys()),
            {
                "logistic_regression",
                "random_forest",
                "xgboost",
                "lightgbm",
            },
        )

        for result in results.values():

            self.assertGreater(
                result.train_rows,
                0,
            )

            self.assertGreater(
                result.test_rows,
                0,
            )

    def test_missing_target_values_are_rejected(self):

        X = pd.DataFrame({
            "age": [20, 25, 30, 35],
        })

        y = pd.Series([
            0,
            1,
            None,
            1,
        ])

        with self.assertRaises(ValueError):

            train_model(
                X=X,
                y=y,
                model_name="random_forest",
                task=MLTask.CLASSIFICATION,
                target_column="target",
            )

    def test_invalid_test_size(self):

        X = self.classification_df.drop(
            columns=["churn"]
        )

        y = self.classification_df["churn"]

        config = TrainingConfig(
            test_size=1.5
        )

        with self.assertRaises(ValueError):

            train_model(
                X=X,
                y=y,
                model_name="random_forest",
                task=MLTask.CLASSIFICATION,
                target_column="churn",
                config=config,
            )


if __name__ == "__main__":
    unittest.main()