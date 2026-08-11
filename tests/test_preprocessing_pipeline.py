import unittest

import numpy as np
import pandas as pd

from src.ml.preprocessing_pipeline import (
    build_preprocessing_pipeline,
    detect_feature_columns,
    fit_preprocessing_pipeline,
    fit_transform_training_data,
    model_requires_scaling,
    transform_features,
)

from src.ml.task_detector import MLTask


class TestPreprocessingPipeline(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "age": [20, 25, 30, 35, 40, 45],
            "income": [
                20000,
                30000,
                40000,
                50000,
                60000,
                70000,
            ],
            "city": [
                "Delhi",
                "Mumbai",
                "Delhi",
                "Pune",
                "Mumbai",
                "Pune",
            ],
        })

    def test_detect_feature_columns(self):
        columns = detect_feature_columns(
            self.df
        )

        self.assertEqual(
            columns.numeric,
            ["age", "income"],
        )

        self.assertEqual(
            columns.categorical,
            ["city"],
        )

        self.assertEqual(
            columns.datetime,
            [],
        )

    def test_scaling_for_logistic_regression(self):
        self.assertTrue(
            model_requires_scaling(
                "logistic_regression"
            )
        )

    def test_scaling_for_linear_regression(self):
        self.assertTrue(
            model_requires_scaling(
                "linear_regression"
            )
        )

    def test_no_scaling_for_random_forest(self):
        self.assertFalse(
            model_requires_scaling(
                "random_forest"
            )
        )

    def test_no_scaling_for_xgboost(self):
        self.assertFalse(
            model_requires_scaling(
                "xgboost"
            )
        )

    def test_no_scaling_for_lightgbm(self):
        self.assertFalse(
            model_requires_scaling(
                "lightgbm"
            )
        )

    def test_mixed_pipeline(self):
        result = build_preprocessing_pipeline(
            X=self.df,
            task=MLTask.CLASSIFICATION,
            model_name="logistic_regression",
        )

        self.assertEqual(
            result.feature_columns.numeric,
            ["age", "income"],
        )

        self.assertEqual(
            result.feature_columns.categorical,
            ["city"],
        )

        self.assertTrue(
            result.scaling_applied
        )

    def test_missing_values(self):
        df = pd.DataFrame({
            "age": [20, None, 30, 35],
            "income": [20000, 30000, None, 50000],
            "city": [
                "Delhi",
                None,
                "Mumbai",
                "Pune",
            ],
        })

        result = build_preprocessing_pipeline(
            X=df,
            task=MLTask.CLASSIFICATION,
            model_name="logistic_regression",
        )

        transformed = (
            fit_transform_training_data(
                result,
                df,
            )
        )

        self.assertEqual(
            transformed.shape[0],
            4,
        )

        self.assertFalse(
            np.isnan(
                transformed.toarray()
                if hasattr(transformed, "toarray")
                else transformed
            ).any()
        )

    def test_transform_test_data(self):
        train = self.df.iloc[:4].copy()
        test = self.df.iloc[4:].copy()

        result = build_preprocessing_pipeline(
            X=train,
            task=MLTask.CLASSIFICATION,
            model_name="logistic_regression",
        )

        fit_preprocessing_pipeline(
            result,
            train,
        )

        transformed_test = transform_features(
            result,
            test,
        )

        self.assertEqual(
            transformed_test.shape[0],
            len(test),
        )

    def test_unknown_category_is_supported(self):
        train = pd.DataFrame({
            "age": [20, 25, 30],
            "city": [
                "Delhi",
                "Mumbai",
                "Pune",
            ],
        })

        test = pd.DataFrame({
            "age": [40],
            "city": ["Bangalore"],
        })

        result = build_preprocessing_pipeline(
            X=train,
            task=MLTask.CLASSIFICATION,
            model_name="logistic_regression",
        )

        fit_preprocessing_pipeline(
            result,
            train,
        )

        transformed = transform_features(
            result,
            test,
        )

        self.assertEqual(
            transformed.shape[0],
            1,
        )

    def test_datetime_is_excluded(self):
        df = pd.DataFrame({
            "age": [20, 25, 30],
            "signup_date": pd.to_datetime([
                "2025-01-01",
                "2025-01-02",
                "2025-01-03",
            ]),
        })

        result = build_preprocessing_pipeline(
            X=df,
            task=MLTask.CLASSIFICATION,
            model_name="random_forest",
        )

        self.assertIn(
            "signup_date",
            result.dropped_columns,
        )

        self.assertTrue(
            any(
                "Datetime columns" in warning
                for warning in result.warnings
            )
        )

    def test_empty_dataframe(self):
        df = pd.DataFrame()

        with self.assertRaises(ValueError):
            build_preprocessing_pipeline(
                X=df,
                task=MLTask.CLASSIFICATION,
                model_name="random_forest",
            )


if __name__ == "__main__":
    unittest.main()