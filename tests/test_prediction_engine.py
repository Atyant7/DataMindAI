import tempfile
import unittest

import pandas as pd

from src.ml.ml_pipeline import run_ml_pipeline
from src.ml.model_persistence import (
    save_best_model,
)
from src.ml.prediction_engine import (
    PredictionResult,
    predict,
    predict_from_artifact,
)


class TestPredictionEngine(unittest.TestCase):

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

    def _train_model(self, tmp):

        result = run_ml_pipeline(
            self.df,
            "churn",
            model_names=[
                "random_forest",
            ],
        )

        artifact = save_best_model(
            result,
            model_directory=tmp,
        )

        return artifact

    def test_prediction_result(self):

        with tempfile.TemporaryDirectory() as tmp:

            artifact = self._train_model(
                tmp
            )

            input_data = pd.DataFrame({
                "age": [45],
                "income": [45],
                "city": ["Delhi"],
            })

            result = predict(
                model_path=artifact.model_path,
                metadata_path=artifact.metadata_path,
                input_data=input_data,
            )

            self.assertIsInstance(
                result,
                PredictionResult,
            )

            self.assertIn(
                result.prediction,
                [0, 1],
            )

            self.assertEqual(
                result.task,
                "classification",
            )

            self.assertEqual(
                result.target_column,
                "churn",
            )

    def test_prediction_from_artifact(self):

        with tempfile.TemporaryDirectory() as tmp:

            artifact = self._train_model(
                tmp
            )

            input_data = pd.DataFrame({
                "age": [45],
                "income": [45],
                "city": ["Delhi"],
            })

            result = predict_from_artifact(
                artifact,
                input_data,
            )

            self.assertIsInstance(
                result,
                PredictionResult,
            )

            self.assertIsNotNone(
                result.prediction,
            )

    def test_probability_generation(self):

        with tempfile.TemporaryDirectory() as tmp:

            artifact = self._train_model(
                tmp
            )

            input_data = pd.DataFrame({
                "age": [45],
                "income": [45],
                "city": ["Delhi"],
            })

            result = predict(
                artifact.model_path,
                artifact.metadata_path,
                input_data,
            )

            self.assertIsNotNone(
                result.probabilities
            )

            self.assertIsNotNone(
                result.confidence
            )

            self.assertGreaterEqual(
                result.confidence,
                0.0,
            )

            self.assertLessEqual(
                result.confidence,
                1.0,
            )

    def test_missing_feature(self):

        with tempfile.TemporaryDirectory() as tmp:

            artifact = self._train_model(
                tmp
            )

            input_data = pd.DataFrame({
                "age": [45],
                "city": ["Delhi"],
            })

            with self.assertRaises(
                ValueError
            ):

                predict(
                    artifact.model_path,
                    artifact.metadata_path,
                    input_data,
                )

    def test_empty_input(self):

        with tempfile.TemporaryDirectory() as tmp:

            artifact = self._train_model(
                tmp
            )

            input_data = pd.DataFrame()

            with self.assertRaises(
                ValueError
            ):

                predict(
                    artifact.model_path,
                    artifact.metadata_path,
                    input_data,
                )


if __name__ == "__main__":
    unittest.main()