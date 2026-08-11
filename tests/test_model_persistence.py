import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.ml.evaluator import evaluate_models
from src.ml.ml_pipeline import run_ml_pipeline
from src.ml.model_persistence import (
    load_artifact,
    load_model,
    load_model_metadata,
    save_best_model,
)
from src.ml.task_detector import MLTask


class TestModelPersistence(unittest.TestCase):

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

    def test_save_and_load_best_model(self):

        result = run_ml_pipeline(
            self.df,
            "churn",
            model_names=[
                "random_forest",
                "xgboost",
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:

            artifact = save_best_model(
                result,
                model_directory=tmp,
            )

            self.assertTrue(
                Path(
                    artifact.model_path
                ).exists()
            )

            self.assertTrue(
                Path(
                    artifact.metadata_path
                ).exists()
            )

            model = load_model(
                artifact.model_path
            )

            metadata = load_model_metadata(
                artifact.metadata_path
            )

            self.assertIsNotNone(
                model
            )

            self.assertIsInstance(
                metadata,
                dict,
            )

    def test_loaded_model_predicts(self):

        result = run_ml_pipeline(
            self.df,
            "churn",
            model_names=[
                "random_forest",
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:

            artifact = save_best_model(
                result,
                model_directory=tmp,
            )

            model = load_model(
                artifact.model_path
            )

            X = self.df.drop(
                columns=["churn"]
            )

            predictions = model.predict(
                X
            )

            self.assertEqual(
                len(predictions),
                len(X),
            )

    def test_load_artifact(self):

        result = run_ml_pipeline(
            self.df,
            "churn",
            model_names=[
                "random_forest",
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:

            artifact = save_best_model(
                result,
                model_directory=tmp,
            )

            loaded_artifact = load_artifact(
                artifact.model_path,
                artifact.metadata_path,
            )

            self.assertEqual(
                loaded_artifact.model_name,
                artifact.model_name,
            )

            self.assertEqual(
                loaded_artifact.target_column,
                "churn",
            )

            self.assertEqual(
                loaded_artifact.task,
                MLTask.CLASSIFICATION.value,
            )


if __name__ == "__main__":
    unittest.main()