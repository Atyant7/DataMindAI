"""
Tests for dynamic prediction logic: tabular models and image models.
"""

import unittest
import numpy as np
import pandas as pd
from PIL import Image

from src.ml.model_trainer_base import ImageModelTrainer, TabularModelTrainer, GeospatialModelTrainer
from src.ml.predictor_base import ImagePredictor, TabularPredictor
from src.ml.model_persistence import save_best_model


class TestDynamicPredictionUI(unittest.TestCase):

    def setUp(self):
        self.df_tabular = pd.DataFrame({
            "age": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
            "salary": [25000, 32000, 45000, 50000, 62000, 70000, 85000, 90000, 105000, 120000],
            "department": ["IT", "HR", "IT", "Sales", "IT", "HR", "Sales", "IT", "HR", "Sales"],
            "promoted": [0, 0, 1, 0, 1, 0, 1, 1, 1, 1],
        })

    def test_tabular_model_trainer_and_predictor(self):
        trainer = TabularModelTrainer()
        result = trainer.train(
            data=self.df_tabular,
            target="promoted",
            model_names=["random_forest"],
        )
        artifact = save_best_model(result, model_directory="artifacts/models")

        predictor = TabularPredictor(artifact)
        input_row = {
            "age": 32,
            "salary": 55000,
            "department": "IT",
        }
        pred_res = predictor.predict(input_row)

        self.assertIn(pred_res.prediction, [0, 1])
        self.assertEqual(pred_res.target_column, "promoted")
        self.assertIsNotNone(pred_res.confidence)
        self.assertIsNotNone(pred_res.probabilities)

    def test_image_model_trainer_and_predictor(self):
        # Create synthetic images (e.g. red image vs blue image)
        red_img = Image.new("RGB", (32, 32), color=(255, 0, 0))
        blue_img = Image.new("RGB", (32, 32), color=(0, 0, 255))

        dataset = [
            (red_img, "red_class"),
            (red_img, "red_class"),
            (blue_img, "blue_class"),
            (blue_img, "blue_class"),
        ]

        trainer = ImageModelTrainer(image_size=(32, 32))
        train_res = trainer.train(dataset, target="color")

        self.assertIn("model", train_res)
        self.assertEqual(len(train_res["classes"]), 2)

        # Predict on new red image
        test_red = Image.new("RGB", (32, 32), color=(240, 10, 10))
        predictor = ImagePredictor(
            model=train_res["model"],
            classes=train_res["classes"],
            display_name="Color Classifier",
        )
        pred = predictor.predict(test_red)

        self.assertEqual(pred.prediction, "red_class")
        self.assertEqual(pred.task, "classification")
        self.assertIsNotNone(pred.confidence)
        self.assertIn("red_class", pred.probabilities)

    def test_geospatial_model_trainer(self):
        df_geo = pd.DataFrame({
            "latitude": [40.7, 34.0, 41.8, 29.7, 33.4, 39.9, 29.4, 32.7, 37.3, 47.6],
            "longitude": [-74.0, -118.2, -87.6, -95.3, -112.0, -75.1, -98.4, -96.7, -121.8, -122.3],
            "value": [10, 20, 15, 30, 25, 12, 28, 22, 18, 14],
            "target": [0, 1, 0, 1, 1, 0, 1, 1, 0, 0],
        })

        trainer = GeospatialModelTrainer()
        result = trainer.train(df_geo, target="target", model_names=["random_forest"])

        self.assertIsNotNone(result.best_model_name)
        self.assertEqual(result.target_column, "target")
