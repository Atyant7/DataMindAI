"""
Tests for central chat workflows: in-place cleaning, model training, and geospatial mapping.
"""

import unittest
import numpy as np
import pandas as pd

from src.agent.agent import DataMindAgent
from src.core.app_state import app_state


class TestChatWorkflows(unittest.TestCase):

    def setUp(self):
        self.original_dataset = app_state.dataset
        self.original_dataset_name = app_state.dataset_name
        self.original_profile = app_state.dataset_profile
        self.original_artifact = app_state.model_artifact

        # Dataset with missing values and geo coordinates
        self.df = pd.DataFrame({
            "city": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"],
            "latitude": [40.7128, 34.0522, 41.8781, 29.7604, 33.4484, 39.9526, 29.4241, 32.7157, 32.7767, 37.3382],
            "longitude": [-74.0060, -118.2437, -87.6298, -95.3698, -112.0740, -75.1652, -98.4936, -117.1611, -96.7970, -121.8863],
            "population": [8.3, 3.9, 2.7, 2.3, 1.6, 1.5, 1.4, 1.3, 1.2, 1.0],
            "score": [85.0, np.nan, 78.0, np.nan, 65.0, 72.0, np.nan, 80.0, 75.0, 88.0],
            "risk": [1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        })

        app_state.dataset = self.df.copy()
        app_state.dataset_name = "test_cities.csv"

    def tearDown(self):
        app_state.dataset = self.original_dataset
        app_state.dataset_name = self.original_dataset_name
        app_state.dataset_profile = self.original_profile
        app_state.model_artifact = self.original_artifact

    def test_in_place_data_cleaning_from_chat(self):
        agent = DataMindAgent()

        # Check missing values before
        self.assertGreater(app_state.dataset["score"].isna().sum(), 0)

        # User says "Handle them"
        response = agent.chat("Handle them", history=[])
        self.assertEqual(response["type"], "text")
        self.assertEqual(response["operation"], "data_cleaning")
        self.assertIn("Done", response["content"])

        # Check missing values after
        self.assertEqual(app_state.dataset["score"].isna().sum(), 0)

    def test_geospatial_mapping_from_chat(self):
        agent = DataMindAgent()

        response = agent.chat("Show me these locations on a map", history=[])
        self.assertEqual(response["type"], "visualization")
        self.assertEqual(response["operation"], "geospatial_map")
        self.assertIsNotNone(response.get("figure"))
        self.assertIsNotNone(app_state.current_figure)

    def test_train_model_from_chat_with_target_detection(self):
        agent = DataMindAgent()

        # Handle missing values first for clean training
        agent.chat("Handle them", history=[])

        # Train model with explicit target
        response = agent.chat("Train a model to predict risk", history=[])

        self.assertEqual(response["type"], "text")
        self.assertEqual(response["operation"], "model_training")
        self.assertIn("AutoML Training Complete", response["content"])
        self.assertIn("risk", response["content"])
        self.assertIn("You can now go to the Prediction section", response["content"])
        self.assertIsNotNone(app_state.model_artifact)
