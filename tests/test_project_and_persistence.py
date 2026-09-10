"""
Tests for projects, multi-user isolation, chat persistence, and model registry.
"""

import unittest
import uuid

import pandas as pd

from src.database.init_db import init_db
from src.services.auth_service import AuthService
from src.services.chat_service import ChatService
from src.services.dataset_service import DatasetService
from src.services.model_registry_service import ModelRegistryService
from src.services.prediction_service import PredictionService
from src.services.project_service import ProjectService


class TestProjectAndPersistence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()

        # Create two isolated users
        s1 = uuid.uuid4().hex[:6]
        s2 = uuid.uuid4().hex[:6]
        cls.user_a = AuthService.register_user(f"usera_{s1}", f"usera_{s1}@test.com", "pass123")
        cls.user_b = AuthService.register_user(f"userb_{s2}", f"userb_{s2}@test.com", "pass123")

    def test_project_lifecycle_and_user_isolation(self):
        # User A creates a project
        proj_a = ProjectService.create_project(
            user_id=self.user_a["id"],
            name="Heart Disease Project A",
            description="Analysis for User A",
        )
        self.assertIsNotNone(proj_a["id"])

        # User B cannot see User A's project
        projects_b = ProjectService.list_user_projects(self.user_b["id"])
        b_ids = [p["id"] for p in projects_b]
        self.assertNotIn(proj_a["id"], b_ids)

        # User B cannot get User A's project by ID
        fetched_by_b = ProjectService.get_project(self.user_b["id"], proj_a["id"])
        self.assertIsNone(fetched_by_b)

        # User A can retrieve it
        fetched_by_a = ProjectService.get_project(self.user_a["id"], proj_a["id"])
        self.assertIsNotNone(fetched_by_a)
        self.assertEqual(fetched_by_a["name"], "Heart Disease Project A")

    def test_chat_persistence(self):
        proj = ProjectService.create_project(self.user_a["id"], "Chat Project")
        chat = ChatService.get_or_create_default_chat(self.user_a["id"], proj["id"])
        chat_id = chat["id"]

        # Add messages
        msg1 = ChatService.add_message(chat_id, "user", "How many rows are in the dataset?")
        msg2 = ChatService.add_message(
            chat_id,
            "assistant",
            "There are 500 rows.",
            operation="overview",
            evidence={"rows": 500},
        )

        self.assertIsNotNone(msg1["id"])
        self.assertIsNotNone(msg2["id"])

        # Retrieve messages
        history = ChatService.get_chat_messages(chat_id)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[0]["content"], "How many rows are in the dataset?")
        self.assertEqual(history[1]["role"], "assistant")
        self.assertEqual(history[1]["evidence"]["rows"], 500)

        # Clear messages
        ChatService.clear_chat_messages(chat_id)
        cleared_history = ChatService.get_chat_messages(chat_id)
        self.assertEqual(len(cleared_history), 0)

    def test_model_registry_and_versioning(self):
        proj = ProjectService.create_project(self.user_a["id"], "AutoML Registry Project")

        # Register v1
        m1 = ModelRegistryService.register_model(
            user_id=self.user_a["id"],
            project_id=proj["id"],
            model_name="random_forest",
            display_name="Random Forest",
            task="classification",
            target_column="churn",
            artifact_path="artifacts/models/rf_churn.joblib",
            metadata_path="artifacts/models/rf_churn_meta.joblib",
            feature_names=["age", "salary"],
            metrics={"f1": 0.82},
        )
        self.assertEqual(m1["version"], 1)
        self.assertTrue(m1["is_best"])

        # Register v2 for same target
        m2 = ModelRegistryService.register_model(
            user_id=self.user_a["id"],
            project_id=proj["id"],
            model_name="xgboost",
            display_name="XGBoost",
            task="classification",
            target_column="churn",
            artifact_path="artifacts/models/xgb_churn.joblib",
            metadata_path="artifacts/models/xgb_churn_meta.joblib",
            feature_names=["age", "salary"],
            metrics={"f1": 0.89},
        )
        self.assertEqual(m2["version"], 2)
        self.assertTrue(m2["is_best"])

        # List project models
        all_models = ModelRegistryService.list_project_models(self.user_a["id"], proj["id"])
        self.assertEqual(len(all_models), 2)
        versions = [m["version"] for m in all_models]
        self.assertIn(1, versions)
        self.assertIn(2, versions)

    def test_prediction_service_history(self):
        proj = ProjectService.create_project(self.user_a["id"], "Pred History Project")
        m = ModelRegistryService.register_model(
            user_id=self.user_a["id"],
            project_id=proj["id"],
            model_name="random_forest",
            display_name="Random Forest",
            task="classification",
            target_column="churn",
            artifact_path="dummy.joblib",
            metadata_path="dummy_meta.joblib",
        )

        # Record prediction
        rec = PredictionService.record_prediction(
            user_id=self.user_a["id"],
            project_id=proj["id"],
            model_id=m["id"],
            input_data={"age": 45, "salary": 65000},
            prediction=1,
            confidence=0.88,
            probabilities={"0": 0.12, "1": 0.88},
        )
        self.assertIsNotNone(rec["id"])
        self.assertEqual(rec["prediction_result"], "1")

        # Retrieve history
        history = PredictionService.get_project_predictions(self.user_a["id"], proj["id"])
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["prediction_result"], "1")
        self.assertEqual(history[0]["input_data"]["age"], 45)
