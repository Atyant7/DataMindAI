import tempfile
import unittest
import pandas as pd

from src.agent.agent import DataMindAgent
from src.core.app_state import app_state
from src.ml.ml_pipeline import run_ml_pipeline
from src.ml.model_persistence import save_best_model


class TestAgentML(unittest.TestCase):

    def setUp(self):
        self.original_dataset = app_state.dataset
        self.original_dataset_name = app_state.dataset_name
        self.original_profile = app_state.dataset_profile
        self.original_plan = app_state.preprocessing_plan
        self.original_artifact = app_state.model_artifact
        self.original_result = app_state.automl_result

        # Sample dataset for testing ML operations
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
                "Delhi", "Mumbai", "Delhi", "Pune", "Mumbai",
                "Delhi", "Pune", "Mumbai", "Delhi", "Pune",
                "Mumbai", "Delhi", "Pune", "Mumbai", "Delhi",
                "Pune", "Mumbai", "Delhi", "Pune", "Mumbai",
            ],
            "churn": [
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 1,
            ],
        })

        app_state.dataset = self.df
        app_state.dataset_name = "test_churn.csv"
        app_state.model_artifact = None
        app_state.automl_result = None

    def tearDown(self):
        app_state.dataset = self.original_dataset
        app_state.dataset_name = self.original_dataset_name
        app_state.dataset_profile = self.original_profile
        app_state.preprocessing_plan = self.original_plan
        app_state.model_artifact = self.original_artifact
        app_state.automl_result = self.original_result

    def test_agent_train_model_tool(self):
        agent = DataMindAgent()
        result = agent._execute_model_training({
            "target_column": "churn",
            "model_names": ["random_forest"],
        })

        self.assertTrue(result["success"])
        self.assertIn("best_model", result)
        self.assertEqual(result["target_column"], "churn")
        self.assertEqual(result["task"], "classification")
        self.assertIsNotNone(app_state.model_artifact)
        self.assertIsNotNone(app_state.automl_result)

    def test_agent_explain_model_tool(self):
        agent = DataMindAgent()
        # First train model
        agent._execute_model_training({
            "target_column": "churn",
            "model_names": ["random_forest"],
        })

        explanation_result = agent._execute_model_explanation({"top_n": 2})
        self.assertTrue(explanation_result["success"])
        self.assertIn("top_features", explanation_result)
        self.assertEqual(len(explanation_result["top_features"]), 2)

    def test_agent_get_preprocessing_plan_tool(self):
        agent = DataMindAgent()
        plan_result = agent._execute_preprocessing_plan()
        self.assertTrue(plan_result["success"])
        self.assertIn("answer", plan_result)

    def test_agent_predict_tool(self):
        agent = DataMindAgent()
        # Train model first
        agent._execute_model_training({
            "target_column": "churn",
            "model_names": ["random_forest"],
        })

        pred_result = agent._execute_prediction({
            "values": {
                "age": 55,
                "income": 55,
                "city": "Mumbai",
            }
        })

        self.assertTrue(pred_result["success"])
        self.assertIn(pred_result["prediction"], [0, 1])
        self.assertEqual(pred_result["target_column"], "churn")
        self.assertIsNotNone(pred_result["confidence"])
        self.assertIn("Prediction Result", pred_result["answer"])

    def test_agent_chat_predict_natural_language(self):
        agent = DataMindAgent()
        # Train model
        agent._execute_model_training({
            "target_column": "churn",
            "model_names": ["random_forest"],
        })

        # Test natural language prediction prompt
        response = agent.chat(
            "Predict for these values: age=60, income=62, city=Delhi",
            history=[],
        )

        self.assertEqual(response["type"], "text")
        self.assertEqual(response["operation"], "prediction")
        self.assertIn("Prediction Result", response["content"])
        self.assertIn("churn", response["content"])

    def test_agent_chat_train_model_intent(self):
        agent = DataMindAgent()
        response = agent.chat(
            "Train the best model for churn",
            history=[],
        )

        self.assertEqual(response["type"], "text")
        self.assertEqual(response["operation"], "model_training")
        self.assertIn("AutoML Training Complete", response["content"])

    def test_agent_chat_explain_model_intent(self):
        agent = DataMindAgent()
        # Train model first
        agent._execute_model_training({
            "target_column": "churn",
            "model_names": ["random_forest"],
        })

        response = agent.chat(
            "What are the most important features?",
            history=[],
        )

        self.assertEqual(response["type"], "text")
        self.assertEqual(response["operation"], "explain_model")
        self.assertIn("Model Explainability", response["content"])

    def test_agent_chat_preprocessing_plan_intent(self):
        agent = DataMindAgent()
        response = agent.chat(
            "Explain the preprocessing plan",
            history=[],
        )

        self.assertEqual(response["type"], "text")
        self.assertEqual(response["operation"], "preprocessing_plan")
        self.assertIn("Automated Preprocessing Plan", response["content"])

    def test_agent_predict_partial_features(self):
        agent = DataMindAgent()
        agent._execute_model_training({
            "target_column": "churn",
            "model_names": ["random_forest"],
        })

        # Only provide age and city, income omitted
        response = agent.chat(
            "Predict for these values: age=22, city=Mumbai",
            history=[],
        )

        self.assertEqual(response["type"], "text")
        self.assertEqual(response["operation"], "prediction")
        self.assertIn("Prediction Result", response["content"])
        self.assertIn("Default values were imputed", response["content"])

    def test_agent_predict_regression(self):
        # Setup regression dataset
        reg_df = pd.DataFrame({
            "experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "education": ["BS", "BS", "MS", "MS", "PhD", "BS", "MS", "PhD", "MS", "PhD"],
            "salary": [40000, 45000, 55000, 60000, 75000, 68000, 80000, 95000, 92000, 110000],
        })
        app_state.dataset = reg_df
        app_state.dataset_name = "salaries.csv"

        agent = DataMindAgent()
        train_res = agent._execute_model_training({
            "target_column": "salary",
            "model_names": ["linear_regression"],
        })
        self.assertEqual(train_res["task"], "regression")

        response = agent.chat(
            "Predict for these values: experience=5, education=MS",
            history=[],
        )
        self.assertEqual(response["operation"], "prediction")
        self.assertIn("Prediction Result", response["content"])
        self.assertIn("salary", response["content"])


if __name__ == "__main__":
    unittest.main()
