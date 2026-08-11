import unittest

import pandas as pd

from src.backend.dataset_intelligence import DatasetIntelligence
from src.backend.preprocessing_intelligence import PreprocessingIntelligence


class TestDataMindAIPhase1(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataframe = pd.DataFrame({
            "customer_id": [1, 2, 3, 4, 5, 6],
            "age": [21, 24, 28, 31, 35, 40],
            "salary": [30000, 35000, 42000, 50000, 55000, 65000],
            "department": ["A", "A", "B", "B", "C", "C"],
            "churn": [0, 0, 1, 0, 1, 1],
        })

    def test_profile_is_generated(self):
        profile = DatasetIntelligence(
            self.dataframe,
            "test.csv",
        ).generate_profile()

        self.assertEqual(profile.rows, 6)
        self.assertEqual(profile.columns, 5)
        self.assertEqual(profile.dataset_name, "test.csv")
        self.assertEqual(profile.recommended_target, "churn")
        self.assertEqual(profile.detected_task, "Classification")
        self.assertIsNotNone(profile.correlation_matrix)

    def test_feature_selection_contains_every_column(self):
        profile = DatasetIntelligence(
            self.dataframe,
            "test.csv",
        ).generate_profile()

        plan = PreprocessingIntelligence(
            self.dataframe,
            profile,
        ).generate_plan()

        self.assertEqual(
            len(plan.feature_selection_plan),
            len(self.dataframe.columns),
        )

    def test_identifier_is_not_treated_as_a_normal_feature(self):
        profile = DatasetIntelligence(
            self.dataframe,
            "test.csv",
        ).generate_profile()

        identifier = next(
            item
            for item in profile.feature_quality_summary
            if item["column"] == "customer_id"
        )

        qualities = {quality["quality"] for quality in identifier["qualities"]}
        self.assertIn("Unique Identifier", qualities)


if __name__ == "__main__":
    unittest.main()
