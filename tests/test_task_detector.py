import unittest

import pandas as pd

from src.ml.task_detector import (
    MLTask,
    TargetType,
    detect_task,
)


class TestTaskDetector(unittest.TestCase):

    def test_binary_classification(self):
        df = pd.DataFrame({
            "age": [20, 25, 30, 35, 40, 45],
            "income": [20, 30, 40, 50, 60, 70],
            "churn": [0, 1, 0, 1, 0, 1],
        })

        result = detect_task(df, "churn")

        self.assertEqual(result.task, MLTask.CLASSIFICATION)
        self.assertEqual(result.target_type, TargetType.BINARY)

    def test_multiclass_classification(self):
        df = pd.DataFrame({
            "age": [20, 25, 30, 35, 40, 45],
            "income": [20, 30, 40, 50, 60, 70],
            "segment": ["A", "B", "C", "A", "B", "C"],
        })

        result = detect_task(df, "segment")

        self.assertEqual(result.task, MLTask.CLASSIFICATION)
        self.assertEqual(result.target_type, TargetType.MULTICLASS)

    def test_regression(self):
        df = pd.DataFrame({
            "area": [500, 700, 900, 1100, 1300, 1500],
            "rooms": [1, 2, 2, 3, 3, 4],
            "price": [100000, 140000, 180000, 220000, 260000, 300000],
        })

        result = detect_task(df, "price")

        self.assertEqual(result.task, MLTask.REGRESSION)
        self.assertEqual(result.target_type, TargetType.REGRESSION)

    def test_missing_target_column(self):
        df = pd.DataFrame({
            "age": [20, 25, 30],
            "income": [20, 30, 40],
        })

        with self.assertRaises(ValueError):
            detect_task(df, "target")

    def test_constant_target(self):
        df = pd.DataFrame({
            "age": [20, 25, 30],
            "target": [1, 1, 1],
        })

        with self.assertRaises(ValueError):
            detect_task(df, "target")


if __name__ == "__main__":
    unittest.main()