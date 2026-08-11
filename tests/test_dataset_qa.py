import unittest

import pandas as pd

from src.backend.dataset_qa import (
    DatasetQA,
    DatasetQAResult,
    answer_dataset_question,
)


class TestDatasetQA(unittest.TestCase):

    def setUp(self):

        self.df = pd.DataFrame(
            {
                "age": [
                    20,
                    30,
                    40,
                    50,
                    60,
                ],
                "salary": [
                    20000,
                    30000,
                    40000,
                    50000,
                    60000,
                ],
                "department": [
                    "IT",
                    "HR",
                    "IT",
                    "Sales",
                    "HR",
                ],
                "city": [
                    "Delhi",
                    "Mumbai",
                    "Delhi",
                    "Pune",
                    "Mumbai",
                ],
            }
        )

        self.qa = DatasetQA(
            self.df
        )

    # ========================================================
    # OVERVIEW
    # ========================================================

    def test_overview(self):

        result = self.qa.answer(
            "Tell me about the dataset"
        )

        self.assertIsInstance(
            result,
            DatasetQAResult,
        )

        self.assertTrue(
            result.success
        )

        self.assertEqual(
            result.operation,
            "overview",
        )

        self.assertEqual(
            result.evidence["rows"],
            5,
        )

        self.assertEqual(
            result.evidence["columns"],
            4,
        )

    # ========================================================
    # COLUMNS
    # ========================================================

    def test_columns(self):

        result = self.qa.answer(
            "What columns are in the dataset?"
        )

        self.assertTrue(
            result.success
        )

        self.assertEqual(
            result.operation,
            "columns",
        )

        self.assertIn(
            "age",
            result.evidence["columns"],
        )

        self.assertIn(
            "salary",
            result.evidence["columns"],
        )

    # ========================================================
    # MEAN
    # ========================================================

    def test_mean(self):

        result = self.qa.answer(
            "What is the average salary?"
        )

        self.assertTrue(
            result.success
        )

        self.assertEqual(
            result.operation,
            "mean",
        )

        self.assertEqual(
            result.evidence["column"],
            "salary",
        )

        self.assertEqual(
            result.evidence["value"],
            40000.0,
        )

    # ========================================================
    # MEDIAN
    # ========================================================

    def test_median(self):

        result = self.qa.answer(
            "What is the median age?"
        )

        self.assertTrue(
            result.success
        )

        self.assertEqual(
            result.operation,
            "median",
        )

        self.assertEqual(
            result.evidence["value"],
            40.0,
        )

    # ========================================================
    # MAXIMUM
    # ========================================================

    def test_maximum(self):

        result = self.qa.answer(
            "What is the highest salary?"
        )

        self.assertTrue(
            result.success
        )

        self.assertEqual(
            result.operation,
            "maximum",
        )

        self.assertEqual(
            result.evidence["value"],
            60000.0,
        )

    # ========================================================
    # MINIMUM
    # ========================================================

    def test_minimum(self):

        result = self.qa.answer(
            "What is the lowest age?"
        )

        self.assertTrue(
            result.success
        )

        self.assertEqual(
            result.operation,
            "minimum",
        )

        self.assertEqual(
            result.evidence["value"],
            20.0,
        )

    # ========================================================
    # UNIQUE
    # ========================================================

    def test_unique_values(self):

        result = self.qa.answer(
            "How many unique cities are there?"
        )

        self.assertTrue(
            result.success
        )

        self.assertEqual(
            result.operation,
            "unique_values",
        )

        self.assertEqual(
            result.evidence["unique_count"],
            3,
        )

    # ========================================================
    # VALUE COUNTS
    # ========================================================

    def test_value_counts(self):

        result = self.qa.answer(
            "What is the frequency of each city?"
        )

        self.assertTrue(
            result.success
        )

        self.assertEqual(
            result.operation,
            "value_counts",
        )

        self.assertIsNotNone(
            result.dataframe
        )

        self.assertEqual(
            len(
                result.dataframe
            ),
            3,
        )

    # ========================================================
    # CORRELATION
    # ========================================================

    def test_correlation(self):

        result = self.qa.answer(
            "What is the correlation between age and salary?"
        )

        self.assertTrue(
            result.success
        )

        self.assertEqual(
            result.operation,
            "correlation",
        )

        self.assertAlmostEqual(
            result.evidence["correlation"],
            1.0,
        )

    # ========================================================
    # GROUPBY
    # ========================================================

    def test_groupby(self):

        result = self.qa.answer(
            "What is the average salary for each department?"
        )

        self.assertTrue(
            result.success
        )

        self.assertEqual(
            result.operation,
            "groupby",
        )

        self.assertEqual(
            result.evidence[
                "group_column"
            ],
            "department",
        )

        self.assertEqual(
            result.evidence[
                "value_column"
            ],
            "salary",
        )

        self.assertEqual(
            result.evidence[
                "aggregation"
            ],
            "mean",
        )

    # ========================================================
    # MISSING VALUES
    # ========================================================

    def test_missing_values(self):

        df = self.df.copy()

        df.loc[
            0,
            "salary",
        ] = None

        qa = DatasetQA(
            df
        )

        result = qa.answer(
            "How many missing values are there?"
        )

        self.assertTrue(
            result.success
        )

        self.assertEqual(
            result.operation,
            "missing_values",
        )

        self.assertEqual(
            result.evidence[
                "total_missing_values"
            ],
            1,
        )

    # ========================================================
    # DUPLICATES
    # ========================================================

    def test_duplicates(self):

        df = pd.concat(
            [
                self.df,
                self.df.iloc[
                    [0]
                ],
            ],
            ignore_index=True,
        )

        qa = DatasetQA(
            df
        )

        result = qa.answer(
            "Are there duplicate rows?"
        )

        self.assertTrue(
            result.success
        )

        self.assertEqual(
            result.operation,
            "duplicates",
        )

        self.assertEqual(
            result.evidence[
                "duplicate_rows"
            ],
            1,
        )

    # ========================================================
    # UNKNOWN QUESTION
    # ========================================================

    def test_unknown_question(self):

        result = self.qa.answer(
            "What will happen tomorrow?"
        )

        self.assertFalse(
            result.success
        )

        self.assertEqual(
            result.operation,
            "unknown",
        )

    # ========================================================
    # EMPTY QUESTION
    # ========================================================

    def test_empty_question(self):

        result = self.qa.answer(
            ""
        )

        self.assertFalse(
            result.success
        )

        self.assertEqual(
            result.operation,
            "unknown",
        )

    # ========================================================
    # CONVENIENCE FUNCTION
    # ========================================================

    def test_convenience_function(self):

        result = answer_dataset_question(
            self.df,
            "What is the average salary?",
        )

        self.assertTrue(
            result.success
        )

        self.assertEqual(
            result.evidence[
                "value"
            ],
            40000.0,
        )


if __name__ == "__main__":
    unittest.main()