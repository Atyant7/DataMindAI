import unittest

import pandas as pd

from src.tools.data_analysis import (
    DataAnalysisTool,
    DataAnalysisResult,
)


class TestDataAnalysisTool(
    unittest.TestCase
):

    def setUp(self):

        self.df = pd.DataFrame(
            {
                "company": [
                    "AMD",
                    "AMD",
                    "Intel",
                    "Intel",
                    "Microsoft",
                ],
                "year": [
                    2020,
                    2021,
                    2020,
                    2021,
                    2021,
                ],
                "revenue": [
                    10.0,
                    15.0,
                    70.0,
                    75.0,
                    180.0,
                ],
                "layoffs": [
                    100,
                    500,
                    1000,
                    2000,
                    500,
                ],
                "employees": [
                    1000,
                    1200,
                    10000,
                    9000,
                    50000,
                ],
            }
        )

        self.tool = DataAnalysisTool(
            self.df
        )

    # ========================================================
    # OVERVIEW
    # ========================================================

    def test_overview(self):

        result = self.tool.execute(
            "overview"
        )

        self.assertTrue(
            result.success
        )

        self.assertEqual(
            result.evidence["rows"],
            5,
        )

        self.assertEqual(
            result.evidence["columns"],
            5,
        )

    # ========================================================
    # STATISTICS
    # ========================================================

    def test_mean(self):

        result = self.tool.execute(
            "statistics",
            column="revenue",
            aggregation="mean",
        )

        self.assertTrue(
            result.success
        )

        self.assertAlmostEqual(
            result.evidence["value"],
            70.0,
        )

    def test_maximum(self):

        result = self.tool.execute(
            "statistics",
            column="layoffs",
            aggregation="max",
        )

        self.assertTrue(
            result.success
        )

        self.assertEqual(
            result.evidence["value"],
            2000.0,
        )

    # ========================================================
    # GROUPBY
    # ========================================================

    def test_groupby(self):

        result = self.tool.execute(
            "groupby",
            group_column="company",
            value_column="revenue",
            aggregation="mean",
        )

        self.assertTrue(
            result.success
        )

        records = (
            result.evidence["groups"]
        )

        self.assertEqual(
            records[0]["company"],
            "Microsoft",
        )

        self.assertEqual(
            records[0]["revenue"],
            180.0,
        )

    # ========================================================
    # RANK
    # ========================================================

    def test_rank(self):

        result = self.tool.execute(
            "rank",
            column="revenue",
            group_column="company",
            aggregation="mean",
            order="descending",
            limit=2,
        )

        self.assertTrue(
            result.success
        )

        records = (
            result.evidence["ranking"]
        )

        self.assertEqual(
            len(records),
            2,
        )

        self.assertEqual(
            records[0]["company"],
            "Microsoft",
        )

    # ========================================================
    # FILTER
    # ========================================================

    def test_filter(self):

        result = self.tool.execute(
            "filter",
            column="layoffs",
            operator=">",
            value=1000,
        )

        self.assertTrue(
            result.success
        )

        self.assertEqual(
            result.evidence[
                "matching_rows"
            ],
            1,
        )

        self.assertEqual(
            result.evidence[
                "rows"
            ][0]["company"],
            "Intel",
        )

    # ========================================================
    # CORRELATION
    # ========================================================

    def test_correlation(self):

        result = self.tool.execute(
            "correlation",
            column1="revenue",
            column2="employees",
        )

        self.assertTrue(
            result.success
        )

        self.assertGreater(
            result.evidence[
                "correlation"
            ],
            0,
        )

    # ========================================================
    # VALUE COUNTS
    # ========================================================

    def test_value_counts(self):

        result = self.tool.execute(
            "value_counts",
            column="company",
            limit=3,
        )

        self.assertTrue(
            result.success
        )

        self.assertEqual(
            result.evidence[
                "values"
            ][0]["company"],
            "AMD",
        )

        self.assertEqual(
            result.evidence[
                "values"
            ][0]["count"],
            2,
        )

    # ========================================================
    # UNIQUE COUNT
    # ========================================================

    def test_unique_count(self):

        result = self.tool.execute(
            "unique_count",
            column="company",
        )

        self.assertTrue(
            result.success
        )

        self.assertEqual(
            result.evidence[
                "unique_count"
            ],
            3,
        )

    # ========================================================
    # ROW EXTREME
    # ========================================================

    def test_row_extreme(self):

        result = self.tool.execute(
            "row_extreme",
            column="layoffs",
            direction="max",
        )

        self.assertTrue(
            result.success
        )

        self.assertEqual(
            result.evidence[
                "value"
            ],
            2000,
        )

        self.assertEqual(
            result.evidence[
                "row"
            ]["company"],
            "Intel",
        )

        self.assertEqual(
            result.evidence[
                "row"
            ]["year"],
            2021,
        )

    # ========================================================
    # COMPARE
    # ========================================================

    def test_compare(self):

        result = self.tool.execute(
            "compare",
            column="revenue",
            groups=[
                "AMD",
                "Intel",
            ],
            group_column="company",
            aggregation="mean",
        )

        self.assertTrue(
            result.success
        )

        records = (
            result.evidence[
                "comparison"
            ]
        )

        self.assertEqual(
            len(records),
            2,
        )

    # ========================================================
    # INVALID COLUMN
    # ========================================================

    def test_invalid_column(self):

        result = self.tool.execute(
            "statistics",
            column="salary",
            aggregation="mean",
        )

        self.assertFalse(
            result.success
        )

        self.assertIn(
            "does not exist",
            result.error,
        )

    # ========================================================
    # INVALID OPERATION
    # ========================================================

    def test_invalid_operation(self):

        result = self.tool.execute(
            "something_random"
        )

        self.assertFalse(
            result.success
        )

    # ========================================================
    # RESULT TYPE
    # ========================================================

    def test_result_type(self):

        result = self.tool.execute(
            "statistics",
            column="revenue",
            aggregation="mean",
        )

        self.assertIsInstance(
            result,
            DataAnalysisResult,
        )


if __name__ == "__main__":
    unittest.main()