import unittest
import numpy as np
import pandas as pd

from src.backend.preprocessing import (
    clean_dataset,
    encode_categoricals,
    handle_missing_values,
    remove_outliers_iqr,
    scale_features,
)


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            " name ": [" Alice ", " Bob ", " Charlie ", " Alice "],
            "age": [25.0, 30.0, 35.0, 25.0],
            "salary": [50000.0, 60000.0, 150000.0, 50000.0],
            "department": ["IT", "HR", "IT", "IT"],
        })
        self.df_missing = pd.DataFrame({
            "age": [25.0, np.nan, 35.0, 25.0],
            "department": ["IT", "HR", np.nan, "IT"],
        })

    def test_clean_dataset(self):
        cleaned = clean_dataset(self.df, drop_duplicates=True, strip_strings=True, clean_column_names=True)
        self.assertEqual(len(cleaned), 3)  # Duplicate "Alice" row dropped
        self.assertIn("name", cleaned.columns)
        self.assertEqual(cleaned["name"].iloc[0], "Alice")

    def test_handle_missing_values_median(self):
        imputed = handle_missing_values(self.df_missing, numeric_strategy="median", categorical_strategy="most_frequent")
        self.assertFalse(imputed["age"].isna().any())
        self.assertFalse(imputed["department"].isna().any())
        self.assertEqual(imputed["age"].iloc[1], 25.0)  # Median of [25, 35, 25] is 25
        self.assertEqual(imputed["department"].iloc[2], "IT")  # Mode of ["IT", "HR", "IT"] is "IT"

    def test_handle_missing_values_custom(self):
        imputed = handle_missing_values(
            self.df_missing,
            custom_strategies={"age": 99.0, "department": "Unknown"},
        )
        self.assertEqual(imputed["age"].iloc[1], 99.0)
        self.assertEqual(imputed["department"].iloc[2], "Unknown")

    def test_remove_outliers_iqr_clip(self):
        # Salary 150000 is an outlier
        treated = remove_outliers_iqr(self.df, columns=["salary"], factor=1.5, method="clip")
        self.assertLess(treated["salary"].max(), 150000.0)

    def test_remove_outliers_iqr_filter(self):
        treated = remove_outliers_iqr(self.df, columns=["salary"], factor=1.5, method="filter")
        self.assertNotIn(150000.0, treated["salary"].values)

    def test_encode_categoricals_onehot(self):
        df_cats = pd.DataFrame({"department": ["IT", "HR", "Sales"]})
        encoded = encode_categoricals(df_cats, columns=["department"], method="onehot")
        self.assertIn("department_IT", encoded.columns)
        self.assertIn("department_HR", encoded.columns)

    def test_encode_categoricals_label(self):
        df_clean = self.df.dropna()
        encoded = encode_categoricals(df_clean, columns=["department"], method="label")
        self.assertTrue(pd.api.types.is_numeric_dtype(encoded["department"]))

    def test_scale_features_standard(self):
        df_num = pd.DataFrame({"val": [10.0, 20.0, 30.0, 40.0, 50.0]})
        scaled = scale_features(df_num, columns=["val"], method="standard")
        self.assertAlmostEqual(scaled["val"].mean(), 0.0, places=5)
        self.assertAlmostEqual(scaled["val"].std(ddof=0), 1.0, places=5)

    def test_scale_features_minmax(self):
        df_num = pd.DataFrame({"val": [10.0, 20.0, 30.0, 40.0, 50.0]})
        scaled = scale_features(df_num, columns=["val"], method="minmax")
        self.assertAlmostEqual(scaled["val"].min(), 0.0, places=5)
        self.assertAlmostEqual(scaled["val"].max(), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
