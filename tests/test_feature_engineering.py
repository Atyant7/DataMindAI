import unittest
import pandas as pd

from src.backend.feature_engineering import (
    bin_numerical_feature,
    create_interaction_features,
    create_polynomial_features,
    extract_datetime_features,
)


class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2023-01-01 10:00:00", "2023-01-07 15:00:00", "2023-06-15 08:00:00"]),
            "price": [10.0, 20.0, 30.0],
            "quantity": [2.0, 5.0, 4.0],
        })

    def test_extract_datetime_features(self):
        result = extract_datetime_features(self.df, columns=["timestamp"])
        self.assertIn("timestamp_year", result.columns)
        self.assertIn("timestamp_month", result.columns)
        self.assertIn("timestamp_day", result.columns)
        self.assertIn("timestamp_dayofweek", result.columns)
        self.assertIn("timestamp_is_weekend", result.columns)
        # 2023-01-01 was a Sunday (dayofweek 6, is_weekend 1)
        self.assertEqual(result["timestamp_is_weekend"].iloc[0], 1)
        # 2023-06-15 was a Thursday (dayofweek 3, is_weekend 0)
        self.assertEqual(result["timestamp_is_weekend"].iloc[2], 0)

    def test_create_interaction_features_multiply(self):
        result = create_interaction_features(self.df, "price", "quantity", operation="multiply")
        self.assertIn("price_x_quantity", result.columns)
        self.assertEqual(result["price_x_quantity"].iloc[0], 20.0)

    def test_create_interaction_features_ratio(self):
        result = create_interaction_features(self.df, "price", "quantity", operation="ratio")
        self.assertIn("price_per_quantity", result.columns)
        self.assertEqual(result["price_per_quantity"].iloc[0], 5.0)

    def test_create_interaction_features_add(self):
        result = create_interaction_features(self.df, "price", "quantity", operation="add")
        self.assertIn("price_plus_quantity", result.columns)
        self.assertEqual(result["price_plus_quantity"].iloc[0], 12.0)

    def test_create_polynomial_features(self):
        result = create_polynomial_features(self.df, columns=["price"], degree=3)
        self.assertIn("price_pow_2", result.columns)
        self.assertIn("price_pow_3", result.columns)
        self.assertEqual(result["price_pow_2"].iloc[0], 100.0)
        self.assertEqual(result["price_pow_3"].iloc[0], 1000.0)

    def test_bin_numerical_feature_quantile(self):
        df_num = pd.DataFrame({"score": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]})
        result = bin_numerical_feature(df_num, "score", n_bins=3, strategy="quantile")
        self.assertIn("score_binned", result.columns)
        self.assertEqual(result["score_binned"].nunique(), 3)

    def test_bin_numerical_feature_uniform(self):
        df_num = pd.DataFrame({"score": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]})
        result = bin_numerical_feature(df_num, "score", n_bins=2, strategy="uniform", labels=["Low", "High"])
        self.assertIn("score_binned", result.columns)
        self.assertEqual(result["score_binned"].iloc[0], "Low")
        self.assertEqual(result["score_binned"].iloc[-1], "High")


if __name__ == "__main__":
    unittest.main()
