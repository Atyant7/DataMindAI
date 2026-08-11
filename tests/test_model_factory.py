import unittest

from src.ml.model_factory import (
    ModelFamily,
    create_model,
    create_models,
    get_model_names,
    get_model_specs,
)
from src.ml.task_detector import MLTask


class TestModelFactory(unittest.TestCase):

    def test_classification_model_specs(self):
        specs = get_model_specs(MLTask.CLASSIFICATION)

        names = [spec.name for spec in specs]

        self.assertIn(
            ModelFamily.LOGISTIC_REGRESSION.value,
            names,
        )
        self.assertIn(
            ModelFamily.RANDOM_FOREST.value,
            names,
        )
        self.assertIn(
            ModelFamily.XGBOOST.value,
            names,
        )
        self.assertIn(
            ModelFamily.LIGHTGBM.value,
            names,
        )

    def test_regression_model_specs(self):
        specs = get_model_specs(MLTask.REGRESSION)

        names = [spec.name for spec in specs]

        self.assertIn(
            ModelFamily.LINEAR_REGRESSION.value,
            names,
        )
        self.assertIn(
            ModelFamily.RANDOM_FOREST.value,
            names,
        )
        self.assertIn(
            ModelFamily.XGBOOST.value,
            names,
        )
        self.assertIn(
            ModelFamily.LIGHTGBM.value,
            names,
        )

    def test_create_classification_models(self):
        models = create_models(
            MLTask.CLASSIFICATION
        )

        self.assertEqual(
            set(models.keys()),
            {
                "logistic_regression",
                "random_forest",
                "xgboost",
                "lightgbm",
            },
        )

    def test_create_regression_models(self):
        models = create_models(
            MLTask.REGRESSION
        )

        self.assertEqual(
            set(models.keys()),
            {
                "linear_regression",
                "random_forest",
                "xgboost",
                "lightgbm",
            },
        )

    def test_create_xgboost_classifier(self):
        model = create_model(
            "xgboost",
            MLTask.CLASSIFICATION,
        )

        self.assertEqual(
            model.__class__.__name__,
            "XGBClassifier",
        )

    def test_create_xgboost_regressor(self):
        model = create_model(
            "xgboost",
            MLTask.REGRESSION,
        )

        self.assertEqual(
            model.__class__.__name__,
            "XGBRegressor",
        )

    def test_invalid_model(self):
        with self.assertRaises(ValueError):
            create_model(
                "invalid_model",
                MLTask.CLASSIFICATION,
            )

    def test_model_names(self):
        names = get_model_names(
            MLTask.CLASSIFICATION
        )

        self.assertEqual(
            len(names),
            4,
        )


if __name__ == "__main__":
    unittest.main()