"""Data-driven preprocessing recommendations."""

import pandas as pd

from src.backend.preprocessing_plan import PreprocessingPlan


class PreprocessingIntelligence:
    """Generate preprocessing recommendations from a DatasetProfile."""

    def __init__(self, df, profile):
        self.df = df
        self.profile = profile

        self.feature_quality_lookup = {
            feature["column"]: feature["qualities"]
            for feature in profile.feature_quality_summary
        }

        self.distribution_lookup = {
            item["column"]: item
            for item in profile.distribution_summary
        }

        self.outlier_lookup = {
            item["column"]: item
            for item in profile.outlier_summary
        }

    def generate_plan(self):
        plan = PreprocessingPlan()

        self._missing_value_strategy(plan)
        self._encoding_strategy(plan)
        self._scaling_strategy(plan)
        self._feature_selection_strategy(plan)
        self._outlier_treatment_strategy(plan)
        self._train_test_recommendation(plan)
        self._pipeline_summary(plan)

        plan.notes.append(
            "Scaling and preprocessing will be made model-aware when the ML engine is added."
        )
        plan.notes.append(
            "Recommendations should be validated before being applied to production data."
        )

        return plan

    def _missing_value_strategy(self, plan):
        """Recommend an imputation or removal strategy."""
        for column in self.df.columns:
            missing_percentage = float(self.df[column].isna().mean() * 100)

            recommendation = {
                "column": column,
                "missing_percentage": round(missing_percentage, 2),
                "strategy": None,
                "reason": None,
            }

            if missing_percentage == 0:
                recommendation["strategy"] = "No Action"
                recommendation["reason"] = "Column has no missing values."

            else:
                if column == self.profile.recommended_target:
                    recommendation["strategy"] = "Drop Rows / Investigate"
                    recommendation["reason"] = (
                        "Missing target values should be handled separately from feature imputation."
                    )
                    plan.missing_value_plan.append(recommendation)
                    continue

                qualities = self.feature_quality_lookup.get(column, [])
                is_empty = any(
                    quality["quality"] == "Empty"
                    for quality in qualities
                )

                if is_empty:
                    recommendation["strategy"] = "Drop Column"
                    recommendation["reason"] = (
                        "Column contains only missing values."
                    )

                elif pd.api.types.is_numeric_dtype(self.df[column]):
                    if missing_percentage <= 5:
                        recommendation["strategy"] = "Mean Imputation"
                        recommendation["reason"] = (
                            "Numeric feature with low missing values."
                        )
                    elif missing_percentage <= 30:
                        recommendation["strategy"] = "Median Imputation"
                        recommendation["reason"] = (
                            "Numeric feature with moderate missing values."
                        )
                    else:
                        recommendation["strategy"] = "Drop Column"
                        recommendation["reason"] = (
                            "Numeric feature has excessive missing values."
                        )

                elif pd.api.types.is_datetime64_any_dtype(self.df[column]):
                    recommendation["strategy"] = "Investigate"
                    recommendation["reason"] = (
                        "Datetime missing values should be handled according to the time-series context."
                    )

                else:
                    if missing_percentage <= 30:
                        recommendation["strategy"] = "Mode Imputation"
                        recommendation["reason"] = (
                            "Categorical feature with acceptable missing values."
                        )
                    else:
                        recommendation["strategy"] = "Drop Column"
                        recommendation["reason"] = (
                            "Categorical feature has excessive missing values."
                        )

            plan.missing_value_plan.append(recommendation)

    def _encoding_strategy(self, plan):
        """Recommend encoding for categorical features."""
        for column in self.df.columns:
            if column == self.profile.recommended_target:
                plan.encoding_plan.append({
                    "column": column,
                    "encoding": "Target - Excluded",
                    "reason": "The target is handled separately from feature encoding.",
                })
                continue

            if pd.api.types.is_numeric_dtype(self.df[column]):
                continue

            if pd.api.types.is_datetime64_any_dtype(self.df[column]):
                plan.encoding_plan.append({
                    "column": column,
                    "encoding": "Feature Engineering Required",
                    "reason": "Datetime columns should be transformed into useful time features.",
                })
                continue

            if pd.api.types.is_bool_dtype(self.df[column]):
                plan.encoding_plan.append({
                    "column": column,
                    "encoding": "No Encoding",
                    "reason": "Boolean features are already machine-readable.",
                })
                continue

            recommendation = {
                "column": column,
                "encoding": None,
                "reason": None,
            }

            qualities = self.feature_quality_lookup.get(column, [])
            stats = self.profile.categorical_statistics.get(column, {})
            unique_values = stats.get("unique_values", 0)

            quality_names = {
                quality["quality"]
                for quality in qualities
            }

            if "Unique Identifier" in quality_names:
                recommendation["encoding"] = "No Encoding"
                recommendation["reason"] = (
                    "Unique identifiers should normally be removed rather than encoded."
                )
                plan.encoding_plan.append(recommendation)
                continue

            is_high_cardinality = "High Cardinality" in quality_names

            if unique_values == 2:
                recommendation["encoding"] = "Binary Encoding"
                recommendation["reason"] = "Binary categorical feature."
            elif is_high_cardinality:
                recommendation["encoding"] = "Frequency Encoding"
                recommendation["reason"] = "High-cardinality categorical feature."
            else:
                recommendation["encoding"] = "One Hot Encoding"
                recommendation["reason"] = (
                    "Categorical feature with a manageable number of categories."
                )

            plan.encoding_plan.append(recommendation)

    def _scaling_strategy(self, plan):
        """Recommend scaling based on distribution and outlier characteristics."""
        for column in self.profile.numerical_columns:
            if column == self.profile.recommended_target:
                plan.scaling_plan.append({
                    "column": column,
                    "scaling": "Target - Excluded",
                    "reason": "The target is never scaled as an input feature.",
                })
                continue

            recommendation = {
                "column": column,
                "scaling": None,
                "reason": None,
            }

            distribution = self.distribution_lookup.get(column, {})
            outlier = self.outlier_lookup.get(column, {})

            severity = distribution.get("severity")
            outlier_status = outlier.get("status")

            if outlier_status in {"Moderate", "High"}:
                recommendation["scaling"] = "RobustScaler"
                recommendation["reason"] = (
                    "Feature contains significant outliers."
                )
            elif severity == "Symmetric":
                recommendation["scaling"] = "StandardScaler"
                recommendation["reason"] = (
                    "Feature is approximately symmetric."
                )
            elif severity in {"Mild", "Moderate", "Severe"}:
                recommendation["scaling"] = "RobustScaler"
                recommendation["reason"] = (
                    "Feature is skewed; robust scaling is a safer general recommendation."
                )
            else:
                recommendation["scaling"] = "StandardScaler"
                recommendation["reason"] = (
                    "Default scaling recommendation because distribution information is limited."
                )

            plan.scaling_plan.append(recommendation)

    def _feature_selection_strategy(self, plan):
        """Recommend whether each feature should be kept or removed."""
        for column in self.df.columns:
            if column == self.profile.recommended_target:
                plan.feature_selection_plan.append({
                    "column": column,
                    "action": "Target - Excluded",
                    "reason": "The target is not treated as an input feature.",
                })
                continue

            recommendation = {
                "column": column,
                "action": "Keep",
                "reason": "No issue detected.",
            }

            qualities = self.feature_quality_lookup.get(column, [])
            quality_names = {
                quality["quality"]
                for quality in qualities
            }

            if "Empty" in quality_names:
                recommendation["action"] = "Drop"
                recommendation["reason"] = (
                    "Feature contains only missing values."
                )
            elif "Constant" in quality_names:
                recommendation["action"] = "Drop"
                recommendation["reason"] = (
                    "Feature contains only one unique value."
                )
            elif "Unique Identifier" in quality_names:
                recommendation["action"] = "Drop"
                recommendation["reason"] = (
                    "Feature uniquely identifies each record."
                )
            elif "Near Constant" in quality_names:
                recommendation["action"] = "Consider Removing"
                recommendation["reason"] = (
                    "Feature is dominated by a single value."
                )
            elif "Low Variance" in quality_names:
                recommendation["action"] = "Consider Removing"
                recommendation["reason"] = (
                    "Feature shows very little variation."
                )

            # IMPORTANT: append inside the loop so every feature is represented.
            plan.feature_selection_plan.append(recommendation)

    def _outlier_treatment_strategy(self, plan):
        """Recommend an outlier treatment strategy."""
        for item in self.profile.outlier_summary:
            status = item["status"]
            recommendation = {
                "column": item["column"],
                "status": status,
                "treatment": None,
                "reason": None,
            }

            if status == "None":
                recommendation["treatment"] = "No Action"
                recommendation["reason"] = "No significant outliers detected."
            elif status == "Low":
                recommendation["treatment"] = "Keep"
                recommendation["reason"] = (
                    "A small number of outliers usually does not require automatic treatment."
                )
            elif status == "Moderate":
                recommendation["treatment"] = "Winsorization"
                recommendation["reason"] = (
                    "Cap extreme values while preserving the observations."
                )
            else:
                recommendation["treatment"] = "Investigate Before Removal"
                recommendation["reason"] = (
                    "A large number of outliers requires investigation before automatic removal."
                )

            plan.outlier_treatment_plan.append(recommendation)

    def _train_test_recommendation(self, plan):
        """Recommend a train/test split and cross-validation strategy."""
        rows = self.profile.rows

        if rows < 100:
            train_size = 80
            test_size = 20
            cross_validation = False
            cv_folds = 0
            reason = (
                "The dataset is very small; validate the target distribution and data quality carefully before training."
            )
        elif rows < 10000:
            train_size = 80
            test_size = 20
            cross_validation = True
            cv_folds = 5
            reason = (
                "An 80/20 split with 5-fold cross-validation provides a robust evaluation for this dataset size."
            )
        else:
            train_size = 90
            test_size = 10
            cross_validation = True
            cv_folds = 3
            reason = (
                "A 90/10 split preserves more training data while 3-fold cross-validation controls evaluation variance."
            )

        plan.train_test_plan = {
            "train_size": train_size,
            "test_size": test_size,
            "cross_validation": cross_validation,
            "cv_folds": cv_folds,
            "reason": reason,
        }

    def _pipeline_summary(self, plan):
        """Create a human-readable summary of the recommended pipeline."""
        pipeline = []

        for item in plan.missing_value_plan:
            if item["strategy"] != "No Action":
                pipeline.append(
                    f"{item['strategy']} -> {item['column']}"
                )

        for item in plan.encoding_plan:
            if item["encoding"] not in {"No Encoding", "Target - Excluded"}:
                pipeline.append(
                    f"{item['encoding']} -> {item['column']}"
                )

        for item in plan.scaling_plan:
            if item["scaling"] != "Target - Excluded":
                pipeline.append(
                    f"{item['scaling']} -> {item['column']}"
                )

        for item in plan.feature_selection_plan:
            if item["action"] not in {"Keep", "Target - Excluded"}:
                pipeline.append(
                    f"{item['action']} -> {item['column']}"
                )

        for item in plan.outlier_treatment_plan:
            if item["treatment"] not in {"No Action", "Keep"}:
                pipeline.append(
                    f"{item['treatment']} -> {item['column']}"
                )

        train_test = plan.train_test_plan
        pipeline.append(
            f"Train/Test Split -> {train_test['train_size']}/{train_test['test_size']}"
        )

        if train_test["cross_validation"]:
            pipeline.append(
                f"Cross Validation -> {train_test['cv_folds']} folds"
            )

        plan.pipeline_summary = pipeline