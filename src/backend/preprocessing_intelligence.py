from src.backend.preprocessing_plan import PreprocessingPlan
import pandas as pd 

class PreprocessingIntelligence:

    def __init__(self, df, profile):
        self.df = df
        self.profile = profile
        self.feature_quality_lookup = {
            feature["column"]: feature["qualities"]
            for feature in profile.feature_quality_summary
        }

    def generate_plan(self):
        plan = PreprocessingPlan()
        self._missing_value_strategy(plan)
        return plan

    def _missing_value_strategy(self, plan):
        """
        Analyze missing values and recommend an imputation strategy.
        """

        for column in self.df.columns:
            missing_percentage = self.df[column].isna().mean() * 100
            recommendation = {
                "column": column,
                "missing_percentage": round(missing_percentage, 2),
                "strategy": None,
                "reason": None
            }
            # ---------------------------------------------
            # No Missing Values
            # ---------------------------------------------
            if missing_percentage == 0:
                recommendation["strategy"] = "No Action"
                recommendation["reason"] = "Column has no missing values."

            else:
                # ---------------------------------------------
                # Check if the column is Empty
                # ---------------------------------------------
                qualities = self.feature_quality_lookup.get(column, [])
                is_empty = any(
                    quality["quality"] == "Empty"
                    for quality in qualities
                )
                # ---------------------------------------------
                # Empty Column
                # ---------------------------------------------
                if is_empty:
                    recommendation["strategy"] = "Drop Column"
                    recommendation["reason"] = (
                        "Column contains only missing values."
                    )
                # ---------------------------------------------
                # Numeric / Categorical Strategy
                # ---------------------------------------------
                else:
                    if pd.api.types.is_numeric_dtype(self.df[column]):
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