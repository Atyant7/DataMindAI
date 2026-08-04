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
        
        self.distribution_lookup = {
            item['column']: item for item in profile.distribution_summary
        }
        
        self.outlier_lookup = {
            item['column'] : item for item in profile.outlier_summary
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
            

    def _encoding_strategy(self, plan):
        """
        Analyze categorical features and recommend an encoding strategy.
        """
        for column in self.df.columns:
            # ---------------------------------------------
            # Skip Numerical Columns
            # ---------------------------------------------
            if pd.api.types.is_numeric_dtype(self.df[column]):
                continue
            recommendation = {
                "column": column,
                "encoding": None,
                "reason": None
            }
            # ---------------------------------------------
            # Retrieve Feature Quality
            # ---------------------------------------------
            qualities = self.feature_quality_lookup.get(column, {})
            
            # ---------------------------------------------
            # Retrieve Column Statistics
            # ---------------------------------------------
            stats = self.profile.categorical_statistics.get(column, {})
            unique_values = stats.get('unique-values' , 0)
            
            # ---------------------------------------------
            # Unique Identifier
            # ---------------------------------------------
            is_identifier = any(quality['quality'] == 'Unique Identifier' for quality in qualities)
            
            if is_identifier:
                recommendation['encoding'] = 'No Encoding'
                recommendation['reason'] = 'Unique identifiers should not be encoded.'
                plan.encoding_plan.append(recommendation)
                continue
            
            # ---------------------------------------------
            # High Cardinality
            # ---------------------------------------------
            is_high_cardinality = any(
                quality['quality'] == 'High Cardinatlity' for quality in qualities
            )
            
            # ---------------------------------------------
            # Binary Feature
            # ---------------------------------------------
            if unique_values == 2:
                recommendation['encoding'] = 'Label Encoding'
                recommendation['reason'] = 'Binary categorical feature.'
            
            # ---------------------------------------------
            # High Cardinality Feature
            # ---------------------------------------------
            elif is_high_cardinality:
                recommendation['encoding'] = 'Frequency Encoding'
                recommendation['reason'] = 'High cardinality categorical feature.'
            
            # ---------------------------------------------
            # Regular Categorical Feature
            # ---------------------------------------------
            else:
                recommendation['encoding'] = 'One Hot Encoding'
                recommendation['reason'] = 'Categorical feature with few unique values.'
            
            plan.encoding_plan.append(recommendation)
            
            
    def _scaling_strategy(self, plan):
        """
        Analyze numerical features and recommend an appropriate scaling strategy.
        """
        for column in self.df.columns:
            # ---------------------------------------------
            # Skip Non-Numerical Columns
            # ---------------------------------------------
            if not pd.api.types.is_numeric_dtype(self.df[column]):
                continue
            recommendation = {
                'column' : column,
                'scaling' : None,
                'reason' : None
            }
            
            distribution = self.distribution_lookup.get(column, {})
            outlier = self.outlier_lookup.get(column, {})
            
            severity = distribution.get('severity')
            outlier_status = outlier.get("status")
            
            if outlier_status in ['Moderate', 'High']:
                recommendation['scaling'] = 'RobustScaler'
                recommendation['reason'] = 'Feature contains significant outliers.'
            elif severity == "Symmetric":
                recommendation['scaling'] = 'StandardScaler'
                recommendation['reason'] = 'Feature is approximately symmetric.'
            else:
                recommendation['scaling'] = 'MinMaxScaler'
                recommendation['reason'] = 'Feature is skewed without significant outliers.'
            
            plan.scaling_plan.append(recommendation)
            
            
    def _feature_selection_strategy(self, plan):
        """
        Analyze feature quality and recommend whether each feature should be kept or removed.
        """
        for column in self.df.columns:
            recommendation = {
                'column' : column,
                'action' : "Keep",
                'reason' : "No issue detected"
            }
            qualities = self.feature_quality_lookup.get(column, [])
            quality_names = {quality["quality"] for quality in qualities}
            
            # ---------------------------------------------
            # Empty Feature
            # ---------------------------------------------
            if "Empty" in quality_names:
                recommendation['action'] = 'Drop'
                recommendation['reason'] = 'Feature contains only missing values.'
            
            # ---------------------------------------------
            # Constant Feature
            # ---------------------------------------------
            elif "Constant" in quality_names:
                recommendation['action'] = 'Drop'
                recommendation['reason'] = 'Feature contains only one unique value.'
            
            # ---------------------------------------------
            # Unique Identifier
            # ---------------------------------------------
            elif "Unique Identifier" in quality_names:
                recommendation["action"] = "Drop"
                recommendation["reason"] = "Feature uniquely identifies each record."
            
            # ---------------------------------------------
            # Near Constant
            # ---------------------------------------------
            elif "Near Constant" in quality_names:
                recommendation['action'] = "Consider Removing"
                recommendation['reason'] = 'Feature is dominated by a single value.'
            
            # ---------------------------------------------
            # Low Variance
            # ---------------------------------------------
            elif "Low Variance" in quality_names:
                recommendation["action"] = "Consider Removing"
                recommendation["reason"] = "Feature shows very little variation."

        plan.feature_selection_plan.append(recommendation)
        
    def _outlier_treatment_strategy(self, plan):
        """
        Recommend an appropriate treatment strategy
        based on detected outliers.
        """

        for item in self.profile.outlier_summary:

            recommendation = {
                "column": item["column"],
                "status": item["status"],
                "treatment": None,
                "reason": None
            }

            status = item["status"]

            # ---------------------------------------------
            # No Outliers
            # ---------------------------------------------
            if status == "None":
                recommendation["treatment"] = "No Action"
                recommendation["reason"] = (
                    "No significant outliers detected."
                )
            # ---------------------------------------------
            # Low Outliers
            # ---------------------------------------------
            elif status == "Low":
                recommendation["treatment"] = "Keep"
                recommendation["reason"] = (
                    "A small number of outliers usually does not require treatment."
                )
            # ---------------------------------------------
            # Moderate Outliers
            # ---------------------------------------------
            elif status == "Moderate":
                recommendation["treatment"] = "Winsorization"

                recommendation["reason"] = (
                    "Cap extreme values while preserving the dataset."
                )

            # ---------------------------------------------
            # High Outliers
            # ---------------------------------------------
            else:
                recommendation["treatment"] = "Investigate Before Removal"
                recommendation["reason"] =  "A large number of outliers requires further analysis." 

            plan.outlier_treatment_plan.append(recommendation)
            
            
    def _train_test_recommendation(self, plan):
        """
        Recommend an appropriate train-test split.
        """
        rows = self.profile.rows

        recommendation = {
            "train_size": None,
            "test_size": None,
            "cross_validation": False,
            "reason": None
        }

        # ---------------------------------------------
        # Small Dataset
        # ---------------------------------------------
        if rows < 1000:

            recommendation["train_size"] = 80
            recommendation["test_size"] = 20
            recommendation["cross_validation"] = True

            recommendation["reason"] = (
                "Small datasets benefit from cross-validation."
            )

        # ---------------------------------------------
        # Medium Dataset
        # ---------------------------------------------
        elif rows < 10000:

            recommendation["train_size"] = 80
            recommendation["test_size"] = 20

            recommendation["reason"] = (
                "80/20 split provides a balanced evaluation."
            )

        # ---------------------------------------------
        # Large Dataset
        # ---------------------------------------------
        else:

            recommendation["train_size"] = 90
            recommendation["test_size"] = 10

            recommendation["reason"] = (
                "Large datasets require a smaller test set."
            )

        plan.train_test_plan = recommendation
        
    def _pipeline_summary(self, plan):
        """
        Generate a preprocessing pipeline summary.
        """
        pipeline = []
        for item in plan.missing_value_plan:
            if item['strategy'] == "No Action":
                pipeline.append(f"{item['strategy']} -> {item['column']}")
        
        # ---------------------------------------------
        # Encoding Strategy
        # ---------------------------------------------
        for item in plan.encoding_plan:
            if item['encoding'] == "No Encoding":
                pipeline.append(f"{item['encoding']} -> {item['column']}")
        
        # ---------------------------------------------
        # Scaling Strategy
        # ---------------------------------------------
        for item in plan.scaling_plan:
            pipeline.append(f"{item['scaling']} -> {item['column']}")
            
        # ---------------------------------------------
        # Feature Selection
        # ---------------------------------------------
        for item in plan.feature_selection_plan:
            if item["action"] != "Keep":
                pipeline.append(f"{item['action']} -> {item['column']}")
                
        # ---------------------------------------------
        # Outlier Treatment
        # ---------------------------------------------
        for item in plan.outlier_treatment_plan:
            if item['treatment'] not in ['No Action', 'Keep']:
                pipeline.append(f"{item['treatment']} -> {item['column']}")
        
        # ---------------------------------------------
        # Train/Test Split
        # ---------------------------------------------
        pipeline.append(f"Train/Test Split -> "f"{plan.train_test_plan['train_size']}/"f"{plan.train_test_plan['test_size']}")
        if plan.train_test_plan["cross_validation"]:
            pipeline.append(
                "Cross Validation Recommended"
            )
        plan.pipeline_summary = pipeline