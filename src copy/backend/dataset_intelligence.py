from src.backend.dataset_profile import DatasetProfile 
import pandas as pd 

class DatasetIntelligence:
    """ Responsible for understanding a dataset and generating a structured DatasetProfile"""
    def __init__(self, dataframe , dataset_name):
        self.df = dataframe
        self.dataset_name = dataset_name
        
        
    def generate_profile(self):
        profile = DatasetProfile()
        profile.dataset_name = self.dataset_name
        self._get_basic_information(profile)
        self._get_quality_information(profile)
        self._cancluate_health_score(profile)
        self._get_column_information(profile)
        self._get_column_statistics(profile)
        self._analyze_correlation(profile)
        self._get_visualization_recommendations(profile)
        self._analyze_outliers(profile)
        self._analyze_distribution(profile)
        self._analyze_feature_quality(profile)
        self._analyze_target(profile)
        return profile 
    
    
    def _get_basic_information(self, profile):
        """Extract the basic information of the dataset"""
        profile.rows = self.df.shape[0]
        profile.columns = self.df.shape[1]
        profile.total_cells = profile.rows * profile.columns
        memory = self.df.memory_usage(deep=True).sum()
        profile.memory_usage = self._format_memory(memory)
        
        
    def _get_quality_information(self, profile):
        """
        Extracts dataset quality metrics.
        """

        # Total missing values in the dataset
        profile.missing_values = int(self.df.isna().sum().sum())
        
        # Total duplicate rows
        profile.duplicate_rows = int(self.df.duplicated().sum())
        
        total_cells = profile.rows * profile.columns
        
        if total_cells > 0:
            profile.missing_percentage = round(
                (profile.missing_values / total_cells) * 100, 2
            )
        if profile.rows > 0:
            profile.duplicate_percentage = round(
                (profile.duplicate_rows / profile.rows) * 100, 2
            )
        
    def _cancluate_health_score(self, profile):
        """Calculate the overall health of the dataset"""
        
        score = 100
        score -= profile.missing_percentage * 0.7
        score -= profile.duplicate_percentage * 0.3
        score = max(0, min(100, score))
        
        profile.health_score = round(score)
        
        if profile.health_score >= 90:
            profile.health_status = "Excellent"

        elif profile.health_score >= 75:
            profile.health_status = "Good"

        elif profile.health_score >= 50:
            profile.health_status = "Fair"

        else:
            profile.health_status = "Poor"
            
    
    def _get_column_information(self, profile):
        """
    I   dentifies different types of columns in the dataset.
        """
        profile.numerical_columns = (
            self.df.select_dtypes(include=["number"]).columns.tolist()
        )
        profile.categorical_columns = (
            self.df.select_dtypes(include=["object", "category"]).columns.tolist()
        )
        profile.boolean_columns = (
            self.df.select_dtypes(include=["bool"]).columns.tolist()
        )
        profile.datetime_columns = (
            self.df.select_dtypes(include=["datetime"]).columns.tolist()
        )
        
    def _get_column_statistics(self, profile):
        """
        Generates statistics for numerical and categorical columns.
        """
        numerical_stats = {}
        categorical_stats = {}        
        # Numerical columns
        for column in profile.numerical_columns:
            numerical_stats[column] = {
                "min" : self.df[column].min(),
                "max" : self.df[column].max(),
                "mean" : self.df[column].mean(),
                "median" : self.df[column].median(),
                "std" : round(self.df[column].std() , 2)
            }
        
        # Categorical Column
        for column in profile.categorical_columns : 
            categorical_stats[column] = {
                "unique-values" : self.df[column].nunique(),
                "most-frequent" : self.df[column].mode().iloc[0] if not self.df[column].mode().empty else None,
                "frequency" : self.df[column].value_counts().iloc[0] if not self.df[column].value_counts().empty else 0
            }
        profile.numerical_statistics = numerical_stats
        profile.categorical_statistics = categorical_stats
        
    def _get_visualization_recommendations(self, profile):
        """
        Generates visualization recommendations based on the dataset.
        """
        recommendations = []
        
        # Numerical Columns
        for column in profile.numerical_columns:
            recommendations.append({
                "chart" : "Histogram",
                "columns" : [column],
                "reason" : f"{column} is a numerical column"
            })
            
            recommendations.append({
                "chart" : "Box Plot",
                "columns" : [column],
                "reason" : f"{column} is a numerical column"
            })
        
        # Categorical Columns
        for column in profile.categorical_columns:
            recommendations.append({
                "chart" : "Bar chart",
                "columns" : [column],
                "reason" : f"{column} is a categorical column"
            })
            recommendations.append({
                "chart" : "Pie chart",
                "columns" : [column],
                "reason" : f"{column} is a categorical column"
            })
        # Scatter plot
        if len(profile.numerical_columns) >= 2:
            recommendations.append({
                "chart" : "Scatter Plot",
                "columns" : profile.numerical_columns[:2],
                "reason" : "Two numerical columns are available"
            })
        
        #Line chart
        if(len(profile.datetime_columns) >= 1 and len(profile.numerical_columns) >= 1):
            recommendations.append({
                "chart" : "Line Chart",
                "columns" : [profile.datetime_columns[0] , profile.numerical_columns[0]],
                "reason" : "Time - series data detected"
            })
        
        profile.recommended_visualizations = recommendations
        
        
    def _analyze_correlation(self, profile):
        # ---------------------------------------
        # 1. Get Numerical Columns
        # ---------------------------------------
        
        numerical_columns = profile.numerical_columns
        
        if len(numerical_columns) < 2:
            return
        
        # ---------------------------------------
        # 2. Compute Correlation Matrix
        # ---------------------------------------
        
        correlation_metrix = self.df[numerical_columns].corr(method='pearson')
        
        # ---------------------------------------
        # 3. Generate Correlation Insights
        # ---------------------------------------
        
        correlation_insights = []
        
        for i in range(len(numerical_columns)):
            for j in range(i+1, len(numerical_columns)):
                column1 = numerical_columns[i]
                column2 = numerical_columns[j]
                correlation = correlation_metrix.loc[column1, column2]
                if abs(correlation) < 0.40:
                    continue
                if correlation > 0:
                    correlation_type = 'Positive'
                    if correlation >= 0.80:
                        strength = "Very Strong Positive"
                    elif correlation >= 0.60:
                        strength = "Strong Positive"
                    else:
                        strength = "Moderate Positive"
                else:
                    correlation_type = "Negative"
                    if correlation <= -0.80:
                        strength = "Very Strong Negative"
                    elif correlation <= -0.60:
                        strength = "Strong Negative"
                    else:
                        strength = "Moderate Negative"
                if correlation > 0:
                    observation = (
                        f"{column1} and {column2} show a positive relationship."
                    )
                else:
                    observation = (
                        f"{column1} and {column2} show a positive relationship."
                    )
                correlation_insights.append({
                    'column_1': column1,
                    'column_2': column2,
                    'correlation': round(correlation , 2),
                    "strength": strength,
                    "type": correlation_type,
                    'observation' : observation
                })
                        

        # ---------------------------------------
        # 4. Sort Insights
        # ---------------------------------------

        correlation_insights.sort(key=lambda insight : abs(insight['correlation']), reverse=True)
        
        # ---------------------------------------
        # 5. Save Results
        # ---------------------------------------
        
        profile.correlation_metrix = correlation_metrix
        profile.correlation_insights = correlation_insights
        
        
    def _analyze_outliers(self, profile):
        
        #-------------------------------------------------------
        # Outlier Summary
        #-------------------------------------------------------
    
        numerical_columns = profile.numerical_columns
        if len(numerical_columns) == 0:
            return
        outlier_summary = []
        for column in numerical_columns:
            data = self.df[column].dropna()
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            outlier_count = len(outliers)
            percentage = (outlier_count / len(data))*100
            
            if outlier_count == 0:
                status = "None"

            elif percentage <= 5:
                status = "Low"

            elif percentage <= 15:
                status = "Moderate"

            else:
                status = "High"
            if status == "None":
                observation = "No outliers were detected."

            elif status == "Low":
                observation = "A small proportion of observations are identified as outliers."

            elif status == "Moderate":
                observation = "A moderate proportion of observations are identified as outliers."

            else:
                observation = "A high proportion of observations are identified as outliers."
            
            outlier_summary.append({
                'column' : column,
                'outliers' : outlier_count,
                'percentage' : round(percentage, 2),
                'lower_bound' : round(lower_bound, 2),
                'upper_bound' : round(upper_bound, 2),
                'status' : status,
                'observation' : observation
            })
        
        outlier_summary.sort(
            key=lambda item : item['percentage'],
            reverse=True
        )
        
        profile.outlier_summary = outlier_summary
    
    
    def _analyze_distribution(self, profile):
        numerical_columns = profile.numerical_columns
        if len(numerical_columns) == 0:
            return
        distribution_summary = []
        for column in numerical_columns:
            data = self.df[column].dropna()
            if len(data) < 8:
                distribution_summary.append(
                    {
                        "column": column,
                        "skewness": None,
                        "distribution": "Insufficient Data",
                        "severity": "Unknown",
                        "observation": "There are not enough observations to analyze the distribution."
                    })
                continue
            
            skewness = data.skew()
            
            absolute_skewness = abs(skewness)
            if absolute_skewness < 0.5:
                severity = "Symmetric"

            elif absolute_skewness < 1:
                severity = "Mild"

            elif absolute_skewness < 2:
                severity = "Moderate"

            else:
                severity = "Severe"
            #-----------------------------------------------------
            if skewness > 0.5:
                distribution = "Right Skewed"

            elif skewness < -0.5:
                distribution = "Left Skewed"

            else:
                distribution = "Approximately Symmetric"
            #---------------------––---–----–------------–-–-–––––––    
            if severity == "Symmetric":
                observation = "The feature is approximately symmetric."

            elif severity == "Mild":
                observation = "The feature is mildly skewed."

            elif severity == "Moderate":
                observation = "The feature is moderately skewed."

            else:
                observation = "The feature is heavily skewed."
                
            distribution_summary.append({
                'column' : column,
                'skewness' : round(skewness, 2),
                'distribution' : distribution,
                'severity' : severity,
                'observation' : observation
            })
        
        distribution_summary.sort(
            key=lambda item : abs(item['skewness']) if item['skewness'] is not None else -1,
            reverse=True
        )
            
        profile.distribution_summary = distribution_summary
     
     
    # Helper Function For _analyze_feature_quality   
    def _add_feature_quality(self, qualities, quality, status, observation):
        qualities.append({
            "quality": quality,
            "status": status,
            "observation": observation
        })
    
    def _analyze_feature_quality(self, profile):
        """
        Analyze the quality of each feature and detect meaningful characteristics.
        """
        feature_quality_summary = []

        for column in self.df.columns:
            data = self.df[column]
            qualities = []
            
            total_rows = len(data)
            non_null = data.dropna()
            
            # ---------------------------------------------
            # Empty Column
            # ---------------------------------------------
            if non_null.empty:
                self._add_feature_quality(
                    qualities,
                    "Empty",
                    "High",
                    "This feature contains no usable values."
                )
            
            # ---------------------------------------------
            # Constant Feature
            # ---------------------------------------------
            if non_null.nunique() == 1:
                self._add_feature_quality(
                    qualities,
                    "Constant",
                    "High",
                    "This feature contains only one unique value."
                )
            
            # ---------------------------------------------
            # Unique Identifier
            # ---------------------------------------------
            if len(non_null) > 0 and non_null.nunique() == len(non_null):
                self._add_feature_quality(
                    qualities,
                    "Unique Identifier",
                    "High",
                    "This feature appears to uniquely identify each record."
                )
            
            # ---------------------------------------------
            # Near Constant
            # ---------------------------------------------
            if len(non_null) > 0:
                dominant_ratio = non_null.value_counts(normalize=True).iloc[0]
                if dominant_ratio >= 0.95 and non_null.nunique() > 1:
                    self._add_feature_quality(
                        qualities,
                        "Near Constant",
                        "Moderate",
                        "The feature is dominated by a single value."
                    )
            
            # ---------------------------------------------
            # High Cardinality
            # --------------------------------------------
            if len(non_null) > 0:
                unique_ratio = non_null.nunique() / len(non_null)
                if unique_ratio >= 0.90 and non_null.nunique() > 20:
                    self._add_feature_quality(
                        qualities,
                        "High Cardinality",
                        "Moderate",
                        "The feature contains large number of unique values."
                    )
                    
            # ---------------------------------------------
            # Low Variance
            # ---------------------------------------------
            if pd.api.types.is_numeric_dtype(data):
                variance = non_null.var()
                if variance is not None and not pd.isna(variance) and variance < 0.01:
                    self._add_feature_quality(
                        qualities,
                        "Low Variance",
                        "Low",
                        "The feature exhibits very little variation."
                    )
                    
            # ---------------------------------------------
            # Store only meaningful columns
            # ---------------------------------------------
            if qualities:
                feature_quality_summary.append({
                    "column" : column,
                    "dtype" : str(data.dtype),
                    "qualities" : qualities
                })
                
        feature_quality_summary.sort(
            key=lambda item: len(item["qualities"]),
            reverse=True
        )
        
        profile.feature_quality_summary = feature_quality_summary
    
    
    def _analyze_target(self, profile):
        """
        Analyze the dataset and identify the most probable target column.
        """

        candidate_scores = {}

        target_analysis = {
            "target_column": None,
            "confidence": "Low",
            "problem_type": None,
            "classification_type": None,
            "num_classes": None,
            "class_distribution": None,
            "is_imbalanced": None,
            "observation": None
        }
        
        target_keywords = {"target","label","class","output","result","outcome","price","salary","sales","profit","purchased","survived", "diagnosis", "income", "response", "default", "loan_status", "fraud", "churn"}

        
        for column in self.df.columns:
            candidate_scores[column] = 0            
            column_name = column.lower()
            for keyword in target_keywords:
                if keyword in column_name:
                    candidate_scores[column] += 40
                    break
            
            if column == self.df.columns[-1]:
                candidate_scores[column] += 15
            
            missing_ratio = self.df[column].isna().mean()
            if missing_ratio <= 0.05:
                candidate_scores[column] += 10
                
            for feature in profile.feature_quality_summary:
                if feature["column"] != column:
                    continue
                
                for quality in feature['qualities']:
                    if quality["quality"] == "Unique Identifier":
                        candidate_scores[column] -= 100
                    if quality['quality'] == "Constant":
                        candidate_scores[column] -= 100
                    if quality['quality'] == 'Empty':
                        candidate_scores[column] -= 100
                    
            series = self.df[column].dropna()
            unique_values = series.nunique()      # <-- CHANGED

            if pd.api.types.is_numeric_dtype(series):
                if unique_values > 20:
                    candidate_scores[column] += 10
                elif 2 <= unique_values <= 20:
                    candidate_scores[column] += 20
            else:                                 # <-- CHANGED
                if 2 <= unique_values <= 20:
                    candidate_scores[column] += 20
        
        # --------------------------------------------------
        # Select the best target column
        # --------------------------------------------------
        best_target = max(candidate_scores, key=candidate_scores.get)

        target_analysis["target_column"] = best_target
        
        # ---------------------------------------------
        # Confidence
        # ---------------------------------------------
        sorted_scores = sorted(candidate_scores.values(), reverse=True)
        if len(sorted_scores) == 1:
            target_analysis['confidence'] = 'High'
        else:
            difference = sorted_scores[0] - sorted_scores[1]
            if difference >= 30:
                target_analysis['confidence'] = 'High'
            elif difference >= 15:
                target_analysis['confidence'] = "Moderate"
            else:
                target_analysis['confidence'] = "Low"
        
        # ---------------------------------------------
        # Analyze Selected Target
        # ---------------------------------------------  
        target_series = self.df[best_target].dropna()
        unique_values = target_series.nunique()
        if pd.api.types.is_numeric_dtype(target_series) and unique_values > 20:
            target_analysis['problem_type'] = 'Regression'
            target_analysis['observation'] = "The selected target appears to represent a continuous regression problem."
        else:
            target_analysis["problem_type"] = "Classification"
            target_analysis["num_classes"] = unique_values
            
            # ---------------------------------------------
            # Classification Type
            # ---------------------------------------------
            if unique_values == 2:
                target_analysis["classification_type"] = "Binary"
            else:
                target_analysis['classification_type'] = "Multiclass"
        
            # ---------------------------------------------
            # Class Distribution
            # ---------------------------------------------
            class_distribution = (
                target_series.value_counts().to_dict()
            )

            target_analysis["class_distribution"] = class_distribution
        
            # ---------------------------------------------
            # Imbalance Detection
            # ---------------------------------------------
            if class_distribution:
                total_samples = sum(class_distribution.values())
                minority_ratio = (
                    min(class_distribution.values()) / total_samples
                )
                target_analysis["is_imbalanced"] = (
                    minority_ratio < 0.20
                )
            else:
                target_analysis["is_imbalanced"] = None
            
            # ---------------------------------------------
            # Observation
            # ---------------------------------------------
            if target_analysis["classification_type"] == "Binary":

                if target_analysis["is_imbalanced"]:

                    target_analysis["observation"] = (
                        "The selected target appears to represent an imbalanced binary classification problem."
                    )

                else:

                    target_analysis["observation"] = (
                        "The selected target appears to represent a balanced binary classification problem."
                    )

            else:

                if target_analysis["is_imbalanced"]:

                    target_analysis["observation"] = (
                        "The selected target appears to represent an imbalanced multiclass classification problem."
                    )

                else:

                    target_analysis["observation"] = (
                        "The selected target appears to represent a balanced multiclass classification problem."
                    )
            
        profile.target_analysis = target_analysis
        
    
    
    def _format_memory(self, memory):
        """
        Converts memory from bytes into a readable format.
        """

        units = ["B", "KB", "MB", "GB"]
        
        size = float(memory)
        for unit in units:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} 2TB"