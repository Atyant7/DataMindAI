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
        self._get_visualization_recommendations(profile)
        self._generate_recommendations(profile)
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
        
    def _generate_recommendations(self, profile):
        """
        Generates intelligent recommendations for the dataset.
        """
        
        recommendations = []
        
        # Missing values
        if profile.missing_values > 0 :
            recommendations.append({
                "type": "Cleaning",
                "priority": "High",
                "message": (
                    "Missing values detected. "
                    "Consider imputing or removing them."
                )   
            }) 
        # Duplicate Rows
        if profile.duplicate_rows > 0:

            recommendations.append({
                "type": "Cleaning",
                "priority": "High",
                "message": (
                    "Duplicate rows detected. "
                    "Consider removing them."
                )
            })

        # Categorical Columns
        if len(profile.categorical_columns) > 0:

            recommendations.append({
                "type": "Machine Learning",
                "priority": "Medium",
                "message": (
                    "Categorical columns should be encoded "
                    "before training most machine learning models."
                )
            })

        # Health Score
        if profile.health_score < 75:

            recommendations.append({
                "type": "Quality",
                "priority": "High",
                "message": (
                    "Dataset quality is below the recommended level. "
                    "Perform data cleaning before analysis."
                )
            })

        # Numerical Columns
        if len(profile.numerical_columns) > 0:

            recommendations.append({
                "type": "Visualization",
                "priority": "Low",
                "message": (
                    "Use histograms and box plots to explore "
                    "the distribution of numerical features."
                )
            })
        
        profile.recommendations = recommendations
    
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