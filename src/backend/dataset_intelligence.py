"""Dataset profiling and data-intelligence logic."""

import pandas as pd

from src.backend.dataset_profile import DatasetProfile


class DatasetIntelligence:
    """Understand a dataset and generate a structured DatasetProfile."""

    TARGET_KEYWORDS = {
        "target",
        "label",
        "class",
        "output",
        "result",
        "outcome",
        "price",
        "salary",
        "sales",
        "profit",
        "purchased",
        "survived",
        "diagnosis",
        "income",
        "response",
        "default",
        "loan_status",
        "fraud",
        "churn",
    }

    def __init__(self, dataframe, dataset_name):
        self.df = dataframe
        self.dataset_name = dataset_name

    def generate_profile(self):
        profile = DatasetProfile()
        profile.dataset_name = self.dataset_name

        self._get_basic_information(profile)
        self._get_quality_information(profile)
        self._calculate_health_score(profile)
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
        profile.rows = int(self.df.shape[0])
        profile.columns = int(self.df.shape[1])
        profile.total_cells = profile.rows * profile.columns

        memory = self.df.memory_usage(deep=True).sum()
        profile.memory_usage = self._format_memory(memory)

    def _get_quality_information(self, profile):
        """Calculate high-level data-quality metrics."""
        profile.missing_values = int(self.df.isna().sum().sum())
        profile.duplicate_rows = int(self.df.duplicated().sum())

        total_cells = profile.total_cells

        profile.missing_percentage = (
            round((profile.missing_values / total_cells) * 100, 2)
            if total_cells
            else 0.0
        )

        profile.duplicate_percentage = (
            round((profile.duplicate_rows / profile.rows) * 100, 2)
            if profile.rows
            else 0.0
        )

        issues = []

        if profile.missing_values:
            issues.append({
                "type": "Missing Values",
                "severity": "High" if profile.missing_percentage > 20 else "Moderate",
                "message": (
                    f"{profile.missing_percentage}% of dataset cells contain missing values."
                ),
            })

        if profile.duplicate_rows:
            issues.append({
                "type": "Duplicate Rows",
                "severity": "Moderate" if profile.duplicate_percentage > 1 else "Low",
                "message": (
                    f"{profile.duplicate_percentage}% of rows are duplicates."
                ),
            })

        profile.quality_issues = issues

    def _calculate_health_score(self, profile):
        """Calculate an initial dataset health score.

        This score is intentionally limited to basic quality indicators at this
        stage. Leakage and model reliability will be evaluated by dedicated
        components in the next architecture phase.
        """
        score = 100.0
        score -= profile.missing_percentage * 0.7
        score -= profile.duplicate_percentage * 0.3

        profile.health_score = round(max(0.0, min(100.0, score)))

        if profile.health_score >= 90:
            profile.health_status = "Excellent"
        elif profile.health_score >= 75:
            profile.health_status = "Good"
        elif profile.health_score >= 50:
            profile.health_status = "Fair"
        else:
            profile.health_status = "Poor"

    def _get_column_information(self, profile):
        """Identify numerical, categorical, boolean and datetime columns."""
        profile.numerical_columns = (
            self.df.select_dtypes(include=["number"]).columns.tolist()
        )

        profile.categorical_columns = (
            self.df.select_dtypes(include=["object", "category", "string"])
            .columns.tolist()
        )

        profile.boolean_columns = (
            self.df.select_dtypes(include=["bool", "boolean"]).columns.tolist()
        )

        profile.datetime_columns = [
            column
            for column in self.df.columns
            if pd.api.types.is_datetime64_any_dtype(self.df[column])
        ]

    def _get_column_statistics(self, profile):
        """Generate statistics for numerical and categorical columns."""
        numerical_stats = {}
        categorical_stats = {}

        for column in profile.numerical_columns:
            series = self.df[column]
            numerical_stats[column] = {
                "min": series.min(),
                "max": series.max(),
                "mean": series.mean(),
                "median": series.median(),
                "std": round(series.std(), 2) if not pd.isna(series.std()) else None,
            }

        for column in profile.categorical_columns:
            series = self.df[column]
            mode = series.mode(dropna=True)
            value_counts = series.value_counts(dropna=True)

            categorical_stats[column] = {
                "unique_values": int(series.nunique(dropna=True)),
                "most_frequent": mode.iloc[0] if not mode.empty else None,
                "frequency": int(value_counts.iloc[0]) if not value_counts.empty else 0,
            }

        profile.numerical_statistics = numerical_stats
        profile.categorical_statistics = categorical_stats

    def _get_visualization_recommendations(self, profile):
        """Generate basic visualization recommendations."""
        recommendations = []

        for column in profile.numerical_columns:
            recommendations.append({
                "chart": "Histogram",
                "columns": [column],
                "reason": f"{column} is a numerical column.",
            })
            recommendations.append({
                "chart": "Box Plot",
                "columns": [column],
                "reason": f"{column} is a numerical column.",
            })

        for column in profile.categorical_columns:
            recommendations.append({
                "chart": "Bar Chart",
                "columns": [column],
                "reason": f"{column} is a categorical column.",
            })

        if len(profile.numerical_columns) >= 2:
            recommendations.append({
                "chart": "Scatter Plot",
                "columns": profile.numerical_columns[:2],
                "reason": "Two numerical columns are available.",
            })

        if profile.datetime_columns and profile.numerical_columns:
            recommendations.append({
                "chart": "Line Chart",
                "columns": [
                    profile.datetime_columns[0],
                    profile.numerical_columns[0],
                ],
                "reason": "Datetime and numerical columns indicate a possible time series.",
            })

        profile.recommended_visualizations = recommendations

    def _analyze_correlation(self, profile):
        numerical_columns = profile.numerical_columns

        if len(numerical_columns) < 2:
            profile.correlation_matrix = None
            profile.correlation_insights = []
            return

        correlation_matrix = self.df[numerical_columns].corr(method="pearson")
        correlation_insights = []

        for i in range(len(numerical_columns)):
            for j in range(i + 1, len(numerical_columns)):
                column1 = numerical_columns[i]
                column2 = numerical_columns[j]
                correlation = correlation_matrix.loc[column1, column2]

                if pd.isna(correlation) or abs(correlation) < 0.40:
                    continue

                if correlation > 0:
                    correlation_type = "Positive"
                    if correlation >= 0.80:
                        strength = "Very Strong Positive"
                    elif correlation >= 0.60:
                        strength = "Strong Positive"
                    else:
                        strength = "Moderate Positive"
                    observation = (
                        f"{column1} and {column2} show a positive relationship."
                    )
                else:
                    correlation_type = "Negative"
                    if correlation <= -0.80:
                        strength = "Very Strong Negative"
                    elif correlation <= -0.60:
                        strength = "Strong Negative"
                    else:
                        strength = "Moderate Negative"
                    observation = (
                        f"{column1} and {column2} show a negative relationship."
                    )

                correlation_insights.append({
                    "column_1": column1,
                    "column_2": column2,
                    "correlation": round(float(correlation), 2),
                    "strength": strength,
                    "type": correlation_type,
                    "observation": observation,
                })

        correlation_insights.sort(
            key=lambda insight: abs(insight["correlation"]),
            reverse=True,
        )

        profile.correlation_matrix = correlation_matrix
        profile.correlation_insights = correlation_insights

    def _analyze_outliers(self, profile):
        """Detect numerical outliers using the IQR rule."""
        outlier_summary = []

        for column in profile.numerical_columns:
            data = self.df[column].dropna()

            if data.empty:
                continue

            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)

            outliers = data[(data < lower_bound) | (data > upper_bound)]
            outlier_count = int(len(outliers))
            percentage = (outlier_count / len(data)) * 100

            if outlier_count == 0:
                status = "None"
                observation = "No outliers were detected."
            elif percentage <= 5:
                status = "Low"
                observation = (
                    "A small proportion of observations are identified as outliers."
                )
            elif percentage <= 15:
                status = "Moderate"
                observation = (
                    "A moderate proportion of observations are identified as outliers."
                )
            else:
                status = "High"
                observation = (
                    "A high proportion of observations are identified as outliers."
                )

            outlier_summary.append({
                "column": column,
                "outliers": outlier_count,
                "percentage": round(percentage, 2),
                "lower_bound": round(float(lower_bound), 2),
                "upper_bound": round(float(upper_bound), 2),
                "status": status,
                "observation": observation,
            })

        outlier_summary.sort(
            key=lambda item: item["percentage"],
            reverse=True,
        )

        profile.outlier_summary = outlier_summary

    def _analyze_distribution(self, profile):
        """Analyze skewness for numerical columns."""
        distribution_summary = []

        for column in profile.numerical_columns:
            data = self.df[column].dropna()

            if len(data) < 8:
                distribution_summary.append({
                    "column": column,
                    "skewness": None,
                    "distribution": "Insufficient Data",
                    "severity": "Unknown",
                    "observation": (
                        "There are not enough observations to analyze the distribution."
                    ),
                })
                continue

            skewness = float(data.skew())
            absolute_skewness = abs(skewness)

            if absolute_skewness < 0.5:
                severity = "Symmetric"
            elif absolute_skewness < 1:
                severity = "Mild"
            elif absolute_skewness < 2:
                severity = "Moderate"
            else:
                severity = "Severe"

            if skewness > 0.5:
                distribution = "Right Skewed"
            elif skewness < -0.5:
                distribution = "Left Skewed"
            else:
                distribution = "Approximately Symmetric"

            if severity == "Symmetric":
                observation = "The feature is approximately symmetric."
            elif severity == "Mild":
                observation = "The feature is mildly skewed."
            elif severity == "Moderate":
                observation = "The feature is moderately skewed."
            else:
                observation = "The feature is heavily skewed."

            distribution_summary.append({
                "column": column,
                "skewness": round(skewness, 2),
                "distribution": distribution,
                "severity": severity,
                "observation": observation,
            })

        distribution_summary.sort(
            key=lambda item: abs(item["skewness"])
            if item["skewness"] is not None
            else -1,
            reverse=True,
        )

        profile.distribution_summary = distribution_summary

    @staticmethod
    def _add_feature_quality(qualities, quality, status, observation):
        qualities.append({
            "quality": quality,
            "status": status,
            "observation": observation,
        })

    def _analyze_feature_quality(self, profile):
        """Detect meaningful quality characteristics for every feature."""
        feature_quality_summary = []

        for column in self.df.columns:
            data = self.df[column]
            qualities = []
            non_null = data.dropna()

            if non_null.empty:
                self._add_feature_quality(
                    qualities,
                    "Empty",
                    "High",
                    "This feature contains no usable values.",
                )

            if non_null.nunique() == 1 and not non_null.empty:
                self._add_feature_quality(
                    qualities,
                    "Constant",
                    "High",
                    "This feature contains only one unique value.",
                )

            # A column being unique does not automatically make it an identifier.
            # Continuous measurements (age, income, sensor values, etc.) can also
            # contain unique values. Require an identifier-like name or a
            # non-numeric unique column before assigning this quality.
            identifier_name = str(column).lower()
            identifier_keywords = (
                "id",
                "uuid",
                "key",
                "identifier",
                "customer_id",
                "user_id",
                "record_id",
            )
            looks_like_identifier = any(
                keyword == identifier_name
                or identifier_name.endswith(f"_{keyword}")
                or identifier_name.startswith(f"{keyword}_")
                for keyword in identifier_keywords
            )

            if (
                len(non_null) > 0
                and non_null.nunique() == len(non_null)
                and (
                    looks_like_identifier
                    or not pd.api.types.is_numeric_dtype(data)
                )
            ):
                self._add_feature_quality(
                    qualities,
                    "Unique Identifier",
                    "High",
                    "This feature appears to uniquely identify each record.",
                )

            if len(non_null) > 0:
                dominant_ratio = non_null.value_counts(normalize=True).iloc[0]
                if dominant_ratio >= 0.95 and non_null.nunique() > 1:
                    self._add_feature_quality(
                        qualities,
                        "Near Constant",
                        "Moderate",
                        "The feature is dominated by a single value.",
                    )

                unique_ratio = non_null.nunique() / len(non_null)
                if unique_ratio >= 0.90 and non_null.nunique() > 20:
                    self._add_feature_quality(
                        qualities,
                        "High Cardinality",
                        "Moderate",
                        "The feature contains a large number of unique values.",
                    )

            if pd.api.types.is_numeric_dtype(data) and not non_null.empty:
                variance = non_null.var()
                if (
                    variance is not None
                    and not pd.isna(variance)
                    and variance < 0.01
                ):
                    self._add_feature_quality(
                        qualities,
                        "Low Variance",
                        "Low",
                        "The feature exhibits very little variation.",
                    )

            if qualities:
                feature_quality_summary.append({
                    "column": column,
                    "dtype": str(data.dtype),
                    "qualities": qualities,
                })

        feature_quality_summary.sort(
            key=lambda item: len(item["qualities"]),
            reverse=True,
        )

        profile.feature_quality_summary = feature_quality_summary

    def _analyze_target(self, profile):
        """Rank possible target columns without blindly trusting one column."""
        candidate_scores = {}
        candidate_details = {}

        for column in self.df.columns:
            score = 0
            column_name = str(column).lower()

            if any(keyword in column_name for keyword in self.TARGET_KEYWORDS):
                score += 40

            if column == self.df.columns[-1]:
                score += 15

            missing_ratio = self.df[column].isna().mean()
            if missing_ratio <= 0.05:
                score += 10

            qualities = next(
                (
                    feature["qualities"]
                    for feature in profile.feature_quality_summary
                    if feature["column"] == column
                ),
                [],
            )
            quality_names = {quality["quality"] for quality in qualities}

            if quality_names & {"Unique Identifier", "Constant", "Empty"}:
                score -= 100

            series = self.df[column].dropna()
            unique_values = series.nunique()

            if unique_values == 0:
                score -= 100
            elif 2 <= unique_values <= 20:
                score += 20
            elif pd.api.types.is_numeric_dtype(series) and unique_values > 20:
                score += 10

            problem_type = self._infer_problem_type(series)

            candidate_scores[column] = score
            candidate_details[column] = {
                "column": column,
                "score": score,
                "problem_type": problem_type,
                "unique_values": int(unique_values),
            }

        ranked_candidates = sorted(
            candidate_details.values(),
            key=lambda item: item["score"],
            reverse=True,
        )

        profile.possible_target_columns = ranked_candidates[:5]

        if not ranked_candidates:
            profile.recommended_target = None
            profile.detected_task = ""
            profile.target_analysis = {
                "target_column": None,
                "confidence": "Low",
                "problem_type": None,
                "recommendation_requires_confirmation": True,
                "observation": "No suitable target candidate was found.",
            }
            return

        best_target = ranked_candidates[0]
        best_score = best_target["score"]
        second_score = (
            ranked_candidates[1]["score"]
            if len(ranked_candidates) > 1
            else best_score
        )
        difference = best_score - second_score

        if len(ranked_candidates) == 1 or difference >= 30:
            confidence = "High"
        elif difference >= 15:
            confidence = "Moderate"
        else:
            confidence = "Low"

        target_series = self.df[best_target["column"]].dropna()
        problem_type = best_target["problem_type"]

        target_analysis = {
            "target_column": best_target["column"],
            "confidence": confidence,
            "problem_type": problem_type,
            "recommendation_requires_confirmation": confidence == "Low",
            "num_classes": None,
            "class_distribution": None,
            "is_imbalanced": None,
            "candidate_scores": candidate_scores,
            "observation": None,
        }

        if problem_type == "Regression":
            target_analysis["observation"] = (
                "The selected target appears to represent a continuous regression problem."
            )
        else:
            unique_values = target_series.nunique()
            target_analysis["num_classes"] = int(unique_values)
            target_analysis["classification_type"] = (
                "Binary" if unique_values == 2 else "Multiclass"
            )

            class_distribution = target_series.value_counts().to_dict()
            target_analysis["class_distribution"] = class_distribution

            if class_distribution:
                total_samples = sum(class_distribution.values())
                minority_ratio = min(class_distribution.values()) / total_samples
                target_analysis["is_imbalanced"] = minority_ratio < 0.20

            if target_analysis["classification_type"] == "Binary":
                target_analysis["observation"] = (
                    "The selected target appears to represent an "
                    + (
                        "imbalanced "
                        if target_analysis["is_imbalanced"]
                        else "balanced "
                    )
                    + "binary classification problem."
                )
            else:
                target_analysis["observation"] = (
                    "The selected target appears to represent an "
                    + (
                        "imbalanced "
                        if target_analysis["is_imbalanced"]
                        else "balanced "
                    )
                    + "multiclass classification problem."
                )

        profile.recommended_target = (
            best_target["column"] if confidence != "Low" else None
        )
        profile.detected_task = problem_type
        profile.target_analysis = target_analysis

    @staticmethod
    def _infer_problem_type(series):
        """Infer a likely ML problem type from a target-like series."""
        unique_values = series.nunique()

        if pd.api.types.is_numeric_dtype(series) and unique_values > 20:
            return "Regression"

        if 2 <= unique_values <= 20:
            return "Classification"

        return "Unknown"

    @staticmethod
    def _format_memory(memory):
        """Convert memory from bytes into a readable format."""
        units = ["B", "KB", "MB", "GB", "TB"]
        size = float(memory)

        for unit in units:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024

        return f"{size:.2f} PB"