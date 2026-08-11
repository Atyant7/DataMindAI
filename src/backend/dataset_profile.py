from dataclasses import dataclass, field


@dataclass
class DatasetProfile:
    """Stores information extracted from a dataset."""

    dataset_name: str = ""

    rows: int = 0
    columns: int = 0
    total_cells: int = 0

    # Basic Information
    memory_usage: str = ""

    # Dataset Health
    missing_values: int = 0
    duplicate_rows: int = 0
    missing_percentage: float = 0.0
    duplicate_percentage: float = 0.0
    health_score: int = 100
    health_status: str = "Excellent"
    quality_issues: list = field(default_factory=list)

    # Column Information
    numerical_columns: list = field(default_factory=list)
    categorical_columns: list = field(default_factory=list)
    datetime_columns: list = field(default_factory=list)
    boolean_columns: list = field(default_factory=list)
    numerical_statistics: dict = field(default_factory=dict)
    categorical_statistics: dict = field(default_factory=dict)

    # Target Analysis
    possible_target_columns: list = field(default_factory=list)
    recommended_target: str | None = None
    detected_task: str = ""
    target_analysis: dict = field(default_factory=dict)

    # Visualization
    recommended_visualizations: list = field(default_factory=list)

    # Correlation Analysis
    correlation_matrix: object = None
    correlation_insights: list = field(default_factory=list)

    # Outliers
    outlier_summary: list = field(default_factory=list)

    # Distribution
    distribution_summary: list = field(default_factory=list)

    # Feature Quality
    feature_quality_summary: list = field(default_factory=list)

    def to_dict(self):
        """Return the serializable parts of the profile for prompts/reports."""
        return {
            "dataset_name": self.dataset_name,
            "rows": self.rows,
            "columns": self.columns,
            "total_cells": self.total_cells,
            "memory_usage": self.memory_usage,
            "missing_values": self.missing_values,
            "duplicate_rows": self.duplicate_rows,
            "missing_percentage": self.missing_percentage,
            "duplicate_percentage": self.duplicate_percentage,
            "health_score": self.health_score,
            "health_status": self.health_status,
            "quality_issues": self.quality_issues,
            "numerical_columns": self.numerical_columns,
            "categorical_columns": self.categorical_columns,
            "datetime_columns": self.datetime_columns,
            "boolean_columns": self.boolean_columns,
            "numerical_statistics": self.numerical_statistics,
            "categorical_statistics": self.categorical_statistics,
            "possible_target_columns": self.possible_target_columns,
            "recommended_target": self.recommended_target,
            "detected_task": self.detected_task,
            "target_analysis": self.target_analysis,
            "recommended_visualizations": self.recommended_visualizations,
            "correlation_insights": self.correlation_insights,
            "outlier_summary": self.outlier_summary,
            "distribution_summary": self.distribution_summary,
            "feature_quality_summary": self.feature_quality_summary,
        }