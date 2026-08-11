from dataclasses import dataclass, field


@dataclass
class DatasetReport:
    """Structured dataset information used by the UI and AI layers."""

    dataset_name: str = ""

    rows: int = 0
    columns: int = 0
    total_cells: int = 0

    missing_values: int = 0
    duplicate_rows: int = 0

    missing_percentage: float = 0.0
    duplicate_percentage: float = 0.0

    health_score: int = 100
    health_status: str = "Excellent"
    quality_issues: list = field(default_factory=list)

    memory_usage: str = ""

    numerical_columns: list = field(default_factory=list)
    categorical_columns: list = field(default_factory=list)
    datetime_columns: list = field(default_factory=list)
    boolean_columns: list = field(default_factory=list)

    numerical_statistics: dict = field(default_factory=dict)
    categorical_statistics: dict = field(default_factory=dict)

    possible_target_columns: list = field(default_factory=list)
    recommended_target: str | None = None
    recommended_task: str = ""

    target_analysis: dict = field(default_factory=dict)
    recommended_visualizations: list = field(default_factory=list)

    correlation_matrix: object = None
    correlation_insights: list = field(default_factory=list)

    outlier_summary: list = field(default_factory=list)
    distribution_summary: list = field(default_factory=list)
    feature_quality_summary: list = field(default_factory=list)
