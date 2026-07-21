from dataclasses import dataclass, field


@dataclass
class DatasetProfile:
    """
    Stores all information extracted from a dataset.
    """

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

    warnings: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)

    # Column Information
    numerical_columns: list = field(default_factory=list)
    categorical_columns: list = field(default_factory=list)
    datetime_columns: list = field(default_factory=list)
    boolean_columns: list = field(default_factory=list)
    numerical_statistics: dict = field(default_factory=dict)
    categorical_statistics: dict = field(default_factory=dict)

    # AI Suggestions
    possible_target_columns: list = field(default_factory=list)

    recommended_task: str = ""

    recommended_visualizations: list = field(default_factory=list)