from dataclasses import dataclass, field


@dataclass
class DatasetProfile:
    """
    Stores all information extracted from a dataset.
    """

    dataset_name: str = ""

    rows: int = 0
    columns: int = 0

    missing_values: int = 0
    duplicate_rows: int = 0

    memory_usage: str = ""

    numerical_columns: list = field(default_factory=list)
    categorical_columns: list = field(default_factory=list)
    datetime_columns: list = field(default_factory=list)
    boolean_columns: list = field(default_factory=list)

    possible_target_columns: list = field(default_factory=list)

    recommended_task: str = ""

    recommended_visualizations: list = field(default_factory=list)