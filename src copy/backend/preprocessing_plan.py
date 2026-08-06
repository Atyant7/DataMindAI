from dataclasses import dataclass, field

@dataclass
class PreprocessingPlan:
    missing_value_plan : list = field(default_factory=list)
    encoding_plan : list = field(default_factory=list)
    scaling_plan : list = field(default_factory=list)
    feature_selection_plan : list = field(default_factory=list)
    outlier_treatment_plan : list = field(default_factory=list)
    train_test_plan : dict = field(default_factory=dict)
    pipeline_summary : list = field(default_factory=list)
    