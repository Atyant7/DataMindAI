"""
DataMindAI ML Engine.

This package contains deterministic machine-learning components
independent of the LLM and agent layers.
"""

from .task_detector import (
    MLTask,
    TargetType,
    TaskDetectionResult,
    detect_task,
)
from .preprocessing_pipeline import (
    FeatureColumns,
    PreprocessingResult,
    build_preprocessing_pipeline,
    detect_feature_columns,
    fit_preprocessing_pipeline,
    fit_transform_training_data,
    get_transformed_feature_names,
    model_requires_scaling,
    transform_features,
)

from .trainer import (
    TrainingConfig,
    TrainingResult,
    predict,
    predict_proba,
    train_model,
    train_models,
)

from .model_factory import (
    ModelFamily,
    ModelSpec,
    create_model,
    create_models,
    get_model_names,
    get_model_spec,
    get_model_specs,
    model_supports_feature_importance,
    model_supports_probability,
)

from .evaluator import (
    EvaluationResult,
    create_comparison_table,
    evaluate_model,
    evaluate_models,
)

from .model_selector import (
    ModelRanking,
    ModelSelectionResult,
    create_leaderboard,
    default_selection_metric,
    get_selected_model,
    metric_direction,
    normalize_score,
    select_best_model,
)

from .ml_pipeline import (
    MLPipelineResult,
    run_ml_pipeline,
)

from .model_persistence import (
    ModelArtifact,
    load_artifact,
    load_model,
    load_model_metadata,
    save_best_model,
    save_model,
)

from .prediction_engine import (
    PredictionResult,
    predict,
    predict_from_artifact,
)
from src.backend.dataset_qa import (
    DatasetQA,
    DatasetQAResult,
    answer_dataset_question,
)
from src.tools.data_analysis import (
    DataAnalysisTool,
    DataAnalysisResult,
)


__all__ = [
    "MLTask",
    "TargetType",
    "TaskDetectionResult",
    "detect_task",
    
    "ModelFamily",
    "ModelSpec",
    "create_model",
    "create_models",
    "get_model_names",
    "get_model_spec",
    "get_model_specs",
    "model_supports_feature_importance",
    "model_supports_probability",
    
    # Preprocessing
    "FeatureColumns",
    "PreprocessingResult",
    "build_preprocessing_pipeline",
    "detect_feature_columns",
    "fit_preprocessing_pipeline",
    "fit_transform_training_data",
    "get_transformed_feature_names",
    "model_requires_scaling",
    "transform_features",
    
    # Training
    "TrainingConfig",
    "TrainingResult",
    "predict",
    "predict_proba",
    "train_model",
    "train_models",
    
    # Evaluation
    "EvaluationResult",
    "create_comparison_table",
    "evaluate_model",
    "evaluate_models",
    
    # Model selection
    "ModelRanking",
    "ModelSelectionResult",
    "create_leaderboard",
    "default_selection_metric",
    "get_selected_model",
    "metric_direction",
    "normalize_score",
    "select_best_model",
    
    # High-level ML pipeline
    "MLPipelineResult",
    "run_ml_pipeline",
    
    # Model persistence
    "ModelArtifact",
    "load_artifact",
    "load_model",
    "load_model_metadata",
    "save_best_model",
    "save_model",
    
    # Prediction
    "PredictionResult",
    "predict",
    "predict_from_artifact",
    
    "DatasetQA",
    "DatasetQAResult",
    "answer_dataset_question",
    
    "DataAnalysisTool",
    "DataAnalysisResult",

]