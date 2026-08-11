"""Application-wide constants used by DataMindAI."""

# Supported machine-learning task types.
CLASSIFICATION = "Classification"
REGRESSION = "Regression"

# Common experiment states.
EXPERIMENT_PENDING = "Pending"
EXPERIMENT_RUNNING = "Running"
EXPERIMENT_COMPLETED = "Completed"
EXPERIMENT_FAILED = "Failed"

# Model lifecycle states.
MODEL_TRAINED = "Trained"
MODEL_REGISTERED = "Registered"
MODEL_RETIRED = "Retired"

# Default evaluation metrics. The ML engine introduced later will select
# the final metric dynamically from the task and target characteristics.
CLASSIFICATION_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
)

REGRESSION_METRICS = (
    "mae",
    "mse",
    "rmse",
    "r2",
)

SUPPORTED_MODELS = (
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "LightGBM",
)