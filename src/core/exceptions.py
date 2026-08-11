"""Custom exceptions used across DataMindAI."""


class DataMindAIError(Exception):
    """Base exception for expected DataMindAI application errors."""


class DatasetLoadError(DataMindAIError):
    """Raised when a dataset cannot be loaded or is invalid."""


class UnsupportedFileTypeError(DatasetLoadError):
    """Raised when an uploaded file type is not supported."""


class EmptyDatasetError(DatasetLoadError):
    """Raised when an uploaded dataset contains no rows or columns."""


class InvalidDatasetError(DataMindAIError):
    """Raised when a dataset does not satisfy basic application requirements."""


class VisualizationError(DataMindAIError):
    """Raised when a visualization cannot be generated."""


class ModelTrainingError(DataMindAIError):
    """Reserved for the ML training engine introduced in a later phase."""


class PredictionError(DataMindAIError):
    """Reserved for the prediction engine introduced in a later phase."""


class WorkflowError(DataMindAIError):
    """Reserved for LangGraph workflow failures introduced in a later phase."""