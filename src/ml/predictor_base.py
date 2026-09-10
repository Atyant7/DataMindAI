"""
Base and specialized predictors for DataMind AI.

Supports dynamic prediction dispatch based on model metadata:
- TabularPredictor (tabular/numeric/categorical)
- ImagePredictor (computer vision / image upload)
- GeospatialPredictor (coordinates/location)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from src.core.logger import get_logger
from src.ml.model_persistence import ModelArtifact, load_artifact, load_model
from src.ml.prediction_engine import PredictionResult, predict_from_artifact

logger = get_logger(__name__)


class BasePredictor(ABC):
    """Abstract base class for all DataMind AI predictors."""

    @abstractmethod
    def predict(self, input_data: Any) -> PredictionResult:
        """Generate prediction for input data."""
        pass


class TabularPredictor(BasePredictor):
    """Predictor for tabular machine learning models."""

    def __init__(self, artifact: ModelArtifact) -> None:
        self.artifact = artifact

    def predict(self, input_data: pd.DataFrame | dict[str, Any]) -> PredictionResult:
        """Validate features and generate prediction from artifact."""
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            df = input_data
        else:
            raise TypeError("Expected dict or pandas DataFrame.")

        return predict_from_artifact(self.artifact, df)


class ImagePredictor(BasePredictor):
    """Predictor for image classification models."""

    def __init__(
        self,
        model: Any,
        classes: list[str],
        display_name: str = "Image Classifier",
        target_column: str = "label",
        image_size: tuple[int, int] = (64, 64),
    ) -> None:
        self.model = model
        self.classes = classes
        self.display_name = display_name
        self.target_column = target_column
        self.image_size = image_size

    def _extract_features(self, img: Image.Image) -> np.ndarray:
        """Extract standardized color-histogram and thumbnail features."""
        img = img.convert("RGB").resize(self.image_size)
        arr = np.array(img, dtype=np.float32) / 255.0

        r_hist, _ = np.histogram(arr[:, :, 0], bins=32, range=(0, 1), density=True)
        g_hist, _ = np.histogram(arr[:, :, 1], bins=32, range=(0, 1), density=True)
        b_hist, _ = np.histogram(arr[:, :, 2], bins=32, range=(0, 1), density=True)

        thumb = np.array(img.resize((8, 8)).convert("RGB"), dtype=np.float32).flatten() / 255.0
        return np.concatenate([r_hist, g_hist, b_hist, thumb]).reshape(1, -1)

    def predict(self, input_data: Image.Image | bytes | str | Path) -> PredictionResult:
        """Predict class and probabilities from image."""
        if isinstance(input_data, bytes):
            img = Image.open(io.BytesIO(input_data))
        elif isinstance(input_data, (str, Path)):
            img = Image.open(input_data)
        elif isinstance(input_data, Image.Image):
            img = input_data
        else:
            raise TypeError("Expected PIL Image, bytes, or file path.")

        features = self._extract_features(img)
        pred_label = str(self.model.predict(features)[0])

        confidence = None
        probabilities = None
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features)[0]
            confidence = float(np.max(probs))
            classes = getattr(self.model, "classes_", self.classes)
            probabilities = {str(c): float(p) for c, p in zip(classes, probs)}

        return PredictionResult(
            prediction=pred_label,
            task="classification",
            model_name="image_classifier",
            display_name=self.display_name,
            target_column=self.target_column,
            confidence=confidence,
            probabilities=probabilities,
            input_data={"format": "image", "size": f"{img.size[0]}x{img.size[1]}"},
        )


class GeospatialPredictor(BasePredictor):
    """Predictor for geospatial tabular models."""

    def __init__(self, artifact: ModelArtifact) -> None:
        self.artifact = artifact

    def predict(self, input_data: pd.DataFrame | dict[str, Any]) -> PredictionResult:
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            df = input_data
        else:
            raise TypeError("Expected dict or pandas DataFrame.")

        # Ensure spatial features exist if missing
        if "latitude" in df.columns and "longitude" in df.columns:
            if "geo_dist_center" in self.artifact.feature_names and "geo_dist_center" not in df.columns:
                df["geo_dist_center"] = 0.0

        return predict_from_artifact(self.artifact, df)
