"""
Base and specialized model trainers for DataMind AI.

Provides modular trainers for:
- Tabular / Numeric / Categorical models
- Computer Vision / Image models
- Geospatial / Location models
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from src.core.logger import get_logger
from src.ml.ml_pipeline import MLPipelineResult, run_ml_pipeline
from src.ml.model_persistence import ModelArtifact, save_best_model
from src.ml.prediction_engine import PredictionResult

logger = get_logger(__name__)


class BaseModelTrainer(ABC):
    """Abstract base class for all DataMind AI model trainers."""

    @abstractmethod
    def train(self, data: Any, target: str, **kwargs) -> Any:
        """Train candidate models and return result."""
        pass


class TabularModelTrainer(BaseModelTrainer):
    """AutoML trainer for tabular, numeric, and categorical datasets."""

    def train(
        self,
        data: pd.DataFrame,
        target: str,
        model_names: list[str] | None = None,
        **kwargs,
    ) -> MLPipelineResult:
        """Execute automated tabular machine-learning pipeline."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame for TabularModelTrainer.")
        return run_ml_pipeline(
            df=data,
            target_column=target,
            model_names=model_names,
        )


class ImageModelTrainer(BaseModelTrainer):
    """Trainer for computer vision and image classification tasks."""

    def __init__(self, image_size: tuple[int, int] = (64, 64)) -> None:
        self.image_size = image_size

    def extract_features(self, image_input: Image.Image | str | Path) -> np.ndarray:
        """Extract a normalized color-histogram + thumbnail vector from an image."""
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise TypeError("Expected PIL Image or file path.")

        # Resize to standardized dimensions
        resized = img.resize(self.image_size)
        arr = np.array(resized, dtype=np.float32) / 255.0

        # Compute RGB histogram (32 bins per channel = 96 features)
        r_hist, _ = np.histogram(arr[:, :, 0], bins=32, range=(0, 1), density=True)
        g_hist, _ = np.histogram(arr[:, :, 1], bins=32, range=(0, 1), density=True)
        b_hist, _ = np.histogram(arr[:, :, 2], bins=32, range=(0, 1), density=True)

        # Spatial thumbnail features (subsampled 8x8x3 = 192 features)
        thumb = np.array(img.resize((8, 8)).convert("RGB"), dtype=np.float32).flatten() / 255.0

        return np.concatenate([r_hist, g_hist, b_hist, thumb])

    def train(
        self,
        data: list[tuple[Image.Image | str, str]],
        target: str = "label",
        **kwargs,
    ) -> Any:
        """
        Train an image classifier from a list of (image, label) pairs.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        if not data or len(data) < 2:
            raise ValueError("At least 2 image samples are required for training.")

        X_list = []
        y_list = []
        for img, label in data:
            X_list.append(self.extract_features(img))
            y_list.append(str(label))

        X = np.array(X_list)
        y = np.array(y_list)

        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        if len(y) >= 4:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            clf.fit(X_train, y_train)
            acc = float(accuracy_score(y_test, clf.predict(X_test)))
        else:
            clf.fit(X, y)
            acc = 1.0

        return {
            "model": clf,
            "classes": clf.classes_.tolist(),
            "accuracy": acc,
            "image_size": self.image_size,
        }


class GeospatialModelTrainer(BaseModelTrainer):
    """Trainer for datasets with latitude and longitude features."""

    def train(
        self,
        data: pd.DataFrame,
        target: str,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        model_names: list[str] | None = None,
        **kwargs,
    ) -> MLPipelineResult:
        """
        Enrich dataset with spatial features and execute ML pipeline.
        """
        df_geo = data.copy()
        if lat_col in df_geo.columns and lon_col in df_geo.columns:
            # Add polar and distance features from center
            lat_center = df_geo[lat_col].mean()
            lon_center = df_geo[lon_col].mean()
            df_geo["geo_dist_center"] = np.sqrt(
                (df_geo[lat_col] - lat_center) ** 2 + (df_geo[lon_col] - lon_center) ** 2
            )
        return run_ml_pipeline(
            df=df_geo,
            target_column=target,
            model_names=model_names,
        )
