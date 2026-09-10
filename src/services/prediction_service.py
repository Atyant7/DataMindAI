"""
Prediction history service for DataMind AI.

Persists and retrieves predictions made by models.
"""

from __future__ import annotations

import json
from typing import Any

from src.core.logger import get_logger
from src.database.models import PredictionRecord
from src.database.session import get_db

logger = get_logger(__name__)


class PredictionService:
    """Manages prediction audit logs and history."""

    @staticmethod
    def record_prediction(
        user_id: str,
        project_id: str,
        model_id: str,
        input_data: dict[str, Any] | str,
        prediction: Any,
        confidence: float | None = None,
        probabilities: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Record a new prediction in the database.
        """
        input_json = input_data if isinstance(input_data, str) else json.dumps(input_data, default=str)
        prob_json = json.dumps(probabilities, default=str) if probabilities else None

        with get_db() as db:
            record = PredictionRecord(
                user_id=user_id,
                project_id=project_id,
                model_id=model_id,
                input_data_json=input_json,
                prediction_result=str(prediction),
                confidence=confidence,
                probabilities_json=prob_json,
            )
            db.add(record)
            db.flush()
            result = record.to_dict()

        logger.info("Recorded prediction for model %s: %s", model_id, prediction)
        return result

    @staticmethod
    def get_project_predictions(
        user_id: str,
        project_id: str,
        model_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Retrieve recent predictions for a project, optionally filtered by model.
        """
        with get_db() as db:
            query = db.query(PredictionRecord).filter(
                PredictionRecord.user_id == user_id,
                PredictionRecord.project_id == project_id,
            )
            if model_id:
                query = query.filter(PredictionRecord.model_id == model_id)

            records = query.order_by(PredictionRecord.created_at.desc()).limit(limit).all()

            result = []
            for r in records:
                d = r.to_dict()
                try:
                    d["input_data"] = json.loads(d["input_data_json"])
                except Exception:
                    d["input_data"] = d["input_data_json"]

                if d.get("probabilities_json"):
                    try:
                        d["probabilities"] = json.loads(d["probabilities_json"])
                    except Exception:
                        d["probabilities"] = None
                else:
                    d["probabilities"] = None
                result.append(d)
            return result
