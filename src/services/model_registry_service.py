"""
Model Registry service for DataMind AI.

Manages registered ML models, versioning, artifacts, and metadata.
"""

from __future__ import annotations

import json
from typing import Any

from src.core.logger import get_logger
from src.database.models import ModelRecord
from src.database.session import get_db

logger = get_logger(__name__)


class ModelRegistryService:
    """Model Registry managing trained models, versions, and metadata."""

    @staticmethod
    def register_model(
        user_id: str,
        project_id: str,
        model_name: str,
        display_name: str,
        task: str,
        target_column: str,
        artifact_path: str,
        metadata_path: str,
        dataset_id: str | None = None,
        model_type: str = "tabular",
        framework: str = "scikit-learn",
        feature_names: list[str] | None = None,
        metrics: dict[str, Any] | None = None,
        training_config: dict[str, Any] | None = None,
        is_best: bool = True,
        is_image: bool = False,
        is_geospatial: bool = False,
    ) -> dict[str, Any]:
        """
        Register a new model run in the registry. Automatically calculates version.
        """
        with get_db() as db:
            # Find current highest version for this target in this project
            latest = db.query(ModelRecord).filter(
                ModelRecord.project_id == project_id,
                ModelRecord.target_column == target_column,
            ).order_by(ModelRecord.version.desc()).first()

            version = (latest.version + 1) if latest else 1

            # If this is marked as best, unset previous is_best for this target
            if is_best:
                db.query(ModelRecord).filter(
                    ModelRecord.project_id == project_id,
                    ModelRecord.target_column == target_column,
                ).update({"is_best": False})

            record = ModelRecord(
                user_id=user_id,
                project_id=project_id,
                dataset_id=dataset_id,
                model_name=model_name,
                display_name=display_name,
                model_type=model_type,
                framework=framework,
                task=task,
                target_column=target_column,
                feature_names_json=json.dumps(feature_names or []),
                metrics_json=json.dumps(metrics or {}, default=str),
                training_config_json=json.dumps(training_config or {}, default=str),
                artifact_path=str(artifact_path),
                metadata_path=str(metadata_path),
                version=version,
                is_best=is_best,
                is_image=is_image,
                is_geospatial=is_geospatial,
            )
            db.add(record)
            db.flush()
            result = record.to_dict()

        logger.info(
            "Registered model '%s' v%d (id: %s) for project %s",
            display_name,
            version,
            result["id"],
            project_id,
        )
        return result

    @staticmethod
    def list_project_models(user_id: str, project_id: str) -> list[dict[str, Any]]:
        """List all models registered in a user's project."""
        with get_db() as db:
            models = db.query(ModelRecord).filter(
                ModelRecord.user_id == user_id,
                ModelRecord.project_id == project_id,
            ).order_by(ModelRecord.created_at.desc()).all()

            result = []
            for m in models:
                d = m.to_dict()
                d["feature_names"] = json.loads(d["feature_names_json"]) if d.get("feature_names_json") else []
                d["metrics"] = json.loads(d["metrics_json"]) if d.get("metrics_json") else {}
                d["training_config"] = json.loads(d["training_config_json"]) if d.get("training_config_json") else {}
                result.append(d)
            return result

    @staticmethod
    def get_model(user_id: str, model_id: str) -> dict[str, Any] | None:
        """Get model details verifying user ownership."""
        with get_db() as db:
            m = db.query(ModelRecord).filter(
                ModelRecord.id == model_id,
                ModelRecord.user_id == user_id,
            ).first()

            if m:
                d = m.to_dict()
                d["feature_names"] = json.loads(d["feature_names_json"]) if d.get("feature_names_json") else []
                d["metrics"] = json.loads(d["metrics_json"]) if d.get("metrics_json") else {}
                d["training_config"] = json.loads(d["training_config_json"]) if d.get("training_config_json") else {}
                return d
        return None

    @staticmethod
    def get_best_model(user_id: str, project_id: str) -> dict[str, Any] | None:
        """
        Retrieve the best model (or most recent model) registered for the user's project.
        Enforces strict user_id and project_id multi-tenant isolation.
        Returns None if no model has been registered for this project yet.
        """
        with get_db() as db:
            # 1. Look for the model explicitly marked is_best=True
            model = (
                db.query(ModelRecord)
                .filter(
                    ModelRecord.user_id == user_id,
                    ModelRecord.project_id == project_id,
                    ModelRecord.is_best == True,  # noqa: E712
                )
                .order_by(ModelRecord.created_at.desc())
                .first()
            )

            # 2. Fallback to most recent model in this project if none explicitly marked is_best
            if not model:
                model = (
                    db.query(ModelRecord)
                    .filter(
                        ModelRecord.user_id == user_id,
                        ModelRecord.project_id == project_id,
                    )
                    .order_by(ModelRecord.created_at.desc())
                    .first()
                )

            if model:
                d = model.to_dict()
                d["feature_names"] = json.loads(d["feature_names_json"]) if d.get("feature_names_json") else []
                d["metrics"] = json.loads(d["metrics_json"]) if d.get("metrics_json") else {}
                d["training_config"] = json.loads(d["training_config_json"]) if d.get("training_config_json") else {}
                return d

        return None

    @staticmethod
    def delete_model(user_id: str, model_id: str) -> bool:
        """Delete a model record from registry."""
        with get_db() as db:
            m = db.query(ModelRecord).filter(
                ModelRecord.id == model_id,
                ModelRecord.user_id == user_id,
            ).first()

            if m:
                db.delete(m)
                logger.info("Deleted model record %s for user %s", model_id, user_id)
                return True
        return False
