"""
Dataset service for DataMind AI.

Handles dataset persistence, metadata tracking, and retrieval
within user-isolated projects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd

from src.backend.data_loader import get_file_signature, load_data
from src.core.logger import get_logger
from src.database.models import DatasetRecord
from src.database.session import get_db
from src.services.storage_service import sanitize_filename, storage_service

logger = get_logger(__name__)


class DatasetService:
    """Handles dataset storage, database metadata, and retrieval."""

    @staticmethod
    def save_dataset(
        user_id: str,
        project_id: str,
        file_obj: Any,
        filename: str,
    ) -> dict[str, Any]:
        """
        Store dataset file and persist its metadata in the database.
        """
        clean_name = sanitize_filename(filename)
        file_type = Path(clean_name).suffix.lstrip(".").lower()

        # Read bytes
        if hasattr(file_obj, "getvalue"):
            file_bytes = file_obj.getvalue()
        elif hasattr(file_obj, "read"):
            file_bytes = file_obj.read()
        elif isinstance(file_obj, bytes):
            file_bytes = file_obj
        else:
            raise TypeError("Unsupported file object type.")

        # Save to storage abstraction
        subfolder = f"datasets_{project_id}"
        stored_rel_path = storage_service.save_file(file_bytes, subfolder, clean_name)
        full_path = storage_service.get_file_path(stored_rel_path)

        # Load DataFrame to calculate shape and signature
        df = load_data(str(full_path))
        signature = get_file_signature(str(full_path))

        meta = {
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        }

        with get_db() as db:
            record = DatasetRecord(
                user_id=user_id,
                project_id=project_id,
                name=clean_name,
                file_path=stored_rel_path,
                file_type=file_type,
                rows=len(df),
                columns=len(df.columns),
                signature=signature,
                metadata_json=json.dumps(meta),
            )
            db.add(record)
            db.flush()
            result = record.to_dict()

        logger.info("Saved dataset '%s' (id: %s) for project %s", clean_name, result["id"], project_id)
        return result

    @staticmethod
    def list_project_datasets(user_id: str, project_id: str) -> list[dict[str, Any]]:
        """List all datasets in a user's project."""
        with get_db() as db:
            datasets = db.query(DatasetRecord).filter(
                DatasetRecord.user_id == user_id,
                DatasetRecord.project_id == project_id,
            ).order_by(DatasetRecord.created_at.desc()).all()

            return [d.to_dict() for d in datasets]

    @staticmethod
    def get_dataset(user_id: str, dataset_id: str) -> dict[str, Any] | None:
        """Get dataset metadata verifying user ownership."""
        with get_db() as db:
            record = db.query(DatasetRecord).filter(
                DatasetRecord.id == dataset_id,
                DatasetRecord.user_id == user_id,
            ).first()

            if record:
                return record.to_dict()
        return None

    @staticmethod
    def load_dataframe(user_id: str, dataset_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Load DataFrame from storage for a dataset record."""
        meta = DatasetService.get_dataset(user_id, dataset_id)
        if not meta:
            raise ValueError(f"Dataset {dataset_id} not found or access denied.")

        file_path = storage_service.get_file_path(meta["file_path"])
        df = load_data(str(file_path))
        return df, meta
