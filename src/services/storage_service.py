"""
Production storage abstraction for DataMind AI.

Supports local filesystem storage and provides a clean interface
for swapping to S3 or cloud object storage in production.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import io
from pathlib import Path
import re
import shutil
import uuid
from typing import BinaryIO

from src.core.config import MAX_FILE_SIZE_MB, STORAGE_PATH
from src.core.logger import get_logger

logger = get_logger(__name__)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent path traversal or unsafe characters.
    """
    # Remove directory paths if provided
    clean = Path(filename).name
    # Replace spaces and non-alphanumeric chars except dots, dashes, underscores
    clean = re.sub(r"[^\w\.\-]", "_", clean)
    # Disallow starting with dot
    clean = clean.lstrip(".")
    if not clean:
        clean = f"file_{uuid.uuid4().hex[:8]}"
    return clean


class BaseStorageService(ABC):
    """Abstract interface for artifact storage."""

    @abstractmethod
    def save_file(self, file_data: bytes | BinaryIO, subfolder: str, filename: str) -> str:
        """Save a file and return its stored URI or relative path."""
        pass

    @abstractmethod
    def get_file_path(self, relative_path: str) -> Path:
        """Get the filesystem path for reading the file."""
        pass

    @abstractmethod
    def delete_file(self, relative_path: str) -> bool:
        """Delete a stored file."""
        pass


class LocalStorageService(BaseStorageService):
    """Local filesystem storage implementation."""

    def __init__(self, base_path: Path | str | None = None) -> None:
        self.base_path = Path(base_path or STORAGE_PATH).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _resolve_safe_path(self, subfolder: str, filename: str) -> Path:
        """Ensure the target path is strictly within self.base_path to prevent traversal."""
        safe_subfolder = sanitize_filename(subfolder)
        safe_filename = sanitize_filename(filename)
        folder = (self.base_path / safe_subfolder).resolve()
        folder.mkdir(parents=True, exist_ok=True)
        target = (folder / safe_filename).resolve()

        if not str(target).startswith(str(self.base_path)):
            raise ValueError(f"Security error: Attempted path traversal to {target}")
        return target

    def save_file(self, file_data: bytes | BinaryIO, subfolder: str, filename: str) -> str:
        """Save bytes or stream to disk."""
        target_path = self._resolve_safe_path(subfolder, filename)

        if isinstance(file_data, bytes):
            if len(file_data) > MAX_FILE_SIZE_MB * 1024 * 1024:
                raise ValueError(f"File exceeds maximum allowed size of {MAX_FILE_SIZE_MB}MB.")
            target_path.write_bytes(file_data)
        else:
            with open(target_path, "wb") as f_out:
                shutil.copyfileobj(file_data, f_out)

        rel_path = str(target_path.relative_to(self.base_path))
        logger.info("Saved file to storage: %s", rel_path)
        return rel_path

    def get_file_path(self, relative_path: str) -> Path:
        """Get absolute path, validating against traversal."""
        target = (self.base_path / relative_path).resolve()
        if not str(target).startswith(str(self.base_path)):
            raise ValueError(f"Security error: Attempted path traversal to {target}")
        return target

    def delete_file(self, relative_path: str) -> bool:
        """Delete a file from storage."""
        try:
            target = self.get_file_path(relative_path)
            if target.exists() and target.is_file():
                target.unlink()
                logger.info("Deleted file from storage: %s", relative_path)
                return True
        except Exception as exc:
            logger.error("Failed to delete %s: %s", relative_path, exc)
        return False


# Global default storage service
storage_service = LocalStorageService()
