"""Dataset loading and validation utilities."""

import hashlib
from io import BytesIO
from pathlib import Path

import pandas as pd

from src.core.config import MAX_FILE_SIZE_MB
from src.core.exceptions import (
    DatasetLoadError,
    EmptyDatasetError,
    UnsupportedFileTypeError,
)


def _get_raw_bytes(upload_file):
    try:
        return upload_file.getvalue()
    except Exception as error:
        raise DatasetLoadError(
            f"Could not read the uploaded file: {error}"
        ) from error


def get_file_signature(upload_file) -> str:
    """Return a stable hash of the uploaded file contents."""
    raw_bytes = _get_raw_bytes(upload_file)
    return hashlib.sha256(raw_bytes).hexdigest()


def load_data(upload_file):
    """Load a CSV/XLSX Streamlit UploadedFile into a pandas DataFrame."""
    if upload_file is None:
        raise DatasetLoadError("No dataset was provided.")

    filename = getattr(upload_file, "name", "")
    extension = Path(filename).suffix.lower().lstrip(".")

    if extension not in {"csv", "xlsx"}:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '.{extension}'. "
            "Only CSV and XLSX files are supported."
        )

    raw_bytes = _get_raw_bytes(upload_file)

    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if len(raw_bytes) > max_bytes:
        raise DatasetLoadError(
            f"File is too large. Maximum allowed size is {MAX_FILE_SIZE_MB} MB."
        )

    try:
        buffer = BytesIO(raw_bytes)

        if extension == "csv":
            dataframe = pd.read_csv(buffer)
        else:
            dataframe = pd.read_excel(buffer)

    except Exception as error:
        raise DatasetLoadError(
            f"Could not load '{filename}': {error}"
        ) from error

    if dataframe.empty:
        raise EmptyDatasetError(
            "The uploaded dataset is empty. Please upload a dataset "
            "containing at least one row."
        )

    if dataframe.shape[1] == 0:
        raise EmptyDatasetError(
            "The uploaded dataset contains no columns."
        )

    return dataframe