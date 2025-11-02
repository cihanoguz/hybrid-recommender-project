"""Utility functions for validation and error handling."""

import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pandas as pd

from logging_config import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""

    pass


class DataLoadError(Exception):
    """Custom exception for data loading errors."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


def validate_user_id(user_id: int, available_user_ids: List[int]) -> Tuple[bool, Optional[str]]:
    """Validate user ID."""
    if not isinstance(user_id, int):
        return False, f"User ID must be an integer, got {type(user_id).__name__}"

    if user_id < 1:
        return False, f"User ID must be positive, got {user_id}"

    if user_id not in available_user_ids:
        return False, f"User ID {user_id} not found in dataset"

    return True, None


def validate_overlap_ratio(ratio: float) -> Tuple[bool, Optional[str]]:
    """Validate overlap ratio percentage."""
    if not isinstance(ratio, (int, float)):
        return False, f"Overlap ratio must be a number, got {type(ratio).__name__}"

    if ratio < 0.0 or ratio > 100.0:
        return False, f"Overlap ratio must be between 0 and 100, got {ratio}"

    return True, None


def validate_correlation_threshold(threshold: float) -> Tuple[bool, Optional[str]]:
    """
    Validate correlation threshold value.

    Args:
        threshold: Correlation threshold to validate (0.0 to 1.0)

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    if not isinstance(threshold, (int, float)):
        return False, f"Correlation threshold must be a number, got {type(threshold).__name__}"

    if threshold < 0.0 or threshold > 1.0:
        return False, f"Correlation threshold must be between 0.0 and 1.0, got {threshold}"

    return True, None


def validate_file_exists(file_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Validate file exists and is readable

    Args:
        file_path: Path to file

    Returns:
        Tuple of (exists, error_message)
    """
    if not file_path.exists():
        return False, f"File not found: {file_path}"

    if not file_path.is_file():
        return False, f"Path is not a file: {file_path}"

    if not os.access(file_path, os.R_OK):
        return False, f"File is not readable: {file_path}"

    return True, None


def safe_load_pickle(file_path: Path, required_keys: Optional[List[str]] = None) -> Any:
    """
    Safely load pickle file with error handling

    Args:
        file_path: Path to pickle file
        required_keys: List of required keys in pickle dict (if applicable)

    Returns:
        Loaded data

    Raises:
        DataLoadError: If file cannot be loaded or validated
    """
    import pickle

    # Validate file exists
    exists, error = validate_file_exists(file_path)
    if not exists:
        raise DataLoadError(error, details={"file_path": str(file_path)})

    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        # Validate required keys if specified
        if required_keys and isinstance(data, dict):
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                raise DataLoadError(
                    f"Missing required keys in pickle file: {missing_keys}",
                    details={"missing_keys": missing_keys, "file_path": str(file_path)},
                )

        logger.info(f"Successfully loaded pickle file: {file_path}")
        return data

    except pickle.UnpicklingError as e:
        raise DataLoadError(
            f"Corrupted pickle file: {e}",
            details={"file_path": str(file_path), "error_type": "UnpicklingError"},
        )
    except DataLoadError:
        # Re-raise DataLoadError as-is
        raise
    except Exception as e:
        raise DataLoadError(
            f"Unexpected error loading pickle file: {e}",
            details={"file_path": str(file_path), "error_type": type(e).__name__},
        )


def validate_dataframe(
    df: pd.DataFrame, required_columns: Optional[List[str]] = None, min_rows: int = 0
) -> Tuple[bool, Optional[str]]:
    """Validate DataFrame structure."""
    if df.empty and min_rows > 0:
        return False, f"DataFrame is empty, but minimum {min_rows} rows required"

    if len(df) < min_rows:
        return False, f"DataFrame has {len(df)} rows, but minimum {min_rows} required"

    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"

    return True, None
