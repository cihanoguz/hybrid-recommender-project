"""
Data Loading Functions.

This module provides functions for loading and validating data from pickle files.
Includes comprehensive error handling and data validation.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from logging_config import get_logger

logger = get_logger(__name__)

try:
    from security_utils import sanitize_file_path, validate_pickle_file_integrity

    SECURITY_UTILS_AVAILABLE = True
except ImportError:
    SECURITY_UTILS_AVAILABLE = False
    logger.warning("security_utils not available, skipping advanced security checks")

try:
    from utils import DataLoadError, safe_load_pickle, validate_dataframe, validate_file_exists
except ImportError:
    logger.warning("utils.py not found, using fallback")
    # Fallback implementations would go here


@st.cache_resource(show_spinner=True)
def load_data(pickle_path: Path) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    List[int],
    np.ndarray,
]:
    """
    Load and validate data from pickle file.

    This function loads preprocessed data from a pickle file, validates
    the structure and contents, and returns the dataframes and matrices
    required for recommendations.

    Args:
        pickle_path: Path to the pickle file containing preprocessed data

    Returns:
        Tuple containing:
            - movie: DataFrame with movie information (movieId, title, genres)
            - rating: DataFrame with user ratings (userId, movieId, rating)
            - df_full: Full rating dataset
            - common_movies: Filtered movies with sufficient ratings
            - user_movie_df: User-movie rating matrix
            - all_user_ids: List of all user IDs in the dataset
            - cosine_sim_genre: Cosine similarity matrix for genre-based similarity

    Raises:
        DataLoadError: If file cannot be loaded or data validation fails

    Note:
        This function is cached using Streamlit's cache_resource decorator
        to avoid reloading data on every app rerun.
    """
    required_keys = [
        "movie",
        "rating",
        "df_full",
        "common_movies",
        "user_movie_df",
        "cosine_sim_genre",
    ]

    # Security: Sanitize file path (prevent path traversal)
    if SECURITY_UTILS_AVAILABLE:
        base_dir = pickle_path.parent.resolve()
        is_valid, sanitized_path = sanitize_file_path(pickle_path, allowed_base=base_dir.parent)
        if not is_valid:
            logger.error(f"Security: Path traversal attempt detected: {pickle_path}")
            raise DataLoadError(
                "Invalid file path: potential path traversal attack",
                details={"file_path": str(pickle_path), "security_check": "path_sanitization"},
            )
        if sanitized_path:
            pickle_path = sanitized_path
            logger.debug(f"Path sanitized: {pickle_path}")

    # Validate file exists
    exists, error = validate_file_exists(pickle_path)
    if not exists:
        logger.error(f"Data file validation failed: {error}")
        raise DataLoadError(
            f"Data file not found: {pickle_path}",
            details={"file_path": str(pickle_path), "error": error},
        )

    # Security: Validate pickle file integrity (with increased max size for large datasets)
    if SECURITY_UTILS_AVAILABLE:
        is_valid, integrity_error = validate_pickle_file_integrity(
            pickle_path, max_file_size=1024 * 1024 * 1024  # 1 GB (increased for large datasets)
        )
        if not is_valid:
            logger.error(f"Pickle file integrity check failed: {integrity_error}")
            raise DataLoadError(
                f"File integrity validation failed: {integrity_error}",
                details={"file_path": str(pickle_path), "security_check": "file_integrity"},
            )

    logger.info(f"Loading data from: {pickle_path}")

    try:
        # Load data with validation
        data = safe_load_pickle(pickle_path, required_keys)

        # Validate data structure
        movie = data.get("movie")
        rating = data.get("rating")
        df_full = data.get("df_full")
        common_movies = data.get("common_movies")
        user_movie_df = data.get("user_movie_df")
        cosine_sim_genre = data.get("cosine_sim_genre")

        if movie is None or rating is None:
            error_msg = "Pickle file missing required data keys"
            logger.error(f"{error_msg}. Expected keys: movie, rating")
            raise DataLoadError(
                error_msg,
                details={
                    "expected_keys": required_keys,
                    "available_keys": list(data.keys()) if isinstance(data, dict) else "not a dict",
                },
            )

        # Validate DataFrames
        is_valid, error = validate_dataframe(movie, required_columns=["movieId", "title", "genres"])
        if not is_valid:
            logger.error(f"Movie DataFrame validation failed: {error}")
            raise DataLoadError(
                f"Movie data validation failed: {error}",
                details={"dataframe": "movie", "error": error},
            )

        is_valid, error = validate_dataframe(
            rating, required_columns=["userId", "movieId", "rating"]
        )
        if not is_valid:
            logger.error(f"Rating DataFrame validation failed: {error}")
            raise DataLoadError(
                f"Rating data validation failed: {error}",
                details={"dataframe": "rating", "error": error},
            )

        # Get user ID list from rating data
        all_user_ids = sorted(rating["userId"].unique().tolist())

        if not all_user_ids:
            raise DataLoadError(
                "No users found in rating data", details={"rating_rows": len(rating)}
            )

        logger.info(
            f"Data loaded successfully: {len(movie)} movies, {len(rating)} ratings, {len(all_user_ids)} users"
        )

        return movie, rating, df_full, common_movies, user_movie_df, all_user_ids, cosine_sim_genre

    except DataLoadError:
        # Re-raise DataLoadError as-is
        raise
    except Exception as e:
        logger.exception(f"Unexpected error during data loading: {e}")
        raise DataLoadError(
            f"Unexpected error loading data: {e}",
            details={"original_error": str(e), "error_type": type(e).__name__},
        ) from e
