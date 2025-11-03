"""Centralized configuration management."""

import os
from pathlib import Path
from typing import List, Optional

from error_handling import ConfigurationError
from logging_config import get_logger

logger = get_logger(__name__)

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
PICKLE_PATH = Path(os.getenv("PICKLE_PATH", str(DATA_DIR / "prepare_data_demo.pkl")))
LOGO_PATH = Path(os.getenv("LOGO_PATH", str(BASE_DIR / "datahub_logo.jpeg")))

DEFAULT_USER_ID = int(os.getenv("DEFAULT_USER_ID", "108170"))
DEFAULT_OVERLAP_RATIO_PCT = float(os.getenv("DEFAULT_OVERLAP_RATIO_PCT", "60"))
DEFAULT_CORR_THRESHOLD = float(os.getenv("DEFAULT_CORR_THRESHOLD", "0.65"))
DEFAULT_MAX_NEIGHBORS = int(os.getenv("DEFAULT_MAX_NEIGHBORS", "7"))
DEFAULT_WEIGHTED_SCORE_THRESHOLD = float(os.getenv("DEFAULT_WEIGHTED_SCORE_THRESHOLD", "3.5"))
DEFAULT_TOP_N = int(os.getenv("DEFAULT_TOP_N", "5"))

# Render.com sets PORT env var, fallback to SERVER_PORT or 8080
SERVER_PORT = int(os.getenv("PORT", os.getenv("SERVER_PORT", "8080")))
SERVER_ADDRESS = os.getenv("SERVER_ADDRESS", "0.0.0.0")

MIN_USER_ID = 1
MIN_OVERLAP_RATIO = 0.0
MAX_OVERLAP_RATIO = 100.0
MIN_CORR_THRESHOLD = 0.0
MAX_CORR_THRESHOLD = 1.0
MIN_NEIGHBORS = 1
MAX_NEIGHBORS = 200
MIN_WEIGHTED_SCORE = 0.0
MAX_WEIGHTED_SCORE = 5.0
MIN_TOP_N = 1
MAX_TOP_N = 50

MAX_RATING = 5.0
FIVE_STAR_RATING = 5.0
SIMILARITY_PERCENTAGE_DIVISOR = 100.0


def _validate_user_id_config(errors: List[str]) -> None:
    """Validate DEFAULT_USER_ID configuration."""
    if not (MIN_USER_ID <= DEFAULT_USER_ID):
        errors.append(f"DEFAULT_USER_ID ({DEFAULT_USER_ID}) must be >= {MIN_USER_ID}")


def _validate_overlap_ratio_config(errors: List[str]) -> None:
    """Validate DEFAULT_OVERLAP_RATIO_PCT configuration."""
    if not (MIN_OVERLAP_RATIO <= DEFAULT_OVERLAP_RATIO_PCT <= MAX_OVERLAP_RATIO):
        errors.append(
            f"DEFAULT_OVERLAP_RATIO_PCT ({DEFAULT_OVERLAP_RATIO_PCT}) "
            f"must be between {MIN_OVERLAP_RATIO} and {MAX_OVERLAP_RATIO}"
        )


def _validate_correlation_threshold_config(errors: List[str]) -> None:
    """Validate DEFAULT_CORR_THRESHOLD configuration."""
    if not (MIN_CORR_THRESHOLD <= DEFAULT_CORR_THRESHOLD <= MAX_CORR_THRESHOLD):
        errors.append(
            f"DEFAULT_CORR_THRESHOLD ({DEFAULT_CORR_THRESHOLD}) "
            f"must be between {MIN_CORR_THRESHOLD} and {MAX_CORR_THRESHOLD}"
        )


def _validate_max_neighbors_config(errors: List[str]) -> None:
    """Validate DEFAULT_MAX_NEIGHBORS configuration."""
    if not (MIN_NEIGHBORS <= DEFAULT_MAX_NEIGHBORS <= MAX_NEIGHBORS):
        errors.append(
            f"DEFAULT_MAX_NEIGHBORS ({DEFAULT_MAX_NEIGHBORS}) "
            f"must be between {MIN_NEIGHBORS} and {MAX_NEIGHBORS}"
        )


def _validate_weighted_score_config(errors: List[str]) -> None:
    """Validate DEFAULT_WEIGHTED_SCORE_THRESHOLD configuration."""
    if not (MIN_WEIGHTED_SCORE <= DEFAULT_WEIGHTED_SCORE_THRESHOLD <= MAX_WEIGHTED_SCORE):
        errors.append(
            f"DEFAULT_WEIGHTED_SCORE_THRESHOLD ({DEFAULT_WEIGHTED_SCORE_THRESHOLD}) "
            f"must be between {MIN_WEIGHTED_SCORE} and {MAX_WEIGHTED_SCORE}"
        )


def _validate_top_n_config(errors: List[str]) -> None:
    """Validate DEFAULT_TOP_N configuration."""
    if not (MIN_TOP_N <= DEFAULT_TOP_N <= MAX_TOP_N):
        errors.append(
            f"DEFAULT_TOP_N ({DEFAULT_TOP_N}) " f"must be between {MIN_TOP_N} and {MAX_TOP_N}"
        )


def _validate_server_config(errors: List[str]) -> None:
    """Validate SERVER_PORT configuration."""
    if not (1 <= SERVER_PORT <= 65535):
        errors.append(f"SERVER_PORT ({SERVER_PORT}) must be between 1 and 65535")


def _validate_paths() -> None:
    """Validate paths exist (only warn, don't fail)."""
    if not DATA_DIR.exists():
        logger.warning(f"Data directory does not exist: {DATA_DIR}")


def validate_config() -> bool:
    """Validate configuration values at startup."""
    errors = []

    _validate_user_id_config(errors)
    _validate_overlap_ratio_config(errors)
    _validate_correlation_threshold_config(errors)
    _validate_max_neighbors_config(errors)
    _validate_weighted_score_config(errors)
    _validate_top_n_config(errors)
    _validate_server_config(errors)
    _validate_paths()

    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        logger.error(error_msg)
        raise ConfigurationError(
            error_msg,
            details={
                "validation_errors": errors,
                "config_values": {
                    "DEFAULT_USER_ID": DEFAULT_USER_ID,
                    "DEFAULT_OVERLAP_RATIO_PCT": DEFAULT_OVERLAP_RATIO_PCT,
                    "DEFAULT_CORR_THRESHOLD": DEFAULT_CORR_THRESHOLD,
                    "DEFAULT_MAX_NEIGHBORS": DEFAULT_MAX_NEIGHBORS,
                    "SERVER_PORT": SERVER_PORT,
                },
            },
        )

    logger.info("Configuration validation passed")
    return True


try:
    validate_config()
except ConfigurationError as e:
    logger.warning(f"Configuration validation failed, but continuing anyway: {e.message}")
except Exception as e:
    logger.warning(f"Unexpected error during configuration validation: {e}")
