"""Download data file for cloud deployment."""

import os
from pathlib import Path
from urllib.request import urlretrieve

from logging_config import get_logger

logger = get_logger(__name__)

DATA_URL = os.getenv(
    "DATA_URL",
    "https://github.com/cihanoguz/hybrid-recommender-project/releases/download/v1.0.0/prepare_data_demo.pkl",
)
DATA_DIR = Path(__file__).parent / "data"
DATA_FILE = DATA_DIR / "prepare_data_demo.pkl"


def download_data_if_needed() -> bool:
    """Download data file if it doesn't exist."""
    if DATA_FILE.exists():
        logger.info(f"Data file already exists: {DATA_FILE}")
        return True

    try:
        DATA_DIR.mkdir(exist_ok=True)
        logger.info(f"Downloading data from {DATA_URL}...")
        urlretrieve(DATA_URL, DATA_FILE)
        logger.info(f"Data downloaded successfully: {DATA_FILE}")
        return True
    except Exception as e:
        logger.error(f"Failed to download data: {e}")
        return False


if __name__ == "__main__":
    download_data_if_needed()

