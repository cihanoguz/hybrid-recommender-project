"""Download data file for cloud deployment."""

import os
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

from logging_config import get_logger

logger = get_logger(__name__)

DATA_URL = os.getenv(
    "DATA_URL",
    "https://github.com/cihanoguz/hybrid-recommender-project/releases/download/v1.0.0/prepare_data_demo.pkl",
)
DATA_DIR = Path(__file__).parent / "data"
DATA_FILE = DATA_DIR / "prepare_data_demo.pkl"

# Chunk size for downloading large files (10MB chunks)
CHUNK_SIZE = 10 * 1024 * 1024


def download_data_if_needed() -> bool:
    """Download data file if it doesn't exist."""
    if DATA_FILE.exists():
        logger.info(f"Data file already exists: {DATA_FILE}")
        return True

    try:
        DATA_DIR.mkdir(exist_ok=True)
        logger.info(f"Downloading data from {DATA_URL}...")

        # Use urlopen for better redirect handling (Google Drive, etc.)
        req = Request(DATA_URL)
        req.add_header("User-Agent", "Mozilla/5.0")

        with urlopen(req) as response:
            file_size = int(response.headers.get("Content-Length", 0))
            logger.info(f"File size: {file_size / (1024 * 1024):.2f} MB")

            downloaded = 0
            with open(DATA_FILE, "wb") as f:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if file_size > 0:
                        progress = (downloaded / file_size) * 100
                        logger.info(f"Download progress: {progress:.1f}%")

        logger.info(f"Data downloaded successfully: {DATA_FILE}")
        return True
    except URLError as e:
        logger.error(f"URL error while downloading data: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to download data: {e}")
        return False


if __name__ == "__main__":
    download_data_if_needed()

