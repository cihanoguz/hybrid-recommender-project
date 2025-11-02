"""Centralized logging configuration."""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels."""

    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        return super().format(record)


def _get_log_level(log_level: Optional[str]) -> Tuple[str, int]:
    """Get and validate log level."""
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    else:
        log_level = log_level.upper()
    
    numeric_level = getattr(logging, log_level, None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    return log_level, numeric_level


def _get_format_string(format_string: Optional[str]) -> str:
    """Get format string with default if not provided."""
    if format_string is None:
        return (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(funcName)s:%(lineno)d - %(message)s'
        )
    return format_string


def _create_console_handler(
    numeric_level: int,
    format_string: str,
    enable_colors: bool,
) -> logging.StreamHandler:
    """Create and configure console handler."""
    console_formatter = (
        ColoredFormatter(format_string) 
        if enable_colors and sys.stdout.isatty() 
        else logging.Formatter(format_string)
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(console_formatter)
    
    return console_handler


def _create_file_handler(
    log_file: Path,
    numeric_level: int,
) -> logging.FileHandler:
    """Create and configure file handler."""
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - '
        '%(pathname)s:%(lineno)d - %(funcName)s - %(message)s'
    )
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(file_formatter)
    
    return file_handler


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[Path] = None,
    enable_colors: bool = True,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Set up centralized logging configuration."""
    log_level_str, numeric_level = _get_log_level(log_level)
    format_str = _get_format_string(format_string)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.handlers.clear()
    
    console_handler = _create_console_handler(numeric_level, format_str, enable_colors)
    root_logger.addHandler(console_handler)
    
    if log_file:
        file_handler = _create_file_handler(log_file, numeric_level)
        root_logger.addHandler(file_handler)
    
    root_logger.propagate = False
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {log_level_str}")
    if log_file:
        logger.info(f"Log file: {log_file}")
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module."""
    return logging.getLogger(name)


_default_log_level = os.getenv("LOG_LEVEL", "INFO")
_default_log_file = os.getenv("LOG_FILE")
if _default_log_file:
    _default_log_file = Path(_default_log_file)

try:
    setup_logging(
        log_level=_default_log_level,
        log_file=_default_log_file if _default_log_file else None,
    )
except Exception as e:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.getLogger(__name__).warning(f"Failed to setup custom logging: {e}")

