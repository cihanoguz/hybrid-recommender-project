"""
Performance Utilities.

Provides utilities for performance monitoring, profiling, and optimization.
"""

import functools
import time
from typing import Any, Callable, Optional, TypeVar

from logging_config import get_logger

logger = get_logger(__name__)

# Type variable for function decorator
F = TypeVar("F", bound=Callable[..., Any])


def measure_execution_time(func: F) -> F:
    """
    Decorator to measure and log function execution time.

    Args:
        func: Function to measure

    Returns:
        Wrapped function that logs execution time

    Example:
        >>> @measure_execution_time
        ... def slow_function():
        ...     time.sleep(1)
        >>> slow_function()  # Logs: "slow_function took 1.002 seconds"
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} took {execution_time:.3f} seconds")

    return wrapper


def log_memory_usage(func: F) -> F:
    """
    Decorator to log memory usage before and after function execution.

    Requires psutil library (optional dependency).

    Args:
        func: Function to monitor

    Returns:
        Wrapped function that logs memory usage
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB

            result = func(*args, **kwargs)

            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_delta = mem_after - mem_before

            logger.debug(
                f"{func.__name__} memory usage: "
                f"{mem_before:.2f} MB -> {mem_after:.2f} MB (delta: {mem_delta:+.2f} MB)"
            )
            return result
        except ImportError:
            # psutil not available, skip memory logging
            logger.debug(f"psutil not available, skipping memory logging for {func.__name__}")
            return func(*args, **kwargs)

    return wrapper


def get_memory_usage() -> Optional[float]:
    """
    Get current memory usage in MB.

    Returns:
        Memory usage in MB, or None if psutil not available
    """
    try:
        import os

        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    except ImportError:
        logger.debug("psutil not available for memory usage monitoring")
        return None


def optimize_dataframe_memory(df, inplace: bool = False):
    """
    Optimize DataFrame memory usage by downcasting numeric types.

    This is a placeholder for future optimization.
    For now, just returns the DataFrame unchanged.

    Args:
        df: DataFrame to optimize
        inplace: Whether to modify in place

    Returns:
        Optimized DataFrame
    """
    # Future: Implement downcasting, category conversion, etc.
    # For now, this is a placeholder
    if inplace:
        return df
    return df.copy()
