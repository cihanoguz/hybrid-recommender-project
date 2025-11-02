"""Unit tests for performance_utils.py."""

import time

import pandas as pd
import pytest

try:
    from performance_utils import (
        get_memory_usage,
        log_memory_usage,
        measure_execution_time,
        optimize_dataframe_memory,
    )
except ImportError:
    pytest.skip("performance_utils not available", allow_module_level=True)


class TestMeasureExecutionTime:
    """Test measure_execution_time decorator."""

    def test_measure_execution_time_decorator(self, caplog):
        """Test that decorator measures and logs execution time."""

        @measure_execution_time
        def test_function():
            time.sleep(0.1)
            return "result"

        result = test_function()

        assert result == "result"
        assert "test_function took" in caplog.text
        assert "seconds" in caplog.text

    def test_measure_execution_time_with_args(self, caplog):
        """Test decorator works with function arguments."""

        @measure_execution_time
        def add_numbers(a, b):
            return a + b

        result = add_numbers(5, 3)

        assert result == 8
        assert "add_numbers took" in caplog.text

    def test_measure_execution_time_with_exception(self, caplog):
        """Test decorator measures time even when exception occurs."""

        @measure_execution_time
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

        assert "failing_function took" in caplog.text


class TestLogMemoryUsage:
    """Test log_memory_usage decorator."""

    def test_log_memory_usage_decorator(self, caplog):
        """Test that decorator logs memory usage when psutil available."""

        @log_memory_usage
        def test_function():
            return "result"

        result = test_function()

        assert result == "result"
        # Memory logging may or may not be present depending on psutil availability

    def test_log_memory_usage_without_psutil(self, monkeypatch, caplog):
        """Test decorator handles missing psutil gracefully."""
        import sys

        if "psutil" in sys.modules:
            monkeypatch.delitem(sys.modules, "psutil", raising=False)

        @log_memory_usage
        def test_function():
            return "result"

        result = test_function()
        assert result == "result"
        # Should not crash if psutil not available

    def test_log_memory_usage_with_exception(self):
        """Test decorator handles exceptions correctly."""

        @log_memory_usage
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()


class TestGetMemoryUsage:
    """Test get_memory_usage function."""

    def test_get_memory_usage_returns_value_or_none(self):
        """Test get_memory_usage returns value or None."""
        result = get_memory_usage()

        # Should return None if psutil not available, or a float if available
        assert result is None or isinstance(result, float)
        if result is not None:
            assert result > 0

    def test_get_memory_usage_without_psutil(self, monkeypatch):
        """Test get_memory_usage handles missing psutil."""
        import sys

        if "psutil" in sys.modules:
            monkeypatch.delitem(sys.modules, "psutil", raising=False)

        result = get_memory_usage()
        assert result is None


class TestOptimizeDataframeMemory:
    """Test optimize_dataframe_memory function."""

    def test_optimize_dataframe_memory_inplace_false(self):
        """Test optimize_dataframe_memory returns copy when inplace=False."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        result = optimize_dataframe_memory(df, inplace=False)

        assert isinstance(result, pd.DataFrame)
        assert not result is df  # Should be a copy
        assert result.equals(df)

    def test_optimize_dataframe_memory_inplace_true(self):
        """Test optimize_dataframe_memory returns same df when inplace=True."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        result = optimize_dataframe_memory(df, inplace=True)

        assert isinstance(result, pd.DataFrame)
        assert result is df  # Should be same object

    def test_optimize_dataframe_memory_preserves_data(self):
        """Test that optimization doesn't change data values."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3, 4, 5],
                "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
                "str_col": ["a", "b", "c", "d", "e"],
            }
        )

        result = optimize_dataframe_memory(df, inplace=False)

        assert result.equals(df)
        assert list(result.columns) == list(df.columns)
        assert len(result) == len(df)

    def test_optimize_dataframe_memory_with_empty_dataframe(self):
        """Test optimize_dataframe_memory with empty DataFrame."""
        df = pd.DataFrame()
        result = optimize_dataframe_memory(df, inplace=False)

        assert isinstance(result, pd.DataFrame)
        assert result.empty == df.empty
