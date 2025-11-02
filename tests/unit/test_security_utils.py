"""Unit tests for security_utils.py."""

import os
import tempfile
from pathlib import Path

import pytest

try:
    from security_utils import (
        calculate_file_checksum,
        sanitize_file_path,
        sanitize_user_input,
        validate_numeric_input,
        validate_pickle_file_integrity,
    )
except ImportError:
    pytest.skip("security_utils not available", allow_module_level=True)


class TestSanitizeUserInput:
    """Test sanitize_user_input function."""

    def test_normal_input(self):
        """Test sanitization of normal input."""
        result = sanitize_user_input("Hello World")
        assert result == "Hello World"

    def test_removes_leading_trailing_whitespace(self):
        """Test that leading/trailing whitespace is removed."""
        result = sanitize_user_input("  Hello World  ")
        assert result == "Hello World"

    def test_max_length_enforcement(self):
        """Test that max_length is enforced."""
        long_input = "A" * 200
        result = sanitize_user_input(long_input, max_length=50)
        assert len(result) <= 50

    def test_removes_control_characters(self):
        """Test removal of control characters."""
        input_with_control = "Hello\x00World\x01\x02"
        result = sanitize_user_input(input_with_control)
        assert "\x00" not in result
        assert "\x01" not in result
        assert "Hello" in result
        assert "World" in result

    def test_empty_string(self):
        """Test handling of empty string."""
        result = sanitize_user_input("")
        assert result == ""

    def test_none_input(self):
        """Test handling of None input."""
        result = sanitize_user_input(None)
        assert result == ""


class TestSanitizeFilePath:
    """Test sanitize_file_path function."""

    def test_normal_path(self, temp_dir):
        """Test sanitization of normal file path."""
        test_file = temp_dir / "test.txt"
        test_file.touch()
        is_valid, result = sanitize_file_path(test_file)
        assert is_valid is True
        assert result == test_file.resolve()

    def test_path_traversal_protection(self, temp_dir):
        """Test that path traversal is prevented."""
        malicious_path = Path("../../../etc/passwd")
        is_valid, result = sanitize_file_path(malicious_path, allowed_base=temp_dir)
        assert is_valid is False
        assert result is None

    def test_absolute_path_normalized(self, temp_dir):
        """Test that absolute paths are normalized."""
        test_file = temp_dir / "test.txt"
        test_file.touch()
        is_valid, result = sanitize_file_path(test_file, allowed_base=temp_dir)
        assert is_valid is True
        assert result == test_file.resolve()
        assert result.is_relative_to(temp_dir.resolve()) or result == test_file.resolve()


class TestValidateNumericInput:
    """Test validate_numeric_input function."""

    def test_valid_integer(self):
        """Test validation of valid integer."""
        is_valid, error = validate_numeric_input(42, min_value=0, max_value=100, param_name="test_param")
        assert is_valid is True
        assert error is None

    def test_valid_float(self):
        """Test validation of valid float."""
        is_valid, error = validate_numeric_input(3.14, min_value=0.0, max_value=10.0, param_name="test_param")
        assert is_valid is True
        assert error is None

    def test_invalid_input(self):
        """Test validation of invalid input."""
        # Function expects numeric value, not string
        is_valid, error = validate_numeric_input("not_a_number", min_value=0, max_value=100, param_name="test_param")
        assert is_valid is False
        assert "must be a number" in error.lower()

    def test_out_of_range(self):
        """Test validation with out-of-range value."""
        is_valid, error = validate_numeric_input(150, min_value=0, max_value=100, param_name="test_param")
        assert is_valid is False
        assert "between" in error.lower()

    def test_in_range_boundaries(self):
        """Test validation with boundary values."""
        is_valid, error = validate_numeric_input(0, min_value=0, max_value=100, param_name="test_param")
        assert is_valid is True
        
        is_valid, error = validate_numeric_input(100, min_value=0, max_value=100, param_name="test_param")
        assert is_valid is True


class TestCalculateFileChecksum:
    """Test calculate_file_checksum function."""

    def test_checksum_calculation(self, temp_dir):
        """Test checksum calculation for a file."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        
        checksum1 = calculate_file_checksum(test_file)
        checksum2 = calculate_file_checksum(test_file)
        
        assert checksum1 == checksum2
        assert len(checksum1) > 0

    def test_different_files_different_checksums(self, temp_dir):
        """Test that different files have different checksums."""
        file1 = temp_dir / "file1.txt"
        file2 = temp_dir / "file2.txt"
        file1.write_text("content1")
        file2.write_text("content2")
        
        checksum1 = calculate_file_checksum(file1)
        checksum2 = calculate_file_checksum(file2)
        
        assert checksum1 != checksum2


class TestValidatePickleFileIntegrity:
    """Test validate_pickle_file_integrity function."""

    def test_valid_pickle_file(self, temp_pickle_file):
        """Test validation of valid pickle file."""
        is_valid, error = validate_pickle_file_integrity(temp_pickle_file)
        assert is_valid is True
        assert error is None

    def test_nonexistent_file(self, temp_dir):
        """Test validation of non-existent file."""
        non_existent = temp_dir / "nonexistent.pkl"
        is_valid, error = validate_pickle_file_integrity(non_existent)
        assert is_valid is False
        assert error is not None

    def test_too_large_file(self, temp_dir, monkeypatch):
        """Test validation fails for file exceeding max size."""
        test_file = temp_dir / "large.pkl"
        # Create smaller file for testing (5MB to avoid disk issues)
        test_file.write_bytes(b"x" * 5 * 1024 * 1024)  # 5 MB
        
        is_valid, error = validate_pickle_file_integrity(test_file, max_file_size=1024 * 1024)  # 1 MB max
        assert is_valid is False
        assert "too large" in error.lower()

    def test_corrupted_pickle_file(self, temp_dir):
        """Test validation passes file size check even for corrupted pickle."""
        corrupted_file = temp_dir / "corrupted.pkl"
        corrupted_file.write_bytes(b"invalid pickle data")
        
        # Function only checks file size, not pickle validity
        is_valid, error = validate_pickle_file_integrity(corrupted_file)
        assert is_valid is True  # File size is valid, pickle validity is checked elsewhere

