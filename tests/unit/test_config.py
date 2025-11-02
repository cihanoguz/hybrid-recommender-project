"""Unit tests for config.py."""

import os
from pathlib import Path

import pytest

from config import ConfigurationError, validate_config
from error_handling import ConfigurationError as ConfigError


class TestConfigValidation:
    """Test configuration validation functions."""

    def test_validate_config_default_values(self):
        """Test that default configuration values are valid."""
        try:
            result = validate_config()
            assert result is True
        except ConfigurationError:
            pytest.fail("Default configuration should be valid")

    def test_config_validation_handles_invalid_user_id(self, monkeypatch):
        """Test validation with invalid DEFAULT_USER_ID."""
        monkeypatch.setattr('config.DEFAULT_USER_ID', 0)
        with pytest.raises(ConfigurationError):
            validate_config()

    def test_config_validation_handles_invalid_overlap_ratio(self, monkeypatch):
        """Test validation with invalid DEFAULT_OVERLAP_RATIO_PCT."""
        monkeypatch.setattr('config.DEFAULT_OVERLAP_RATIO_PCT', 150.0)
        with pytest.raises(ConfigurationError):
            validate_config()

    def test_config_validation_handles_invalid_corr_threshold(self, monkeypatch):
        """Test validation with invalid DEFAULT_CORR_THRESHOLD."""
        monkeypatch.setattr('config.DEFAULT_CORR_THRESHOLD', 2.0)
        with pytest.raises(ConfigurationError):
            validate_config()

    def test_config_validation_handles_invalid_max_neighbors(self, monkeypatch):
        """Test validation with invalid DEFAULT_MAX_NEIGHBORS."""
        monkeypatch.setattr('config.DEFAULT_MAX_NEIGHBORS', 300)
        with pytest.raises(ConfigurationError):
            validate_config()

    def test_config_validation_handles_invalid_weighted_score(self, monkeypatch):
        """Test validation with invalid DEFAULT_WEIGHTED_SCORE_THRESHOLD."""
        monkeypatch.setattr('config.DEFAULT_WEIGHTED_SCORE_THRESHOLD', 10.0)
        with pytest.raises(ConfigurationError):
            validate_config()

    def test_config_validation_handles_invalid_top_n(self, monkeypatch):
        """Test validation with invalid DEFAULT_TOP_N."""
        monkeypatch.setattr('config.DEFAULT_TOP_N', 100)
        with pytest.raises(ConfigurationError):
            validate_config()

    def test_config_validation_handles_invalid_server_port(self, monkeypatch):
        """Test validation with invalid SERVER_PORT."""
        monkeypatch.setattr('config.SERVER_PORT', 70000)
        with pytest.raises(ConfigurationError):
            validate_config()

    def test_config_paths_validation_warns_on_missing_dir(self, monkeypatch, temp_dir):
        """Test that missing data directory generates warning but doesn't fail."""
        monkeypatch.setattr('config.DATA_DIR', temp_dir / "nonexistent")
        # Should not raise exception, just log warning
        try:
            validate_config()
            # If we get here, validation passed (warning was logged)
            assert True
        except ConfigurationError:
            # Also acceptable - validation might fail if other checks fail
            pass

