"""Unit tests for utils.py."""

import pickle
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from utils import (
    DataLoadError,
    ValidationError,
    safe_load_pickle,
    validate_dataframe,
    validate_overlap_ratio,
    validate_user_id,
)


class TestValidateUserId:
    """Test validate_user_id function."""

    def test_valid_user_id(self, sample_user_ids):
        """Test validation with valid user ID."""
        is_valid, error = validate_user_id(1, sample_user_ids)
        assert is_valid is True
        assert error is None

    def test_invalid_type(self):
        """Test validation with invalid type."""
        is_valid, error = validate_user_id("1", [1, 2, 3])
        assert is_valid is False
        assert "integer" in error.lower()

    def test_negative_user_id(self, sample_user_ids):
        """Test validation with negative user ID."""
        is_valid, error = validate_user_id(-1, sample_user_ids)
        assert is_valid is False
        assert "positive" in error.lower()

    def test_user_id_not_in_list(self, sample_user_ids):
        """Test validation with user ID not in available list."""
        is_valid, error = validate_user_id(999, sample_user_ids)
        assert is_valid is False
        assert "not found" in error.lower()


class TestValidateOverlapRatio:
    """Test validate_overlap_ratio function."""

    def test_valid_ratio(self):
        """Test validation with valid overlap ratio."""
        is_valid, error = validate_overlap_ratio(50.0)
        assert is_valid is True
        assert error is None

    def test_valid_integer(self):
        """Test validation with integer overlap ratio."""
        is_valid, error = validate_overlap_ratio(60)
        assert is_valid is True
        assert error is None

    def test_invalid_type(self):
        """Test validation with invalid type."""
        is_valid, error = validate_overlap_ratio("50")
        assert is_valid is False
        assert "number" in error.lower()

    def test_negative_ratio(self):
        """Test validation with negative ratio."""
        is_valid, error = validate_overlap_ratio(-10.0)
        assert is_valid is False
        assert "between 0 and 100" in error.lower()

    def test_too_large_ratio(self):
        """Test validation with ratio > 100."""
        is_valid, error = validate_overlap_ratio(150.0)
        assert is_valid is False
        assert "between 0 and 100" in error.lower()

    def test_boundary_values(self):
        """Test validation with boundary values."""
        is_valid, error = validate_overlap_ratio(0.0)
        assert is_valid is True
        
        is_valid, error = validate_overlap_ratio(100.0)
        assert is_valid is True


class TestValidateDataframe:
    """Test validate_dataframe function."""

    def test_valid_dataframe(self, sample_movie_data):
        """Test validation with valid DataFrame."""
        is_valid, error = validate_dataframe(sample_movie_data)
        assert is_valid is True
        assert error is None

    def test_empty_dataframe_no_min_rows(self):
        """Test validation with empty DataFrame but no min_rows requirement."""
        df = pd.DataFrame()
        is_valid, error = validate_dataframe(df)
        assert is_valid is True
        assert error is None

    def test_empty_dataframe_with_min_rows(self):
        """Test validation with empty DataFrame but min_rows required."""
        df = pd.DataFrame()
        is_valid, error = validate_dataframe(df, min_rows=5)
        assert is_valid is False
        assert "empty" in error.lower()

    def test_dataframe_below_min_rows(self):
        """Test validation with DataFrame below minimum rows."""
        df = pd.DataFrame({'col1': [1, 2]})
        is_valid, error = validate_dataframe(df, min_rows=5)
        assert is_valid is False
        assert "minimum" in error.lower()

    def test_missing_required_columns(self, sample_movie_data):
        """Test validation with missing required columns."""
        is_valid, error = validate_dataframe(
            sample_movie_data,
            required_columns=['movieId', 'title', 'missing_col']
        )
        assert is_valid is False
        assert "Missing required columns" in error

    def test_all_required_columns_present(self, sample_movie_data):
        """Test validation with all required columns present."""
        is_valid, error = validate_dataframe(
            sample_movie_data,
            required_columns=['movieId', 'title', 'genres']
        )
        assert is_valid is True
        assert error is None


class TestSafeLoadPickle:
    """Test safe_load_pickle function."""

    def test_load_valid_pickle(self, temp_pickle_file, sample_pickle_data):
        """Test loading a valid pickle file."""
        data = safe_load_pickle(temp_pickle_file)
        assert isinstance(data, dict)
        assert 'movie' in data
        assert 'rating' in data

    def test_load_nonexistent_file(self, temp_dir):
        """Test loading a non-existent file."""
        non_existent = temp_dir / "nonexistent.pkl"
        with pytest.raises(DataLoadError) as exc_info:
            safe_load_pickle(non_existent)
        assert "not found" in str(exc_info.value).lower()

    def test_load_corrupted_pickle(self, temp_dir):
        """Test loading a corrupted pickle file."""
        corrupted_file = temp_dir / "corrupted.pkl"
        with open(corrupted_file, 'wb') as f:
            f.write(b"invalid pickle data")
        
        with pytest.raises(DataLoadError) as exc_info:
            safe_load_pickle(corrupted_file)
        assert "corrupted" in str(exc_info.value).lower() or "unpickling" in str(exc_info.value).lower()

    def test_load_pickle_with_required_keys(self, temp_pickle_file):
        """Test loading pickle file with required keys validation."""
        data = safe_load_pickle(
            temp_pickle_file,
            required_keys=['movie', 'rating', 'user_movie_df']
        )
        assert isinstance(data, dict)
        assert 'movie' in data

    def test_load_pickle_missing_required_keys(self, temp_dir, sample_pickle_data):
        """Test loading pickle file with missing required keys."""
        pickle_file = temp_dir / "partial.pkl"
        partial_data = {'movie': sample_pickle_data['movie']}
        
        with open(pickle_file, 'wb') as f:
            pickle.dump(partial_data, f)
        
        with pytest.raises(DataLoadError) as exc_info:
            safe_load_pickle(pickle_file, required_keys=['movie', 'rating'])
        assert "Missing required keys" in str(exc_info.value)

    def test_load_non_dict_pickle(self, temp_dir):
        """Test loading a pickle file that's not a dict."""
        pickle_file = temp_dir / "list.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump([1, 2, 3], f)
        
        # Should work if no required_keys specified
        data = safe_load_pickle(pickle_file)
        assert isinstance(data, list)
        assert data == [1, 2, 3]

