"""Integration tests for data loading."""

import pickle
from pathlib import Path

import pandas as pd
import pytest

from data_loader.loader import load_data
from utils import DataLoadError


class TestDataLoading:
    """Integration tests for data loading pipeline."""

    def test_load_data_success(self, temp_pickle_file):
        """Test successful data loading from pickle file."""
        result = load_data(temp_pickle_file)
        
        assert len(result) == 7
        movie, rating, df_full, common_movies, user_movie_df, all_user_ids, cosine_sim_genre = result
        
        assert isinstance(movie, pd.DataFrame)
        assert isinstance(rating, pd.DataFrame)
        assert isinstance(df_full, pd.DataFrame)
        assert isinstance(common_movies, pd.DataFrame)
        assert isinstance(user_movie_df, pd.DataFrame)
        assert isinstance(all_user_ids, list)
        assert isinstance(cosine_sim_genre, (pd.DataFrame, type(None))) or hasattr(cosine_sim_genre, 'shape')

    def test_load_data_file_not_found(self, temp_dir):
        """Test data loading with non-existent file."""
        non_existent = temp_dir / "nonexistent.pkl"
        
        with pytest.raises(DataLoadError):
            load_data(non_existent)

    def test_load_data_corrupted_file(self, temp_dir):
        """Test data loading with corrupted pickle file."""
        corrupted_file = temp_dir / "corrupted.pkl"
        corrupted_file.write_bytes(b"invalid pickle data")
        
        with pytest.raises(DataLoadError):
            load_data(corrupted_file)

    def test_load_data_missing_keys(self, temp_dir):
        """Test data loading with pickle file missing required keys."""
        incomplete_data = {
            'movie': pd.DataFrame({'movieId': [1], 'title': ['Test']}),
            'rating': pd.DataFrame({'userId': [1], 'movieId': [1], 'rating': [5.0]})
        }
        
        incomplete_file = temp_dir / "incomplete.pkl"
        with open(incomplete_file, 'wb') as f:
            pickle.dump(incomplete_data, f)
        
        # Should either raise error or handle gracefully
        try:
            result = load_data(incomplete_file)
            # If it succeeds, verify structure
            assert len(result) == 7
        except (DataLoadError, KeyError, ValueError):
            # Expected behavior - missing required keys
            pass

