"""Unit tests for recommendation algorithms."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mock_streamlit_cache(monkeypatch):
    """Mock Streamlit cache decorator for testing."""
    def no_cache(func):
        return func
    import sys
    if 'streamlit' not in sys.modules:
        import types
        mock_st = types.ModuleType('streamlit')
        mock_st.cache_data = lambda **kwargs: lambda f: f
        sys.modules['streamlit'] = mock_st
    else:
        import streamlit as st
        monkeypatch.setattr(st, 'cache_data', lambda **kwargs: lambda f: f)


class TestUserBasedRecommender:
    """Test user-based recommendation functions."""

    def test_precompute_for_user_userbased_invalid_user(self, sample_user_ids, sample_user_movie_df, sample_rating_data, mock_streamlit_cache):
        """Test precompute with invalid user ID."""
        from recommenders.user_based import precompute_for_user_userbased
        
        result = precompute_for_user_userbased(
            chosen_user=999,
            all_user_ids=sample_user_ids,
            user_movie_df=sample_user_movie_df,
            rating=sample_rating_data
        )
        
        assert result["status"] == "no_user"
        assert len(result["movies_watched"]) == 0

    def test_precompute_for_user_userbased_no_movies(self, sample_user_ids, sample_rating_data, mock_streamlit_cache):
        """Test precompute for user with no watched movies."""
        from recommenders.user_based import precompute_for_user_userbased
        
        empty_df = pd.DataFrame({}, index=[1])
        result = precompute_for_user_userbased(
            chosen_user=1,
            all_user_ids=sample_user_ids,
            user_movie_df=empty_df,
            rating=sample_rating_data
        )
        
        assert result["status"] == "no_movies"

    def test_precompute_for_user_userbased_valid(self, sample_user_ids, sample_user_movie_df, sample_rating_data, mock_streamlit_cache):
        """Test precompute for valid user."""
        from recommenders.user_based import precompute_for_user_userbased
        
        result = precompute_for_user_userbased(
            chosen_user=1,
            all_user_ids=sample_user_ids,
            user_movie_df=sample_user_movie_df,
            rating=sample_rating_data
        )
        
        assert result["status"] in ["ok", "no_corr"]
        assert "movies_watched" in result
        assert "candidate_users_df" in result
        assert "corr_df" in result


class TestItemBasedRecommender:
    """Test item-based recommendation functions."""

    def test_precompute_for_user_itembased_invalid_user(self, sample_user_ids, sample_movie_data, sample_rating_data, sample_user_movie_df, mock_streamlit_cache):
        """Test precompute with invalid user ID."""
        from recommenders.item_based import precompute_for_user_itembased
        
        result = precompute_for_user_itembased(
            chosen_user=999,
            all_user_ids=sample_user_ids,
            rating=sample_rating_data,
            movie=sample_movie_data,
            user_movie_df=sample_user_movie_df
        )
        
        assert result["status"] == "no_user"

    def test_precompute_for_user_itembased_no_five_star(self, sample_user_ids, sample_movie_data, sample_user_movie_df, mock_streamlit_cache):
        """Test precompute for user with no 5-star ratings."""
        from recommenders.item_based import precompute_for_user_itembased
        
        low_ratings = pd.DataFrame({
            'userId': [1, 1],
            'movieId': [1, 2],
            'rating': [3.0, 4.0],
            'timestamp': [1000, 2000]
        })
        
        result = precompute_for_user_itembased(
            chosen_user=1,
            all_user_ids=sample_user_ids,
            rating=low_ratings,
            movie=sample_movie_data,
            user_movie_df=sample_user_movie_df
        )
        
        assert result["status"] == "no_five_star"

    def test_precompute_for_user_itembased_valid(self, sample_user_ids, sample_movie_data, sample_user_movie_df, mock_streamlit_cache):
        """Test precompute for valid user with 5-star rating."""
        from recommenders.item_based import precompute_for_user_itembased
        
        high_ratings = pd.DataFrame({
            'userId': [1, 1],
            'movieId': [1, 2],
            'rating': [5.0, 4.5],
            'timestamp': [1000, 2000]
        })
        
        # Ensure movie exists in user_movie_df
        user_movie_df_with_movie = sample_user_movie_df.copy()
        if 1 not in user_movie_df_with_movie.index:
            user_movie_df_with_movie.loc[1] = {1: 5.0, 2: 4.5}
        
        result = precompute_for_user_itembased(
            chosen_user=1,
            all_user_ids=sample_user_ids,
            rating=high_ratings,
            movie=sample_movie_data,
            user_movie_df=user_movie_df_with_movie
        )
        
        assert result["status"] in ["ok", "no_five_star", "not_in_matrix"]
        assert "reference_movie" in result


class TestContentBasedRecommender:
    """Test content-based recommendation functions."""

    def test_content_based_recommender_cached_movie_not_found(self, sample_movie_data, mock_streamlit_cache):
        """Test content-based recommender with non-existent movie."""
        from recommenders.content_based import content_based_recommender_cached
        
        cosine_sim = np.eye(len(sample_movie_data))
        result = content_based_recommender_cached(
            movie_title="Non-existent Movie",
            top_n=5,
            movie=sample_movie_data,
            cosine_sim_genre=cosine_sim
        )
        
        assert result["status"] != "ok"

    def test_content_based_recommender_cached_valid(self, sample_movie_data, mock_streamlit_cache):
        """Test content-based recommender with valid movie."""
        from recommenders.content_based import content_based_recommender_cached
        
        cosine_sim = np.eye(len(sample_movie_data))
        cosine_sim[0, 1] = 0.8
        cosine_sim[1, 0] = 0.8
        
        result = content_based_recommender_cached(
            movie_title="Movie A",
            top_n=5,
            movie=sample_movie_data,
            cosine_sim_genre=cosine_sim
        )
        
        assert result["status"] == "ok"
        assert "reference_movie" in result
        assert "recommendations" in result
        assert len(result["recommendations"]) <= 5

