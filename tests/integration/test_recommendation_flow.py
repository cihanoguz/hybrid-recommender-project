"""Integration tests for recommendation flow."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mock_streamlit_cache(monkeypatch):
    """Mock Streamlit cache decorator."""
    def no_cache(func):
        return func
    import sys
    if 'streamlit' not in sys.modules:
        import types
        mock_st = types.ModuleType('streamlit')
        mock_st.cache_data = lambda **kwargs: lambda f: f
        sys.modules['streamlit'] = mock_st


class TestUserBasedFlow:
    """Integration tests for user-based recommendation flow."""

    def test_full_user_based_flow(self, sample_user_movie_df, sample_rating_data, sample_user_ids, sample_movie_data, mock_streamlit_cache):
        """Test complete user-based recommendation flow."""
        from recommenders.user_based import (
            finalize_user_based_from_cache,
            precompute_for_user_userbased,
        )
        
        # Step 1: Precompute
        precomputed = precompute_for_user_userbased(
            chosen_user=1,
            all_user_ids=sample_user_ids,
            user_movie_df=sample_user_movie_df,
            rating=sample_rating_data
        )
        
        # Step 2: Finalize recommendations
        if precomputed["status"] == "ok":
            result = finalize_user_based_from_cache(
                precomputed=precomputed,
                min_overlap_ratio_pct=50.0,
                corr_threshold=0.5,
                max_neighbors=5,
                weighted_score_threshold=3.0,
                top_n=5,
                chosen_user=1,
                rating=sample_rating_data,
                movie=sample_movie_data
            )
            
            assert "status" in result
            assert "recommendations" in result
            assert isinstance(result["recommendations"], pd.DataFrame)


class TestItemBasedFlow:
    """Integration tests for item-based recommendation flow."""

    def test_full_item_based_flow(self, sample_user_ids, sample_movie_data, sample_user_movie_df, mock_streamlit_cache):
        """Test complete item-based recommendation flow."""
        from recommenders.item_based import (
            finalize_item_based_from_cache,
            precompute_for_user_itembased,
        )
        
        high_ratings = pd.DataFrame({
            'userId': [1, 1, 2, 2],
            'movieId': [1, 2, 1, 3],
            'rating': [5.0, 4.5, 5.0, 4.0],
            'timestamp': [1000, 2000, 3000, 4000]
        })
        
        # Step 1: Precompute
        precomputed = precompute_for_user_itembased(
            chosen_user=1,
            all_user_ids=sample_user_ids,
            rating=high_ratings,
            movie=sample_movie_data,
            user_movie_df=sample_user_movie_df
        )
        
        # Step 2: Finalize recommendations
        if precomputed["status"] == "ok":
            status, ref_movie, recommendations = finalize_item_based_from_cache(
                precomputed,
                top_n_item=5
            )
            
            assert status in ["ok", "no_five_star", "not_in_matrix"]
            if status == "ok":
                assert ref_movie is not None
                assert isinstance(recommendations, pd.DataFrame)


class TestContentBasedFlow:
    """Integration tests for content-based recommendation flow."""

    def test_full_content_based_flow(self, sample_movie_data, mock_streamlit_cache):
        """Test complete content-based recommendation flow."""
        from recommenders.content_based import content_based_recommender_cached
        
        cosine_sim = np.eye(len(sample_movie_data))
        cosine_sim[0, 1] = 0.9
        cosine_sim[1, 0] = 0.9
        
        result = content_based_recommender_cached(
            movie_title="Movie A",
            top_n=5,
            movie=sample_movie_data,
            cosine_sim_genre=cosine_sim
        )
        
        if result["status"] == "ok":
            assert "reference_movie" in result
            assert "reference_genres" in result
            assert "recommendations" in result
            assert isinstance(result["recommendations"], pd.DataFrame)
            assert len(result["recommendations"]) <= 5


class TestHybridFlow:
    """Integration tests for hybrid recommendation flow."""

    def test_hybrid_combines_all_approaches(self, sample_user_ids, sample_movie_data, sample_rating_data, sample_user_movie_df, mock_streamlit_cache):
        """Test that hybrid approach can combine all three recommendation methods."""
        from recommenders.user_based import (
            finalize_user_based_from_cache,
            precompute_for_user_userbased,
        )
        from recommenders.item_based import (
            finalize_item_based_from_cache,
            precompute_for_user_itembased,
        )
        from recommenders.content_based import content_based_recommender_cached
        
        # User-based
        pre_u = precompute_for_user_userbased(1, sample_user_ids, sample_user_movie_df, sample_rating_data)
        if pre_u["status"] == "ok":
            result_u = finalize_user_based_from_cache(
                pre_u, 50.0, 0.5, 5, 3.0, 5, 1, sample_rating_data, sample_movie_data
            )
            assert "recommendations" in result_u
        
        # Item-based
        high_ratings = sample_rating_data.copy()
        high_ratings.loc[0, 'rating'] = 5.0
        pre_i = precompute_for_user_itembased(1, sample_user_ids, high_ratings, sample_movie_data, sample_user_movie_df)
        if pre_i["status"] == "ok":
            status_i, ref_movie, result_i = finalize_item_based_from_cache(pre_i, 5)
            assert status_i in ["ok", "no_five_star"]
        
        # Content-based
        cosine_sim = np.eye(len(sample_movie_data))
        result_cb = content_based_recommender_cached(
            "Movie A", 5, sample_movie_data, cosine_sim
        )
        assert "status" in result_cb

