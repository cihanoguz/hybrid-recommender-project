"""
Content-Based Recommendation System.

Implements recommendation based on genre similarity using cosine similarity.
This module provides functions for generating content-based recommendations
by comparing genre vectors of movies.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
import streamlit as st

from logging_config import get_logger

try:
    import config
    from error_handling import RecommendationError
except ImportError:
    import logging
    logging.warning("config.py not found, using fallback")
    config = None
    RecommendationError = Exception

logger = get_logger(__name__)


@st.cache_data(show_spinner=False)
def content_based_recommender_cached(
    movie_title: str,
    top_n: int,
    movie: pd.DataFrame,
    cosine_sim_genre: np.ndarray,
) -> Dict[str, Any]:
    """
    Content-based recommendation using genre similarity.

    Generates recommendations by finding movies with similar genres to
    the reference movie using cosine similarity on genre vectors.

    Args:
        movie_title: Title of the reference movie to find similar movies for
        top_n: Number of top recommendations to return
        movie: DataFrame containing movie information (must have 'title' and 'genres' columns)
        cosine_sim_genre: Precomputed cosine similarity matrix where [i][j] represents
            similarity between movie i and movie j based on genres

    Returns:
        Dictionary containing:
            - status: Status string ("ok", "not_found", "no_similarities", "error", etc.)
            - reference_movie: Title of the reference movie
            - reference_genres: Genres of the reference movie
            - recommendations: DataFrame with similar movies, genres, and similarity scores

    Note:
        This function is cached using Streamlit's cache_data decorator to avoid
        recalculating similarity scores for the same inputs.
    """
    try:
        # Security: Sanitize input
        try:
            from security_utils import sanitize_user_input
            movie_title = sanitize_user_input(str(movie_title), max_length=500)
        except ImportError:
            # Fallback: basic sanitization
            movie_title = str(movie_title).strip()[:500]
        
        # Validate inputs
        if not isinstance(movie_title, str) or not movie_title.strip():
            logger.warning(f"Invalid movie title: {movie_title}")
            return {
                "status": "invalid_title",
                "reference_movie": movie_title,
                "reference_genres": None,
                "recommendations": pd.DataFrame()
            }
        
        if not isinstance(top_n, int) or top_n < 1:
            logger.warning(f"Invalid top_n value: {top_n}")
            if config:
                top_n = config.DEFAULT_TOP_N
            else:
                top_n = 5
        
        # Does film exist?
        if movie_title not in movie['title'].values:
            logger.warning(f"Movie not found: {movie_title}")
            return {
                "status": "not_found",
                "reference_movie": movie_title,
                "reference_genres": None,
                "recommendations": pd.DataFrame()
            }

        # Find movie index
        movie_idx = movie[movie['title'] == movie_title].index[0]

        # Genre information
        ref_genres = movie.iloc[movie_idx]['genres']

        # Validate cosine similarity matrix
        if movie_idx >= len(cosine_sim_genre):
            logger.error(f"Movie index {movie_idx} out of range for similarity matrix")
            return {
                "status": "index_error",
                "reference_movie": movie_title,
                "reference_genres": ref_genres,
                "recommendations": pd.DataFrame()
            }

        # Get Cosine similarity scores
        sim_scores = list(enumerate(cosine_sim_genre[movie_idx]))

        # Sort by score, exclude itself
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1: top_n + 1]

        if not sim_scores:
            logger.warning(f"No similar movies found for: {movie_title}")
            return {
                "status": "no_similarities",
                "reference_movie": movie_title,
                "reference_genres": ref_genres,
                "recommendations": pd.DataFrame()
            }

        # Related movie indices
        movie_indices = [i[0] for i in sim_scores]

        # Validate indices
        if any(idx >= len(movie) for idx in movie_indices):
            logger.error("Movie indices out of range")
            return {
                "status": "index_error",
                "reference_movie": movie_title,
                "reference_genres": ref_genres,
                "recommendations": pd.DataFrame()
            }

        # Result DF
        result_df = movie.iloc[movie_indices][['title', 'genres']].copy()
        result_df['Similarity Score'] = [round(i[1], 3) for i in sim_scores]
        result_df = result_df.rename(columns={'title': 'Film', 'genres': 'Genres'})

        logger.info(f"Content-based recommendations generated for '{movie_title}': {len(result_df)} results")
        
        return {
            "status": "ok",
            "reference_movie": movie_title,
            "reference_genres": ref_genres,
            "recommendations": result_df
        }
    except Exception as e:
        logger.exception(f"Error in content_based_recommender_cached for '{movie_title}': {e}")
        return {
            "status": "error",
            "error_message": str(e),
            "reference_movie": movie_title,
            "reference_genres": None,
            "recommendations": pd.DataFrame()
        }

