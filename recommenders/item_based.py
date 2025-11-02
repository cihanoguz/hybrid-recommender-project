"""
Item-Based Recommendation System.

Implements collaborative filtering based on item similarity.
This module provides functions for finding similar items based on
user rating patterns and generating item-based recommendations.
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from logging_config import get_logger

try:
    import config
    from error_handling import RecommendationError
    from utils import validate_user_id
except ImportError:
    import logging

    logging.warning("utils.py or config.py not found, using fallback")
    config = None
    RecommendationError = Exception

logger = get_logger(__name__)


@st.cache_data(show_spinner=False)
def precompute_for_user_itembased(
    chosen_user: int,
    all_user_ids: List[int],
    rating: pd.DataFrame,
    movie: pd.DataFrame,
    user_movie_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Pre-compute item-based recommendation data for a given user.

    Finds the user's most recently rated 5-star movie and calculates
    similarity scores with other movies based on user rating patterns.

    Args:
        chosen_user: User ID to generate recommendations for
        all_user_ids: List of all valid user IDs in the dataset
        rating: DataFrame containing user ratings with columns: userId, movieId, rating
            (optionally: timestamp for determining most recent rating)
        movie: DataFrame containing movie information (must have 'movieId' and 'title')
        user_movie_df: User-movie rating matrix (users as rows, movies as columns)

    Returns:
        Dictionary containing:
            - status: Status string ("ok", "no_user", "no_five_star", "not_in_matrix", etc.)
            - reference_movie: Title of the reference movie (user's last 5-star rating)
            - similarity_df: DataFrame with similarity scores for all movies

    Note:
        - If user has no 5-star ratings, status will be "no_five_star"
        - If reference movie is not in user_movie_df matrix, status will be "not_in_matrix"
        - This function is cached using Streamlit's cache_data decorator
    """
    # Validate user ID
    is_valid, error = validate_user_id(chosen_user, all_user_ids)
    if not is_valid:
        logger.warning(f"Invalid user ID for item-based: {chosen_user} - {error}")
        return {"status": "no_user", "reference_movie": None, "similarity_df": pd.DataFrame()}

    try:
        # Movies user gave 5 stars to (using config constant)
        five_star_rating = config.FIVE_STAR_RATING if config else 5.0
        user_5 = rating[(rating["userId"] == chosen_user) & (rating["rating"] == five_star_rating)]

        if user_5.empty:
            return {
                "status": "no_five_star",
                "reference_movie": None,
                "similarity_df": pd.DataFrame(),
            }

        # most recently given 5★
        if "timestamp" in user_5.columns:
            last_fav = user_5.sort_values("timestamp", ascending=False).iloc[0]
        else:
            last_fav = user_5.iloc[0]

        ref_movie_id = last_fav["movieId"]
        ref_title_arr = movie.loc[movie["movieId"] == ref_movie_id, "title"].values
        if len(ref_title_arr) == 0:
            return {"status": "no_title", "reference_movie": None, "similarity_df": pd.DataFrame()}

        ref_title = ref_title_arr[0]

        # item-based correlation: between movies
        if ref_title not in user_movie_df.columns:
            return {
                "status": "not_in_matrix",
                "reference_movie": ref_title,
                "similarity_df": pd.DataFrame(),
            }

        ref_vector = user_movie_df[ref_title]
        sims = user_movie_df.corrwith(ref_vector).dropna()  # movie-movie similarity
        sims = sims[sims.index != ref_title]  # remove itself

        similarity_df = (
            sims.sort_values(ascending=False)
            .reset_index()
            .rename(columns={"index": "Similar Film", 0: "Similarity"})
        )

        return {"status": "ok", "reference_movie": ref_title, "similarity_df": similarity_df}
    except Exception as e:
        logger.exception(f"Error in precompute_for_user_itembased for user {chosen_user}: {e}")
        return {
            "status": "error",
            "error_message": str(e),
            "reference_movie": None,
            "similarity_df": pd.DataFrame(),
        }


def finalize_item_based_from_cache(
    precomputed_item: Dict[str, Any], top_n_item: int
) -> Tuple[str, Optional[str], pd.DataFrame]:
    """
    Finalize item-based recommendations from precomputed data.

    Processes precomputed similarity data to generate final recommendations.
    Filters and sorts similar movies based on correlation scores.

    Args:
        precomputed_item: Dictionary containing precomputed item-based data:
            - status: Status of precomputation ("ok", "no_five_star", etc.)
            - reference_movie: Title of the reference movie
            - similarity_df: DataFrame with similarity scores
        top_n_item: Number of top recommendations to return

    Returns:
        Tuple containing:
            - status: Status string ("ok", "no_similarities", "error", etc.)
            - reference_movie: Title of the reference movie (None if error)
            - recommendations_df: DataFrame with top N similar movies and scores

    Note:
        If status is not "ok", returns empty DataFrame and appropriate status.
    """
    try:
        status = precomputed_item.get("status", "unknown")
        if status != "ok":
            return status, None, pd.DataFrame()

        ref_movie = precomputed_item.get("reference_movie")
        sim_df_all = precomputed_item.get("similarity_df", pd.DataFrame())

        if sim_df_all.empty:
            logger.warning("Empty similarity dataframe in finalize_item_based_from_cache")
            return "no_similarities", ref_movie, pd.DataFrame()

        sim_df_head = sim_df_all.head(top_n_item).copy()

        return "ok", ref_movie, sim_df_head
    except KeyError as e:
        logger.exception(f"Missing key in precomputed_item: {e}")
        return "error", None, pd.DataFrame()
    except Exception as e:
        logger.exception(f"Error in finalize_item_based_from_cache: {e}")
        return "error", None, pd.DataFrame()
