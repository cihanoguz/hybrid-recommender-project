"""
User-Based Recommendation System.

Implements collaborative filtering based on user similarity.
This module provides functions for precomputing user similarity data
and generating user-based recommendations.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from logging_config import get_logger

try:
    import config
    from error_handling import RecommendationError
    from utils import validate_correlation_threshold, validate_overlap_ratio, validate_user_id
except ImportError:
    import logging

    logging.warning("utils.py or config.py not found, using fallback")
    config = None
    RecommendationError = Exception

logger = get_logger(__name__)


@st.cache_data(show_spinner=False)
def precompute_for_user_userbased(
    chosen_user: int,
    all_user_ids: List[int],
    user_movie_df: pd.DataFrame,
    rating: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Pre-compute user-based recommendation data for a given user.

    This function performs the initial computation for user-based collaborative
    filtering by finding candidate users, calculating correlations, and preparing
    the data needed for final recommendation generation.

    Args:
        chosen_user: User ID to generate recommendations for
        all_user_ids: List of all valid user IDs in the dataset
        user_movie_df: User-movie rating matrix (users as rows, movies as columns)
        rating: DataFrame containing user ratings with columns: userId, movieId, rating

    Returns:
        Dictionary containing:
            - status: Status string ("ok", "no_user", "no_movies", "no_corr", etc.)
            - movies_watched: List of movie IDs watched by the chosen user
            - candidate_users_df: DataFrame with users who watched common movies
            - corr_df: DataFrame with correlation scores between chosen_user and others
            - top_users_ratings: DataFrame with ratings from candidate neighbors

    Note:
        This function is cached using Streamlit's cache_data decorator to avoid
        recomputing for the same user on every app rerun.
    """
    # Validate user ID
    is_valid, error = validate_user_id(chosen_user, all_user_ids)
    if not is_valid:
        logger.warning(f"Invalid user ID: {chosen_user} - {error}")
        return {
            "status": "no_user",
            "movies_watched": [],
            "candidate_users_df": pd.DataFrame(),
            "corr_df": pd.DataFrame(),
            "top_users_ratings": pd.DataFrame(),
        }

    try:
        # Movies watched by target user
        if chosen_user not in user_movie_df.index:
            return {
                "status": "no_user",
                "movies_watched": [],
                "candidate_users_df": pd.DataFrame(),
                "corr_df": pd.DataFrame(),
                "top_users_ratings": pd.DataFrame(),
            }

        # Movies watched by target user
        row = user_movie_df.loc[[chosen_user]]
        movies_watched = row.columns[row.notna().any()].to_list()

        if len(movies_watched) == 0:
            return {
                "status": "no_movies",
                "movies_watched": [],
                "candidate_users_df": pd.DataFrame(),
                "corr_df": pd.DataFrame(),
                "top_users_ratings": pd.DataFrame(),
            }

        # Sub-matrix based on target user's watched movies
        movies_watched_df = user_movie_df[movies_watched].copy()

        # How many of these movies each other user has watched
        user_movie_count_series = movies_watched_df.notnull().sum(axis=1)

        candidate_users_df = user_movie_count_series.reset_index().rename(
            columns={"index": "userId", 0: "movie_count"}
        )
        candidate_users_df.columns = ["userId", "movie_count"]

        candidate_users_df = candidate_users_df[candidate_users_df["userId"] != chosen_user].copy()

        # Calculate correlation
        base_vector = user_movie_df.loc[chosen_user]
        corr_series = movies_watched_df.T.corrwith(base_vector).dropna()

        corr_df = corr_series.reset_index().rename(columns={"index": "userId", 0: "corr"})
        corr_df.columns = ["userId", "corr"]
        corr_df = corr_df[corr_df["userId"] != chosen_user].copy()

        if corr_df.empty:
            return {
                "status": "no_corr",
                "movies_watched": movies_watched,
                "candidate_users_df": candidate_users_df,
                "corr_df": corr_df,
                "top_users_ratings": pd.DataFrame(),
            }

        # Ratings from candidate neighbors
        top_users_ratings = corr_df.merge(
            rating[["userId", "movieId", "rating"]], on="userId", how="inner"
        )

        if top_users_ratings.empty:
            return {
                "status": "no_ratings_from_neighbors",
                "movies_watched": movies_watched,
                "candidate_users_df": candidate_users_df,
                "corr_df": corr_df,
                "top_users_ratings": top_users_ratings,
            }

        return {
            "status": "ok",
            "movies_watched": movies_watched,
            "candidate_users_df": candidate_users_df,
            "corr_df": corr_df,
            "top_users_ratings": top_users_ratings,
        }
    except Exception as e:
        logger.exception(f"Error in precompute_for_user_userbased for user {chosen_user}: {e}")
        return {
            "status": "error",
            "error_message": str(e),
            "movies_watched": [],
            "candidate_users_df": pd.DataFrame(),
            "corr_df": pd.DataFrame(),
            "top_users_ratings": pd.DataFrame(),
        }


def _create_error_response(
    status: str,
    candidate_users_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    corr_filtered: Optional[pd.DataFrame] = None,
    neighbor_ratings: Optional[pd.DataFrame] = None,
    debug_info: Optional[Dict[str, int]] = None,
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create standardized error response dictionary.

    Args:
        status: Error status string
        candidate_users_df: Candidate users DataFrame
        corr_df: Correlation DataFrame
        corr_filtered: Filtered correlation DataFrame (optional)
        neighbor_ratings: Neighbor ratings DataFrame (optional)
        debug_info: Debug information dictionary (optional)
        error_message: Error message string (optional)

    Returns:
        Standardized error response dictionary
    """
    response = {
        "status": status,
        "recommendations": pd.DataFrame(),
        "debug_info": debug_info or {},
        "dbg_candidate_users_df": candidate_users_df,
        "dbg_corr_df": corr_df,
        "dbg_corr_filtered": corr_filtered if corr_filtered is not None else pd.DataFrame(),
        "dbg_neighbor_ratings": (
            neighbor_ratings if neighbor_ratings is not None else pd.DataFrame()
        ),
    }
    if error_message:
        response["error_message"] = error_message
    return response


def _apply_overlap_filter(
    candidate_users_df: pd.DataFrame,
    movies_watched: List[int],
    min_overlap_ratio_pct: float,
) -> pd.Series:
    """
    Apply overlap filter to candidate users.

    Args:
        candidate_users_df: DataFrame with candidate users and movie counts
        movies_watched: List of movie IDs watched by target user
        min_overlap_ratio_pct: Minimum overlap ratio percentage

    Returns:
        Series of user IDs that pass the overlap filter
    """
    percentage_divisor = config.SIMILARITY_PERCENTAGE_DIVISOR if config else 100.0
    threshold_common = len(movies_watched) * (min_overlap_ratio_pct / percentage_divisor)

    good_overlap_users = candidate_users_df[candidate_users_df["movie_count"] >= threshold_common][
        "userId"
    ]

    return good_overlap_users


def _apply_correlation_filter(
    corr_df: pd.DataFrame,
    good_overlap_users: pd.Series,
    corr_threshold: float,
) -> pd.DataFrame:
    """
    Apply correlation filter to users.

    Args:
        corr_df: DataFrame with correlation scores
        good_overlap_users: Series of user IDs that passed overlap filter
        corr_threshold: Minimum correlation threshold

    Returns:
        Filtered DataFrame with correlation scores
    """
    return corr_df[
        (corr_df["userId"].isin(good_overlap_users)) & (corr_df["corr"] >= corr_threshold)
    ].copy()


def _apply_neighbor_limit(
    corr_filtered: pd.DataFrame,
    max_neighbors: int,
) -> pd.DataFrame:
    """
    Limit the number of neighbors to consider.

    Args:
        corr_filtered: Filtered correlation DataFrame
        max_neighbors: Maximum number of neighbors

    Returns:
        DataFrame with top N neighbors by correlation
    """
    return corr_filtered.sort_values("corr", ascending=False).head(max_neighbors).copy()


def _calculate_weighted_ratings(
    neighbor_ratings: pd.DataFrame,
    chosen_user: int,
    rating: pd.DataFrame,
    weighted_score_threshold: float,
    top_n: int,
) -> pd.DataFrame:
    """
    Calculate weighted ratings and filter recommendations.

    Args:
        neighbor_ratings: DataFrame with neighbor ratings and correlations
        chosen_user: Target user ID
        rating: Rating DataFrame
        weighted_score_threshold: Minimum weighted score threshold
        top_n: Number of top recommendations

    Returns:
        DataFrame with filtered and sorted recommendations
    """
    # Find correlation column
    possible_corr_cols = [c for c in neighbor_ratings.columns if "corr" in c]
    if not possible_corr_cols:
        return pd.DataFrame()

    corr_col = possible_corr_cols[0]
    neighbor_ratings["weighted_rating"] = neighbor_ratings["rating"] * neighbor_ratings[corr_col]

    # Aggregate by movie
    recommendation_df = (
        neighbor_ratings.groupby("movieId")
        .agg(weighted_rating=("weighted_rating", "mean"))
        .reset_index()
    )

    # Remove movies user has already watched
    seen_ids = rating.loc[rating["userId"] == chosen_user, "movieId"].unique().tolist()
    recommendation_df = recommendation_df[~recommendation_df["movieId"].isin(seen_ids)]

    # Apply weighted score threshold
    recommendation_df = recommendation_df[
        recommendation_df["weighted_rating"] >= weighted_score_threshold
    ]

    # Sort and limit to top N
    return recommendation_df.sort_values("weighted_rating", ascending=False).head(top_n).copy()


def _build_final_recommendations(
    recommendation_df: pd.DataFrame,
    movie: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build final recommendations with movie titles.

    Args:
        recommendation_df: DataFrame with movie IDs and scores
        movie: Movie DataFrame with movieId and title

    Returns:
        DataFrame with movie titles and scores
    """
    recommendation_df = recommendation_df.merge(
        movie[["movieId", "title"]], on="movieId", how="left"
    )

    return recommendation_df[["title", "weighted_rating"]].rename(
        columns={"title": "Film", "weighted_rating": "Score"}
    )


def finalize_user_based_from_cache(
    precomputed: Dict[str, Any],
    min_overlap_ratio_pct: float,
    corr_threshold: float,
    max_neighbors: int,
    weighted_score_threshold: float,
    top_n: int,
    chosen_user: int,
    rating: pd.DataFrame,
    movie: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Finalize user-based recommendations from precomputed data.

    This function processes precomputed similarity data and applies filtering
    criteria to generate final user-based recommendations.

    Args:
        precomputed: Dictionary containing precomputed data from
            precompute_for_user_userbased() function
        min_overlap_ratio_pct: Minimum percentage of common movies required
            between users (0.0 to 100.0)
        corr_threshold: Minimum correlation threshold for similarity (0.0 to 1.0)
        max_neighbors: Maximum number of similar users to consider
        weighted_score_threshold: Minimum weighted score for recommendations
            (weighted_score = rating × correlation)
        top_n: Number of top recommendations to return
        chosen_user: Target user ID for recommendations
        rating: DataFrame containing user ratings
        movie: DataFrame containing movie information (must have 'movieId' and 'title')

    Returns:
        Dictionary containing:
            - status: Status string ("ok", "no_candidates_after_overlap", etc.)
            - recommendations: DataFrame with top N recommended movies and scores
            - debug_info: Dictionary with intermediate calculation statistics
            - dbg_candidate_users_df: Debug DataFrame with candidate users
            - dbg_corr_df: Debug DataFrame with correlation scores
            - dbg_corr_filtered: Debug DataFrame with filtered correlations
            - dbg_neighbor_ratings: Debug DataFrame with neighbor ratings

    Note:
        This function applies multiple filtering steps:
        1. Overlap filter: Users must have watched at least min_overlap_ratio_pct% of
           the same movies as the target user
        2. Correlation filter: Only users with correlation >= corr_threshold
        3. Neighbor limit: Only top max_neighbors users by correlation
        4. Score threshold: Only movies with weighted_score >= weighted_score_threshold
    """
    # Validate inputs
    overlap_valid, overlap_error = validate_overlap_ratio(min_overlap_ratio_pct)
    if not overlap_valid:
        logger.warning(f"Invalid overlap ratio: {overlap_error}")

    corr_valid, corr_error = validate_correlation_threshold(corr_threshold)
    if not corr_valid:
        logger.warning(f"Invalid correlation threshold: {corr_error}")

    try:
        # Check precomputed status
        status = precomputed["status"]
        if status != "ok":
            return _create_error_response(
                status=status,
                candidate_users_df=precomputed.get("candidate_users_df", pd.DataFrame()),
                corr_df=precomputed.get("corr_df", pd.DataFrame()),
            )

        # Extract precomputed data
        movies_watched = precomputed["movies_watched"]
        candidate_users_df = precomputed["candidate_users_df"].copy()
        corr_df = precomputed["corr_df"].copy()
        top_users_ratings = precomputed["top_users_ratings"].copy()

        # Validate data availability
        if (
            len(movies_watched) == 0
            or candidate_users_df.empty
            or corr_df.empty
            or top_users_ratings.empty
        ):
            return _create_error_response(
                status="not_enough_data",
                candidate_users_df=candidate_users_df,
                corr_df=corr_df,
            )

        # Step 1: Apply overlap filter
        good_overlap_users = _apply_overlap_filter(
            candidate_users_df, movies_watched, min_overlap_ratio_pct
        )
        if good_overlap_users.empty:
            return _create_error_response(
                status="no_candidates_after_overlap",
                candidate_users_df=candidate_users_df,
                corr_df=corr_df,
                debug_info={
                    "movies_watched": len(movies_watched),
                    "candidate_users": len(candidate_users_df),
                    "after_overlap_users": 0,
                    "after_corr_users": 0,
                    "used_neighbors": 0,
                },
            )

        # Step 2: Apply correlation filter
        corr_filtered = _apply_correlation_filter(corr_df, good_overlap_users, corr_threshold)
        if corr_filtered.empty:
            return _create_error_response(
                status="no_similar_users",
                candidate_users_df=candidate_users_df,
                corr_df=corr_df,
                corr_filtered=corr_filtered,
                debug_info={
                    "movies_watched": len(movies_watched),
                    "candidate_users": len(candidate_users_df),
                    "after_overlap_users": len(good_overlap_users),
                    "after_corr_users": 0,
                    "used_neighbors": 0,
                },
            )

        # Step 3: Apply neighbor limit
        corr_filtered = _apply_neighbor_limit(corr_filtered, max_neighbors)
        if corr_filtered.empty:
            return _create_error_response(
                status="no_similar_users_after_limit",
                candidate_users_df=candidate_users_df,
                corr_df=corr_df,
                corr_filtered=corr_filtered,
                debug_info={
                    "movies_watched": len(movies_watched),
                    "candidate_users": len(candidate_users_df),
                    "after_overlap_users": len(good_overlap_users),
                    "after_corr_users": 0,
                    "used_neighbors": 0,
                },
            )

        # Step 4: Get neighbor movie ratings
        neighbor_ratings = top_users_ratings.merge(
            corr_filtered[["userId", "corr"]], on="userId", how="inner"
        )
        if neighbor_ratings.empty:
            return _create_error_response(
                status="no_neighbor_ratings_after_filter",
                candidate_users_df=candidate_users_df,
                corr_df=corr_df,
                corr_filtered=corr_filtered,
                neighbor_ratings=neighbor_ratings,
                debug_info={
                    "movies_watched": len(movies_watched),
                    "candidate_users": len(candidate_users_df),
                    "after_overlap_users": len(good_overlap_users),
                    "after_corr_users": len(corr_filtered["userId"].unique()),
                    "used_neighbors": 0,
                },
            )

        # Step 5: Calculate weighted ratings
        recommendation_df = _calculate_weighted_ratings(
            neighbor_ratings, chosen_user, rating, weighted_score_threshold, top_n
        )
        if recommendation_df.empty:
            return _create_error_response(
                status="no_corr_column_after_merge",
                candidate_users_df=candidate_users_df,
                corr_df=corr_df,
                corr_filtered=corr_filtered,
                neighbor_ratings=neighbor_ratings,
            )

        # Step 6: Build final recommendations with movie titles
        out_df = _build_final_recommendations(recommendation_df, movie)

        # Prepare debug info
        debug_info = {
            "movies_watched": len(movies_watched),
            "candidate_users": len(candidate_users_df),
            "after_overlap_users": len(good_overlap_users),
            "after_corr_users": len(corr_filtered["userId"].unique()),
            "used_neighbors": len(corr_filtered["userId"].unique()),
        }

        return {
            "status": "ok",
            "recommendations": out_df,
            "debug_info": debug_info,
            "dbg_candidate_users_df": candidate_users_df,
            "dbg_corr_df": corr_df,
            "dbg_corr_filtered": corr_filtered,
            "dbg_neighbor_ratings": neighbor_ratings,
        }
    except KeyError as e:
        logger.exception(f"Missing key in precomputed data: {e}")
        return _create_error_response(
            status="error",
            candidate_users_df=pd.DataFrame(),
            corr_df=pd.DataFrame(),
            error_message=f"Missing key in precomputed data: {e}",
        )
    except Exception as e:
        logger.exception(f"Error in finalize_user_based_from_cache: {e}")
        return _create_error_response(
            status="error",
            candidate_users_df=pd.DataFrame(),
            corr_df=pd.DataFrame(),
            error_message=str(e),
        )
