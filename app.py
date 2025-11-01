# -------------------------------------------------
# HYBRID RECOMMENDER STREAMLIT APP
# -------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import base64

# -------------------------------------------------
# PATH / GLOBAL SETTINGS
# -------------------------------------------------

# Page config (call early)
st.set_page_config(
    page_title="Hybrid Recommender Case Study",
    layout="wide"
)

# Project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Pickle file
PICKLE_PATH = os.path.join(BASE_DIR, "data/prepare_data_demo.pkl")

# Logo file (change name if needed)
LOGO_PATH = os.path.join(BASE_DIR, "datahub_logo.jpeg")
# example: LOGO_PATH = os.path.join(BASE_DIR, "datahub logo.jpeg")


def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# Encode logo (silently fallback if logo doesn't exist)
logo_b64 = None
if os.path.exists(LOGO_PATH):
    logo_b64 = img_to_base64(LOGO_PATH)


# -------------------------------------------------
# DATAHUB HEADER (top banner)
# -------------------------------------------------
def render_header():
    # Make logo optional so it doesn't throw errors
    if logo_b64:
        logo_html = (
            f"<img src='data:image/png;base64,{logo_b64}' "
            "style=\"height:40px; border-radius:.5rem; "
            "background:rgba(0,0,0,.15); padding:4px;"
            "box-shadow:0 10px 20px rgba(0,0,0,0.4);\"/>"
        )
    else:
        logo_html = (
            "<div style=\"height:40px; width:40px; border-radius:.5rem; "
            "background:rgba(0,0,0,.15); display:flex; align-items:center; "
            "justify-content:center; font-size:.6rem; font-weight:600; "
            "box-shadow:0 10px 20px rgba(0,0,0,0.4); color:white;\">DH</div>"
        )

    datahub_banner_html = (
        "<div style=\""
        "background: linear-gradient(90deg, rgba(37,99,235,1) 0%, rgba(16,185,129,1) 100%);"
        "padding: .75rem 1rem;"
        "border-radius: .5rem;"
        "color: white;"
        "font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Inter', Roboto, 'Segoe UI', sans-serif;"
        "font-size: .9rem;"
        "font-weight: 500;"
        "display: flex;"
        "align-items: center;"
        "gap: .75rem;"
        "margin-bottom: 1rem;"
        "border: 1px solid rgba(255,255,255,0.3);"
        "box-shadow: 0 20px 40px -10px rgba(0,0,0,0.4);"
        "\">"

        # logo
        + logo_html +

        # badge
        "<div style=\""
        "background: rgba(255,255,255,0.15);"
        "border-radius: .5rem;"
        "padding: .5rem .75rem;"
        "font-size: .8rem;"
        "font-weight: 600;"
        "line-height: 1;"
        "display: flex;"
        "align-items: center;"
        "\">"
        "DataHub"
        "</div>"

        # description
        "<div style=\"flex:1; font-size:.9rem; font-weight:500;\">"
        "In the real world, hybrid approach combines these three ideas: "
        "community taste (user-based), product similarity (item-based), "
        "content similarity (content-based)."
        "</div>"

        "</div>"
    )

    st.markdown(datahub_banner_html, unsafe_allow_html=True)


render_header()


# -------------------------------------------------
# GENERAL CSS
# -------------------------------------------------
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #ffffff;
        border-radius: 0.75rem;
        padding: 0.9rem 1rem;
        border: 1px solid rgba(0,0,0,0.07);
        box-shadow: 0 12px 24px -12px rgba(0,0,0,0.20);
        margin-bottom: 1rem;
        min-height: 5.5rem;
    }
    .metric-title {
        font-weight: 500;
        font-size: .8rem;
        color: #6b7280;
        margin-bottom: .25rem;
    }
    .metric-value {
        font-size: 1.15rem;
        font-weight: 600;
        color: #111827;
        line-height: 1.4rem;
        word-break: break-word;
    }

    .header-badge-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: .5rem 1rem;
        margin-bottom: .75rem;
    }
    .header-badge {
        background: #fdf8c7;
        color: #111827;
        display: inline-block;
        padding: .4rem .6rem;
        border-radius: .25rem;
        font-size: .8rem;
        font-weight: 500;
        border: 1px solid #e2e0a8;
    }

    table.var-table {
        border-collapse: collapse;
        width: 100%;
        margin-bottom: 2rem;
        font-size: 0.9rem;
        background: #ffffff;
        color: #111827;
    }
    table.var-table th {
        text-align: left;
        background: #fdf8c7;
        color: #111827;
        font-weight: 600;
        padding: .5rem .6rem;
        border: 1px solid #d4d4d4;
        white-space: nowrap;
    }
    table.var-table td {
        padding: .5rem .6rem;
        border: 1px solid #d4d4d4;
        vertical-align: top;
        background: #ffffff;
        color: #111827;
    }

    table.stage-table {
        border-collapse: collapse;
        width: 100%;
        margin-top: 1rem;
        font-size: .85rem;
        background: #ffffff;
        color: #111827;
    }
    table.stage-table th {
        text-align: left;
        background: #eef2ff;
        color: #111827;
        font-weight: 600;
        padding: .5rem .6rem;
        border: 1px solid #c7c9df;
        white-space: nowrap;
    }
    table.stage-table td {
        padding: .5rem .6rem;
        border: 1px solid #c7c9df;
        vertical-align: top;
        background: #ffffff;
        color: #111827;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_data(pickle_path: str):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    movie = data["movie"]
    rating = data["rating"]
    df_full = data["df_full"]
    common_movies = data["common_movies"]
    user_movie_df = data["user_movie_df"]
    cosine_sim_genre = data["cosine_sim_genre"]

    # Get user ID list from matrix
    all_user_ids = user_movie_df.index.tolist()

    return movie, rating, df_full, common_movies, user_movie_df, all_user_ids, cosine_sim_genre


movie, rating, df_full, common_movies, user_movie_df, all_user_ids, cosine_sim_genre = load_data(PICKLE_PATH)


# -------------------------------------------------
# USER-BASED PRE-COMPUTATION
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def precompute_for_user_userbased(chosen_user: int):
    # Does user exist?
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

    candidate_users_df = (
        user_movie_count_series
        .reset_index()
        .rename(columns={"index": "userId", 0: "movie_count"})
    )
    candidate_users_df.columns = ["userId", "movie_count"]

    candidate_users_df = candidate_users_df[candidate_users_df["userId"] != chosen_user].copy()

    # Calculate correlation
    base_vector = user_movie_df.loc[chosen_user]
    corr_series = movies_watched_df.T.corrwith(base_vector).dropna()

    corr_df = (
        corr_series
        .reset_index()
        .rename(columns={"index": "userId", 0: "corr"})
    )
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
        rating[["userId", "movieId", "rating"]],
        on="userId",
        how="inner"
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


def finalize_user_based_from_cache(
    precomputed,
    min_overlap_ratio_pct: float,
    corr_threshold: float,
    max_neighbors: int,
    weighted_score_threshold: float,
    top_n: int,
    chosen_user: int
):
    status = precomputed["status"]
    if status != "ok":
        return {
            "status": status,
            "recommendations": pd.DataFrame(),
            "debug_info": {},
            "dbg_candidate_users_df": precomputed.get("candidate_users_df", pd.DataFrame()),
            "dbg_corr_df": precomputed.get("corr_df", pd.DataFrame()),
            "dbg_corr_filtered": pd.DataFrame(),
            "dbg_neighbor_ratings": pd.DataFrame(),
        }

    movies_watched = precomputed["movies_watched"]
    candidate_users_df = precomputed["candidate_users_df"].copy()
    corr_df = precomputed["corr_df"].copy()
    top_users_ratings = precomputed["top_users_ratings"].copy()

    if len(movies_watched) == 0 or candidate_users_df.empty or corr_df.empty or top_users_ratings.empty:
        return {
            "status": "not_enough_data",
            "recommendations": pd.DataFrame(),
            "debug_info": {},
            "dbg_candidate_users_df": candidate_users_df,
            "dbg_corr_df": corr_df,
            "dbg_corr_filtered": pd.DataFrame(),
            "dbg_neighbor_ratings": pd.DataFrame(),
        }

    # 1. overlap filter
    threshold_common = len(movies_watched) * (min_overlap_ratio_pct / 100.0)

    good_overlap_users = candidate_users_df[
        candidate_users_df["movie_count"] >= threshold_common
    ]["userId"]

    if good_overlap_users.empty:
        return {
            "status": "no_candidates_after_overlap",
            "recommendations": pd.DataFrame(),
            "debug_info": {
                "movies_watched": len(movies_watched),
                "candidate_users": len(candidate_users_df),
                "after_overlap_users": 0,
                "after_corr_users": 0,
                "used_neighbors": 0
            },
            "dbg_candidate_users_df": candidate_users_df,
            "dbg_corr_df": corr_df,
            "dbg_corr_filtered": pd.DataFrame(),
            "dbg_neighbor_ratings": pd.DataFrame(),
        }

    # 2. correlation filter
    corr_filtered = corr_df[
        (corr_df["userId"].isin(good_overlap_users)) &
        (corr_df["corr"] >= corr_threshold)
    ].copy()

    if corr_filtered.empty:
        return {
            "status": "no_similar_users",
            "recommendations": pd.DataFrame(),
            "debug_info": {
                "movies_watched": len(movies_watched),
                "candidate_users": len(candidate_users_df),
                "after_overlap_users": len(good_overlap_users),
                "after_corr_users": 0,
                "used_neighbors": 0
            },
            "dbg_candidate_users_df": candidate_users_df,
            "dbg_corr_df": corr_df,
            "dbg_corr_filtered": corr_filtered,
            "dbg_neighbor_ratings": pd.DataFrame(),
        }

    # 3. max_neighbors limit
    corr_filtered = (
        corr_filtered
        .sort_values("corr", ascending=False)
        .head(max_neighbors)
        .copy()
    )

    if corr_filtered.empty:
        return {
            "status": "no_similar_users_after_limit",
            "recommendations": pd.DataFrame(),
            "debug_info": {
                "movies_watched": len(movies_watched),
                "candidate_users": len(candidate_users_df),
                "after_overlap_users": len(good_overlap_users),
                "after_corr_users": 0,
                "used_neighbors": 0
            },
            "dbg_candidate_users_df": candidate_users_df,
            "dbg_corr_df": corr_df,
            "dbg_corr_filtered": corr_filtered,
            "dbg_neighbor_ratings": pd.DataFrame(),
        }

    # 4. get neighbor movie ratings
    neighbor_ratings = top_users_ratings.merge(
        corr_filtered[["userId", "corr"]],
        on="userId",
        how="inner"
    )

    if neighbor_ratings.empty:
        return {
            "status": "no_neighbor_ratings_after_filter",
            "recommendations": pd.DataFrame(),
            "debug_info": {
                "movies_watched": len(movies_watched),
                "candidate_users": len(candidate_users_df),
                "after_overlap_users": len(good_overlap_users),
                "after_corr_users": len(corr_filtered["userId"].unique()),
                "used_neighbors": 0
            },
            "dbg_candidate_users_df": candidate_users_df,
            "dbg_corr_df": corr_df,
            "dbg_corr_filtered": corr_filtered,
            "dbg_neighbor_ratings": neighbor_ratings,
        }

    # 5. weighting = rating * corr
    possible_corr_cols = [c for c in neighbor_ratings.columns if "corr" in c]
    if not possible_corr_cols:
        return {
            "status": "no_corr_column_after_merge",
            "recommendations": pd.DataFrame(),
            "debug_info": {},
            "dbg_candidate_users_df": candidate_users_df,
            "dbg_corr_df": corr_df,
            "dbg_corr_filtered": corr_filtered,
            "dbg_neighbor_ratings": neighbor_ratings,
        }

    corr_col = possible_corr_cols[0]
    neighbor_ratings["weighted_rating"] = (
        neighbor_ratings["rating"] * neighbor_ratings[corr_col]
    )

    recommendation_df = (
        neighbor_ratings
        .groupby("movieId")
        .agg(weighted_rating=("weighted_rating", "mean"))
        .reset_index()
    )

    # Remove movies user has already watched
    seen_ids = rating.loc[rating["userId"] == chosen_user, "movieId"].unique().tolist()
    recommendation_df = recommendation_df[
        ~recommendation_df["movieId"].isin(seen_ids)
    ]

    # apply weighted score threshold
    recommendation_df = recommendation_df[
        recommendation_df["weighted_rating"] >= weighted_score_threshold
    ]

    # sort + top_n
    recommendation_df = (
        recommendation_df
        .sort_values("weighted_rating", ascending=False)
        .head(top_n)
        .copy()
    )

    # add movie names
    recommendation_df = recommendation_df.merge(
        movie[["movieId", "title"]],
        on="movieId",
        how="left"
    )

    out_df = recommendation_df[["title", "weighted_rating"]].rename(
        columns={"title": "Film", "weighted_rating": "Score"}
    )

    debug_info = {
        "movies_watched": len(movies_watched),
        "candidate_users": len(candidate_users_df),
        "after_overlap_users": len(good_overlap_users),
        "after_corr_users": len(corr_filtered["userId"].unique()),
        "used_neighbors": len(corr_filtered["userId"].unique())
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


# -------------------------------------------------
# ITEM-BASED
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def precompute_for_user_itembased(chosen_user: int):
    # Movies user gave 5 stars to
    user_5 = rating[
        (rating["userId"] == chosen_user) &
        (rating["rating"] == 5.0)
    ]

    if user_5.empty:
        return {
            "status": "no_five_star",
            "reference_movie": None,
            "similarity_df": pd.DataFrame()
        }

    # most recently given 5★
    if "timestamp" in user_5.columns:
        last_fav = user_5.sort_values("timestamp", ascending=False).iloc[0]
    else:
        last_fav = user_5.iloc[0]

    ref_movie_id = last_fav["movieId"]
    ref_title_arr = movie.loc[movie["movieId"] == ref_movie_id, "title"].values
    if len(ref_title_arr) == 0:
        return {
            "status": "no_title",
            "reference_movie": None,
            "similarity_df": pd.DataFrame()
        }

    ref_title = ref_title_arr[0]

    # item-based correlation: between movies
    if ref_title not in user_movie_df.columns:
        return {
            "status": "not_in_matrix",
            "reference_movie": ref_title,
            "similarity_df": pd.DataFrame()
        }

    ref_vector = user_movie_df[ref_title]
    sims = user_movie_df.corrwith(ref_vector).dropna()  # movie-movie similarity
    sims = sims[sims.index != ref_title]  # remove itself

    similarity_df = (
        sims.sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "Similar Film", 0: "Similarity"})
    )

    return {
        "status": "ok",
        "reference_movie": ref_title,
        "similarity_df": similarity_df
    }


def finalize_item_based_from_cache(precomputed_item, top_n_item: int):
    status = precomputed_item["status"]
    if status != "ok":
        return status, None, pd.DataFrame()

    ref_movie = precomputed_item["reference_movie"]
    sim_df_all = precomputed_item["similarity_df"]

    sim_df_head = sim_df_all.head(top_n_item).copy()

    return "ok", ref_movie, sim_df_head


# -------------------------------------------------
# CONTENT-BASED
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def content_based_recommender_cached(movie_title: str, top_n: int):
    # Does film exist?
    if movie_title not in movie['title'].values:
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

    # Get Cosine similarity scores
    sim_scores = list(enumerate(cosine_sim_genre[movie_idx]))

    # Sort by score, exclude itself
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1: top_n + 1]

    # Related movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Result DF
    result_df = movie.iloc[movie_indices][['title', 'genres']].copy()
    result_df['Similarity Score'] = [round(i[1], 3) for i in sim_scores]
    result_df = result_df.rename(columns={'title': 'Film', 'genres': 'Genres'})

    return {
        "status": "ok",
        "reference_movie": movie_title,
        "reference_genres": ref_genres,
        "recommendations": result_df
    }


# -------------------------------------------------
# HELPER FUNCTION
# -------------------------------------------------
def get_matrix_shape(df):
    return df.shape


# -------------------------------------------------
# TABS
# -------------------------------------------------
tab_problem, tab_dataset, tab_tasks = st.tabs([
    "1. Business Problem",
    "2. Dataset Story",
    "3. Project Tasks"
])

# -------------------------------------------------
# TAB 1
# -------------------------------------------------
with tab_problem:
    st.title("Case Study: Hybrid Recommender Project")
    st.header("Business Problem")

    st.info(
        "💡 Scenario: 'Recommend films for a given user ID using user-based and item-based recommendation methods.' "
        "This problem was defined using MovieLens data."
    )

    st.write(
        """
        Objectives:
        1. Recommend films liked by users similar to the target user (User-Based).
        2. Recommend films similar to the film the user last rated 5⭐ (Item-Based).
        3. BONUS: Find similar films based on genre similarity of a selected film (Content-Based).
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🧑‍🤝‍🧑 User-Based
        - Find users with similar taste to me
        - Bring films they liked but I haven't watched
        - Sort by weighted score
        """)

    with col2:
        st.markdown("""
        ### 🎬 Item-Based
        - Find the film I last rated 5⭐
        - Calculate similar films using correlation
        - Sort the most similar ones
        """)

    with col3:
        st.markdown("""
        ### 🏷️ Content-Based
        - Select a film
        - Check genre information
        - Bring films in the same tone using cosine similarity
        """)

    st.success(
        "These three approaches are usually combined into one package in real life. This is what we call a hybrid system."
    )

# -------------------------------------------------
# TAB 2
# -------------------------------------------------
with tab_dataset:
    st.title("Dataset Story")

    total_users_full = df_full["userId"].nunique()
    total_movies_full = df_full["movieId"].nunique()
    total_ratings_full = df_full.shape[0]

    total_users_common = common_movies["userId"].nunique()
    total_movies_common = common_movies["movieId"].nunique()
    total_ratings_common = common_movies.shape[0]

    um_users, um_movies = get_matrix_shape(user_movie_df)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>Total Users (raw)</div>"
            f"<div class='metric-value'>{total_users_full:,}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>Total Movies (raw)</div>"
            f"<div class='metric-value'>{total_movies_full:,}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>Total Ratings (raw)</div>"
            f"<div class='metric-value'>{total_ratings_full:,}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.write(
        "This dataset is provided by MovieLens. It contains tens of thousands of films and millions of ratings. "
        "Each user has rated at least 20 films; time range is 1995-2015."
    )

    st.subheader("Variables")

    st.markdown("**movie.csv**")
    st.markdown(
        """
        <div class='header-badge-wrap'>
            <div class='header-badge'>3 Variables</div>
            <div class='header-badge'>~27K Observations</div>
            <div class='header-badge'>Movie information</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <table class="var-table">
        <tr><th>movieId</th><td>Unique movie number.</td></tr>
        <tr><th>title</th><td>Movie title.</td></tr>
        <tr><th>genres</th><td>Genre information (Action|Comedy|Drama ...)</td></tr>
        </table>
        """,
        unsafe_allow_html=True
    )

    st.markdown("**rating.csv**")
    st.markdown(
        """
        <div class='header-badge-wrap'>
            <div class='header-badge'>4 Variables</div>
            <div class='header-badge'>~20M Observations</div>
            <div class='header-badge'>User ratings</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <table class="var-table">
        <tr><th>userId</th><td>User ID (unique)</td></tr>
        <tr><th>movieId</th><td>Movie ID</td></tr>
        <tr><th>rating</th><td>Given rating</td></tr>
        <tr><th>timestamp</th><td>Time when rated</td></tr>
        </table>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Data Preparation Steps")

    st.markdown(
        """
        MovieLens data is very large as-is. Therefore, we performed layered reduction:

        1. Full universe (original, ~20M ratings / ~138K users / ~27K movies)  
        2. Popular movies filter (remove movies with very few ratings)  
        3. Snapshot: subset universe keeping a specific user's network and most meaningful interactions (~2.3K users)

        The table below compares the two worlds:
        """
    )

    st.markdown(
        f"""
        <table class="stage-table">
        <tr>
            <th>Stage</th>
            <th>Description</th>
            <th>Rating Rows</th>
            <th>Movie Count</th>
            <th>User Count</th>
        </tr>

        <tr>
            <td>Raw Data (MovieLens Original)</td>
            <td>movie.merge(rating)<br/>Ratings between 1995-2015<br/>Each user rated ≥20 movies</td>
            <td>{20_000_263:,}+</td>
            <td>~27,000</td>
            <td>~138,000</td>
        </tr>

        <tr>
            <td>Filtered by Popular Movies</td>
            <td>Remove movies with less than 1000 ratings</td>
            <td>{17_766_015:,}</td>
            <td>~3,000 active movies</td>
            <td>~138,000</td>
        </tr>

        <tr>
            <td>Demo Full (snapshot)</td>
            <td>Interactions around example user</td>
            <td>{1_793_782:,}</td>
            <td>{6_818:,}</td>
            <td>{2_326:,}</td>
        </tr>

        <tr>
            <td>Demo Popular Movies (common_movies)</td>
            <td>Removed movies with less than 1000 ratings</td>
            <td>{1_572_589:,}</td>
            <td>{1_986:,}</td>
            <td>{2_326:,}</td>
        </tr>

        <tr>
            <td>Demo User-Movie Matrix (user_movie_df)</td>
            <td>pivot: user x movie (rating matrix)</td>
            <td>{2_326:,} rows x {1_982:,} columns</td>
            <td>{1_982:,} active movies</td>
            <td>{2_326:,} active users</td>
        </tr>
        </table>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "Removing movies with few ratings both speeds up calculations and makes similarity measurements more reliable."
    )

# -------------------------------------------------
# TAB 3 (LIVE DEMO)
# -------------------------------------------------
with tab_tasks:
    st.title("Project Tasks and Live Recommendation Engine")

    st.subheader("Approaches")
    st.markdown("""
    **User-Based**  
    Find users similar to me, bring films they liked but I haven't watched.  
    Correlation (corr) = taste similarity. Weighted Score = corr * average rating.

    **Item-Based**  
    Base on the film the user last rated 5⭐.  
    Find other films most similar to that film (movie-movie correlation).

    **Content-Based**  
    Only look at content. Calculate cosine similarity between genre vectors.  
    'What other films are closest to this film's genre DNA?'
    """)

    st.info(
        """
        Hybrid system: if the same film is recommended by multiple signals
        (user-based + item-based + content-based),
        we consider that film more reliable.
        """
    )

    # LEFT and RIGHT columns
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.markdown("### Parameters / Control Panel")

        chosen_user = st.number_input(
            "Target User ID",
            min_value=1,
            value=108170,
            step=1,
            help="User ID to analyze"
        )

        rec_type = st.radio(
            "Which method should we run?",
            [
                "User-Based (Users Like Me)",
                "Item-Based (Similar to This Film)",
                "Content-Based (Similar Genre Films)",
                "Hybrid (Combine All)"
            ],
            key="rec_type_radio"
        )

        st.markdown("---")

        # Parameter blocks
        if rec_type.startswith("User-Based"):
            st.markdown("#### 🧑‍🤝‍🧑 User-Based Parameters")

            min_overlap_ratio_pct = st.slider(
                "Common viewing percentage (%)",
                min_value=0,
                max_value=100,
                value=60,
                step=5,
                help="Consider users who watched at least 60% of the films I watched as 'similar'."
            )

            corr_threshold = st.slider(
                "Correlation threshold (taste similarity)",
                min_value=0.0,
                max_value=1.0,
                value=0.65,
                step=0.05,
                help="0.65 and above: really similar to me."
            )

            max_neighbors = st.slider(
                "Maximum neighbor count",
                min_value=1,
                max_value=200,
                value=7,
                step=1,
                help="How many similar users should we use?"
            )

            weighted_score_threshold = st.slider(
                "Weighted score threshold",
                min_value=0.0,
                max_value=5.0,
                value=3.5,
                step=0.1,
                help="Recommend if (corr * rating) average is above 3.5."
            )

            top_n_user_based = st.slider(
                "How many films to recommend? (Top-N)",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                help="How many films to list?"
            )

            # default others
            top_n_item_based = 5
            top_n_content = 5
            hybrid_top_n = 5
            selected_movie_title = None

        elif rec_type.startswith("Item-Based"):
            st.markdown("#### 🎬 Item-Based Parameters")

            top_n_item_based = st.slider(
                "How many similar films to show?",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                help="Show the top N most similar films for the film the user last rated 5⭐."
            )

            # default others
            min_overlap_ratio_pct = 60
            corr_threshold = 0.65
            max_neighbors = 7
            weighted_score_threshold = 3.5
            top_n_user_based = 5
            top_n_content = 5
            hybrid_top_n = 5
            selected_movie_title = None

        elif rec_type.startswith("Content-Based"):
            st.markdown("#### 🏷️ Content-Based Parameters")

            # quick search box
            search_term = st.text_input(
                "Search / type film (for autocomplete):",
                value="",
                help="Type first few letters. The box below will filter accordingly."
            )

            # filtered film list
            if search_term.strip():
                filtered_titles = sorted(
                    [t for t in movie['title'].tolist() if search_term.lower() in t.lower()]
                )
            else:
                filtered_titles = sorted(movie['title'].tolist())

            selected_movie_title = st.selectbox(
                "Select Reference Film",
                options=filtered_titles,
                help="We will find films most similar to this film by genre."
            )

            top_n_content = st.slider(
                "How many similar films to show?",
                min_value=1,
                max_value=20,
                value=5,
                step=1
            )

            # default others
            min_overlap_ratio_pct = 60
            corr_threshold = 0.65
            max_neighbors = 7
            weighted_score_threshold = 3.5
            top_n_user_based = 5
            top_n_item_based = 5
            hybrid_top_n = 5

        else:  # Hybrid
            st.markdown("#### 🔀 Hybrid Parameters")

            # User-Based defaults
            min_overlap_ratio_pct = 60
            corr_threshold = 0.65
            max_neighbors = 7
            weighted_score_threshold = 3.5
            top_n_user_based = 5

            # Item-Based defaults
            top_n_item_based = 5

            # Content-Based defaults
            top_n_content = 5
            selected_movie_title = None

            hybrid_top_n = st.slider(
                "How many films should Hybrid show in total?",
                min_value=3,
                max_value=15,
                value=5,
                step=1,
                help="List the top N films from common/strong candidates of all three approaches."
            )

        run_button = st.button("🎬 Calculate Recommendations", type="primary")

    # RIGHT PANEL - RESULT
    with right_col:
        st.markdown("### Solution Output")

        if not run_button:
            st.info("Select parameters and press the button.")
        else:
            # USER-BASED
            # USER-BASED
            if rec_type.startswith("User-Based"):
                with st.spinner("Calculating User-Based..."):
                    pre_u = precompute_for_user_userbased(chosen_user)
                    result_user = finalize_user_based_from_cache(
                        precomputed=pre_u,
                        min_overlap_ratio_pct=min_overlap_ratio_pct,
                        corr_threshold=corr_threshold,
                        max_neighbors=max_neighbors,
                        weighted_score_threshold=weighted_score_threshold,
                        top_n=top_n_user_based,
                        chosen_user=chosen_user
                    )

                status = result_user["status"]
                debug_info = result_user.get("debug_info", {})
                recs_df = result_user["recommendations"]

                # prepare debug dataframes locally (so they're always defined)
                cand_df = result_user.get("dbg_candidate_users_df", pd.DataFrame())
                corr_df_dbg = result_user.get("dbg_corr_df", pd.DataFrame())
                corr_filtered_dbg = result_user.get("dbg_corr_filtered", pd.DataFrame())
                neigh_dbg = result_user.get("dbg_neighbor_ratings", pd.DataFrame())

                # --- status check ---
                if status != "ok":
                    st.warning(f"User-Based recommendations could not be generated. Status: {status}")

                    # even if status is not ok, you can see debug to explain technically
                    with st.expander("🔎 Debug / Intermediate Steps (detailed calculation steps)"):
                        st.write("• candidate_users_df = 'Everyone who watched common films with target user'")
                        st.write("Shape:", cand_df.shape)
                        st.dataframe(cand_df.head(20))

                        st.write("• corr_df = 'Each candidate user's correlation with target user (taste similarity)'")
                        st.write("Shape:", corr_df_dbg.shape)
                        st.dataframe(corr_df_dbg.head(20))

                        st.write("• corr_filtered = 'Both sufficient common film count AND corr above threshold'")
                        st.write("Shape:", corr_filtered_dbg.shape)
                        st.dataframe(corr_filtered_dbg.head(20))

                        st.write("• neighbor_ratings = 'Which films these similar users rated how much'")
                        st.write("Shape:", neigh_dbg.shape)
                        st.dataframe(neigh_dbg.head(20))

                else:
                    # give separate message if no recommendations
                    if recs_df.empty:
                        st.info("No recommendations found matching the parameters.")

                        # even if no recommendations, we can explain 'why not' by showing debug
                        with st.expander("🔎 Debug / Intermediate Steps (detailed calculation steps)"):
                            st.write("• candidate_users_df = 'Everyone who watched common films with target user'")
                            st.write("Shape:", cand_df.shape)
                            st.dataframe(cand_df.head(20))

                            st.write(
                                "• corr_df = 'Each candidate user's correlation with target user (taste similarity)'")
                            st.write("Shape:", corr_df_dbg.shape)
                            st.dataframe(corr_df_dbg.head(20))

                            st.write("• corr_filtered = 'Both sufficient common film count AND corr above threshold'")
                            st.write("Shape:", corr_filtered_dbg.shape)
                            st.dataframe(corr_filtered_dbg.head(20))

                            st.write("• neighbor_ratings = 'Which films these similar users rated how much'")
                            st.write("Shape:", neigh_dbg.shape)
                            st.dataframe(neigh_dbg.head(20))

                    else:
                        # SUCCESS STATUS
                        st.success("User-Based recommendations ready.")

                        # short explanation card
                        st.markdown(
                            """
                            <div class='metric-card'>
                                <div class='metric-title'>💬 Comment</div>
                                <div class='metric-value'>
                                    This list was calculated as follows:<br/>
                                    1) We found users with very similar movie history to this user.<br/>
                                    2) We took films that these similar users rated highly.<br/>
                                    3) We excluded films the user has already watched.<br/>
                                    4) We recommended those with high weighted score (corr × rating average).
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        # metrics (how many neighbors we used, etc.)
                        col_a, col_b, col_c, col_d = st.columns(4)

                        with col_a:
                            st.metric(
                                label="Movies Watched Count",
                                value=debug_info.get("movies_watched", "?"),
                                help="Number of films the target user rated (example: 186)."
                            )

                        with col_b:
                            st.metric(
                                label="Initial Candidate Pool",
                                value=debug_info.get("candidate_users", "?"),
                                help="Number of users sharing at least one common film with this user. Raw initial pool."
                            )

                        with col_c:
                            st.metric(
                                label="Similar Users (Overlap+Corr)",
                                value=debug_info.get("after_corr_users", "?"),
                                help="Number of users who passed both common viewing percentage threshold and correlation threshold. (top_users)"
                            )

                        with col_d:
                            st.metric(
                                label="Neighbors Included in Score",
                                value=debug_info.get("used_neighbors", "?"),
                                help="Number of most similar neighbors used when calculating recommendation score."
                            )

                        # RESULT TABLE (final recommendations)
                        st.markdown("#### 🎬 User-Based Recommendations")
                        st.dataframe(
                            recs_df.reset_index(drop=True),
                            use_container_width=True
                        )

                        # DEBUG EXPANDER (presentation mode 💅)
                        with st.expander("🔎 Debug / Intermediate Steps (detailed calculation steps)"):
                            st.write("STAGE 1 · candidate_users_df")
                            st.caption(
                                "How many of the films watched by target user did other users also watch? movie_count shows this.")
                            st.write("Shape:", cand_df.shape)
                            st.dataframe(cand_df.head(20))

                            st.write("STAGE 2 · corr_df")
                            st.caption(
                                "Correlation between target user and other users (taste similarity). corr = 1 → same taste, 0 → no relationship.")
                            st.write("Shape:", corr_df_dbg.shape)
                            st.dataframe(corr_df_dbg.head(20))

                            st.write("STAGE 3 · corr_filtered (neighbors)")
                            st.caption(
                                "Users who watched sufficient common films AND exceeded corr threshold. So really 'like me'.")
                            st.write("Shape:", corr_filtered_dbg.shape)
                            st.dataframe(corr_filtered_dbg.head(20))

                            st.write("STAGE 4 · neighbor_ratings")
                            st.caption("Which films these neighbors rated how much, and weighted score = corr × rating.")
                            st.write("Shape:", neigh_dbg.shape)
                            st.dataframe(neigh_dbg.head(20))


            # ITEM-BASED
            elif rec_type.startswith("Item-Based"):
                with st.spinner("Calculating Item-Based..."):
                    pre_i = precompute_for_user_itembased(chosen_user)
                    status_i, ref_movie, sim_df = finalize_item_based_from_cache(
                        pre_i,
                        top_n_item_based
                    )

                if status_i != "ok":
                    if status_i == "no_five_star":
                        st.warning("This user has never given 5 stars.")
                    elif status_i == "not_in_matrix":
                        st.warning(
                            f"Reference film not in user-movie matrix: {pre_i['reference_movie']}"
                        )
                    else:
                        st.warning(f"Item-Based recommendations could not be generated. Status: {status_i}")
                else:
                    st.success("Item-Based recommendations ready.")

                    st.markdown(
                        f"<div class='metric-card'>"
                        f"<div class='metric-title'>Reference Film (user's last 5⭐ rating)</div>"
                        f"<div class='metric-value'>{ref_movie}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        "This approach works with the logic 'Those who liked this film also liked these'. "
                        "It measures similarity between films based on users' rating behavior."
                    )

                    st.markdown(f"Total {len(sim_df)} similar films found.")
                    st.dataframe(
                        sim_df.head(top_n_item_based).reset_index(drop=True),
                        use_container_width=True
                    )

            # CONTENT-BASED
            elif rec_type.startswith("Content-Based"):
                with st.spinner("Calculating Content-Based..."):
                    cb_result = content_based_recommender_cached(
                        movie_title=selected_movie_title,
                        top_n=top_n_content
                    )

                status_c = cb_result["status"]

                if status_c != "ok":
                    st.warning(f"Film not found: {selected_movie_title}")
                else:
                    ref_movie_cb = cb_result["reference_movie"]
                    ref_genres_cb = cb_result["reference_genres"]
                    rec_df_cb = cb_result["recommendations"]

                    st.success("Content-Based recommendations ready.")

                    st.markdown(
                        f"<div class='metric-card'>"
                        f"<div class='metric-title'>Reference Film</div>"
                        f"<div class='metric-value'>{ref_movie_cb}</div>"
                        f"<div class='metric-title'>Genres: {ref_genres_cb}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                    with st.expander("How does Content-Based work?"):
                        st.info(
                            f"""
                            This method looks at the content itself, not user behavior.

                            1) We took the genres of the reference film: `{ref_genres_cb}`
                            2) We represent each film as a genre vector
                            3) We ask 'which films are most similar to this film by genre?' using Cosine Similarity
                            4) We bring the {top_n_content} films with highest similarity

                            Plus side: Works even in cold start (new user problem is smaller).
                            Minus side: Can return similar tastes all the time (filter bubble).
                            """
                        )

                    st.markdown(f"Total {len(rec_df_cb)} similar films found.")
                    st.dataframe(
                        rec_df_cb.head(top_n_content).reset_index(drop=True),
                        use_container_width=True
                    )

            # HYBRID
            else:
                with st.spinner("Calculating Hybrid..."):

                    # USER-BASED tarafı
                    pre_u = precompute_for_user_userbased(chosen_user)
                    result_user = finalize_user_based_from_cache(
                        precomputed=pre_u,
                        min_overlap_ratio_pct=min_overlap_ratio_pct,
                        corr_threshold=corr_threshold,
                        max_neighbors=max_neighbors,
                        weighted_score_threshold=weighted_score_threshold,
                        top_n=top_n_user_based,
                        chosen_user=chosen_user
                    )
                    if result_user["status"] == "ok" and not result_user["recommendations"].empty:
                        df_user_part = result_user["recommendations"].copy()
                        df_user_part["Source"] = "User-Based"
                        df_user_part = df_user_part.rename(
                            columns={"Film": "Film_Name", "Score": "Model_Score"}
                        )
                    else:
                        df_user_part = pd.DataFrame(columns=["Film_Name", "Model_Score", "Source"])

                    # ITEM-BASED side
                    pre_i = precompute_for_user_itembased(chosen_user)
                    status_i, ref_movie_i, sim_df_i = finalize_item_based_from_cache(
                        pre_i,
                        top_n_item_based
                    )
                    if status_i == "ok" and sim_df_i is not None and not sim_df_i.empty:
                        df_item_part = sim_df_i.copy()
                        df_item_part["Source"] = "Item-Based"
                        df_item_part = df_item_part.rename(
                            columns={"Similar Film": "Film_Name", "Similarity": "Model_Score"}
                        )
                    else:
                        df_item_part = pd.DataFrame(columns=["Film_Name", "Model_Score", "Source"])

                    # CONTENT-BASED side
                    if status_i == "ok" and ref_movie_i is not None:
                        cb_result = content_based_recommender_cached(
                            movie_title=ref_movie_i,
                            top_n=top_n_content
                        )
                        if cb_result["status"] == "ok" and not cb_result["recommendations"].empty:
                            df_cb_part = cb_result["recommendations"].copy()
                            df_cb_part["Source"] = "Content-Based"
                            df_cb_part = df_cb_part.rename(
                                columns={"Film": "Film_Name", "Similarity Score": "Model_Score"}
                            )
                            df_cb_part = df_cb_part[["Film_Name", "Model_Score", "Source"]]
                        else:
                            df_cb_part = pd.DataFrame(columns=["Film_Name", "Model_Score", "Source"])
                    else:
                        df_cb_part = pd.DataFrame(columns=["Film_Name", "Model_Score", "Source"])

                    combined_all = pd.concat(
                        [df_user_part, df_item_part, df_cb_part],
                        ignore_index=True
                    )

                if combined_all.empty:
                    st.warning("Hybrid system could not generate recommendations (insufficient signals).")
                else:
                    st.markdown(
                        f"<div class='metric-card'>"
                        f"<div class='metric-title'>Hybrid Logic</div>"
                        f"<div class='metric-value'>"
                        f"If a film is recommended by multiple models (User / Item / Content), "
                        f"we consider that film more reliable. "
                        f"'Source_Count' shows this."
                        f"</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                    hybrid_summary = (
                        combined_all
                        .groupby("Film_Name")
                        .agg(
                            Source_Count=("Source", "nunique"),
                            Average_Score=("Model_Score", "mean")
                        )
                        .reset_index()
                    )

                    # confidence metric
                    hybrid_summary["Hybrid_Confidence"] = (
                        hybrid_summary["Source_Count"] * hybrid_summary["Average_Score"]
                    )

                    hybrid_summary = hybrid_summary.sort_values(
                        by=["Hybrid_Confidence", "Source_Count", "Average_Score"],
                        ascending=False
                    ).head(hybrid_top_n)

                    st.success("Hybrid (Common Candidates from All Models)")

                    st.dataframe(
                        hybrid_summary.reset_index(drop=True),
                        use_container_width=True
                    )

                    # Show sub-sources
                    with st.expander("User-Based details"):
                        if df_user_part.empty:
                            st.write("No User-Based results.")
                        else:
                            st.dataframe(df_user_part.reset_index(drop=True),
                                         use_container_width=True)

                    with st.expander("Item-Based details"):
                        if df_item_part.empty:
                            st.write("No Item-Based results.")
                        else:
                            st.dataframe(df_item_part.reset_index(drop=True),
                                         use_container_width=True)

                    with st.expander("Content-Based details"):
                        if df_cb_part.empty:
                            st.write("No Content-Based results.")
                        else:
                            st.dataframe(df_cb_part.reset_index(drop=True),
                                         use_container_width=True)

                    # also write reference film
                    if status_i == "ok" and ref_movie_i is not None:
                        st.info(
                            f"Film the user last rated 5⭐: {ref_movie_i} "
                            f"→ Item-Based looked for similarity around this, "
                            f"Content-Based also looked at this film's genre DNA."
                        )
                    else:
                        st.info(
                            "Reference film that user rated 5⭐ could not be found. "
                            "Therefore, Content-Based signal may be missing."
                        )


# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6b7280; font-size: 0.85rem;'>
        <p>DataHub · Hybrid Recommender System</p>
        <p>You can see how the recommendation logic changes as you play with the parameters.</p>
    </div>
    """,
    unsafe_allow_html=True
)
