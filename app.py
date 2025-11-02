from pathlib import Path
from typing import Tuple

import pandas as pd
import streamlit as st

from error_handling import DataLoadError, handle_errors, handle_streamlit_exception
from logging_config import get_logger

logger = get_logger(__name__)

try:
    import config
    from utils import ValidationError
except ImportError:
    logger.warning("utils.py or config.py not found, using fallback configuration")
    import config

    ValidationError = Exception

from data_loader import load_data
from recommenders import (
    content_based_recommender_cached,
    finalize_item_based_from_cache,
    finalize_user_based_from_cache,
    precompute_for_user_itembased,
    precompute_for_user_userbased,
)
from ui import img_to_base64, render_header, render_styles

st.set_page_config(page_title="Hybrid Recommender Case Study", layout="wide")

PICKLE_PATH = config.PICKLE_PATH
LOGO_PATH = config.LOGO_PATH

logo_b64 = None
if LOGO_PATH.exists():
    logo_b64 = img_to_base64(LOGO_PATH)
    if logo_b64:
        logger.info(f"Logo loaded successfully from: {LOGO_PATH}")
    else:
        logger.warning(f"Logo file not found: {LOGO_PATH}, using fallback")

render_header(logo_b64)
render_styles()

# JavaScript for tab persistence after Streamlit rerun
st.markdown(
    """
<script>
(function() {
    'use strict';
    
    const win = window.self !== window.top ? window.top : window.self;
    const doc = win.document;
    
    let isTabMaintaining = false;
    let tabCheckAttempts = 0;
    const maxAttempts = 50;
    let lastCheckTime = 0;
    const checkInterval = 25;
    
    function forceSelectProjectTasksTab() {
        let tabButtons = doc.querySelectorAll('button[data-baseweb="tab"]');
        if (tabButtons.length === 0) {
            tabButtons = doc.querySelectorAll('[role="tab"]');
        }
        if (tabButtons.length === 0) {
            tabButtons = doc.querySelectorAll('button[class*="tab"]');
        }
        
        if (tabButtons.length >= 3) {
            const thirdTab = tabButtons[2];
            const isSelected = thirdTab.getAttribute('aria-selected') === 'true' || 
                              thirdTab.classList.contains('stTabs-1a2j6c5') ||
                              thirdTab.getAttribute('aria-current') === 'true';
            
            if (!isSelected) {
                thirdTab.focus();
                thirdTab.click();
                const clickEvent = new MouseEvent('click', {
                    bubbles: true,
                    cancelable: true,
                    view: win
                });
                thirdTab.dispatchEvent(clickEvent);
                tabButtons.forEach((btn, idx) => {
                    if (idx === 2) {
                        btn.setAttribute('aria-selected', 'true');
                        btn.classList.add('stTabs-1a2j6c5');
                    } else {
                        btn.setAttribute('aria-selected', 'false');
                    }
                });
                return true;
            }
            return true;
        }
        return false;
    }
    
    function checkAndSelectTab() {
        const now = Date.now();
        if (now - lastCheckTime < checkInterval) {
            return;
        }
        lastCheckTime = now;
        
        if (win.sessionStorage.getItem('stayOnProjectTasks') === 'true') {
            if (!isTabMaintaining) {
                isTabMaintaining = true;
                tabCheckAttempts = 0;
            }
            
            const success = forceSelectProjectTasksTab();
            
            if (success) {
                setTimeout(function() {
                    if (forceSelectProjectTasksTab()) {
                        win.sessionStorage.removeItem('stayOnProjectTasks');
                        isTabMaintaining = false;
                        tabCheckAttempts = 0;
                    }
                }, 100);
            } else if (tabCheckAttempts < maxAttempts) {
                tabCheckAttempts++;
                setTimeout(checkAndSelectTab, checkInterval);
            } else {
                win.sessionStorage.removeItem('stayOnProjectTasks');
                isTabMaintaining = false;
                tabCheckAttempts = 0;
            }
        } else {
            isTabMaintaining = false;
            tabCheckAttempts = 0;
        }
    }
    
    function startTabMonitoring() {
        checkAndSelectTab();
        setInterval(checkAndSelectTab, checkInterval);
    }
    
    if (doc.readyState === 'loading') {
        doc.addEventListener('DOMContentLoaded', startTabMonitoring);
    } else {
        startTabMonitoring();
    }
    
    const observer = new MutationObserver(function(mutations) {
        if (win.sessionStorage.getItem('stayOnProjectTasks') === 'true') {
            checkAndSelectTab();
        }
    });
    
    observer.observe(doc.body, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['aria-selected', 'aria-current', 'class', 'data-baseweb']
    });
    
    doc.addEventListener('click', function(e) {
        let target = e.target;
        let button = null;
        
        while (target && target !== doc.body) {
            if (target.tagName === 'BUTTON') {
                const text = target.textContent || target.innerText || '';
                if (text.includes('Calculate Recommendations')) {
                    button = target;
                    break;
                }
            }
            target = target.parentElement;
        }
        
        if (button) {
            win.sessionStorage.setItem('stayOnProjectTasks', 'true');
            isTabMaintaining = false;
            setTimeout(function() {
                for (let i = 0; i < 10; i++) {
                    setTimeout(function() {
                        checkAndSelectTab();
                    }, i * 50);
                }
            }, 10);
        }
    }, true);
})();
</script>
""",
    unsafe_allow_html=True,
)

# Load data
movie = None
rating = None
df_full = None
common_movies = None
user_movie_df = None
all_user_ids = None
cosine_sim_genre = None

try:
    with handle_errors(DataLoadError, "Data loading error", log_details=True):
        movie, rating, df_full, common_movies, user_movie_df, all_user_ids, cosine_sim_genre = (
            load_data(PICKLE_PATH)
        )
        logger.info("Data loaded successfully")
except DataLoadError as e:
    handle_streamlit_exception(e, show_to_user=True)
    st.info("Please check that data/prepare_data_demo.pkl file exists.")
    st.code(f"Technical details: {str(e)}", language="text")
    st.stop()
except Exception as e:
    handle_streamlit_exception(e, show_to_user=True)
    st.stop()

if df_full is None or common_movies is None or user_movie_df is None:
    st.error("❌ Data could not be loaded. Application cannot run.")
    st.stop()


# Tabs
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "business-problem"
if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False

query_params = st.query_params
if "page" in query_params:
    url_page = query_params.get("page", ["business-problem"])
    if isinstance(url_page, list):
        url_page = url_page[0]
    previous_tab = st.session_state.get("active_tab", "business-problem")
    if st.session_state.get("active_tab") != url_page:
        st.session_state.active_tab = url_page
        if previous_tab == "project-tasks" and url_page != "project-tasks":
            st.session_state.button_clicked = False

tab_names = ["1. Business Problem", "2. Dataset Story", "3. Project Tasks"]
tab_problem, tab_dataset, tab_tasks = st.tabs(tab_names)
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
        st.markdown(
            """
        ### 🧑‍🤝‍🧑 User-Based
        - Find users with similar taste to me
        - Bring films they liked but I haven't watched
        - Sort by weighted score
        """
        )

    with col2:
        st.markdown(
            """
        ### 🎬 Item-Based
        - Find the film I last rated 5⭐
        - Calculate similar films using correlation
        - Sort the most similar ones
        """
        )

    with col3:
        st.markdown(
            """
        ### 🏷️ Content-Based
        - Select a film
        - Check genre information
        - Bring films in the same tone using cosine similarity
        """
        )

    st.success(
        "These three approaches are usually combined into one package in real life. This is what we call a hybrid system."
    )

with tab_dataset:
    st.title("Dataset Story")

    total_users_full = df_full["userId"].nunique()
    total_movies_full = df_full["movieId"].nunique()
    total_ratings_full = df_full.shape[0]

    total_users_common = common_movies["userId"].nunique()
    total_movies_common = common_movies["movieId"].nunique()
    total_ratings_common = common_movies.shape[0]

    um_users, um_movies = user_movie_df.shape

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>Total Users (raw)</div>"
            f"<div class='metric-value'>{total_users_full:,}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>Total Movies (raw)</div>"
            f"<div class='metric-value'>{total_movies_full:,}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>Total Ratings (raw)</div>"
            f"<div class='metric-value'>{total_ratings_full:,}</div>"
            f"</div>",
            unsafe_allow_html=True,
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
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <table class="var-table">
        <tr><th>movieId</th><td>Unique movie number.</td></tr>
        <tr><th>title</th><td>Movie title.</td></tr>
        <tr><th>genres</th><td>Genre information (Action|Comedy|Drama ...)</td></tr>
        </table>
        """,
        unsafe_allow_html=True,
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
        unsafe_allow_html=True,
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
        unsafe_allow_html=True,
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
        unsafe_allow_html=True,
    )

    st.info(
        "Removing movies with few ratings both speeds up calculations and makes similarity measurements more reliable."
    )

with tab_tasks:
    st.title("Project Tasks and Live Recommendation Engine")

    st.subheader("Approaches")
    st.markdown(
        """
    **User-Based**  
    Find users similar to me, bring films they liked but I haven't watched.  
    Correlation (corr) = taste similarity. Weighted Score = corr * average rating.

    **Item-Based**  
    Base on the film the user last rated 5⭐.  
    Find other films most similar to that film (movie-movie correlation).

    **Content-Based**  
    Only look at content. Calculate cosine similarity between genre vectors.  
    'What other films are closest to this film's genre DNA?'
    """
    )

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

        if all_user_ids is not None:
            st.caption(f"📊 Total users in dataset: {len(all_user_ids)}")
            st.caption(f"💡 Example User IDs: {', '.join(map(str, sorted(all_user_ids)[:10]))}...")

        chosen_user = st.number_input(
            "Target User ID",
            min_value=1,
            value=config.DEFAULT_USER_ID,
            step=1,
            help=f"User ID to analyze. Available user IDs range from {min(all_user_ids) if all_user_ids is not None else 1} to {max(all_user_ids) if all_user_ids is not None else 1}",
        )

        rec_type = st.radio(
            "Which method should we run?",
            [
                "User-Based (Users Like Me)",
                "Item-Based (Similar to This Film)",
                "Content-Based (Similar Genre Films)",
                "Hybrid (Combine All)",
            ],
            key="rec_type_radio",
        )

        st.markdown("---")

        if rec_type.startswith("User-Based"):
            st.markdown("#### 🧑‍🤝‍🧑 User-Based Parameters")

            min_overlap_ratio_pct = st.slider(
                "Common viewing percentage (%)",
                min_value=config.MIN_OVERLAP_RATIO,
                max_value=config.MAX_OVERLAP_RATIO,
                value=config.DEFAULT_OVERLAP_RATIO_PCT,
                step=5.0,
                help="Consider users who watched at least 60% of the films I watched as 'similar'.",
            )

            corr_threshold = st.slider(
                "Correlation threshold (taste similarity)",
                min_value=config.MIN_CORR_THRESHOLD,
                max_value=config.MAX_CORR_THRESHOLD,
                value=config.DEFAULT_CORR_THRESHOLD,
                step=0.05,
                help="0.65 and above: really similar to me.",
            )

            max_neighbors = st.slider(
                "Maximum neighbor count",
                min_value=config.MIN_NEIGHBORS,
                max_value=config.MAX_NEIGHBORS,
                value=config.DEFAULT_MAX_NEIGHBORS,
                step=1,
                help="How many similar users should we use?",
            )

            weighted_score_threshold = st.slider(
                "Weighted score threshold",
                min_value=config.MIN_WEIGHTED_SCORE,
                max_value=config.MAX_WEIGHTED_SCORE,
                value=config.DEFAULT_WEIGHTED_SCORE_THRESHOLD,
                step=0.1,
                help="Recommend if (corr * rating) average is above 3.5.",
            )

            top_n_user_based = st.slider(
                "How many films to recommend? (Top-N)",
                min_value=config.MIN_TOP_N,
                max_value=config.MAX_TOP_N,
                value=config.DEFAULT_TOP_N,
                step=1,
                help="How many films to list?",
            )

            top_n_item_based = config.DEFAULT_TOP_N
            top_n_content = config.DEFAULT_TOP_N
            hybrid_top_n = config.DEFAULT_TOP_N
            selected_movie_title = None

        elif rec_type.startswith("Item-Based"):
            st.markdown("#### 🎬 Item-Based Parameters")

            top_n_item_based = st.slider(
                "How many similar films to show?",
                min_value=config.MIN_TOP_N,
                max_value=config.MAX_TOP_N,
                value=config.DEFAULT_TOP_N,
                step=1,
                help="Show the top N most similar films for the film the user last rated 5⭐.",
            )

            min_overlap_ratio_pct = config.DEFAULT_OVERLAP_RATIO_PCT
            corr_threshold = config.DEFAULT_CORR_THRESHOLD
            max_neighbors = config.DEFAULT_MAX_NEIGHBORS
            weighted_score_threshold = config.DEFAULT_WEIGHTED_SCORE_THRESHOLD
            top_n_user_based = config.DEFAULT_TOP_N
            top_n_content = config.DEFAULT_TOP_N
            hybrid_top_n = config.DEFAULT_TOP_N
            selected_movie_title = None

        elif rec_type.startswith("Content-Based"):
            st.markdown("#### 🏷️ Content-Based Parameters")

            try:
                from security_utils import sanitize_user_input
            except ImportError:
                logger.warning("security_utils not available, using basic input validation")
                sanitize_user_input = lambda x, **kwargs: x.strip()[:200] if x else ""

            search_term_raw = st.text_input(
                "Search / type film (for autocomplete):",
                value="",
                help="Type first few letters. The box below will filter accordingly.",
                max_chars=200,
            )

            search_term = sanitize_user_input(search_term_raw, max_length=200)

            if search_term.strip():
                filtered_titles = sorted(
                    [t for t in movie["title"].tolist() if search_term.lower() in t.lower()]
                )
            else:
                filtered_titles = sorted(movie["title"].tolist())

            selected_movie_title = st.selectbox(
                "Select Reference Film",
                options=filtered_titles,
                help="We will find films most similar to this film by genre.",
            )

            top_n_content = st.slider(
                "How many similar films to show?",
                min_value=config.MIN_TOP_N,
                max_value=config.MAX_TOP_N,
                value=config.DEFAULT_TOP_N,
                step=1,
            )

            min_overlap_ratio_pct = config.DEFAULT_OVERLAP_RATIO_PCT
            corr_threshold = config.DEFAULT_CORR_THRESHOLD
            max_neighbors = config.DEFAULT_MAX_NEIGHBORS
            weighted_score_threshold = config.DEFAULT_WEIGHTED_SCORE_THRESHOLD
            top_n_user_based = config.DEFAULT_TOP_N
            top_n_item_based = config.DEFAULT_TOP_N
            hybrid_top_n = config.DEFAULT_TOP_N

        else:  # Hybrid
            st.markdown("#### 🔀 Hybrid Parameters")

            min_overlap_ratio_pct = config.DEFAULT_OVERLAP_RATIO_PCT
            corr_threshold = config.DEFAULT_CORR_THRESHOLD
            max_neighbors = config.DEFAULT_MAX_NEIGHBORS
            weighted_score_threshold = config.DEFAULT_WEIGHTED_SCORE_THRESHOLD
            top_n_user_based = config.DEFAULT_TOP_N
            top_n_item_based = config.DEFAULT_TOP_N
            top_n_content = config.DEFAULT_TOP_N
            selected_movie_title = None

            hybrid_top_n = st.slider(
                "How many films should Hybrid show in total?",
                min_value=config.MIN_TOP_N,
                max_value=config.MAX_TOP_N,
                value=config.DEFAULT_TOP_N,
                step=1,
                help="List the top N films from common/strong candidates of all three approaches.",
            )

        is_project_tasks_page = True
        st.session_state.active_tab = "project-tasks"

        st.markdown(
            """
        <script>
        sessionStorage.setItem('stayOnProjectTasks', 'true');
        </script>
        """,
            unsafe_allow_html=True,
        )

        run_button = st.button(
            "🎬 Calculate Recommendations", type="primary", key="calculate_button"
        )

        if run_button:
            st.session_state.button_clicked = True
            st.session_state.active_tab = "project-tasks"
            st.markdown(
                """
            <script>
            if (typeof sessionStorage !== 'undefined') {
                sessionStorage.setItem('stayOnProjectTasks', 'true');
            }
            </script>
            """,
                unsafe_allow_html=True,
            )

    with right_col:
        st.markdown("### Solution Output")
        show_results = run_button or (is_project_tasks_page and st.session_state.button_clicked)

        if not show_results:
            st.info("Select parameters and press the button.")
        else:
            if rec_type.startswith("User-Based"):
                with st.spinner("Calculating User-Based..."):
                    pre_u = precompute_for_user_userbased(
                        chosen_user, all_user_ids, user_movie_df, rating
                    )
                    result_user = finalize_user_based_from_cache(
                        precomputed=pre_u,
                        min_overlap_ratio_pct=min_overlap_ratio_pct,
                        corr_threshold=corr_threshold,
                        max_neighbors=max_neighbors,
                        weighted_score_threshold=weighted_score_threshold,
                        top_n=top_n_user_based,
                        chosen_user=chosen_user,
                        rating=rating,
                        movie=movie,
                    )

                status = result_user["status"]
                debug_info = result_user.get("debug_info", {})
                recs_df = result_user["recommendations"]

                cand_df = result_user.get("dbg_candidate_users_df", pd.DataFrame())
                corr_df_dbg = result_user.get("dbg_corr_df", pd.DataFrame())
                corr_filtered_dbg = result_user.get("dbg_corr_filtered", pd.DataFrame())
                neigh_dbg = result_user.get("dbg_neighbor_ratings", pd.DataFrame())

                if status != "ok":
                    st.warning(
                        f"User-Based recommendations could not be generated. Status: {status}"
                    )
                    with st.expander("🔎 Debug / Intermediate Steps (detailed calculation steps)"):
                        st.write(
                            "• candidate_users_df = 'Everyone who watched common films with target user'"
                        )
                        st.write("Shape:", cand_df.shape)
                        st.dataframe(cand_df.head(20))

                        st.write(
                            "• corr_df = 'Each candidate user's correlation with target user (taste similarity)'"
                        )
                        st.write("Shape:", corr_df_dbg.shape)
                        st.dataframe(corr_df_dbg.head(20))

                        st.write(
                            "• corr_filtered = 'Both sufficient common film count AND corr above threshold'"
                        )
                        st.write("Shape:", corr_filtered_dbg.shape)
                        st.dataframe(corr_filtered_dbg.head(20))

                        st.write(
                            "• neighbor_ratings = 'Which films these similar users rated how much'"
                        )
                        st.write("Shape:", neigh_dbg.shape)
                        st.dataframe(neigh_dbg.head(20))

                else:
                    if recs_df.empty:
                        st.info("No recommendations found matching the parameters.")
                        with st.expander(
                            "🔎 Debug / Intermediate Steps (detailed calculation steps)"
                        ):
                            st.write(
                                "• candidate_users_df = 'Everyone who watched common films with target user'"
                            )
                            st.write("Shape:", cand_df.shape)
                            st.dataframe(cand_df.head(20))

                            st.write(
                                "• corr_df = 'Each candidate user's correlation with target user (taste similarity)'"
                            )
                            st.write("Shape:", corr_df_dbg.shape)
                            st.dataframe(corr_df_dbg.head(20))

                            st.write(
                                "• corr_filtered = 'Both sufficient common film count AND corr above threshold'"
                            )
                            st.write("Shape:", corr_filtered_dbg.shape)
                            st.dataframe(corr_filtered_dbg.head(20))

                            st.write(
                                "• neighbor_ratings = 'Which films these similar users rated how much'"
                            )
                            st.write("Shape:", neigh_dbg.shape)
                            st.dataframe(neigh_dbg.head(20))

                    else:
                        st.success("User-Based recommendations ready.")
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
                            unsafe_allow_html=True,
                        )

                        col_a, col_b, col_c, col_d = st.columns(4)

                        with col_a:
                            st.metric(
                                label="Movies Watched Count",
                                value=debug_info.get("movies_watched", "?"),
                                help="Number of films the target user rated (example: 186).",
                            )

                        with col_b:
                            st.metric(
                                label="Initial Candidate Pool",
                                value=debug_info.get("candidate_users", "?"),
                                help="Number of users sharing at least one common film with this user. Raw initial pool.",
                            )

                        with col_c:
                            st.metric(
                                label="Similar Users (Overlap+Corr)",
                                value=debug_info.get("after_corr_users", "?"),
                                help="Number of users who passed both common viewing percentage threshold and correlation threshold. (top_users)",
                            )

                        with col_d:
                            st.metric(
                                label="Neighbors Included in Score",
                                value=debug_info.get("used_neighbors", "?"),
                                help="Number of most similar neighbors used when calculating recommendation score.",
                            )

                        st.markdown("#### 🎬 User-Based Recommendations")
                        st.dataframe(recs_df.reset_index(drop=True), width="stretch")

                        with st.expander(
                            "🔎 Debug / Intermediate Steps (detailed calculation steps)"
                        ):
                            st.write("STAGE 1 · candidate_users_df")
                            st.caption(
                                "How many of the films watched by target user did other users also watch? movie_count shows this."
                            )
                            st.write("Shape:", cand_df.shape)
                            st.dataframe(cand_df.head(20))

                            st.write("STAGE 2 · corr_df")
                            st.caption(
                                "Correlation between target user and other users (taste similarity). corr = 1 → same taste, 0 → no relationship."
                            )
                            st.write("Shape:", corr_df_dbg.shape)
                            st.dataframe(corr_df_dbg.head(20))

                            st.write("STAGE 3 · corr_filtered (neighbors)")
                            st.caption(
                                "Users who watched sufficient common films AND exceeded corr threshold. So really 'like me'."
                            )
                            st.write("Shape:", corr_filtered_dbg.shape)
                            st.dataframe(corr_filtered_dbg.head(20))

                            st.write("STAGE 4 · neighbor_ratings")
                            st.caption(
                                "Which films these neighbors rated how much, and weighted score = corr × rating."
                            )
                            st.write("Shape:", neigh_dbg.shape)
                            st.dataframe(neigh_dbg.head(20))

            elif rec_type.startswith("Item-Based"):
                with st.spinner("Calculating Item-Based..."):
                    pre_i = precompute_for_user_itembased(
                        chosen_user, all_user_ids, rating, movie, user_movie_df
                    )
                    status_i, ref_movie, sim_df = finalize_item_based_from_cache(
                        pre_i, top_n_item_based
                    )

                if status_i != "ok":
                    if status_i == "no_five_star":
                        st.warning("This user has never given 5 stars.")
                    elif status_i == "not_in_matrix":
                        st.warning(
                            f"Reference film not in user-movie matrix: {pre_i['reference_movie']}"
                        )
                    else:
                        st.warning(
                            f"Item-Based recommendations could not be generated. Status: {status_i}"
                        )
                else:
                    st.success("Item-Based recommendations ready.")

                    st.markdown(
                        f"<div class='metric-card'>"
                        f"<div class='metric-title'>Reference Film (user's last 5⭐ rating)</div>"
                        f"<div class='metric-value'>{ref_movie}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        "This approach works with the logic 'Those who liked this film also liked these'. "
                        "It measures similarity between films based on users' rating behavior."
                    )

                    st.markdown(f"Total {len(sim_df)} similar films found.")
                    st.dataframe(
                        sim_df.head(top_n_item_based).reset_index(drop=True),
                        width="stretch",
                    )

            elif rec_type.startswith("Content-Based"):
                with st.spinner("Calculating Content-Based..."):
                    cb_result = content_based_recommender_cached(
                        movie_title=selected_movie_title,
                        top_n=top_n_content,
                        movie=movie,
                        cosine_sim_genre=cosine_sim_genre,
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
                        unsafe_allow_html=True,
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
                        width="stretch",
                    )

            else:  # Hybrid
                with st.spinner("Calculating Hybrid..."):
                    pre_u = precompute_for_user_userbased(
                        chosen_user, all_user_ids, user_movie_df, rating
                    )
                    result_user = finalize_user_based_from_cache(
                        precomputed=pre_u,
                        min_overlap_ratio_pct=min_overlap_ratio_pct,
                        corr_threshold=corr_threshold,
                        max_neighbors=max_neighbors,
                        weighted_score_threshold=weighted_score_threshold,
                        top_n=top_n_user_based,
                        chosen_user=chosen_user,
                        rating=rating,
                        movie=movie,
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
                    pre_i = precompute_for_user_itembased(
                        chosen_user, all_user_ids, rating, movie, user_movie_df
                    )
                    status_i, ref_movie_i, sim_df_i = finalize_item_based_from_cache(
                        pre_i, top_n_item_based
                    )
                    if status_i == "ok" and sim_df_i is not None and not sim_df_i.empty:
                        df_item_part = sim_df_i.copy()
                        df_item_part["Source"] = "Item-Based"
                        df_item_part = df_item_part.rename(
                            columns={"Similar Film": "Film_Name", "Similarity": "Model_Score"}
                        )
                    else:
                        df_item_part = pd.DataFrame(columns=["Film_Name", "Model_Score", "Source"])

                    if status_i == "ok" and ref_movie_i is not None:
                        cb_result = content_based_recommender_cached(
                            movie_title=ref_movie_i,
                            top_n=top_n_content,
                            movie=movie,
                            cosine_sim_genre=cosine_sim_genre,
                        )
                        if cb_result["status"] == "ok" and not cb_result["recommendations"].empty:
                            df_cb_part = cb_result["recommendations"].copy()
                            df_cb_part["Source"] = "Content-Based"
                            df_cb_part = df_cb_part.rename(
                                columns={"Film": "Film_Name", "Similarity Score": "Model_Score"}
                            )
                            df_cb_part = df_cb_part[["Film_Name", "Model_Score", "Source"]]
                        else:
                            df_cb_part = pd.DataFrame(
                                columns=["Film_Name", "Model_Score", "Source"]
                            )
                    else:
                        df_cb_part = pd.DataFrame(columns=["Film_Name", "Model_Score", "Source"])

                    combined_all = pd.concat(
                        [df_user_part, df_item_part, df_cb_part], ignore_index=True
                    )

                if combined_all.empty:
                    st.warning(
                        "Hybrid system could not generate recommendations (insufficient signals)."
                    )
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
                        unsafe_allow_html=True,
                    )

                    hybrid_summary = (
                        combined_all.groupby("Film_Name")
                        .agg(
                            Source_Count=("Source", "nunique"),
                            Average_Score=("Model_Score", "mean"),
                        )
                        .reset_index()
                    )

                    # confidence metric
                    hybrid_summary["Hybrid_Confidence"] = (
                        hybrid_summary["Source_Count"] * hybrid_summary["Average_Score"]
                    )

                    hybrid_summary = hybrid_summary.sort_values(
                        by=["Hybrid_Confidence", "Source_Count", "Average_Score"], ascending=False
                    ).head(hybrid_top_n)

                    st.success("Hybrid (Common Candidates from All Models)")
                    st.dataframe(hybrid_summary.reset_index(drop=True), width="stretch")

                    with st.expander("User-Based details"):
                        if df_user_part.empty:
                            st.write("No User-Based results.")
                        else:
                            st.dataframe(
                                df_user_part.reset_index(drop=True), width="stretch"
                            )

                    with st.expander("Item-Based details"):
                        if df_item_part.empty:
                            st.write("No Item-Based results.")
                        else:
                            st.dataframe(
                                df_item_part.reset_index(drop=True), width="stretch"
                            )

                    with st.expander("Content-Based details"):
                        if df_cb_part.empty:
                            st.write("No Content-Based results.")
                        else:
                            st.dataframe(
                                df_cb_part.reset_index(drop=True), width="stretch"
                            )

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

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6b7280; font-size: 0.85rem;'>
        <p>DataHub · Hybrid Recommender System</p>
        <p>You can see how the recommendation logic changes as you play with the parameters.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
