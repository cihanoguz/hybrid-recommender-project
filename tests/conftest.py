"""Pytest configuration and shared fixtures."""

import os
import pickle
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_movie_data():
    """Sample movie DataFrame for testing."""
    return pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4, 5],
            "title": ["Movie A", "Movie B", "Movie C", "Movie D", "Movie E"],
            "genres": ["Action|Comedy", "Drama", "Action|Thriller", "Comedy", "Drama|Romance"],
        }
    )


@pytest.fixture
def sample_rating_data():
    """Sample rating DataFrame for testing."""
    return pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 2, 3, 3],
            "movieId": [1, 2, 3, 1, 2, 4, 2, 3],
            "rating": [5.0, 4.0, 3.0, 4.5, 5.0, 4.0, 3.5, 4.5],
            "timestamp": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000],
        }
    )


@pytest.fixture
def sample_user_movie_df():
    """Sample user-movie rating matrix for testing."""
    data = {1: {1: 5.0, 2: 4.0, 3: 3.0}, 2: {1: 4.5, 2: 5.0, 4: 4.0}, 3: {2: 3.5, 3: 4.5}}
    return pd.DataFrame(data).T.fillna(0)


@pytest.fixture
def sample_user_ids():
    """Sample list of user IDs."""
    return [1, 2, 3, 4, 5]


@pytest.fixture
def sample_pickle_data(sample_movie_data, sample_rating_data, sample_user_movie_df):
    """Sample pickle data structure matching application format."""
    return {
        "movie": sample_movie_data,
        "rating": sample_rating_data,
        "df_full": pd.concat(
            [
                sample_rating_data,
                sample_movie_data.merge(sample_rating_data, on="movieId", how="inner"),
            ],
            ignore_index=True,
        ),
        "common_movies": sample_rating_data,
        "user_movie_df": sample_user_movie_df,
        "all_user_ids": [1, 2, 3],
        "cosine_sim_genre": np.eye(5),
    }


@pytest.fixture
def temp_pickle_file(temp_dir, sample_pickle_data):
    """Create a temporary pickle file with sample data."""
    pickle_path = temp_dir / "test_data.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(sample_pickle_data, f)
    return pickle_path


@pytest.fixture(autouse=True)
def mock_streamlit(monkeypatch):
    """Mock Streamlit for testing without UI dependencies."""
    import sys
    import types

    # Create mock streamlit module if it doesn't exist
    if "streamlit" not in sys.modules:
        mock_st = types.ModuleType("streamlit")
        sys.modules["streamlit"] = mock_st

    mock_st = sys.modules["streamlit"]

    class MockSessionState:
        def __init__(self):
            self._state = {}

        def get(self, key, default=None):
            return self._state.get(key, default)

        def __getitem__(self, key):
            return self._state[key]

        def __setitem__(self, key, value):
            self._state[key] = value

        def __contains__(self, key):
            return key in self._state

    class MockCacheData:
        def __call__(self, **kwargs):
            def decorator(func):
                return func

            return decorator

    mock_st.session_state = MockSessionState()
    mock_st.query_params = {}
    mock_st.error = lambda x: None
    mock_st.warning = lambda x: None
    mock_st.info = lambda x: None
    mock_st.stop = lambda: None
    mock_st.cache_data = MockCacheData()
    mock_st.cache_resource = MockCacheData()
    mock_st.set_page_config = lambda **kwargs: None

    def mock_columns(n):
        if isinstance(n, list):
            # Handle st.columns([1, 2]) format
            num_cols = len(n)
        else:
            # Handle st.columns(2) format
            num_cols = n
        return [
            type("Col", (), {"__enter__": lambda s: s, "__exit__": lambda s, *a: None})()
            for _ in range(num_cols)
        ]

    mock_st.columns = mock_columns
    mock_st.markdown = lambda *args, **kwargs: None
    mock_st.tabs = lambda names: [
        type("Tab", (), {"__enter__": lambda s: s, "__exit__": lambda s, *a: None})() for _ in names
    ]
    mock_st.number_input = lambda *args, **kwargs: 1
    mock_st.radio = lambda *args, **kwargs: args[1][0] if isinstance(args[1], list) else args[1]
    mock_st.slider = lambda *args, **kwargs: kwargs.get("value", args[3] if len(args) > 3 else 5)
    mock_st.text_input = lambda *args, **kwargs: ""
    mock_st.selectbox = lambda *args, **kwargs: (
        args[1][0] if isinstance(args[1], list) and args[1] else None
    )
    mock_st.button = lambda *args, **kwargs: False
    mock_st.caption = lambda *args: None
    mock_st.title = lambda *args: None
    mock_st.header = lambda *args: None
    mock_st.subheader = lambda *args: None
    mock_st.write = lambda *args: None
    mock_st.info = lambda *args: None
    mock_st.success = lambda *args: None
    mock_st.warning = lambda *args: None
    mock_st.error = lambda *args: None
    mock_st.code = lambda *args, **kwargs: None
    mock_st.dataframe = lambda *args, **kwargs: None
    mock_st.expander = lambda *args, **kwargs: type(
        "Expander", (), {"__enter__": lambda s: s, "__exit__": lambda s, *a: None}
    )()
    mock_st.spinner = lambda *args, **kwargs: type(
        "Spinner", (), {"__enter__": lambda s: s, "__exit__": lambda s, *a: None}
    )()

    return mock_st


@pytest.fixture(autouse=True)
def reset_environment(monkeypatch):
    """Reset environment variables before each test."""
    env_vars_to_clear = [
        "LOG_LEVEL",
        "LOG_FILE",
        "PICKLE_PATH",
        "LOGO_PATH",
        "DEFAULT_USER_ID",
        "SERVER_PORT",
    ]
    for var in env_vars_to_clear:
        monkeypatch.delenv(var, raising=False)
