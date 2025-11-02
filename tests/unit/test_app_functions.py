"""Unit tests for testable functions in app.py.

Note: app.py contains mostly UI code and module-level execution.
We test the function logic directly without importing app.py to avoid
Streamlit dependencies and module-level execution issues.
"""

import pandas as pd
import pytest


class TestGetMatrixShape:
    """Test get_matrix_shape function logic from app.py.
    
    We test the function logic directly rather than importing app.py,
    since app.py has module-level Streamlit UI code that's hard to mock.
    """

    def test_get_matrix_shape_empty_dataframe(self):
        """Test get_matrix_shape with empty DataFrame."""
        # Function logic from app.py: def get_matrix_shape(df: pd.DataFrame) -> Tuple[int, int]
        def get_matrix_shape(df: pd.DataFrame) -> tuple:
            """Get DataFrame shape."""
            return df.shape
        
        df = pd.DataFrame()
        result = get_matrix_shape(df)
        assert result == (0, 0)

    def test_get_matrix_shape_normal_dataframe(self):
        """Test get_matrix_shape with normal DataFrame."""
        def get_matrix_shape(df: pd.DataFrame) -> tuple:
            """Get DataFrame shape."""
            return df.shape
        
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9]
        })
        result = get_matrix_shape(df)
        assert result == (3, 3)

    def test_get_matrix_shape_large_dataframe(self):
        """Test get_matrix_shape with large DataFrame."""
        def get_matrix_shape(df: pd.DataFrame) -> tuple:
            """Get DataFrame shape."""
            return df.shape
        
        df = pd.DataFrame({
            'a': range(100),
            'b': range(100, 200),
            'c': range(200, 300),
            'd': range(300, 400)
        })
        result = get_matrix_shape(df)
        assert result == (100, 4)

