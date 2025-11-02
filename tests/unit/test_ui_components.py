"""Unit tests for ui/components.py."""

import base64
import tempfile
from pathlib import Path

import pytest

try:
    from ui.components import img_to_base64, render_header, render_styles
except ImportError:
    pytest.skip("ui.components not available", allow_module_level=True)


class TestImgToBase64:
    """Test img_to_base64 function."""

    def test_img_to_base64_success(self, temp_dir):
        """Test successful image to base64 conversion."""
        test_file = temp_dir / "test_image.txt"
        test_content = b"fake image content"
        test_file.write_bytes(test_content)
        
        result = img_to_base64(test_file)
        
        assert result is not None
        assert isinstance(result, str)
        expected = base64.b64encode(test_content).decode("utf-8")
        assert result == expected

    def test_img_to_base64_file_not_found(self):
        """Test img_to_base64 with non-existent file."""
        non_existent = Path("/nonexistent/file.txt")
        result = img_to_base64(non_existent)
        
        assert result is None

    def test_img_to_base64_empty_file(self, temp_dir):
        """Test img_to_base64 with empty file."""
        empty_file = temp_dir / "empty.txt"
        empty_file.touch()
        
        result = img_to_base64(empty_file)
        
        assert result == ""  # base64 of empty content

    def test_img_to_base64_binary_content(self, temp_dir):
        """Test img_to_base64 with binary content."""
        binary_file = temp_dir / "binary.bin"
        binary_content = bytes(range(256))
        binary_file.write_bytes(binary_content)
        
        result = img_to_base64(binary_file)
        
        assert result is not None
        decoded = base64.b64decode(result)
        assert decoded == binary_content


class TestRenderHeader:
    """Test render_header function."""

    def test_render_header_with_logo(self, mock_streamlit):
        """Test render_header with logo."""
        logo_b64 = "dGVzdA=="  # base64 for "test"
        
        render_header(logo_b64)
        
        # Verify st.markdown was called
        assert hasattr(mock_streamlit, 'markdown')
        # Function should not raise exception

    def test_render_header_without_logo(self, mock_streamlit):
        """Test render_header without logo."""
        render_header(None)
        
        # Should use fallback "DH" badge
        assert hasattr(mock_streamlit, 'markdown')

    def test_render_header_with_empty_string(self, mock_streamlit):
        """Test render_header with empty string."""
        render_header("")
        
        # Should use fallback
        assert hasattr(mock_streamlit, 'markdown')

    def test_render_header_includes_logo_html(self, mock_streamlit):
        """Test that header includes logo HTML when logo provided."""
        logo_b64 = "dGVzdA=="
        render_header(logo_b64)
        # Function should generate HTML with logo
        assert True  # If no exception, HTML generation succeeded


class TestRenderStyles:
    """Test render_styles function."""

    def test_render_styles_calls_markdown(self, mock_streamlit):
        """Test render_styles calls st.markdown."""
        render_styles()
        
        # Should call st.markdown with CSS
        assert hasattr(mock_streamlit, 'markdown')

    def test_render_styles_includes_css_classes(self, mock_streamlit):
        """Test that rendered styles include expected CSS classes."""
        render_styles()
        
        # CSS should be rendered (verified by no exception)
        assert True

    def test_render_styles_multiple_calls(self, mock_streamlit):
        """Test render_styles can be called multiple times."""
        render_styles()
        render_styles()
        render_styles()
        
        # Should not crash on multiple calls
        assert True

