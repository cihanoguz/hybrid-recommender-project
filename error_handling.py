"""Centralized error handling and custom exceptions."""

import logging
import traceback
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Optional, Type, TypeVar

import streamlit as st

from utils import DataLoadError, ValidationError
from logging_config import get_logger

logger = get_logger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


class ApplicationError(Exception):
    """Base exception for all application-specific errors."""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        logger.error(f"{self.__class__.__name__}: {message}", extra=self.details)


class DataProcessingError(ApplicationError):
    """Raised when data processing operations fail."""
    pass


class RecommendationError(ApplicationError):
    """Raised when recommendation generation fails."""
    pass


class ConfigurationError(ApplicationError):
    """Raised when configuration is invalid or missing."""
    pass


@contextmanager
def handle_errors(
    error_type: Type[Exception] = ApplicationError,
    user_message: Optional[str] = None,
    log_details: bool = True,
    reraise: bool = False,
):
    """Context manager for centralized error handling."""
    try:
        yield
    except Exception as e:
        error_msg = user_message or str(e)
        
        if log_details:
            logger.exception(f"{error_type.__name__}: {error_msg}")
        else:
            logger.error(f"{error_type.__name__}: {error_msg}")
        
        if not isinstance(e, error_type):
            raise error_type(error_msg, details={"original_error": str(e)}) from e
        
        if reraise:
            raise


def _handle_validation_error(
    error: ValidationError,
    func_name: str,
    show_user_message: bool,
    show_technical_details: bool,
    log_error: bool,
) -> None:
    """Handle ValidationError in Streamlit context."""
    error_msg = f"❌ Validation Error: {str(error)}"
    if log_error:
        logger.warning(f"Validation error in {func_name}: {error}")
    if show_user_message:
        st.error(error_msg)
    if show_technical_details:
        st.code(str(error), language="text")


def _handle_data_load_error(
    error: DataLoadError,
    func_name: str,
    show_user_message: bool,
    show_technical_details: bool,
    log_error: bool,
) -> None:
    """Handle DataLoadError in Streamlit context."""
    error_msg = "❌ Data loading error. Please contact system administrator."
    if log_error:
        logger.error(f"Data load error in {func_name}: {error}", exc_info=True)
    if show_user_message:
        st.error(error_msg)
        st.info("Data file not found or cannot be read.")
    if show_technical_details:
        with st.expander("🔍 Technical Details"):
            st.code(f"{type(error).__name__}: {str(error)}", language="text")
            st.code(traceback.format_exc(), language="python")
    st.stop()


def _handle_recommendation_error(
    error: RecommendationError,
    func_name: str,
    show_user_message: bool,
    show_technical_details: bool,
    log_error: bool,
) -> None:
    """Handle RecommendationError in Streamlit context."""
    error_msg = f"⚠️ An error occurred while generating recommendations: {str(error)}"
    if log_error:
        logger.error(f"Recommendation error in {func_name}: {error}", exc_info=True)
    if show_user_message:
        st.warning(error_msg)
    if show_technical_details:
        with st.expander("🔍 Technical Details"):
            st.code(str(error), language="text")


def _handle_application_error(
    error: ApplicationError,
    func_name: str,
    show_user_message: bool,
    show_technical_details: bool,
    log_error: bool,
) -> None:
    """Handle ApplicationError in Streamlit context."""
    error_msg = f"❌ Error: {error.message}"
    if log_error:
        logger.error(f"Application error in {func_name}: {error.message}", extra=error.details, exc_info=True)
    if show_user_message:
        st.error(error_msg)
    if show_technical_details:
        with st.expander("🔍 Technical Details"):
            st.json(error.details)
            st.code(traceback.format_exc(), language="python")


def _handle_generic_exception(
    error: Exception,
    func_name: str,
    show_user_message: bool,
    show_technical_details: bool,
    log_error: bool,
) -> None:
    """Handle generic Exception in Streamlit context."""
    error_msg = "❌ An unexpected error occurred."
    if log_error:
        logger.exception(f"Unexpected error in {func_name}: {error}")
    if show_user_message:
        st.error(error_msg)
        st.info("Please check logs or contact system administrator.")
    if show_technical_details:
        with st.expander("🔍 Technical Details (Debug Mode)"):
            st.code(f"{type(error).__name__}: {str(error)}", language="text")
            st.code(traceback.format_exc(), language="python")


def streamlit_error_handler(
    show_user_message: bool = True,
    show_technical_details: bool = False,
    log_error: bool = True,
):
    """Decorator for Streamlit-specific error handling."""
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ValidationError as e:
                _handle_validation_error(e, func.__name__, show_user_message, show_technical_details, log_error)
                return None
            except DataLoadError as e:
                _handle_data_load_error(e, func.__name__, show_user_message, show_technical_details, log_error)
                return None
            except RecommendationError as e:
                _handle_recommendation_error(e, func.__name__, show_user_message, show_technical_details, log_error)
                return None
            except ApplicationError as e:
                _handle_application_error(e, func.__name__, show_user_message, show_technical_details, log_error)
                return None
            except Exception as e:
                _handle_generic_exception(e, func.__name__, show_user_message, show_technical_details, log_error)
                return None
        
        return wrapper
    return decorator


def safe_execute(
    func: Callable[..., Any],
    error_type: Type[Exception] = ApplicationError,
    default_return: Any = None,
    log_error: bool = True,
) -> Any:
    """Safely execute a function with error handling."""
    try:
        return func()
    except error_type as e:
        if log_error:
            logger.error(f"Error executing {func.__name__}: {e}", exc_info=True)
        return default_return
    except Exception as e:
        if log_error:
            logger.exception(f"Unexpected error executing {func.__name__}: {e}")
        return default_return


def format_error_message(error: Exception, include_traceback: bool = False) -> str:
    """Format error message for display or logging."""
    error_type = type(error).__name__
    error_msg = str(error)
    
    formatted = f"{error_type}: {error_msg}"
    
    if include_traceback:
        formatted += f"\n\nTraceback:\n{traceback.format_exc()}"
    
    return formatted


def handle_streamlit_exception(error: Exception, show_to_user: bool = True):
    """Handle exception in Streamlit context."""
    error_type = type(error).__name__
    
    if isinstance(error, ApplicationError):
        logger.error(f"{error_type}: {error.message}", extra=error.details, exc_info=True)
    else:
        logger.exception(f"Unexpected error: {error_type}: {str(error)}")
    
    if show_to_user:
        if isinstance(error, ValidationError):
            st.error(f"❌ Validation Error: {str(error)}")
        elif isinstance(error, DataLoadError):
            st.error("❌ Data loading error. Please contact system administrator.")
            st.info("Data file not found or cannot be read.")
        elif isinstance(error, RecommendationError):
            st.warning(f"⚠️ An error occurred while generating recommendations: {str(error)}")
        elif isinstance(error, ApplicationError):
            st.error(f"❌ Error: {error.message}")
        else:
            st.error("❌ An unexpected error occurred. Please check logs.")

