# =============================================================================
# src/utils/error_handler.py
# Error Handling Utilities
# =============================================================================
"""
Centralized error handling utilities for the application.

Provides:
- Decorator for standardized error handling
- Custom exception classes
- Error logging utilities
"""
import functools
import traceback
from typing import Callable, Optional, Type, Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Custom Exception Classes
# =============================================================================

class BaseApplicationError(Exception):
    """Base exception for application-specific errors"""

    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(BaseApplicationError):
    """Raised when input validation fails"""
    pass


class DatabaseError(BaseApplicationError):
    """Raised when database operations fail"""
    pass


class NetworkError(BaseApplicationError):
    """Raised when network operations fail"""
    pass


class TimeoutError(BaseApplicationError):
    """Raised when operations timeout"""
    pass


class ConfigurationError(BaseApplicationError):
    """Raised when configuration is invalid"""
    pass


class NotFoundError(BaseApplicationError):
    """Raised when a resource is not found"""
    pass


class PermissionError(BaseApplicationError):
    """Raised when permission is denied"""
    pass


# =============================================================================
# Error Handling Decorator
# =============================================================================

def handle_errors(
    *,
    reraise: bool = True,
    default_return: Any = None,
    log_level: str = "error",
    include_traceback: bool = True,
    custom_exceptions: Optional[dict[Type[Exception], str]] = None
):
    """
    Decorator for standardized error handling.

    Args:
        reraise: Whether to re-raise the exception after logging (default: True)
        default_return: Value to return if reraise is False (default: None)
        log_level: Logging level to use ('debug', 'info', 'warning', 'error', 'critical')
        include_traceback: Whether to include traceback in logs (default: True)
        custom_exceptions: Dict mapping exception types to custom error messages

    Example:
        @handle_errors(reraise=False, default_return={"error": "Operation failed"})
        async def my_function():
            # Function code
            pass
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Check for custom exception messages
                error_message = None
                if custom_exceptions:
                    for exc_type, msg in custom_exceptions.items():
                        if isinstance(e, exc_type):
                            error_message = msg
                            break

                if error_message is None:
                    error_message = str(e)

                # Log the error
                log_func = getattr(logger, log_level.lower(), logger.error)

                log_data = {
                    "function": func.__name__,
                    "error_type": type(e).__name__,
                    "error_message": error_message
                }

                if include_traceback:
                    log_data["traceback"] = traceback.format_exc()

                log_func(f"Error in {func.__name__}: {error_message}", extra=log_data)

                if reraise:
                    raise
                else:
                    return default_return

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Check for custom exception messages
                error_message = None
                if custom_exceptions:
                    for exc_type, msg in custom_exceptions.items():
                        if isinstance(e, exc_type):
                            error_message = msg
                            break

                if error_message is None:
                    error_message = str(e)

                # Log the error
                log_func = getattr(logger, log_level.lower(), logger.error)

                log_data = {
                    "function": func.__name__,
                    "error_type": type(e).__name__,
                    "error_message": error_message
                }

                if include_traceback:
                    log_data["traceback"] = traceback.format_exc()

                log_func(f"Error in {func.__name__}: {error_message}", extra=log_data)

                if reraise:
                    raise
                else:
                    return default_return

        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# =============================================================================
# Error Logging Utilities
# =============================================================================

def log_error(
    error: Exception,
    context: Optional[dict] = None,
    log_level: str = "error",
    include_traceback: bool = True
):
    """
    Log an error with context information.

    Args:
        error: The exception to log
        context: Additional context information
        log_level: Logging level to use
        include_traceback: Whether to include traceback
    """
    log_func = getattr(logger, log_level.lower(), logger.error)

    log_data = {
        "error_type": type(error).__name__,
        "error_message": str(error)
    }

    if context:
        log_data.update(context)

    if include_traceback:
        log_data["traceback"] = traceback.format_exc()

    log_func(f"Error: {str(error)}", extra=log_data)


def format_error_response(error: Exception, include_details: bool = False) -> dict:
    """
    Format an exception as a standardized error response.

    Args:
        error: The exception to format
        include_details: Whether to include detailed error information

    Returns:
        Dictionary with error information
    """
    response = {
        "error": True,
        "error_type": type(error).__name__,
        "message": str(error)
    }

    if include_details:
        response["traceback"] = traceback.format_exc()

    if isinstance(error, BaseApplicationError) and error.details:
        response["details"] = error.details

    return response


# =============================================================================
# Import asyncio for async function detection
# =============================================================================

import asyncio
