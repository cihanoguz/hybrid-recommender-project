"""
Security Utilities.

Provides security-related functions for input validation, path sanitization,
and file integrity checks.
"""

import hashlib
import os
from pathlib import Path
from typing import Optional, Tuple

from logging_config import get_logger

logger = get_logger(__name__)


def sanitize_file_path(file_path: Path, allowed_base: Optional[Path] = None) -> Tuple[bool, Optional[Path]]:
    """
    Sanitize and validate file path to prevent path traversal attacks.
    
    Resolves the path and ensures it's within the allowed base directory.
    This prevents directory traversal attacks (e.g., ../../../etc/passwd).
    
    Args:
        file_path: Path to sanitize
        allowed_base: Base directory that the file must be within.
                     If None, resolves to absolute path without restriction.
    
    Returns:
        Tuple of (is_valid, sanitized_path)
        - is_valid: True if path is safe
        - sanitized_path: Resolved absolute path if valid, None otherwise
    
    Example:
        >>> is_valid, safe_path = sanitize_file_path(Path("../etc/passwd"), Path("/app/data"))
        >>> is_valid  # False
        >>> is_valid, safe_path = sanitize_file_path(Path("data/file.pkl"), Path("/app"))
        >>> safe_path  # PosixPath('/app/data/file.pkl')
    """
    try:
        # Resolve to absolute path
        resolved_path = file_path.resolve()
        
        # If no base restriction, just return resolved path
        if allowed_base is None:
            return True, resolved_path
        
        # Ensure base is also absolute
        allowed_base = allowed_base.resolve()
        
        # Check if resolved path is within allowed base
        try:
            resolved_path.relative_to(allowed_base)
            return True, resolved_path
        except ValueError:
            # Path is outside allowed base
            logger.warning(
                f"Path traversal attempt detected: {file_path} "
                f"(resolved to {resolved_path}, base: {allowed_base})"
            )
            return False, None
    
    except (OSError, RuntimeError) as e:
        logger.error(f"Error resolving path {file_path}: {e}")
        return False, None


def calculate_file_checksum(file_path: Path, algorithm: str = "sha256") -> Optional[str]:
    """
    Calculate file checksum for integrity verification.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm to use ('md5', 'sha256', 'sha512')
    
    Returns:
        Hex digest of file checksum, or None if error
    """
    if not file_path.exists() or not file_path.is_file():
        logger.warning(f"Cannot calculate checksum: file does not exist: {file_path}")
        return None
    
    try:
        hash_obj = hashlib.new(algorithm)
        
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        
        checksum = hash_obj.hexdigest()
        logger.debug(f"Calculated {algorithm} checksum for {file_path}: {checksum[:16]}...")
        return checksum
    
    except Exception as e:
        logger.error(f"Error calculating checksum for {file_path}: {e}")
        return None


def validate_pickle_file_integrity(
    file_path: Path,
    expected_checksum: Optional[str] = None,
    max_file_size: int = 500 * 1024 * 1024,  # 500 MB default
) -> Tuple[bool, Optional[str]]:
    """
    Validate pickle file integrity and safety.
    
    Performs multiple safety checks:
    1. File size validation (prevents memory exhaustion)
    2. File path sanitization
    3. Optional checksum verification
    
    Args:
        file_path: Path to pickle file
        expected_checksum: Optional expected checksum for verification
        max_file_size: Maximum allowed file size in bytes (default: 500 MB)
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check file size
    try:
        file_size = file_path.stat().st_size
        if file_size > max_file_size:
            return False, f"File too large: {file_size / 1024 / 1024:.2f} MB (max: {max_file_size / 1024 / 1024:.2f} MB)"
        
        if file_size == 0:
            return False, "File is empty"
    
    except OSError as e:
        return False, f"Cannot read file size: {e}"
    
    # Validate checksum if provided
    if expected_checksum:
        actual_checksum = calculate_file_checksum(file_path, algorithm="sha256")
        if actual_checksum is None:
            return False, "Failed to calculate file checksum"
        
        if actual_checksum != expected_checksum:
            logger.warning(f"Checksum mismatch for {file_path}")
            return False, "File checksum verification failed (file may be corrupted or tampered)"
    
    return True, None


def sanitize_user_input(input_str: str, max_length: int = 1000, allowed_chars: Optional[str] = None) -> str:
    """
    Sanitize user input string to prevent injection attacks.
    
    Args:
        input_str: Input string to sanitize
        max_length: Maximum allowed length
        allowed_chars: Optional regex pattern of allowed characters.
                      If None, allows alphanumeric, spaces, and common punctuation.
    
    Returns:
        Sanitized string
    """
    if not isinstance(input_str, str):
        return ""
    
    # Truncate if too long
    if len(input_str) > max_length:
        logger.warning(f"Input string truncated from {len(input_str)} to {max_length} characters")
        input_str = input_str[:max_length]
    
    # Remove null bytes and control characters
    sanitized = "".join(char for char in input_str if ord(char) >= 32 or char == "\n" or char == "\t")
    
    # If allowed_chars pattern provided, filter accordingly
    if allowed_chars:
        import re
        sanitized = re.sub(f"[^{allowed_chars}]", "", sanitized)
    
    return sanitized.strip()


def validate_numeric_input(value: float, min_value: float, max_value: float, param_name: str) -> Tuple[bool, Optional[str]]:
    """
    Validate numeric input is within acceptable range.
    
    Args:
        value: Value to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        param_name: Name of parameter (for error messages)
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(value, (int, float)):
        return False, f"{param_name} must be a number"
    
    if value < min_value or value > max_value:
        return False, f"{param_name} must be between {min_value} and {max_value}, got {value}"
    
    # Check for NaN or Infinity
    import math
    if math.isnan(value) or math.isinf(value):
        return False, f"{param_name} cannot be NaN or Infinity"
    
    return True, None

