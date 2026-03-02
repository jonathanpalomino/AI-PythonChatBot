# =============================================================================
# src/utils/file_utils.py
# File Operations Utilities
# =============================================================================
from pathlib import Path

from src.document_loaders import DocumentLoaderFactory


def validate_file_extension(filename: str) -> bool:
    """
    Validate if the file extension is supported.

    Args:
        filename (str): The name of the file.

    Returns:
        bool: True if supported, raises ValueError otherwise.

    Raises:
        ValueError: If file type is not allowed.
    """
    if not filename:
        return False

    extension = Path(filename).suffix.lower()
    allowed_extensions = DocumentLoaderFactory.get_supported_extensions()

    if extension not in allowed_extensions:
        raise ValueError(
            f"File type {extension} not allowed. Allowed: {', '.join(sorted(allowed_extensions))}"
        )

    return True
