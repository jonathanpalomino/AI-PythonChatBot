# =============================================================================
# src/utils/date_utils.py
# Date and Time Utilities
# =============================================================================
from datetime import datetime, timezone

def get_current_utc() -> datetime:
    """
    Get current UTC datetime (naive, for DB compatibility).
    
    Returns:
        datetime: Current datetime in UTC without timezone info.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)

def get_current_utc_iso() -> str:
    """
    Get current UTC datetime as ISO format string.
    
    Returns:
        str: ISO 8601 formatted string.
    """
    return get_current_utc().isoformat()
