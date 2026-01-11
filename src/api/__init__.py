# =============================================================================
# api/v1/__init__.py
# API v1 package initialization
# =============================================================================
"""
API v1 endpoints
"""

# Import all routers for easy access
from src.api.v1 import conversations, prompts, collections, files, tools

__all__ = [
    "conversations",
    "prompts",
    "collections",
    "files",
    "tools"
]
