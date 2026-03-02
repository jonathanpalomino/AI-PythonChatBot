# =============================================================================
# src/repositories/project_repository.py
# Project Repository
# =============================================================================
"""
Repository for Project entity operations.
"""

from src.models.models import Project
from src.repositories.base_repository import BaseRepository
from sqlalchemy.ext.asyncio import AsyncSession

class ProjectRepository(BaseRepository[Project]):
    """Repository for managing projects."""

    def __init__(self, db: AsyncSession):
        super().__init__(Project, db)
