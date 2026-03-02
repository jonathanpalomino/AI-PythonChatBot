# =============================================================================
# src/repositories/prompt_template_repository.py
# PromptTemplate Repository
# =============================================================================
"""
Repository for PromptTemplate entity operations.
"""
from typing import List, Optional
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from src.models.models import PromptTemplate, VisibilityType
from src.repositories.base_repository import BaseRepository


class PromptTemplateRepository(BaseRepository[PromptTemplate]):
    """Repository for managing prompt templates."""
    
    def __init__(self, db: AsyncSession):
        super().__init__(PromptTemplate, db)
    
    async def get_by_name(self, name: str) -> Optional[PromptTemplate]:
        """Get template by name."""
        try:
            stmt = select(PromptTemplate).where(PromptTemplate.name == name)
            result = await self.db.execute(stmt)
            return result.scalars().first()
        except Exception as e:
            self.logger.error(f"Error getting template by name {name}: {e}")
            raise
    
    async def list_with_filters(
        self,
        category: Optional[str] = None,
        visibility: Optional[VisibilityType] = None,
        search: Optional[str] = None,
        is_active: Optional[bool] = True,
        skip: int = 0,
        limit: int = 20
    ) -> tuple[List[PromptTemplate], int]:
        """
        List templates with filters and return total count.
        
        Returns:
            Tuple of (templates, total_count)
        """
        try:
            query = select(PromptTemplate)
            
            # Apply filters
            if category:
                query = query.filter(PromptTemplate.category == category)
            if visibility:
                query = query.filter(PromptTemplate.visibility == visibility)
            if is_active is not None:
                query = query.filter(PromptTemplate.is_active == is_active)
            if search:
                query = query.filter(
                    or_(
                        PromptTemplate.name.ilike(f"%{search}%"),
                        PromptTemplate.description.ilike(f"%{search}%")
                    )
                )
            
            # Get total count
            total = await self.db.scalar(
                select(func.count()).select_from(query.subquery())
            )
            
            # Get paginated results
            result = await self.db.execute(
                query
                .order_by(PromptTemplate.created_at.desc())
                .offset(skip)
                .limit(limit)
            )
            templates = result.scalars().all()
            
            return list(templates), total or 0
            
        except Exception as e:
            self.logger.error(f"Error listing templates with filters: {e}")
            raise
    
    async def get_categories(self) -> List[str]:
        """Get list of distinct active categories."""
        try:
            result = await self.db.execute(
                select(PromptTemplate.category)
                .filter(PromptTemplate.is_active == True)
                .distinct()
            )
            categories = result.all()
            return [cat[0] for cat in categories if cat[0]]
        except Exception as e:
            self.logger.error(f"Error getting categories: {e}")
            raise
