# =============================================================================
# src/repositories/qdrant_collection_repository.py
# QdrantCollection Repository
# =============================================================================
"""
Repository for QdrantCollection entity operations.
"""
from typing import List, Optional, Tuple
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from src.models.models import QdrantCollection, VisibilityType
from src.repositories.base_repository import BaseRepository


class QdrantCollectionRepository(BaseRepository[QdrantCollection]):
    """Repository for managing Qdrant collections."""
    
    def __init__(self, db: AsyncSession):
        super().__init__(QdrantCollection, db)
    
    async def get_by_name(self, name: str) -> Optional[QdrantCollection]:
        """Get collection by name."""
        try:
            stmt = select(QdrantCollection).where(QdrantCollection.name == name)
            result = await self.db.execute(stmt)
            return result.scalars().first()
        except Exception as e:
            self.logger.error(f"Error getting collection by name {name}: {e}")
            raise
    
    async def list_with_filters(
        self,
        category: Optional[str] = None,
        visibility: Optional[VisibilityType] = None,
        is_active: Optional[bool] = True,
        search: Optional[str] = None,
        skip: int = 0,
        limit: int = 20
    ) -> Tuple[List[QdrantCollection], int]:
        """
        List collections with filters and return total count.
        
        Returns:
            Tuple of (collections, total_count)
        """
        try:
            query = select(QdrantCollection)
            
            # Apply filters
            if category:
                query = query.filter(QdrantCollection.category == category)
            if visibility:
                query = query.filter(QdrantCollection.visibility == visibility)
            if is_active is not None:
                query = query.filter(QdrantCollection.is_active == is_active)
            if search:
                query = query.filter(
                    or_(
                        QdrantCollection.name.ilike(f"%{search}%"),
                        QdrantCollection.display_name.ilike(f"%{search}%"),
                        QdrantCollection.description.ilike(f"%{search}%")
                    )
                )
            
            # Get total count
            total = await self.db.scalar(
                select(func.count()).select_from(query.subquery())
            )
            
            # Get paginated results
            result = await self.db.execute(
                query.offset(skip).limit(limit)
            )
            collections = result.scalars().all()
            
            return list(collections), total or 0
            
        except Exception as e:
            self.logger.error(f"Error listing collections with filters: {e}")
            raise
    
    async def get_categories(self) -> List[str]:
        """Get list of distinct active categories."""
        try:
            result = await self.db.execute(
                select(QdrantCollection.category)
                .filter(QdrantCollection.is_active == True)
                .distinct()
            )
            categories = result.all()
            return [cat[0] for cat in categories if cat[0]]
        except Exception as e:
            self.logger.error(f"Error getting categories: {e}")
            raise

    # =========================================================================
    # REPOSITORY PATTERN: Methods to encapsulate DB operations
    # Services should NEVER access self.db directly
    # =========================================================================

    async def create_collection(
        self,
        name: str,
        display_name: str,
        description: str,
        category: str,
        visibility: VisibilityType,
        extra_metadata: Optional[dict] = None
    ) -> QdrantCollection:
        """
        Create a new QdrantCollection record.
        
        Args:
            name: Collection name
            display_name: Human-readable name
            description: Collection description
            category: Category (e.g., 'chat', 'project')
            visibility: Visibility type
            extra_metadata: Optional metadata
            
        Returns:
            Created collection record
        """
        return await self.create(
            name=name,
            display_name=display_name,
            description=description,
            category=category,
            visibility=visibility,
            extra_metadata=extra_metadata or {}
        )

    async def get_or_create(
        self,
        name: str,
        display_name: str,
        description: str,
        category: str,
        visibility: VisibilityType,
        extra_metadata: Optional[dict] = None
    ) -> Tuple[QdrantCollection, bool]:
        """
        Get existing collection or create new one.
        
        Args:
            name: Collection name
            display_name: Human-readable name
            description: Collection description
            category: Category
            visibility: Visibility type
            extra_metadata: Optional metadata
            
        Returns:
            Tuple of (collection, created) where created is True if new
        """
        existing = await self.get_by_name(name)
        if existing:
            return existing, False
        
        new_collection = await self.create_collection(
            name=name,
            display_name=display_name,
            description=description,
            category=category,
            visibility=visibility,
            extra_metadata=extra_metadata
        )
        return new_collection, True
