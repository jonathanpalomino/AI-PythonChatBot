# =============================================================================
# src/repositories/base_repository.py
# Base Repository Pattern Implementation
# =============================================================================
"""
Generic base repository providing common CRUD operations.
All specific repositories inherit from this base class.

Benefits:
- DRY: No duplicate CRUD code
- Type-safe: Generic typing for model operations
- Testable: Easy to mock for unit tests
- Async-native: Built for FastAPI/AsyncIO
"""

from typing import Generic, TypeVar, Type, List, Optional
from uuid import UUID

from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.utils.logger import get_logger

# Generic type for SQLAlchemy models
ModelType = TypeVar("ModelType")

logger = get_logger(__name__)


class BaseRepository(Generic[ModelType]):
    """
    Generic async repository for database operations.

    Usage:
        class UserRepository(BaseRepository[User]):
            def __init__(self, db: AsyncSession):
                super().__init__(User, db)
    """

    def __init__(self, model: Type[ModelType], db: AsyncSession):
        """
        Initialize repository.

        Args:
            model: SQLAlchemy model class
            db: Async database session
        """
        self.model = model
        self.db = db
        self.logger = logger

    async def get_by_id(self, id: UUID) -> Optional[ModelType]:
        """
        Get single entity by ID.

        Args:
            id: Entity UUID

        Returns:
            Model instance or None if not found
        """
        try:
            stmt = select(self.model).where(self.model.id == id)
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            self.logger.error(f"Error getting {self.model.__name__} by id={id}: {e}")
            raise

    async def get_by_ids(self, ids: List[UUID]) -> List[ModelType]:
        """
        Get multiple entities by their IDs.

        Args:
            ids: List of UUIDs

        Returns:
            List of model instances
        """
        if not ids:
            return []
        try:
            stmt = select(self.model).where(self.model.id.in_(ids))
            result = await self.db.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            self.logger.error(f"Error getting {self.model.__name__} by ids: {e}")
            raise

    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        order_by: Optional[str] = None,
        descending: bool = True,
        **filters
    ) -> List[ModelType]:
        """
        Get all entities with pagination and optional filtering.

        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            order_by: Field name to order by
            descending: Order by descending if True
            **filters: Optional filters (field=value)

        Returns:
            List of model instances
        """
        try:
            stmt = select(self.model)

            # Apply filters
            for field, value in filters.items():
                if value is not None and hasattr(self.model, field):
                    stmt = stmt.where(getattr(self.model, field) == value)

            # Apply ordering
            if order_by:
                order_field = getattr(self.model, order_by, None)
                if order_field is not None:
                    stmt = stmt.order_by(order_field.desc() if descending else order_field.asc())
            elif hasattr(self.model, 'created_at'):
                stmt = stmt.order_by(self.model.created_at.desc() if descending else self.model.created_at.asc())

            stmt = stmt.offset(skip).limit(limit)

            result = await self.db.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            self.logger.error(f"Error getting filtered {self.model.__name__}: {e}")
            raise

    async def create(self, **kwargs) -> ModelType:
        """
        Create new entity.

        Args:
            **kwargs: Model fields as keyword arguments

        Returns:
            Created model instance
        """
        try:
            instance = self.model(**kwargs)
            self.db.add(instance)
            await self.db.flush()  # Generate ID without committing
            await self.db.refresh(instance)  # Load defaults
            return instance
        except Exception as e:
            self.logger.error(f"Error creating {self.model.__name__}: {e}")
            raise

    async def update(self, id: UUID, **kwargs) -> Optional[ModelType]:
        """
        Update entity by ID.

        Args:
            id: Entity UUID
            **kwargs: Fields to update

        Returns:
            Updated model instance or None if not found
        """
        try:
            stmt = (
                update(self.model)
                .where(self.model.id == id)
                .values(**kwargs)
                .returning(self.model)
            )
            result = await self.db.execute(stmt)
            updated = result.scalar_one_or_none()

            if updated:
                await self.db.refresh(updated)

            return updated
        except Exception as e:
            self.logger.error(f"Error updating {self.model.__name__} id={id}: {e}")
            raise

    async def delete(self, id: UUID) -> bool:
        """
        Delete entity by ID.

        Args:
            id: Entity UUID

        Returns:
            True if deleted, False if not found
        """
        try:
            stmt = delete(self.model).where(self.model.id == id)
            result = await self.db.execute(stmt)
            return result.rowcount > 0
        except Exception as e:
            self.logger.error(f"Error deleting {self.model.__name__} id={id}: {e}")
            raise

    async def exists(self, id: UUID) -> bool:
        """
        Check if entity exists by ID.

        Args:
            id: Entity UUID

        Returns:
            True if exists, False otherwise
        """
        try:
            stmt = select(func.count()).where(self.model.id == id).select_from(self.model)
            result = await self.db.execute(stmt)
            count = result.scalar()
            return count > 0
        except Exception as e:
            self.logger.error(f"Error checking existence for {self.model.__name__} id={id}: {e}")
            raise

    async def count(self, **filters) -> int:
        """
        Count entities matching filters.

        Args:
            **filters: Field=value filters

        Returns:
            Count of matching entities
        """
        try:
            stmt = select(func.count()).select_from(self.model)

            # Apply filters
            for field, value in filters.items():
                if hasattr(self.model, field):
                    stmt = stmt.where(getattr(self.model, field) == value)

            result = await self.db.execute(stmt)
            return result.scalar()
        except Exception as e:
            self.logger.error(f"Error counting {self.model.__name__}: {e}")
            raise

    async def commit(self):
        """Commit transaction."""
        await self.db.commit()

    async def rollback(self):
        """Rollback transaction."""
        await self.db.rollback()

    async def refresh(self, instance: ModelType):
        """Refresh instance from database."""
        await self.db.refresh(instance)

    async def flush(self):
        """Flush pending changes to the database."""
        await self.db.flush()

    async def save(self, instance: ModelType) -> ModelType:
        """
        Save (persist) a model instance.
        Encapsulates flush/refresh for Repository pattern compliance.
        
        Args:
            instance: Model instance to persist
            
        Returns:
            Refreshed model instance
        """
        try:
            await self.db.flush()
            await self.db.refresh(instance)
            return instance
        except Exception as e:
            self.logger.error(f"Error saving {self.model.__name__}: {e}")
            raise
