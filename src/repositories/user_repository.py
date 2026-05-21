# =============================================================================
# src/repositories/user_repository.py
# =============================================================================
from typing import Optional
from uuid import UUID

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.models.user import User
from src.models.role import UserRole
from src.repositories.base_repository import BaseRepository
from src.utils.logger import get_logger

logger = get_logger(__name__)


class UserRepository(BaseRepository[User]):

    def __init__(self, db: AsyncSession):
        super().__init__(User, db)

    async def get_by_email(self, email: str) -> Optional[User]:
        stmt = select(User).where(User.email == email.lower().strip())
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_email_with_roles(self, email: str) -> Optional[User]:
        """Carga usuario + user_roles + role en una sola query."""
        stmt = (
            select(User)
            .where(User.email == email.lower().strip())
            .options(selectinload(User.user_roles).selectinload(UserRole.role))
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_id_with_roles(self, user_id: UUID) -> Optional[User]:
        stmt = (
            select(User)
            .where(User.id == user_id)
            .options(selectinload(User.user_roles).selectinload(UserRole.role))
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def email_exists(self, email: str) -> bool:
        stmt = (
            select(func.count())
            .select_from(User)
            .where(User.email == email.lower().strip())
        )
        result = await self.db.execute(stmt)
        return (result.scalar() or 0) > 0

    async def increment_token_version(self, user_id: UUID) -> Optional[User]:
        """
        Incrementa token_version invalidando todos los refresh tokens activos.
        """
        user = await self.get_by_id(user_id)
        if not user:
            return None
        user.token_version += 1
        await self.db.flush()
        await self.db.refresh(user)
        return user
