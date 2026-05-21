# =============================================================================
# src/repositories/role_repository.py
# =============================================================================
from typing import Optional, List
from uuid import UUID

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.models.role import Role, UserRole, Permission, role_permissions
from src.repositories.base_repository import BaseRepository
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RoleRepository(BaseRepository[Role]):

    def __init__(self, db: AsyncSession):
        super().__init__(Role, db)

    async def get_by_name(self, name: str) -> Optional[Role]:
        stmt = select(Role).where(Role.name == name)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_name_with_permissions(self, name: str) -> Optional[Role]:
        stmt = (
            select(Role)
            .where(Role.name == name)
            .options(selectinload(Role.permissions))
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_all_active(self) -> List[Role]:
        stmt = select(Role).where(Role.is_active == True)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def assign_role_to_user(self, user_id: UUID, role_id: UUID) -> UserRole:
        """Asigna un rol a un usuario. Upsert silencioso si ya existe."""
        stmt = select(UserRole).where(
            UserRole.user_id == user_id,
            UserRole.role_id == role_id,
        )
        result = await self.db.execute(stmt)
        existing = result.scalar_one_or_none()
        if existing:
            return existing
        user_role = UserRole(user_id=user_id, role_id=role_id)
        self.db.add(user_role)
        await self.db.flush()
        return user_role

    async def remove_role_from_user(self, user_id: UUID, role_id: UUID) -> bool:
        stmt = delete(UserRole).where(
            UserRole.user_id == user_id,
            UserRole.role_id == role_id,
        )
        result = await self.db.execute(stmt)
        return result.rowcount > 0

    async def get_user_roles(self, user_id: UUID) -> List[Role]:
        stmt = (
            select(Role)
            .join(UserRole, UserRole.role_id == Role.id)
            .where(UserRole.user_id == user_id)
            .where(Role.is_active == True)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())


class PermissionRepository(BaseRepository[Permission]):

    def __init__(self, db: AsyncSession):
        super().__init__(Permission, db)

    async def get_by_name(self, name: str) -> Optional[Permission]:
        stmt = select(Permission).where(Permission.name == name)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_resource_action(
        self, resource: str, action: str
    ) -> Optional[Permission]:
        stmt = select(Permission).where(
            Permission.resource == resource,
            Permission.action == action,
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_all_for_role(self, role_id: UUID) -> List[Permission]:
        stmt = (
            select(Permission)
            .join(
                role_permissions,
                role_permissions.c.permission_id == Permission.id,
            )
            .where(role_permissions.c.role_id == role_id)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
