# =============================================================================
# src/repositories/auth/role_repository.py
# =============================================================================
"""
Repositorio de Role, Permission, RolePermission y UserRole.

Un solo repositorio agrupa las cuatro entidades RBAC porque son cohesivas
y siempre se usan juntas. Separar en 4 repositorios sería over-engineering
para el volumen esperado de operaciones RBAC.

Métodos propios:
  Role
    · get_by_name            — lookup por nombre único
    · get_with_permissions   — Role + RolePermission + Permission (join)
    · list_active            — roles activos para UI de asignación

  Permission
    · get_by_code            — lookup por código único (ej: "conversations:read")
    · get_by_codes           — bulk lookup para seed

  UserRole (tabla pivot User ↔ Role)
    · assign_role            — asigna un rol a un usuario (INSERT ignore duplicados)
    · revoke_role            — elimina un UserRole específico
    · get_user_role_names    — lista los nombres de roles de un usuario (sin cargar User)
    · user_has_role          — check booleano O(1)

  RolePermission (tabla pivot Role ↔ Permission)
    · assign_permission      — asigna un permiso a un rol
    · revoke_permission      — elimina un RolePermission específico
    · role_has_permission    — check booleano O(1)
"""

from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, delete
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.models.role import Role, Permission, UserRole, role_permissions
from src.repositories.base_repository import BaseRepository
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RoleRepository(BaseRepository[Role]):

    def __init__(self, db: AsyncSession) -> None:
        super().__init__(Role, db)

    # ------------------------------------------------------------------
    # Role lookups
    # ------------------------------------------------------------------

    async def get_by_name(self, name: str) -> Optional[Role]:
        """Busca un rol por nombre (case-insensitive)."""
        try:
            stmt = select(Role).where(Role.name == name.lower().strip())
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"RoleRepository.get_by_name error: {e}")
            raise

    async def get_with_permissions(self, role_id: UUID) -> Optional[Role]:
        """
        Carga Role con sus RolePermission + Permission en un solo query.

        Usar cuando se necesite evaluar permisos de un rol sin
        N+1 queries de lazy load.
        """
        try:
            stmt = (
                select(Role)
                .where(Role.id == role_id)
                .options(
                    selectinload(Role.permissions)
                )
            )
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"RoleRepository.get_with_permissions error: {e}")
            raise

    async def list_active(self) -> List[Role]:
        """Retorna todos los roles activos ordenados por nombre."""
        try:
            stmt = (
                select(Role)
                .where(Role.is_active.is_(True))
                .order_by(Role.name.asc())
            )
            result = await self.db.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            logger.error(f"RoleRepository.list_active error: {e}")
            raise

    # ------------------------------------------------------------------
    # Permission lookups
    # ------------------------------------------------------------------

    async def get_permission_by_code(self, code: str) -> Optional[Permission]:
        """Busca un permiso por código único (ej: 'conversations:read')."""
        try:
            stmt = select(Permission).where(Permission.code == code.lower().strip())
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"RoleRepository.get_permission_by_code error: {e}")
            raise

    async def get_permissions_by_codes(self, codes: List[str]) -> List[Permission]:
        """Bulk lookup de permisos por lista de códigos. Usado en seed."""
        if not codes:
            return []
        try:
            normalized = [c.lower().strip() for c in codes]
            stmt = select(Permission).where(Permission.code.in_(normalized))
            result = await self.db.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            logger.error(f"RoleRepository.get_permissions_by_codes error: {e}")
            raise

    async def create_permission(self, **kwargs) -> Permission:
        """Crea un nuevo permiso. Delega en add+flush+refresh."""
        try:
            perm = Permission(**kwargs)
            self.db.add(perm)
            await self.db.flush()
            await self.db.refresh(perm)
            return perm
        except Exception as e:
            logger.error(f"RoleRepository.create_permission error: {e}")
            raise

    async def list_all_permissions(self) -> List[Permission]:
        """Lista todos los permisos ordenados por resource + action."""
        try:
            stmt = select(Permission).order_by(
                Permission.resource.asc(), Permission.action.asc()
            )
            result = await self.db.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            logger.error(f"RoleRepository.list_all_permissions error: {e}")
            raise

    # ------------------------------------------------------------------
    # UserRole (User ↔ Role)
    # ------------------------------------------------------------------

    async def assign_role(self, user_id: UUID, role_id: UUID) -> UserRole:
        """
        Asigna un rol a un usuario.

        Usa INSERT ... ON CONFLICT DO NOTHING para ser idempotente:
        si el rol ya está asignado, no lanza excepción.

        Args:
            user_id: UUID del usuario.
            role_id: UUID del rol.

        Returns:
            UserRole existente o recién creado.
        """
        try:
            stmt = (
                pg_insert(UserRole)
                .values(user_id=user_id, role_id=role_id)
                .on_conflict_do_nothing(
                    index_elements=["user_id", "role_id"]
                )
                .returning(UserRole)
            )
            result = await self.db.execute(stmt)
            row = result.scalar_one_or_none()
            await self.db.flush()

            if row is None:
                # Ya existía — recuperar la fila existente
                existing = await self.db.execute(
                    select(UserRole).where(
                        UserRole.user_id == user_id,
                        UserRole.role_id == role_id,
                    )
                )
                row = existing.scalar_one()
            return row
        except Exception as e:
            logger.error(f"RoleRepository.assign_role error: {e}")
            raise

    async def revoke_role(self, user_id: UUID, role_id: UUID) -> bool:
        """
        Revoca un rol de un usuario.

        Returns:
            True si se eliminó, False si no existía.
        """
        try:
            stmt = delete(UserRole).where(
                UserRole.user_id == user_id,
                UserRole.role_id == role_id,
            )
            result = await self.db.execute(stmt)
            await self.db.flush()
            return result.rowcount > 0
        except Exception as e:
            logger.error(f"RoleRepository.revoke_role error: {e}")
            raise

    async def get_user_role_names(self, user_id: UUID) -> List[str]:
        """
        Retorna los nombres de los roles asignados a un usuario.

        Query directo con JOIN — no carga el objeto User completo.
        Usar en contextos donde solo se necesitan los nombres (ej: JWT claims).

        Args:
            user_id: UUID del usuario.

        Returns:
            Lista de nombres de roles (ej: ["developer", "viewer"]).
        """
        try:
            stmt = (
                select(Role.name)
                .join(UserRole, UserRole.role_id == Role.id)
                .where(UserRole.user_id == user_id)
                .where(Role.is_active.is_(True))
                .order_by(Role.name.asc())
            )
            result = await self.db.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            logger.error(f"RoleRepository.get_user_role_names error: {e}")
            raise

    async def user_has_role(self, user_id: UUID, role_name: str) -> bool:
        """
        Verifica si un usuario tiene un rol específico. Check O(1).

        Args:
            user_id: UUID del usuario.
            role_name: Nombre del rol.

        Returns:
            True si el usuario tiene el rol activo.
        """
        try:
            from sqlalchemy import func
            stmt = (
                select(func.count())
                .select_from(UserRole)
                .join(Role, Role.id == UserRole.role_id)
                .where(UserRole.user_id == user_id)
                .where(Role.name == role_name.lower().strip())
                .where(Role.is_active.is_(True))
            )
            result = await self.db.execute(stmt)
            return result.scalar() > 0
        except Exception as e:
            logger.error(f"RoleRepository.user_has_role error: {e}")
            raise

    # ------------------------------------------------------------------
    # RolePermission (Role ↔ Permission)
    # ------------------------------------------------------------------

    async def assign_permission(
        self, role_id: UUID, permission_id: UUID
    ) -> None:
        """
        Asigna un permiso a un rol. Idempotente via ON CONFLICT DO NOTHING.

        Args:
            role_id: UUID del rol.
            permission_id: UUID del permiso.
        """
        try:
            stmt = (
                pg_insert(role_permissions)
                .values(role_id=role_id, permission_id=permission_id)
                .on_conflict_do_nothing(
                    index_elements=["role_id", "permission_id"]
                )
            )
            await self.db.execute(stmt)
            await self.db.flush()
        except Exception as e:
            logger.error(f"RoleRepository.assign_permission error: {e}")
            raise

    async def revoke_permission(
        self, role_id: UUID, permission_id: UUID
    ) -> bool:
        """Revoca un permiso de un rol. Retorna True si existía."""
        try:
            stmt = delete(role_permissions).where(
                role_permissions.c.role_id == role_id,
                role_permissions.c.permission_id == permission_id,
            )
            result = await self.db.execute(stmt)
            await self.db.flush()
            return result.rowcount > 0
        except Exception as e:
            logger.error(f"RoleRepository.revoke_permission error: {e}")
            raise

    async def role_has_permission(
        self, role_id: UUID, permission_code: str
    ) -> bool:
        """
        Verifica si un rol tiene un permiso por código. Check O(1).

        Args:
            role_id: UUID del rol.
            permission_code: Código del permiso (ej: 'conversations:read').

        Returns:
            True si el rol tiene el permiso.
        """
        try:
            from sqlalchemy import func
            stmt = (
                select(func.count())
                .select_from(role_permissions)
                .join(Permission, Permission.id == role_permissions.c.permission_id)
                .where(role_permissions.c.role_id == role_id)
                .where(Permission.code == permission_code.lower().strip())
            )
            result = await self.db.execute(stmt)
            return result.scalar() > 0
        except Exception as e:
            logger.error(f"RoleRepository.role_has_permission error: {e}")
            raise

    async def get_permissions_for_user(self, user_id: UUID) -> List[str]:
        """
        Retorna todos los códigos de permisos efectivos de un usuario
        (unión de permisos de todos sus roles activos).

        Query con doble JOIN: UserRole → role_permissions → Permission.
        Usado para construir el set de permisos en el JWT o en cache.

        Args:
            user_id: UUID del usuario.

        Returns:
            Lista deduplicada de códigos de permisos.
        """
        try:
            stmt = (
                select(Permission.code)
                .join(role_permissions, role_permissions.c.permission_id == Permission.id)
                .join(Role, Role.id == role_permissions.c.role_id)
                .join(UserRole, UserRole.role_id == Role.id)
                .where(UserRole.user_id == user_id)
                .where(Role.is_active.is_(True))
                .distinct()
                .order_by(Permission.code.asc())
            )
            result = await self.db.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            logger.error(f"RoleRepository.get_permissions_for_user error: {e}")
            raise
