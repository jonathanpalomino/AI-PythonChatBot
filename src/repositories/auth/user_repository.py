# =============================================================================
# src/repositories/auth/user_repository.py
# =============================================================================
"""
Repositorio de User.

Métodos propios (adicionales a BaseRepository):
  · get_by_email            — login, registro, verificación de unicidad
  · get_by_username         — lookup por alias público
  · get_with_roles          — carga User + UserRole + Role en un solo query (join)
  · get_active_by_email     — shortcut: email + is_active=True
  · email_exists            — verificación rápida O(1) sin traer la fila completa
  · increment_token_version — logout everywhere / cambio de contraseña
  · increment_login_count   — post-login exitoso
  · set_email_verified      — confirmar email
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.models.user import User
from src.repositories.base_repository import BaseRepository
from src.utils.logger import get_logger

logger = get_logger(__name__)


class UserRepository(BaseRepository[User]):

    def __init__(self, db: AsyncSession) -> None:
        super().__init__(User, db)

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    async def get_by_email(self, email: str) -> Optional[User]:
        """
        Busca un usuario por email (case-insensitive via lower()).

        Args:
            email: Dirección de email.

        Returns:
            User o None.
        """
        try:
            stmt = select(User).where(User.email == email.lower().strip())
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"UserRepository.get_by_email error: {e}")
            raise

    async def get_by_username(self, username: str) -> Optional[User]:
        """Busca un usuario por username (case-insensitive)."""
        try:
            stmt = select(User).where(User.username == username.lower().strip())
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"UserRepository.get_by_username error: {e}")
            raise

    async def get_active_by_email(self, email: str) -> Optional[User]:
        """
        Busca un usuario activo por email.
        Shortcut para el flujo de login: evita un filtro manual en el servicio.
        """
        try:
            stmt = (
                select(User)
                .where(User.email == email.lower().strip())
                .where(User.is_active.is_(True))
            )
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"UserRepository.get_active_by_email error: {e}")
            raise

    async def get_with_roles(self, user_id: UUID) -> Optional[User]:
        """
        Carga User con sus UserRole y Role en un solo round-trip (selectinload).

        Usar en get_current_user (dependency) y en cualquier lugar donde
        se necesite user.role_names sin lazy load adicional.

        Args:
            user_id: UUID del usuario.

        Returns:
            User con user_roles y user_roles[].role cargados, o None.
        """
        try:
            from src.models.role import UserRole, Role  # import diferido — evita circular
            stmt = (
                select(User)
                .where(User.id == user_id)
                .options(
                    selectinload(User.user_roles).selectinload(UserRole.role)
                )
            )
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"UserRepository.get_with_roles error: {e}")
            raise

    # ------------------------------------------------------------------
    # Verificaciones rápidas (sin traer la fila completa)
    # ------------------------------------------------------------------

    async def email_exists(self, email: str) -> bool:
        """
        Verifica si un email ya está registrado. O(1) — COUNT sin SELECT *.

        Args:
            email: Dirección de email.

        Returns:
            True si existe.
        """
        try:
            from sqlalchemy import func
            stmt = (
                select(func.count())
                .select_from(User)
                .where(User.email == email.lower().strip())
            )
            result = await self.db.execute(stmt)
            return result.scalar() > 0
        except Exception as e:
            logger.error(f"UserRepository.email_exists error: {e}")
            raise

    async def username_exists(self, username: str) -> bool:
        """Verifica si un username ya está en uso."""
        try:
            from sqlalchemy import func
            stmt = (
                select(func.count())
                .select_from(User)
                .where(User.username == username.lower().strip())
            )
            result = await self.db.execute(stmt)
            return result.scalar() > 0
        except Exception as e:
            logger.error(f"UserRepository.username_exists error: {e}")
            raise

    # ------------------------------------------------------------------
    # Operaciones atómicas de actualización parcial
    # ------------------------------------------------------------------

    async def increment_token_version(self, user_id: UUID) -> int:
        """
        Incrementa token_version en 1 atómicamente (UPDATE ... RETURNING).

        Invalida todos los refresh tokens activos del usuario sin
        necesidad de borrar filas ni mantener blacklists.

        Args:
            user_id: UUID del usuario.

        Returns:
            Nuevo valor de token_version.

        Raises:
            ValueError: Si el usuario no existe.
        """
        try:
            stmt = (
                update(User)
                .where(User.id == user_id)
                .values(token_version=User.token_version + 1)
                .returning(User.token_version)
            )
            result = await self.db.execute(stmt)
            new_version = result.scalar_one_or_none()
            if new_version is None:
                raise ValueError(f"User {user_id} no encontrado.")
            await self.db.flush()
            return new_version
        except Exception as e:
            logger.error(f"UserRepository.increment_token_version error: {e}")
            raise

    async def increment_login_count(self, user_id: UUID) -> None:
        """
        Incrementa login_count y actualiza last_login_at = now().

        Llamar después de un login exitoso.

        Args:
            user_id: UUID del usuario.
        """
        try:
            stmt = (
                update(User)
                .where(User.id == user_id)
                .values(
                    login_count=User.login_count + 1,
                    last_login_at=datetime.now(tz=timezone.utc),
                )
            )
            await self.db.execute(stmt)
            await self.db.flush()
        except Exception as e:
            logger.error(f"UserRepository.increment_login_count error: {e}")
            raise

    async def set_email_verified(self, user_id: UUID) -> None:
        """
        Marca el email del usuario como verificado.

        Args:
            user_id: UUID del usuario.
        """
        try:
            stmt = (
                update(User)
                .where(User.id == user_id)
                .values(
                    email_verified=True,
                    email_verified_at=datetime.now(tz=timezone.utc),
                )
            )
            await self.db.execute(stmt)
            await self.db.flush()
        except Exception as e:
            logger.error(f"UserRepository.set_email_verified error: {e}")
            raise
