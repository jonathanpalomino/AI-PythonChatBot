# =============================================================================
# src/repositories/auth/refresh_token_repository.py
# =============================================================================
"""
Repositorio de RefreshToken.

Responsabilidades:
  · Crear tokens (INSERT)
  · Buscar por jti (JWT ID único por token) — lookup principal en refresh flow
  · Revocar uno o todos los tokens de un usuario
  · Limpiar tokens expirados (mantenimiento)
  · Validar token_version — invalida tokens de rotaciones previas

Decisiones de diseño:
  · No hereda get_by_id como lookup primario: el campo de lookup es jti (str),
    no id (UUID). get_by_id sigue disponible via BaseRepository para admin.
  · revoke_all_for_user usa UPDATE masivo (no DELETE): conserva auditoría.
  · purge_expired usa DELETE directo para no inflar el pool de memoria.
"""

from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.refresh_token import RefreshToken
from src.repositories.base_repository import BaseRepository
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RefreshTokenRepository(BaseRepository[RefreshToken]):

    def __init__(self, db: AsyncSession) -> None:
        super().__init__(RefreshToken, db)

    # ------------------------------------------------------------------
    # Lookup principal
    # ------------------------------------------------------------------

    async def get_by_jti(self, jti: str) -> Optional[RefreshToken]:
        """
        Busca un RefreshToken por su JWT ID (jti).

        El jti es el identificador único de cada token emitido.
        Es el campo de lookup en el flujo de refresh.

        Args:
            jti: JWT ID string (UUID como string en el payload JWT).

        Returns:
            RefreshToken o None.
        """
        try:
            stmt = select(RefreshToken).where(RefreshToken.jti == jti)
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"RefreshTokenRepository.get_by_jti error: {e}")
            raise

    # ------------------------------------------------------------------
    # Validación
    # ------------------------------------------------------------------

    async def is_valid(self, jti: str, token_version: int) -> bool:
        """
        Verifica que el token:
          1. Existe y no está revocado (is_revoked=False).
          2. No ha expirado (expires_at > now).
          3. El token_version coincide con el del usuario (no fue invalidado
             por logout-everywhere o cambio de contraseña).

        No carga el objeto completo — usa COUNT para O(1).

        Args:
            jti: JWT ID del token.
            token_version: Valor actual de user.token_version.

        Returns:
            True si el token es válido y usable.
        """
        try:
            from sqlalchemy import func
            now = datetime.now(tz=timezone.utc)
            stmt = (
                select(func.count())
                .select_from(RefreshToken)
                .where(RefreshToken.jti == jti)
                .where(RefreshToken.is_revoked.is_(False))
                .where(RefreshToken.expires_at > now)
                .where(RefreshToken.token_version == token_version)
            )
            result = await self.db.execute(stmt)
            return result.scalar() > 0
        except Exception as e:
            logger.error(f"RefreshTokenRepository.is_valid error: {e}")
            raise

    # ------------------------------------------------------------------
    # Revocación
    # ------------------------------------------------------------------

    async def revoke(self, jti: str) -> bool:
        """
        Revoca un token específico por jti.

        Marca is_revoked=True en lugar de borrar la fila:
        permite auditoría y detección de reuse attacks.

        Args:
            jti: JWT ID del token a revocar.

        Returns:
            True si el token existía y fue revocado.
        """
        try:
            stmt = (
                update(RefreshToken)
                .where(RefreshToken.jti == jti)
                .where(RefreshToken.is_revoked.is_(False))
                .values(is_revoked=True, revoked_at=datetime.now(tz=timezone.utc))
            )
            result = await self.db.execute(stmt)
            await self.db.flush()
            return result.rowcount > 0
        except Exception as e:
            logger.error(f"RefreshTokenRepository.revoke error: {e}")
            raise

    async def revoke_all_for_user(self, user_id: UUID) -> int:
        """
        Revoca todos los refresh tokens activos de un usuario.

        Llamar en: logout-everywhere, cambio de contraseña, suspensión de cuenta.
        Complementa increment_token_version en UserRepository.

        Args:
            user_id: UUID del usuario.

        Returns:
            Número de tokens revocados.
        """
        try:
            stmt = (
                update(RefreshToken)
                .where(RefreshToken.user_id == user_id)
                .where(RefreshToken.is_revoked.is_(False))
                .values(is_revoked=True, revoked_at=datetime.now(tz=timezone.utc))
            )
            result = await self.db.execute(stmt)
            await self.db.flush()
            return result.rowcount
        except Exception as e:
            logger.error(f"RefreshTokenRepository.revoke_all_for_user error: {e}")
            raise

    # ------------------------------------------------------------------
    # Listado
    # ------------------------------------------------------------------

    async def get_active_for_user(self, user_id: UUID) -> List[RefreshToken]:
        """
        Lista los tokens activos (no revocados, no expirados) de un usuario.

        Útil para UI de sesiones activas ("dispositivos conectados").

        Args:
            user_id: UUID del usuario.

        Returns:
            Lista de RefreshToken activos, ordenados por created_at desc.
        """
        try:
            now = datetime.now(tz=timezone.utc)
            stmt = (
                select(RefreshToken)
                .where(RefreshToken.user_id == user_id)
                .where(RefreshToken.is_revoked.is_(False))
                .where(RefreshToken.expires_at > now)
                .order_by(RefreshToken.created_at.desc())
            )
            result = await self.db.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            logger.error(f"RefreshTokenRepository.get_active_for_user error: {e}")
            raise

    # ------------------------------------------------------------------
    # Mantenimiento
    # ------------------------------------------------------------------

    async def purge_expired(self) -> int:
        """
        Elimina físicamente los tokens expirados Y revocados.

        Diseñado para correr en un job de mantenimiento periódico
        (ej: noche, via APScheduler o celery beat).
        No toca tokens revocados pero aún no expirados (auditoría activa).

        Returns:
            Número de filas eliminadas.
        """
        try:
            now = datetime.now(tz=timezone.utc)
            stmt = delete(RefreshToken).where(
                RefreshToken.expires_at < now,
                RefreshToken.is_revoked.is_(True),
            )
            result = await self.db.execute(stmt)
            await self.db.flush()
            logger.info(f"RefreshTokenRepository.purge_expired: {result.rowcount} tokens eliminados")
            return result.rowcount
        except Exception as e:
            logger.error(f"RefreshTokenRepository.purge_expired error: {e}")
            raise
