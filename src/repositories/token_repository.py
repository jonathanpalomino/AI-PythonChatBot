# =============================================================================
# src/repositories/token_repository.py
# =============================================================================
import hashlib
from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.refresh_token import RefreshToken
from src.repositories.base_repository import BaseRepository
from src.utils.date_utils import get_current_utc
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _hash_token(raw_token: str) -> str:
    """SHA-256 del token raw. Nunca almacenar el valor original."""
    return hashlib.sha256(raw_token.encode()).hexdigest()


class TokenRepository(BaseRepository[RefreshToken]):

    def __init__(self, db: AsyncSession):
        super().__init__(RefreshToken, db)

    async def create_refresh_token(
        self,
        user_id: UUID,
        raw_token: str,
        token_version: int,
        expires_at: datetime,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> RefreshToken:
        token = RefreshToken(
            user_id=user_id,
            token_hash=_hash_token(raw_token),
            token_version=token_version,
            expires_at=expires_at,
            user_agent=user_agent,
            ip_address=ip_address,
        )
        self.db.add(token)
        await self.db.flush()
        await self.db.refresh(token)
        return token

    async def get_valid_token(
        self, raw_token: str, user_token_version: int
    ) -> Optional[RefreshToken]:
        """
        Retorna el token si:
          · hash coincide
          · no está revocado
          · no ha expirado
          · token_version coincide con la del usuario
        """
        token_hash = _hash_token(raw_token)
        now = get_current_utc()
        stmt = select(RefreshToken).where(
            RefreshToken.token_hash == token_hash,
            RefreshToken.is_revoked == False,
            RefreshToken.expires_at > now,
            RefreshToken.token_version == user_token_version,
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def revoke_token(self, raw_token: str) -> bool:
        token_hash = _hash_token(raw_token)
        stmt = (
            update(RefreshToken)
            .where(RefreshToken.token_hash == token_hash)
            .values(is_revoked=True)
        )
        result = await self.db.execute(stmt)
        return result.rowcount > 0

    async def revoke_all_for_user(self, user_id: UUID) -> int:
        """Revoca todos los tokens activos del usuario (logout everywhere)."""
        stmt = (
            update(RefreshToken)
            .where(
                RefreshToken.user_id == user_id,
                RefreshToken.is_revoked == False,
            )
            .values(is_revoked=True)
        )
        result = await self.db.execute(stmt)
        return result.rowcount

    async def delete_expired(self, user_id: Optional[UUID] = None) -> int:
        """Limpieza de tokens expirados (opcional: solo de un usuario)."""
        now = get_current_utc()
        stmt = delete(RefreshToken).where(RefreshToken.expires_at <= now)
        if user_id:
            stmt = stmt.where(RefreshToken.user_id == user_id)
        result = await self.db.execute(stmt)
        return result.rowcount
