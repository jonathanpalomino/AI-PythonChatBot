# =============================================================================
# src/repositories/external_oauth_token_repository.py
# CRUD para ExternalOAuthToken con cifrado/descifrado Fernet
# =============================================================================
from datetime import datetime
from typing import Optional
from uuid import UUID

from cryptography.fernet import Fernet
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.external_oauth_token import ExternalOAuthToken
from src.repositories.base_repository import BaseRepository
from src.utils.date_utils import get_current_utc


class ExternalOAuthTokenRepository(BaseRepository[ExternalOAuthToken]):
    """
    Gestiona tokens OAuth de proveedores externos.
    Cifra/descifra access_token y refresh_token con Fernet.

    La fernet_key debe venir de settings.FERNET_KEY (32-byte base64).
    Si no se provee, los tokens se almacenan sin cifrar (solo para dev/test).
    """

    def __init__(self, db: AsyncSession, fernet_key: Optional[str] = None):
        super().__init__(ExternalOAuthToken, db)
        self._fernet: Optional[Fernet] = None
        if fernet_key:
            self._fernet = Fernet(fernet_key.encode() if isinstance(fernet_key, str) else fernet_key)

    # ── Cifrado ────────────────────────────────────────────────────────────

    def _encrypt(self, value: str) -> str:
        if self._fernet:
            return self._fernet.encrypt(value.encode()).decode()
        return value  # plaintext fallback (dev only)

    def _decrypt(self, value: str) -> str:
        if self._fernet:
            return self._fernet.decrypt(value.encode()).decode()
        return value

    # ── Queries ────────────────────────────────────────────────────────────

    async def get_by_user_provider(
        self, user_id: UUID, provider: str
    ) -> Optional[ExternalOAuthToken]:
        result = await self.db.execute(
            select(ExternalOAuthToken).where(
                and_(
                    ExternalOAuthToken.user_id == user_id,
                    ExternalOAuthToken.provider == provider,
                    ExternalOAuthToken.is_active.is_(True),
                )
            )
        )
        return result.scalar_one_or_none()

    # ── Upsert (crea o actualiza) ──────────────────────────────────────────

    async def upsert(
        self,
        user_id: UUID,
        provider: str,
        access_token: str,
        refresh_token: Optional[str],
        expires_at: Optional[datetime],
        scopes: Optional[str] = None,
        token_type: str = "Bearer",
    ) -> ExternalOAuthToken:
        # Buscar registro existente incluyendo los inactivos para evitar violar la restricción única
        result = await self.db.execute(
            select(ExternalOAuthToken).where(
                and_(
                    ExternalOAuthToken.user_id == user_id,
                    ExternalOAuthToken.provider == provider,
                )
            )
        )
        existing = result.scalar_one_or_none()

        enc_access  = self._encrypt(access_token)
        enc_refresh = self._encrypt(refresh_token) if refresh_token else None

        if existing:
            existing.access_token_enc           = enc_access
            existing.refresh_token_enc          = enc_refresh
            existing.expires_at                 = expires_at
            existing.scopes                     = scopes
            existing.token_type                 = token_type
            existing.is_active                  = True
            existing.updated_at                 = get_current_utc()
            await self.db.flush()
            return existing

        token = ExternalOAuthToken(
            user_id=user_id,
            provider=provider,
            access_token_enc=enc_access,
            refresh_token_enc=enc_refresh,
            expires_at=expires_at,
            scopes=scopes,
            token_type=token_type,
        )
        self.db.add(token)
        await self.db.flush()
        return token

    # ── Leer tokens descifrados ────────────────────────────────────────────

    async def get_valid_token(
        self, user_id: UUID, provider: str
    ) -> Optional[dict]:
        """
        Retorna {"access_token": str, "refresh_token": str|None,
                 "expires_at": datetime|None, "token_type": str}
        o None si no existe / está revocado.
        """
        rec = await self.get_by_user_provider(user_id, provider)
        if not rec:
            return None
        return {
            "access_token":  self._decrypt(rec.access_token_enc),
            "refresh_token": self._decrypt(rec.refresh_token_enc) if rec.refresh_token_enc else None,
            "expires_at":    rec.expires_at,
            "token_type":    rec.token_type,
            "scopes":        rec.scopes,
        }

    async def revoke(self, user_id: UUID, provider: str) -> bool:
        rec = await self.get_by_user_provider(user_id, provider)
        if not rec:
            return False
        rec.is_active = False
        await self.db.flush()
        return True
