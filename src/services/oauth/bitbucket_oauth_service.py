# =============================================================================
# src/services/oauth/bitbucket_oauth_service.py
# OAuth 2.0 Authorization Code Flow para Bitbucket Cloud
# =============================================================================
"""
Flujo completo:
  1. GET  /oauth/bitbucket/authorize  → redirect a Bitbucket
  2. GET  /oauth/bitbucket/callback   → intercambia code por tokens
                                        → persiste en ExternalOAuthToken
  3. get_valid_access_token(user_id)  → retorna token vigente,
                                        renovando si expiró

Scopes requeridos en la Bitbucket OAuth Consumer:
  · repository:read
  · repository:write   (si se quiere push/PR)
  · pullrequest:read

Configurar en settings:
  BITBUCKET_CLIENT_ID     = "..."
  BITBUCKET_CLIENT_SECRET = "..."
  BITBUCKET_REDIRECT_URI  = "https://tu-dominio/api/v1/oauth/bitbucket/callback"
  FERNET_KEY              = "..."   # 32-byte base64, generado con Fernet.generate_key()
"""

from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urlencode
from uuid import UUID

import httpx

from src.repositories.external_oauth_token_repository import ExternalOAuthTokenRepository
from src.services.auth.token_service import TokenService
from src.utils.date_utils import get_current_utc
from src.utils.logger import get_logger

logger = get_logger(__name__)

_AUTH_URL   = "https://bitbucket.org/site/oauth2/authorize"
_TOKEN_URL  = "https://bitbucket.org/site/oauth2/access_token"
_SCOPES     = "repository account"


class BitbucketOAuthService:
    """
    Gestiona el ciclo de vida de tokens OAuth de Bitbucket por usuario.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        token_repo: ExternalOAuthTokenRepository,
    ):
        self._client_id     = client_id
        self._client_secret = client_secret
        self._redirect_uri  = redirect_uri
        self._repo          = token_repo

    # ── Paso 1: URL de autorización ────────────────────────────────────────

    def get_authorization_url(self, state: str) -> str:
        """
        Genera la URL a la que se redirige al usuario.
        `state` es un token CSRF de un solo uso (persistir en sesión o BD).
        """
        params = {
            "client_id":     self._client_id,
            "response_type": "code",
            "redirect_uri":  self._redirect_uri,
            "scope":         _SCOPES,
            "state":         state,
        }
        return f"{_AUTH_URL}?{urlencode(params)}"

    # ── Paso 2: Intercambiar code por tokens ───────────────────────────────

    async def exchange_code(
        self, code: str, user_id: UUID
    ) -> dict:
        """
        POST al endpoint de Bitbucket con el authorization_code.
        Persiste los tokens cifrados en BD.
        Retorna el access_token en claro (para uso inmediato).
        """
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                _TOKEN_URL,
                auth=(self._client_id, self._client_secret),
                data={
                    "grant_type":   "authorization_code",
                    "code":         code,
                    "redirect_uri": self._redirect_uri,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        return await self._persist_tokens(user_id, data)

    # ── Renovación automática ──────────────────────────────────────────────

    async def refresh_access_token(self, user_id: UUID) -> Optional[str]:
        """
        Usa el refresh_token almacenado para obtener un nuevo access_token.
        Retorna el nuevo access_token o None si el refresh_token ya expiró/revocado.
        """
        stored = await self._repo.get_valid_token(user_id, "bitbucket")
        if not stored or not stored.get("refresh_token"):
            return None

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                _TOKEN_URL,
                auth=(self._client_id, self._client_secret),
                data={
                    "grant_type":    "refresh_token",
                    "refresh_token": stored["refresh_token"],
                },
            )
            if resp.status_code == 400:
                logger.warning(f"[bb_oauth] refresh_token expirado para user={user_id}")
                await self._repo.revoke(user_id, "bitbucket")
                return None
            resp.raise_for_status()
            data = resp.json()

        result = await self._persist_tokens(user_id, data)
        return result["access_token"]

    # ── get_valid_access_token (entry point para git_tool) ─────────────────

    async def get_valid_access_token(self, user_id: UUID) -> Optional[str]:
        """
        Retorna un access_token válido para el usuario.
        Si expiró, lo renueva automáticamente.
        Si no hay token OAuth almacenado, retorna None.

        Uso desde git_tool:
            token = await bb_svc.get_valid_access_token(user_id)
            if not token:
                raise NeedsOAuthFlow("User must authorize Bitbucket first")
        """
        stored = await self._repo.get_valid_token(user_id, "bitbucket")
        if not stored:
            return None

        expires_at = stored.get("expires_at")
        if expires_at and expires_at.tzinfo is not None:
            expires_at = expires_at.replace(tzinfo=None)
            
        now = get_current_utc()

        # Si expira en menos de 5 minutos, renovar proactivamente
        if expires_at and expires_at <= now + timedelta(minutes=5):
            logger.info(f"[bb_oauth] access_token próximo a expirar para user={user_id}, renovando...")
            return await self.refresh_access_token(user_id)

        return stored["access_token"]

    # ── Revocación ─────────────────────────────────────────────────────────

    async def revoke(self, user_id: UUID) -> bool:
        return await self._repo.revoke(user_id, "bitbucket")

    # ── Helper interno ─────────────────────────────────────────────────────

    async def _persist_tokens(self, user_id: UUID, data: dict) -> dict:
        access_token  = data["access_token"]
        refresh_token = data.get("refresh_token")
        expires_in    = data.get("expires_in")  # segundos
        scopes        = data.get("scopes") or data.get("scope")

        expires_at = None
        if expires_in:
            expires_at = get_current_utc() + timedelta(seconds=int(expires_in))

        await self._repo.upsert(
            user_id=user_id,
            provider="bitbucket",
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            scopes=scopes,
            token_type=data.get("token_type", "Bearer"),
        )
        logger.info(f"[bb_oauth] Tokens persistidos para user={user_id}, expires_at={expires_at}")

        return {
            "access_token":  access_token,
            "refresh_token": refresh_token,
            "expires_at":    expires_at,
        }
