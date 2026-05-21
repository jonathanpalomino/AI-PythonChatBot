# =============================================================================
# src/services/auth/auth_service.py
# Servicio de autenticación: register, login, refresh, logout
# =============================================================================
"""
Dependencias externas requeridas:
  pip install python-jose[cryptography] passlib[bcrypt]

Variables de settings requeridas:
  SECRET_KEY                  str   — clave HMAC para JWT
  JWT_ALGORITHM               str   — default "HS256"
  ACCESS_TOKEN_EXPIRE_MINUTES int   — default 30
  REFRESH_TOKEN_EXPIRE_DAYS   int   — default 7
"""

import secrets
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from fastapi import Request
from jose import jwt, JWTError
from passlib.context import CryptContext

from src.config.settings import settings
from src.models.user import User, UserStatus
from src.repositories.role_repository import RoleRepository
from src.repositories.token_repository import TokenRepository
from src.repositories.user_repository import UserRepository
from src.schemas.auth.requests import LoginRequest, RegisterRequest
from src.schemas.auth.responses import TokenResponse, UserResponse
from src.utils.date_utils import get_current_utc
from src.utils.logger import get_logger

logger = get_logger(__name__)

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------------------------------------------------------------------------
# Constantes de configuración
# ---------------------------------------------------------------------------

_ALGORITHM: str = getattr(settings, "JWT_ALGORITHM", "HS256")
_ACCESS_EXPIRE_MIN: int = getattr(settings, "ACCESS_TOKEN_EXPIRE_MINUTES", 30)
_REFRESH_EXPIRE_DAYS: int = getattr(settings, "REFRESH_TOKEN_EXPIRE_DAYS", 7)


# ---------------------------------------------------------------------------
# AuthService
# ---------------------------------------------------------------------------

class AuthService:
    """
    Orquesta el ciclo de autenticación JWT + refresh token persistido.

    Constructor:
        user_repo  — UserRepository
        role_repo  — RoleRepository
        token_repo — TokenRepository
    """

    def __init__(
        self,
        user_repo: UserRepository,
        role_repo: RoleRepository,
        token_repo: TokenRepository,
    ):
        self._users = user_repo
        self._roles = role_repo
        self._tokens = token_repo

    # ------------------------------------------------------------------
    # Público: register
    # ------------------------------------------------------------------

    async def register(
        self,
        data: RegisterRequest,
        request: Optional[Request] = None,
    ) -> TokenResponse:
        email = data.email.lower().strip()

        if await self._users.email_exists(email):
            from fastapi import HTTPException, status
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="El email ya está registrado.",
            )

        user = await self._users.create(
            email=email,
            password_hash=_pwd_context.hash(data.password),
            full_name=data.full_name,
            status=UserStatus.ACTIVE,
            is_active=True,
        )

        # Asignar rol por defecto si existe
        default_role = await self._roles.get_by_name("user")
        if default_role:
            await self._roles.assign_role_to_user(user.id, default_role.id)

        return await self._issue_token_pair(user, request)

    # ------------------------------------------------------------------
    # Público: login
    # ------------------------------------------------------------------

    async def login(
        self,
        data: LoginRequest,
        request: Optional[Request] = None,
    ) -> TokenResponse:
        from fastapi import HTTPException, status

        user = await self._users.get_by_email(data.email.lower().strip())
        _INVALID = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales inválidas.",
        )

        if not user or not user.is_active:
            raise _INVALID
        if not user.password_hash:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Esta cuenta usa autenticación OAuth. Usa el proveedor correspondiente.",
            )
        if not _pwd_context.verify(data.password, user.password_hash):
            raise _INVALID

        # Actualizar auditoría
        user.last_login_at = get_current_utc()
        user.login_count += 1
        await self._users.flush()

        return await self._issue_token_pair(user, request)

    # ------------------------------------------------------------------
    # Público: refresh
    # ------------------------------------------------------------------

    async def refresh_tokens(
        self,
        raw_refresh_token: str,
        request: Optional[Request] = None,
    ) -> TokenResponse:
        from fastapi import HTTPException, status

        _INVALID = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token inválido o expirado.",
        )

        # Extraer user_id del JWT sin verificar expiración del access token
        try:
            payload = jwt.decode(
                raw_refresh_token,
                settings.SECRET_KEY,
                algorithms=[_ALGORITHM],
                options={"verify_exp": False},
            )
            user_id = UUID(payload["sub"])
        except (JWTError, KeyError, ValueError):
            raise _INVALID

        user = await self._users.get_by_id(user_id)
        if not user or not user.is_active:
            raise _INVALID

        token_record = await self._tokens.get_valid_token(
            raw_refresh_token, user.token_version
        )
        if not token_record:
            raise _INVALID

        # Revocar token usado (rotación)
        await self._tokens.revoke_token(raw_refresh_token)

        return await self._issue_token_pair(user, request)

    # ------------------------------------------------------------------
    # Público: logout
    # ------------------------------------------------------------------

    async def logout(self, raw_refresh_token: str) -> None:
        await self._tokens.revoke_token(raw_refresh_token)

    async def logout_everywhere(self, user_id: UUID) -> None:
        """Invalida todos los refresh tokens + incrementa token_version."""
        await self._tokens.revoke_all_for_user(user_id)
        await self._users.increment_token_version(user_id)

    # ------------------------------------------------------------------
    # Público: get_current_user (para Depends en FastAPI)
    # ------------------------------------------------------------------

    async def get_current_user(self, token: str) -> User:
        from fastapi import HTTPException, status

        _CRED_EXCEPTION = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido o expirado.",
            headers={"WWW-Authenticate": "Bearer"},
        )
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[_ALGORITHM])
            user_id: str = payload.get("sub")
            if not user_id:
                raise _CRED_EXCEPTION
        except JWTError:
            raise _CRED_EXCEPTION

        user = await self._users.get_by_id_with_roles(UUID(user_id))
        if not user or not user.is_active:
            raise _CRED_EXCEPTION
        return user

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _create_access_token(self, user: User) -> str:
        expire = get_current_utc() + timedelta(minutes=_ACCESS_EXPIRE_MIN)
        payload = {
            "sub": str(user.id),
            "email": user.email,
            "exp": expire,
            "type": "access",
            "ver": user.token_version,
        }
        return jwt.encode(payload, settings.SECRET_KEY, algorithm=_ALGORITHM)

    def _create_refresh_token_raw(self) -> str:
        """Genera un token opaco aleatorio (no JWT) para el refresh."""
        return secrets.token_urlsafe(64)

    async def _issue_token_pair(
        self, user: User, request: Optional[Request]
    ) -> TokenResponse:
        raw_refresh = self._create_refresh_token_raw()
        expires_at = get_current_utc() + timedelta(days=_REFRESH_EXPIRE_DAYS)

        ua = request.headers.get("user-agent") if request else None
        ip = request.client.host if request and request.client else None

        await self._tokens.create_refresh_token(
            user_id=user.id,
            raw_token=raw_refresh,
            token_version=user.token_version,
            expires_at=expires_at,
            user_agent=ua,
            ip_address=ip,
        )

        access = self._create_access_token(user)
        return TokenResponse(
            access_token=access,
            refresh_token=raw_refresh,
            expires_in=_ACCESS_EXPIRE_MIN * 60,
        )

    # ------------------------------------------------------------------
    # Helper para construir UserResponse desde User ORM
    # ------------------------------------------------------------------

    @staticmethod
    def build_user_response(user: User) -> UserResponse:
        return UserResponse(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            is_active=user.is_active,
            is_superuser=user.is_superuser,
            email_verified=user.email_verified,
            roles=user.role_names,
            created_at=user.created_at,
        )
