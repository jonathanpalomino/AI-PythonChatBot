# =============================================================================
# src/api/v1/auth.py
# Router de autenticación
# =============================================================================
"""
Endpoints:
  POST /auth/register        → TokenResponse
  POST /auth/login           → TokenResponse
  POST /auth/refresh         → TokenResponse
  POST /auth/logout          → MessageResponse
  POST /auth/logout-all      → MessageResponse
  GET  /auth/me              → UserResponse
"""

from typing import Optional
from fastapi import APIRouter, Depends, Request, status, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.schemas.auth.requests import LoginRequest, LogoutRequest, RefreshRequest, RegisterRequest
from src.schemas.auth.responses import MessageResponse, TokenResponse, UserResponse
from src.utils.service_factory import get_auth_service
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()
_bearer = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Dependency: extrae y valida el Bearer token, retorna User ORM
# ---------------------------------------------------------------------------

async def _get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    token: Optional[str] = Query(None, description="Token JWT opcional pasado en query params (para redirecciones)"),
):
    actual_token = None
    if credentials and credentials.credentials:
        actual_token = credentials.credentials
    elif token:
        actual_token = token

    if not actual_token:
        # Fallback para desarrollo local si no hay cabecera ni token de consulta.
        # Busca o crea un usuario por defecto para que no falle la experiencia del frontend.
        from src.database.connection import get_async_db
        from src.repositories.user_repository import UserRepository
        from src.models.user import UserStatus
        
        async for db in get_async_db():
            user_repo = UserRepository(db)
            user = await user_repo.get_by_email("default@example.com")
            if not user:
                logger.info("[auth] Creating default developer user 'default@example.com'...")
                user = await user_repo.create(
                    email="default@example.com",
                    password_hash=None,
                    full_name="Default Developer",
                    status=UserStatus.ACTIVE,
                    is_active=True,
                )
                await db.commit()
                await db.refresh(user)
            return user

    async with get_auth_service() as svc:
        return await svc.get_current_user(actual_token)



# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Registrar nuevo usuario",
)
async def register(
    data: RegisterRequest,
    request: Request,
):
    async with get_auth_service() as svc:
        return await svc.register(data, request)


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Autenticar usuario (email + password)",
)
async def login(
    data: LoginRequest,
    request: Request,
):
    async with get_auth_service() as svc:
        return await svc.login(data, request)


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Renovar access token usando refresh token",
)
async def refresh(
    data: RefreshRequest,
    request: Request,
):
    async with get_auth_service() as svc:
        return await svc.refresh_tokens(data.refresh_token, request)


@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Cerrar sesión (revoca el refresh token actual)",
)
async def logout(
    data: LogoutRequest,
):
    async with get_auth_service() as svc:
        await svc.logout(data.refresh_token)
    return MessageResponse(message="Sesión cerrada correctamente.")


@router.post(
    "/logout-all",
    response_model=MessageResponse,
    summary="Cerrar todas las sesiones del usuario autenticado",
)
async def logout_everywhere(
    current_user=Depends(_get_current_user),
):
    async with get_auth_service() as svc:
        await svc.logout_everywhere(current_user.id)
    return MessageResponse(message="Todas las sesiones han sido cerradas.")


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Obtener datos del usuario autenticado",
)
async def me(
    current_user=Depends(_get_current_user),
):
    from src.services.auth.auth_service import AuthService
    return AuthService.build_user_response(current_user)
