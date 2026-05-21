# =============================================================================
# src/utils/service_factory.py
# Service Factory — provee servicios configurados sin exponer la sesión DB
# =============================================================================
"""
Patrón: Service → Repository → AsyncSession

Todos los context managers:
  · Abren una sesión async limpia.
  · Instancian los repositorios necesarios.
  · Instancian el servicio con dichos repositorios.
  · Cierran la sesión al salir del bloque.

Uso estándar:
    async with get_tool_service() as svc:
        result = await svc.list_available_tools()
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Callable, TypeVar

from src.database.connection import get_async_db_session
from src.repositories import (
    CustomToolRepository,
    ConversationRepository,
    FileRepository,
    ToolConfigurationRepository,
)
from src.services.tool_service import ToolService
from src.services.utils.model_service import ModelService
from src.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# ToolService
# ---------------------------------------------------------------------------

@asynccontextmanager
async def get_tool_service() -> AsyncGenerator[ToolService, None]:
    async with get_async_db_session() as session:
        yield ToolService(
            custom_tool_repo=CustomToolRepository(session),
            tool_configuration_repo=ToolConfigurationRepository(session),
            conversation_repo=ConversationRepository(session),
            file_repo=FileRepository(session),
        )


async def with_tool_service(func: Callable[[ToolService], T]) -> T:
    async with get_tool_service() as svc:
        return await func(svc)


# ---------------------------------------------------------------------------
# ModelService
# ---------------------------------------------------------------------------

@asynccontextmanager
async def get_model_service() -> AsyncGenerator[ModelService, None]:
    from src.repositories import LLMModelRepository
    async with get_async_db_session() as session:
        yield ModelService(llm_model_repo=LLMModelRepository(session))


async def with_model_service(func: Callable[[ModelService], T]) -> T:
    async with get_model_service() as svc:
        return await func(svc)


# ---------------------------------------------------------------------------
# Provider sync (caso especial — expone la sesión directamente)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def get_session_for_provider_sync():
    async with get_async_db_session() as session:
        yield session


# ---------------------------------------------------------------------------
# AuthService
# ---------------------------------------------------------------------------

@asynccontextmanager
async def get_auth_service():
    """
    Provee un AuthService configurado.

    Uso:
        async with get_auth_service() as svc:
            return await svc.login(data, request)
    """
    from src.repositories.user_repository import UserRepository
    from src.repositories.role_repository import RoleRepository
    from src.repositories.token_repository import TokenRepository
    from src.services.auth.auth_service import AuthService

    async with get_async_db_session() as session:
        yield AuthService(
            user_repo=UserRepository(session),
            role_repo=RoleRepository(session),
            token_repo=TokenRepository(session),
        )


# ---------------------------------------------------------------------------
# PermissionService
# ---------------------------------------------------------------------------

@asynccontextmanager
async def get_permission_service():
    """
    Provee un PermissionService configurado.

    Uso (lifespan de main.py):
        async with get_permission_service() as svc:
            await svc.bootstrap_system_roles()
    """
    from src.repositories.role_repository import RoleRepository, PermissionRepository
    from src.repositories.user_repository import UserRepository
    from src.services.auth.permission_service import PermissionService

    async with get_async_db_session() as session:
        yield PermissionService(
            role_repo=RoleRepository(session),
            permission_repo=PermissionRepository(session),
            user_repo=UserRepository(session),
        )

# BitbucketOAuthService
@asynccontextmanager
async def get_bb_oauth_service():
    """
    Provee un BitbucketOAuthService configurado con Fernet + settings.

    Uso:
        async with get_bb_oauth_service() as svc:
            token = await svc.get_valid_access_token(user_id)
    """
    from src.repositories.external_oauth_token_repository import ExternalOAuthTokenRepository
    from src.services.oauth.bitbucket_oauth_service import BitbucketOAuthService
    from src.config.settings import settings

    async with get_async_db_session() as session:
        token_repo = ExternalOAuthTokenRepository(
            db=session,
            fernet_key=getattr(settings, "FERNET_KEY", None),
        )
        yield BitbucketOAuthService(
            client_id=settings.BITBUCKET_CLIENT_ID,
            client_secret=settings.BITBUCKET_CLIENT_SECRET,
            redirect_uri=settings.BITBUCKET_REDIRECT_URI,
            token_repo=token_repo,
        )


# TokenService singleton (stateless, no necesita DB)
def get_token_service() -> "TokenService":
    from src.services.auth.token_service import TokenService
    from src.config.settings import settings
    return TokenService(
        secret_key=settings.SECRET_KEY,
        algorithm=getattr(settings, "JWT_ALGORITHM", "HS256"),
        access_ttl_minutes=getattr(settings, "ACCESS_TOKEN_EXPIRE_MINUTES", 30),
        refresh_ttl_days=getattr(settings, "REFRESH_TOKEN_EXPIRE_DAYS", 7),
    )
