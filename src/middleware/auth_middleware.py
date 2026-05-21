# =============================================================================
# src/middleware/auth_middleware.py
# FastAPI dependencies: get_current_user, require_permission
# =============================================================================
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError

from src.models.auth_models import ResourceType, ActionType
from src.models.user import User
from src.services.auth.token_service import get_token_service, TokenExpiredError, TokenInvalidError
from src.utils.logger import get_logger

logger = get_logger(__name__)

_bearer = HTTPBearer(auto_error=True)
_optional_bearer = HTTPBearer(auto_error=False)


async def get_token_payload(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer)],
) -> dict:
    """Decode Bearer JWT — no DB hit (fast path)."""
    try:
        return get_token_service().verify_access_token(credentials.credentials)
    except (TokenExpiredError, TokenInvalidError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user_from_db(
    payload: Annotated[dict, Depends(get_token_payload)],
) -> User:
    """Resolve full User ORM object from JWT sub. Use only when ORM access is needed."""
    from src.database.connection import AsyncSessionLocal
    from src.repositories.user_repository import UserRepository

    user_id_str = payload.get("sub")
    if not user_id_str:
        raise HTTPException(status_code=401, detail="Token missing subject")
    try:
        user_id = UUID(user_id_str)
    except ValueError:
        raise HTTPException(status_code=401, detail="Malformed token subject")

    async with AsyncSessionLocal() as session:
        repo = UserRepository(session)
        user = await repo.get_with_roles(user_id)

    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    if user.status.value != "active":
        raise HTTPException(status_code=403, detail="Account inactive")
    return user


def require_permission(resource: ResourceType, action: ActionType):
    """
    Dependency factory — checks permission from JWT payload (no DB hit).

    Usage:
        @router.get("/x", dependencies=[Depends(require_permission(ResourceType.TOOL, ActionType.READ))])
    """
    async def _check(payload: Annotated[dict, Depends(get_token_payload)]) -> dict:
        if payload.get("is_superuser"):
            return payload
        permissions: list[str] = payload.get("permissions", [])
        needed = f"{resource.value}:{action.value}"
        if not any(p in permissions for p in (
            needed,
            f"*:{action.value}",
            f"{resource.value}:*",
            "*:*",
        )):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {needed}",
            )
        return payload
    return _check


async def require_superuser(payload: Annotated[dict, Depends(get_token_payload)]) -> dict:
    if not payload.get("is_superuser"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Superuser access required")
    return payload


async def get_optional_token_payload(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_optional_bearer)],
) -> dict | None:
    if not credentials:
        return None
    try:
        return get_token_service().verify_access_token(credentials.credentials)
    except (TokenExpiredError, TokenInvalidError):
        return None
