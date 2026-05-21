# =============================================================================
# src/api/v1/oauth_router.py
# OAuth 2.0 endpoints — actualmente: Bitbucket
# =============================================================================
"""
Endpoints:
  GET  /oauth/bitbucket/authorize  → redirige al proveedor
  GET  /oauth/bitbucket/callback   → recibe code + state, persiste tokens
  POST /oauth/bitbucket/revoke     → revoca tokens del usuario autenticado
  GET  /oauth/bitbucket/status     → informa si el usuario tiene OAuth activo
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import RedirectResponse

from src.api.v1.auth import _get_current_user
from src.schemas.auth.responses import MessageResponse
from src.services.auth.token_service import TokenService
from src.utils.service_factory import get_bb_oauth_service, get_token_service
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Estado temporal OAuth (en producción usar Redis o tabla oauth_state)
_pending_states: dict[str, str] = {}  # state → user_id (str)


async def get_redis_client():
    """Obtiene un cliente de Redis asíncrono si está disponible."""
    try:
        import redis.asyncio as redis
        from src.config.settings import settings
        return redis.from_url(settings.REDIS_URL)
    except Exception as e:
        logger.warning(f"[oauth_router] Redis no disponible para verificar estado: {e}")
        return None


# ---------------------------------------------------------------------------
# Authorize
# ---------------------------------------------------------------------------

@router.get(
    "/bitbucket/authorize",
    summary="Iniciar flujo OAuth con Bitbucket",
    response_class=RedirectResponse,
)
async def bitbucket_authorize(
    current_user=Depends(_get_current_user),
):
    async with get_bb_oauth_service() as svc:
        ts = get_token_service()
        state = ts.create_state_token()

        redis_client = await get_redis_client()
        if redis_client:
            try:
                async with redis_client:
                    await redis_client.setex(f"oauth_state:{state}", 600, str(current_user.id))
                    logger.info(f"[oauth] state persisted in redis for user={current_user.id}")
            except Exception as e:
                logger.warning(f"[oauth] failed to persist state in redis, falling back to memory: {e}")
                _pending_states[state] = str(current_user.id)
        else:
            _pending_states[state] = str(current_user.id)

        url = svc.get_authorization_url(state)
        logger.info(f"[oauth] authorize redirect user={current_user.id}")
        return RedirectResponse(url)


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

@router.get(
    "/bitbucket/callback",
    summary="Callback OAuth de Bitbucket (Bitbucket redirige aquí)",
)
async def bitbucket_callback(
    code: str  = Query(...),
    state: str = Query(...),
):
    user_id_str = None
    redis_client = await get_redis_client()
    if redis_client:
        try:
            async with redis_client:
                user_id_bytes = await redis_client.get(f"oauth_state:{state}")
                if user_id_bytes:
                    user_id_str = user_id_bytes.decode("utf-8")
                    await redis_client.delete(f"oauth_state:{state}")
        except Exception as e:
            logger.warning(f"[oauth] failed to read/delete state in redis, checking memory: {e}")

    # Fallback to in-memory dict if not found in Redis (or Redis failed)
    if not user_id_str:
        user_id_str = _pending_states.pop(state, None)

    if not user_id_str:
        return RedirectResponse("http://localhost:4200/?oauth_error=true&message=Estado+OAuth+invalido+o+expirado")

    from uuid import UUID
    user_id = UUID(user_id_str)

    try:
        async with get_bb_oauth_service() as svc:
            result = await svc.exchange_code(code, user_id)
    except Exception as e:
        logger.error(f"[oauth] Error exchanging code: {e}", exc_info=True)
        # Urlencode error message
        import urllib.parse
        error_msg = urllib.parse.quote(str(e))
        return RedirectResponse(f"http://localhost:4200/?oauth_error=true&message={error_msg}")

    return RedirectResponse("http://localhost:4200/?oauth_success=true&provider=bitbucket")



# ---------------------------------------------------------------------------
# Revoke
# ---------------------------------------------------------------------------

@router.post(
    "/bitbucket/revoke",
    response_model=MessageResponse,
    summary="Revocar tokens OAuth de Bitbucket del usuario actual",
)
async def bitbucket_revoke(
    current_user=Depends(_get_current_user),
):
    async with get_bb_oauth_service() as svc:
        revoked = await svc.revoke(current_user.id)
    if not revoked:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No se encontraron tokens OAuth de Bitbucket para este usuario.",
        )
    return MessageResponse(message="Tokens de Bitbucket revocados correctamente.")


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@router.get(
    "/bitbucket/status",
    summary="Verificar si el usuario tiene OAuth de Bitbucket activo",
)
async def bitbucket_status(
    current_user=Depends(_get_current_user),
):
    async with get_bb_oauth_service() as svc:
        token = await svc.get_valid_access_token(current_user.id)
    return {
        "provider":    "bitbucket",
        "authorized":  token is not None,
        "user_id":     str(current_user.id),
    }
