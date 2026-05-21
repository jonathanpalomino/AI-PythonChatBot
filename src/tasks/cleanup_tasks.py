# =============================================================================
# src/tasks/cleanup_tasks.py
# Celery Tasks para mantenimiento y limpieza periódica
# =============================================================================
import asyncio
import logging

from sqlalchemy import delete

from src.config.celery import celery_app
from src.dependencies import get_db_for_background_task
from src.models import RefreshToken, ExternalOAuthToken
from src.utils.date_utils import get_current_utc

logger = logging.getLogger(__name__)

# Reutilizar loop de eventos
_CLEANUP_LOOP = None

@celery_app.task(
    bind=True,
    name="tasks.cleanup_expired_tokens",
    max_retries=1,
    acks_late=True
)
def cleanup_expired_tokens_task(self):
    """
    Tarea periódica para limpiar tokens de sesión y OAuth expirados o revocados.
    """
    async def _run():
        db = await get_db_for_background_task()
        try:
            now = get_current_utc()
            
            # 1. Limpiar refresh tokens internos expirados o revocados
            stmt_refresh = delete(RefreshToken).where(
                (RefreshToken.expires_at <= now) | (RefreshToken.is_revoked == True)
            )
            res_refresh = await db.execute(stmt_refresh)
            deleted_refresh = res_refresh.rowcount
            
            # 2. Limpiar tokens OAuth externos inactivos/revocados
            stmt_oauth = delete(ExternalOAuthToken).where(
                ExternalOAuthToken.is_active == False
            )
            res_oauth = await db.execute(stmt_oauth)
            deleted_oauth = res_oauth.rowcount
            
            await db.commit()
            
            logger.info(
                f"[cleanup] Purged {deleted_refresh} expired/revoked refresh tokens "
                f"and {deleted_oauth} inactive OAuth tokens."
            )
            return {
                "status": "success",
                "deleted_refresh_tokens": deleted_refresh,
                "deleted_oauth_tokens": deleted_oauth,
                "timestamp": now.isoformat()
            }
        except Exception as e:
            logger.error(f"[cleanup] Error executing cleanup task: {e}", exc_info=True)
            await db.rollback()
            raise
        finally:
            await db.close()

    global _CLEANUP_LOOP
    if _CLEANUP_LOOP is None:
        _CLEANUP_LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(_CLEANUP_LOOP)
    return _CLEANUP_LOOP.run_until_complete(_run())
