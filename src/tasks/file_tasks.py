
# =============================================================================
# src/tasks/file_tasks.py
# Celery Tasks para procesamiento asincrono de archivos
# =============================================================================
# Celery Tasks para procesamiento asincrono de archivos
# Usa Redis DB 1 y DB 2 - sin conflicto con DB 0
import asyncio
import logging
from datetime import datetime
from uuid import UUID

from src.config.celery import celery_app
from celery.exceptions import SoftTimeLimitExceeded

from src.database.unit_of_work import UnitOfWork
from src.dependencies import get_db_for_background_task
from src.models.models import ProcessingStatus
from src.services.processing.file_processor import FileProcessor, FileProcessingError

logger = logging.getLogger(__name__)

# Prefijo para notificaciones Redis (evita conflictos con DB 0)
REDIS_NOTIFICATION_PREFIX = "pythonchatbot:"

# Reuse a single event loop in the worker process to avoid cross-loop DB connections
_CELERY_LOOP = None


async def update_file_status(uow: UnitOfWork, file_id: UUID, status: ProcessingStatus, error: str = None):
    """Helper para actualizar estado del archivo"""
    try:
        file_record = await uow.files.get_by_id(file_id)
        if file_record:
            file_record.processing_status = status
            metadata = file_record.extra_metadata or {}
            metadata["last_processing_attempt"] = datetime.utcnow().isoformat()
            if error:
                metadata["processing_error"] = error
            file_record.extra_metadata = metadata
            await uow.commit()
    except Exception as e:
        logger.error(f"Failed to update file status: {e}")


def notify_frontend_status_change(
    file_id: str,
    status: ProcessingStatus,
    error: str = None
):
    """
    Notifica al frontend sobre cambio de estado via Redis Pub/Sub.

    Usa canal con prefijo 'pythonchatbot:' para evitar conflictos.
    """
    try:
        import redis
        from src.config.settings import settings

        redis_client = redis.Redis.from_url(settings.REDIS_URL)

        message = {
            "type": "file_processing_status",
            "file_id": file_id,
            "status": status.value,
            "error": error,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Canal con prefijo unico
        channel = f"{REDIS_NOTIFICATION_PREFIX}file_status:{file_id}"
        redis_client.publish(channel, str(message))

        logger.debug(f"Notified frontend about file {file_id} status: {status.value}")

    except Exception as e:
        logger.warning(f"Failed to notify frontend: {e}")
        # No fallar la tarea por esto


@celery_app.task(
    bind=True,
    name="tasks.process_file",
    max_retries=3,
    default_retry_delay=60,
    soft_time_limit=3300,
    acks_late=True
)
def process_file_task(self, file_id: str):
    """
    Procesa archivo de cualquier tipo de forma asincrona.

    Genera embeddings, indexa a Qdrant, y notifica al frontend.
    """
    async def _run():
        fid = UUID(file_id)
        db = await get_db_for_background_task()
        uow = UnitOfWork(db)
        processor = FileProcessor(
            uow.files, uow.conversations, uow.qdrant_collections
        )

        try:
            logger.info(f"Starting async file processing: {fid}")

            # Actualizar estado a PROCESSING
            await update_file_status(uow, fid, ProcessingStatus.PROCESSING)

            # Procesar archivo (embedding + indexacion)
            result = await processor.process_file(fid)

            # Actualizar estado a COMPLETED
            await update_file_status(uow, fid, ProcessingStatus.COMPLETED)

            # Notificar al frontend via WebSocket/Redis
            notify_frontend_status_change(str(fid), ProcessingStatus.COMPLETED)

            logger.info(f"File processing completed: {fid}, chunks: {result.get('chunks', 0)}")

            return {
                "status": "success",
                "file_id": str(fid),
                "chunks": result.get("chunks", 0),
                "completed_at": datetime.utcnow().isoformat()
            }

        except SoftTimeLimitExceeded:
            error_msg = "Task timeout exceeded"
            logger.error(f"Timeout processing file {fid}: {error_msg}")
            await update_file_status(uow, fid, ProcessingStatus.ERROR, error_msg)
            notify_frontend_status_change(str(fid), ProcessingStatus.ERROR, error_msg)
            raise

        except FileProcessingError as e:
            error_msg = str(e)
            logger.error(f"File processing error for {fid}: {error_msg}")
            await update_file_status(uow, fid, ProcessingStatus.ERROR, error_msg)
            notify_frontend_status_change(str(fid), ProcessingStatus.ERROR, error_msg)

            # Retry si no ha alcanzado el limite
            if self.request.retries < self.max_retries:
                raise self.retry(exc=e, countdown=60)
            else:
                logger.error(f"Max retries reached for file {fid}")
                return {
                    "status": "error",
                    "file_id": str(fid),
                    "error": error_msg
                }

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"Unexpected error processing file {fid}: {e}", exc_info=True)
            await update_file_status(uow, fid, ProcessingStatus.ERROR, error_msg)
            notify_frontend_status_change(str(fid), ProcessingStatus.ERROR, error_msg)
            raise

        finally:
            await db.close()

    global _CELERY_LOOP
    if _CELERY_LOOP is None:
        _CELERY_LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(_CELERY_LOOP)
    return _CELERY_LOOP.run_until_complete(_run())
