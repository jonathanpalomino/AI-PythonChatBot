
# =============================================================================
# src/config/celery.py
# Celery Application Configuration
# =============================================================================
# Celery Application Configuration
# Usa Redis DB 1 (broker) y DB 2 (results) - sin conflicto con DB 0 (cache)
import platform
from celery import Celery
from src.config.settings import settings

# Create Celery app
celery_app = Celery(
    "pythonchatbot",
    broker=settings.CELERY_BROKER_URL,       # ej: redis://localhost:6379/1
    backend=settings.CELERY_RESULT_BACKEND,  # ej: redis://localhost:6379/2
    include=["src.tasks.file_tasks", "src.tasks.cleanup_tasks"],
)

# Expose default attribute for Celery autodiscovery
app = celery_app
# Ensure this app becomes the default/current app in this process
celery_app.set_default()

# Configuration
celery_app.conf.update(
    # Serializacion
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Zona horaria
    timezone="UTC",
    enable_utc=True,

    # Ejecucion de tareas
    task_track_started=True,
    task_time_limit=3600,       # 1 hora max por tarea
    task_soft_time_limit=3300,  # 55 min

    # Reintentos
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_max_retries=3,
    default_retry_delay=60,     # 1 minuto

    # Worker
    worker_prefetch_multiplier=1,
    worker_concurrency=4,

    # Resultados
    result_expires=86400,  # 24 horas
    result_extended=True,

    # Redis especifico
    broker_transport_options={
        "max_connections": 50,
        "retry_on_timeout": True,
    },

    # Cola por defecto (alineada con el worker -Q default)
    task_default_queue="default",

    # Ruteo
    task_routes={
        "tasks.process_file": {"queue": "default"},
        "tasks.cleanup_expired_tokens": {"queue": "default"},
    },

    # Programación de tareas periódicas (Celery Beat)
    beat_schedule={
        "cleanup-expired-tokens-every-6-hours": {
            "task": "tasks.cleanup_expired_tokens",
            "schedule": 21600.0,  # 6 horas en segundos
        },
    },

    # Celery 6.x forward-compat: keep startup retry behavior
    broker_connection_retry_on_startup=True,
)

# Windows: use a safe pool to avoid billiard semaphore permission errors
if platform.system() == "Windows":
    celery_app.conf.update(
        worker_pool="solo",
        worker_concurrency=1,
    )


def get_celery_app() -> Celery:
    """Get Celery app instance"""
    return celery_app
