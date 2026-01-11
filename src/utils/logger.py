# =============================================================================
# src/utils/logger.py
# Professional Logging System (similar to Log4j/SLF4J)
# =============================================================================
"""
Sistema de logging profesional con:
- Archivos rotativos (rotating file handler)
- Patrones de formato personalizables
- Contexto de conversación (MDC-like con contextvars)
- Múltiples outputs (console + file)
"""
import logging
import sys
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

from src.config.settings import settings

# =============================================================================
# Context Variables (similar to MDC in Log4j)
# =============================================================================

# Context variables para tracking (thread-safe)
conversation_id_var: ContextVar[Optional[str]] = ContextVar('conversation_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)


# =============================================================================
# Custom Filter para agregar contexto
# =============================================================================

class ContextFilter(logging.Filter):
    """
    Filter que agrega contexto a los logs (similar a MDC en Java)
    """

    def filter(self, record):
        # Agregar conversation_id (chat_id)
        record.conversation_id = conversation_id_var.get() or 'N/A'
        record.user_id = user_id_var.get() or 'N/A'
        record.request_id = request_id_var.get() or 'N/A'
        return True


class CustomFormatter(logging.Formatter):
    """
    Custom formatter that automatically adds 'extra' fields to the log message
    """

    def format(self, record):
        # Format the standard message first
        s = super().format(record)

        # Identify extra fields (those not in standard LogRecord attributes)
        extra_fields = {
            k: v for k, v in record.__dict__.items()
            if k not in [
                'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
                'funcName', 'levelname', 'levelno', 'lineno', 'module',
                'msecs', 'message', 'msg', 'name', 'pathname', 'process',
                'processName', 'relativeCreated', 'stack_info', 'thread',
                'threadName', 'conversation_id', 'user_id', 'request_id'
            ]
        }

        if extra_fields:
            # Append extra fields as JSON-like string
            try:
                import json
                # Use default=str to handle non-serializable objects (like UUIDs)
                extras_str = json.dumps(extra_fields, default=str, ensure_ascii=False)
                s = f"{s} | {extras_str}"
            except Exception:
                # Fallback if JSON serialization fails
                s = f"{s} | {extra_fields}"

        return s


# =============================================================================
# Configuración de Logging
# =============================================================================

def setup_logging(
    app_name: str = "RAG_Chatbot",
    log_dir: Path = Path("./logs"),
    log_level: str = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Configura el sistema de logging profesional

    Args:
        app_name: Nombre de la aplicación
        log_dir: Directorio para archivos de log
        log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR)
        max_bytes: Tamaño máximo de archivo antes de rotar (default 10MB)
        backup_count: Número de archivos de backup a mantener

    Returns:
        Logger configurado
    """
    # Crear directorio de logs
    log_dir.mkdir(parents=True, exist_ok=True)

    # Nivel de logging
    level = getattr(logging, log_level or settings.LOG_LEVEL, logging.INFO)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Patrón de formato (similar a Log4j pattern)
    # Formato: [timestamp] [level] [logger] [conversation_id] - message
    log_format = (
        '[%(asctime)s] [%(levelname)-8s] [%(name)-20s] '
        '[conv:%(conversation_id)s] [req:%(request_id)s] - '
        '%(message)s'
    )

    # Formatter
    formatter = CustomFormatter(
        fmt=log_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # =============================================================================
    # Handler 1: Console (stdout)
    # =============================================================================
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(ContextFilter())

    # =============================================================================
    # Handler 2: Rotating File Handler (por tamaño)
    # =============================================================================
    app_log_file = log_dir / f"{app_name.lower()}.log"
    file_handler = RotatingFileHandler(
        filename=app_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(ContextFilter())

    # =============================================================================
    # Handler 3: Error Log (solo errores)
    # =============================================================================
    error_log_file = log_dir / f"{app_name.lower()}_error.log"
    error_handler = RotatingFileHandler(
        filename=error_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    error_handler.addFilter(ContextFilter())

    # =============================================================================
    # Handler 4: Daily Rotating File (por día)
    # =============================================================================
    daily_log_file = log_dir / f"{app_name.lower()}_daily.log"
    daily_handler = TimedRotatingFileHandler(
        filename=daily_log_file,
        when='midnight',
        interval=1,
        backupCount=30,  # Mantener 30 días
        encoding='utf-8'
    )
    daily_handler.setLevel(level)
    daily_handler.setFormatter(formatter)
    daily_handler.addFilter(ContextFilter())
    daily_handler.suffix = "%Y-%m-%d"  # Sufijo de fecha
    
    # =============================================================================
    # Handler 5: Chat-specific log file
    # =============================================================================
    chat_log_file = log_dir / f"{app_name.lower()}_chat.log"
    chat_handler = RotatingFileHandler(
        filename=chat_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    chat_handler.setLevel(level)
    chat_handler.setFormatter(formatter)
    chat_handler.addFilter(ContextFilter())
    
    # =============================================================================
    # Handler 6: Tools-specific log file
    # =============================================================================
    tools_log_file = log_dir / f"{app_name.lower()}_tools.log"
    tools_handler = RotatingFileHandler(
        filename=tools_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    tools_handler.setLevel(level)
    tools_handler.setFormatter(formatter)
    tools_handler.addFilter(ContextFilter())
    
    # Limpiar handlers existentes y agregar nuevos
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(daily_handler)
    root_logger.addHandler(chat_handler)
    root_logger.addHandler(tools_handler)

    # Log inicial
    root_logger.info(f"Logging system initialized - Level: {logging.getLevelName(level)}")
    root_logger.info(f"Log directory: {log_dir.absolute()}")

    return root_logger


# =============================================================================
# Helper Functions para Context Management
# =============================================================================

def set_conversation_context(conversation_id: str):
    """Establece el conversation_id en el contexto (similar a MDC.put en Java)"""
    conversation_id_var.set(str(conversation_id))


def set_user_context(user_id: str):
    """Establece el user_id en el contexto"""
    user_id_var.set(str(user_id))


def set_request_context(request_id: str):
    """Establece el request_id en el contexto"""
    request_id_var.set(str(request_id))


def clear_context():
    """Limpia todo el contexto (similar a MDC.clear en Java)"""
    conversation_id_var.set(None)
    user_id_var.set(None)
    request_id_var.set(None)


# =============================================================================
# Context Manager para auto-cleanup
# =============================================================================

from contextlib import contextmanager


@contextmanager
def log_context(conversation_id: str = None, user_id: str = None, request_id: str = None):
    """
    Context manager para logging con auto-cleanup

    Uso:
        with log_context(conversation_id="abc-123"):
            logger.info("Processing message")  # Incluirá conversation_id
    """
    # Set context
    if conversation_id:
        set_conversation_context(conversation_id)
    if user_id:
        set_user_context(user_id)
    if request_id:
        set_request_context(request_id)

    try:
        yield
    finally:
        # Auto-cleanup
        clear_context()


# =============================================================================
# Utility Functions
# =============================================================================

def get_logger(name: str) -> logging.Logger:
    """
    Obtiene un logger con el nombre especificado

    Args:
        name: Nombre del logger (usualmente __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def get_chat_logger(name: str = "chat") -> logging.Logger:
    """
    Obtiene un logger específico para interacciones de chat

    Args:
        name: Nombre del logger (default: "chat")

    Returns:
        Logger instance configurado para chat
    """
    chat_logger = logging.getLogger(name)
    chat_logger.propagate = False  # Evitar duplicación de logs
    
    # Añadir handlers específicos para chat
    for handler in logging.getLogger().handlers:
        if isinstance(handler, RotatingFileHandler) and 'chat' in handler.baseFilename:
            chat_logger.addHandler(handler)
    
    # Añadir console handler también
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler):
            chat_logger.addHandler(handler)
    
    return chat_logger


def get_tools_logger(name: str = "tools") -> logging.Logger:
    """
    Obtiene un logger específico para herramientas (tools)

    Args:
        name: Nombre del logger (default: "tools")

    Returns:
        Logger instance configurado para tools
    """
    tools_logger = logging.getLogger(name)
    tools_logger.propagate = False  # Evitar duplicación de logs
    
    # Añadir handlers específicos para tools
    for handler in logging.getLogger().handlers:
        if isinstance(handler, RotatingFileHandler) and 'tools' in handler.baseFilename:
            tools_logger.addHandler(handler)
    
    # Añadir console handler también
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler):
            tools_logger.addHandler(handler)
    
    return tools_logger


def setup_loggers_for_packages():
    """
    Configura loggers específicos para paquetes/clases
    Similar al enfoque de Log4j donde se definen loggers por jerarquía
    """
    # Configurar logger para el paquete de chat
    chat_package_logger = logging.getLogger("src.api.v1.conversations")
    chat_package_logger.propagate = False
    chat_package_logger.setLevel(logging.INFO)
    
    # Añadir handlers específicos para chat
    for handler in logging.getLogger().handlers:
        if isinstance(handler, RotatingFileHandler) and 'chat' in handler.baseFilename:
            chat_package_logger.addHandler(handler)
        if isinstance(handler, logging.StreamHandler):
            chat_package_logger.addHandler(handler)
    
    # Configurar logger para el paquete de tools
    tools_package_logger = logging.getLogger("src.tools")
    tools_package_logger.propagate = False
    tools_package_logger.setLevel(logging.INFO)
    
    # Añadir handlers específicos para tools
    for handler in logging.getLogger().handlers:
        if isinstance(handler, RotatingFileHandler) and 'tools' in handler.baseFilename:
            tools_package_logger.addHandler(handler)
        if isinstance(handler, logging.StreamHandler):
            tools_package_logger.addHandler(handler)


# =============================================================================
# Initialize on import
# =============================================================================

# Auto-inicializar logging al importar el módulo
_root_logger = setup_logging()

# Configurar loggers para paquetes específicos
setup_loggers_for_packages()
