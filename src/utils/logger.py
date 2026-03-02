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
# Global Storage for Specialized Handlers
# =============================================================================

# Almacenamiento global para handlers especializados (no añadidos al root logger)
# Esto permite que cada logger especializado escriba SOLO en su archivo específico
_chat_handler: Optional[RotatingFileHandler] = None
_tools_handler: Optional[RotatingFileHandler] = None
_payload_request_handler: Optional[RotatingFileHandler] = None
_payload_response_handler: Optional[RotatingFileHandler] = None


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
    # NOTA: Este handler NO se añade al root logger para evitar que todos los
    # logs se escriban en este archivo. Se almacena en variable global para
    # uso exclusivo del logger especializado (chat).
    # =============================================================================
    global _chat_handler
    
    chat_log_file = log_dir / f"{app_name.lower()}_chat.log"
    _chat_handler = RotatingFileHandler(
        filename=chat_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    _chat_handler.setLevel(level)
    _chat_handler.setFormatter(formatter)
    _chat_handler.addFilter(ContextFilter())

    # =============================================================================
    # Handler 6: Tools-specific log file
    # NOTA: Este handler NO se añade al root logger para evitar que todos los
    # logs se escriban en este archivo. Se almacena en variable global para
    # uso exclusivo del logger especializado (tools).
    # =============================================================================
    global _tools_handler
    
    tools_log_file = log_dir / f"{app_name.lower()}_tools.log"
    _tools_handler = RotatingFileHandler(
        filename=tools_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    _tools_handler.setLevel(level)
    _tools_handler.setFormatter(formatter)
    _tools_handler.addFilter(ContextFilter())

    # =============================================================================
    # Handler 7: Payloads log file (LLM Requests/Responses)
    # NOTA: Estos handlers NO se añaden al root logger para evitar que todos los
    # logs se escriban en estos archivos. Se almacenan en variables globales para
    # uso exclusivo de los loggers especializados (payload_request y payload_response).
    # =============================================================================
    global _payload_request_handler, _payload_response_handler
    
    payloads_request_log_file = log_dir / f"{app_name.lower()}_payload_request.log"
    payloads_response_log_file = log_dir / f"{app_name.lower()}_payload_response.log"
    
    _payload_request_handler = RotatingFileHandler(
        filename=payloads_request_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    _payload_request_handler.setLevel(level)
    _payload_request_handler.setFormatter(formatter)
    _payload_request_handler.addFilter(ContextFilter())

    _payload_response_handler = RotatingFileHandler(
        filename=payloads_response_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    _payload_response_handler.setLevel(level)
    _payload_response_handler.setFormatter(formatter)
    _payload_response_handler.addFilter(ContextFilter())

    # Limpiar handlers existentes y agregar nuevos
    # NOTA: Los handlers especializados (chat, tools, payload_*) NO se añaden al root logger
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(daily_handler)
    # _chat_handler, _tools_handler, _payload_request_handler, _payload_response_handler
    # NO se añaden al root logger

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


# =============================================================================
# GENERIC LOGGER FACTORY
# =============================================================================

def create_specialized_logger(
    name: str,
    handler_keyword: str,
    include_console: bool = True,
    log_level: int = None
) -> logging.Logger:
    """
    Generic factory function for creating specialized loggers with specific handlers.

    Args:
        name: Name of the logger
        handler_keyword: Keyword to match in handler baseFilename (e.g., 'chat', 'tools', 'payload_request')
        include_console: Whether to include console handler (default: True)
        log_level: Optional log level to set (default: None, uses root logger level)

    Returns:
        Logger instance configured with specific handlers

    Example:
        chat_logger = create_specialized_logger("chat", "chat", include_console=True)
        tools_logger = create_specialized_logger("tools", "tools", include_console=True)
        payload_request_logger = create_specialized_logger("payload_request", "payload_request", include_console=False)
    """
    logger = logging.getLogger(name)
    logger.propagate = False  # Evitar duplicación de logs

    # Set log level if specified
    if log_level is not None:
        logger.setLevel(log_level)

    # Obtener handlers ya añadidos para evitar duplicados
    existing_handler_ids = {id(h) for h in logger.handlers}

    # Añadir handlers específicos basados en keyword (búsqueda exacta del patrón)
    for handler in logging.getLogger().handlers:
        if isinstance(handler, RotatingFileHandler):
            # Usar búsqueda más estricta: el keyword debe estar presente como parte del nombre del archivo
            # Evita coincidencias parciales como "payload" coincidiendo con "payload_request" y "payload_response"
            if handler_keyword in handler.baseFilename and id(handler) not in existing_handler_ids:
                logger.addHandler(handler)

    # Añadir console handler si se solicita
    if include_console:
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, RotatingFileHandler):
                if id(handler) not in existing_handler_ids:
                    logger.addHandler(handler)

    return logger


# =============================================================================
# SPECIALIZED LOGGER WRAPPERS
# =============================================================================

def _create_isolated_logger(
    name: str,
    handler: Optional[RotatingFileHandler],
    include_console: bool = True
) -> logging.Logger:
    """
    Función interna para crear loggers aislados con handlers específicos.
    
    El nivel del logger se hereda del root logger, por lo que respeta la
    configuración global (settings.LOG_LEVEL). Si el nivel global es DEBUG,
    todos los mensajes DEBUG y superiores se mostrarán.
    
    Args:
        name: Nombre del logger
        handler: Handler específico para el archivo de log (puede ser None)
        include_console: Si True, añade el console handler del root logger
    
    Returns:
        Logger configurado con propagate=False para aislamiento
    """
    logger = logging.getLogger(name)
    logger.propagate = False  # Evitar que los logs se propaguen al root logger
    
    # NO establecer nivel fijo - heredar del root logger
    # El root logger ya tiene el nivel configurado desde settings.LOG_LEVEL
    # Esto permite que DEBUG, INFO, WARNING, ERROR funcionen según configuración global
    
    # Obtener handlers ya añadidos para evitar duplicados
    handler_ids = {id(h) for h in logger.handlers}
    
    # Añadir el handler específico si está disponible
    if handler is not None and id(handler) not in handler_ids:
        logger.addHandler(handler)
        handler_ids.add(id(handler))
    
    # Añadir console handler si se solicita
    if include_console:
        for root_handler in logging.getLogger().handlers:
            if isinstance(root_handler, logging.StreamHandler) and not isinstance(root_handler, RotatingFileHandler):
                if id(root_handler) not in handler_ids:
                    logger.addHandler(root_handler)
    
    return logger


def get_chat_logger(name: str = "chat") -> logging.Logger:
    """
    Obtiene un logger específico para interacciones de chat.
    
    Este logger escribe EXCLUSIVAMENTE en el archivo de chat y no
    propaga al root logger, evitando que los logs se mezclen con otros archivos.

    Args:
        name: Nombre del logger (default: "chat")

    Returns:
        Logger instance configurado para chat
    """
    global _chat_handler
    return _create_isolated_logger(name, _chat_handler, include_console=True)


def get_tools_logger(name: str = "tools") -> logging.Logger:
    """
    Obtiene un logger específico para herramientas (tools).
    
    Este logger escribe EXCLUSIVAMENTE en el archivo de tools y no
    propaga al root logger, evitando que los logs se mezclen con otros archivos.

    Args:
        name: Nombre del logger (default: "tools")

    Returns:
        Logger instance configurado para tools
    """
    global _tools_handler
    return _create_isolated_logger(name, _tools_handler, include_console=True)


def get_payload_request_logger(name: str = "payload_request") -> logging.Logger:
    """
    Obtiene un logger específico para Payloads de tipo Request (LLM Requests).
    
    Este logger escribe EXCLUSIVAMENTE en el archivo de payload_request y no
    propaga al root logger, evitando que los logs se mezclen con otros archivos.

    Args:
        name: Nombre del logger (default: "payload_request")

    Returns:
        Logger instance configurado para payload requests
    """
    global _payload_request_handler
    return _create_isolated_logger(name, _payload_request_handler, include_console=False)


def get_payload_response_logger(name: str = "payload_response") -> logging.Logger:
    """
    Obtiene un logger específico para Payloads de tipo Response (LLM Responses).
    
    Este logger escribe EXCLUSIVAMENTE en el archivo de payload_response y no
    propaga al root logger, evitando que los logs se mezclen con otros archivos.

    Args:
        name: Nombre del logger (default: "payload_response")

    Returns:
        Logger instance configurado para payload responses
    """
    global _payload_response_handler
    return _create_isolated_logger(name, _payload_response_handler, include_console=False)


def setup_loggers_for_packages():
    """
    Configura loggers específicos para paquetes/clases.
    Similar al enfoque de Log4j donde se definen loggers por jerarquía.
    Usa _create_isolated_logger para asegurar que cada paquete escriba
    solo en su archivo de log correspondiente.
    """
    # Configurar logger para conversaciones (usa chat handler)
    _create_isolated_logger("src.api.v1.conversations", _chat_handler, include_console=True)
    
    # Configurar logger para tools (usa tools handler)
    _create_isolated_logger("src.tools", _tools_handler, include_console=True)


# =============================================================================
# Initialize on import
# =============================================================================

# Auto-inicializar logging al importar el módulo
_root_logger = setup_logging()

# Configurar loggers para paquetes específicos
setup_loggers_for_packages()
