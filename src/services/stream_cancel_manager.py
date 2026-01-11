# =============================================================================
# src/services/stream_cancel_manager.py
# Stream Cancellation Management
# =============================================================================
"""
Gestor de cancelación de streams de chat
Permite cancelar streams activos de manera segura y eficiente
"""
import asyncio
import threading
import weakref
from typing import Dict, Optional
from uuid import UUID
from dataclasses import dataclass, field
from datetime import datetime

from src.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class StreamCancelToken:
    """Token para controlar la cancelación de un stream"""
    conversation_id: UUID
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    task: Optional[asyncio.Task] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    cancelled_by: Optional[str] = None  # "user", "disconnect", "timeout", None

    def cancel(self, reason: str = "user") -> bool:
        """Cancela el stream"""
        if self.cancel_event.is_set():
            return False  # Ya estaba cancelado

        self.cancelled_by = reason
        self.cancel_event.set()

        if self.task and not self.task.done():
            self.task.cancel()

        logger.info(
            f"Stream cancelled: conversation_id={self.conversation_id}, reason={reason}"
        )
        return True

    def is_cancelled(self) -> bool:
        """Verifica si el stream está cancelado"""
        return self.cancel_event.is_set()

    async def wait_for_cancel(self, timeout: Optional[float] = None) -> bool:
        """Espera a que el stream sea cancelado"""
        try:
            await asyncio.wait_for(self.cancel_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False


class StreamCancelManager:
    """
    Gestor global de streams activos
    Maneja el registro, cancelación y cleanup de streams
    """

    def __init__(self):
        self._active_streams: Dict[UUID, StreamCancelToken] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        # NOTE: No iniciamos cleanup task aquí porque no hay event loop activo durante import
        # Se iniciará lazy cuando se registre el primer stream

    def _start_cleanup_task(self):
        """Inicia la tarea de cleanup automático (lazy initialization)"""
        try:
            # Solo iniciar si hay un event loop activo
            loop = asyncio.get_running_loop()
            if self._cleanup_task is None or self._cleanup_task.done():
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        except RuntimeError:
            # No hay event loop activo, se iniciará cuando se registre el primer stream
            pass

    async def _cleanup_loop(self):
        """Loop de cleanup automático para streams abandonados"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(30)  # Cleanup cada 30 segundos

                current_time = datetime.utcnow()
                to_remove = []

                async with self._lock:
                    for conv_id, token in self._active_streams.items():
                        # Remover streams que llevan más de 5 minutos
                        if (current_time - token.created_at).total_seconds() > 300:
                            to_remove.append(conv_id)
                            logger.warning(
                                f"Removing abandoned stream: conversation_id={conv_id}, "
                                f"age={(current_time - token.created_at).total_seconds():.0f}s"
                            )

                # Remover fuera del lock para evitar deadlocks
                for conv_id in to_remove:
                    await self._remove_stream(conv_id)

            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}", exc_info=True)

    async def register_stream(self, conversation_id: UUID) -> StreamCancelToken:
        """Registra un nuevo stream activo"""
        # Asegurar que el cleanup task esté iniciado (lazy initialization)
        self._start_cleanup_task()
        
        async with self._lock:
            # Si ya existe un stream para esta conversación, cancelarlo primero
            if conversation_id in self._active_streams:
                old_token = self._active_streams[conversation_id]
                old_token.cancel("replaced")
                logger.info(f"Cancelled existing stream for conversation: {conversation_id}")

            # Crear nuevo token
            token = StreamCancelToken(conversation_id=conversation_id)
            self._active_streams[conversation_id] = token

            logger.info(f"Registered new stream: conversation_id={conversation_id}")
            return token

    async def cancel_stream(self, conversation_id: UUID, reason: str = "user") -> bool:
        """Cancela un stream específico"""
        async with self._lock:
            token = self._active_streams.get(conversation_id)
            if token:
                return token.cancel(reason)
            else:
                logger.warning(f"Attempted to cancel non-existent stream: {conversation_id}")
                return False

    async def get_stream_token(self, conversation_id: UUID) -> Optional[StreamCancelToken]:
        """Obtiene el token de un stream activo"""
        async with self._lock:
            return self._active_streams.get(conversation_id)

    async def unregister_stream(self, conversation_id: UUID):
        """Desregistra un stream (llamado al finalizar)"""
        await self._remove_stream(conversation_id)

    async def _remove_stream(self, conversation_id: UUID):
        """Remueve un stream del registro"""
        async with self._lock:
            if conversation_id in self._active_streams:
                token = self._active_streams.pop(conversation_id)
                logger.debug(f"Unregistered stream: conversation_id={conversation_id}")

                # Cancelar el token si aún no lo está
                if not token.is_cancelled():
                    token.cancel("cleanup")

    async def cancel_all_streams(self, reason: str = "shutdown"):
        """Cancela todos los streams activos (para shutdown graceful)"""
        async with self._lock:
            active_ids = list(self._active_streams.keys())

        cancelled_count = 0
        for conv_id in active_ids:
            if await self.cancel_stream(conv_id, reason):
                cancelled_count += 1

        logger.info(f"Cancelled {cancelled_count} active streams due to {reason}")
        return cancelled_count

    async def get_active_streams_count(self) -> int:
        """Obtiene el número de streams activos"""
        async with self._lock:
            return len(self._active_streams)

    async def get_active_stream_ids(self) -> list[UUID]:
        """Obtiene la lista de IDs de conversaciones con streams activos"""
        async with self._lock:
            return list(self._active_streams.keys())

    async def shutdown(self):
        """Shutdown graceful del manager"""
        logger.info("Shutting down StreamCancelManager")

        # Señalar shutdown
        self._shutdown_event.set()

        # Cancelar todos los streams activos
        await self.cancel_all_streams("shutdown")

        # Cancelar la tarea de cleanup
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info("StreamCancelManager shutdown complete")


# =============================================================================
# Global Manager Instance
# =============================================================================

# Crear instancia global del manager
stream_cancel_manager = StreamCancelManager()


# =============================================================================
# Cleanup on exit
# =============================================================================

import atexit

@atexit.register
def cleanup_streams():
    """Cleanup de streams al salir (sincrónico para atexit)"""
    try:
        # Crear un event loop si no existe
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Ejecutar cleanup
        if loop.is_running():
            # Si el loop ya está corriendo, programar el cleanup
            asyncio.create_task(stream_cancel_manager.cancel_all_streams("exit"))
        else:
            # Ejecutar sincrónicamente
            loop.run_until_complete(stream_cancel_manager.cancel_all_streams("exit"))

    except Exception as e:
        # Logging básico si el logger no está disponible
        print(f"Error during stream cleanup: {e}")