# =============================================================================
# src/providers/cancellable_stream.py
# Cancellable Stream Mixin for LLM Providers
# =============================================================================
"""
Mixin para agregar soporte de cancelación a streams de providers
Permite cancelar streams de manera segura y consistente
"""
import asyncio
from typing import AsyncGenerator, Any, Optional
from abc import ABC, abstractmethod

from src.utils.logger import get_logger


logger = get_logger(__name__)


class CancellableStreamMixin(ABC):
    """
    Mixin que agrega soporte de cancelación a streams de providers
    Los providers que heredan de este mixin pueden ser cancelados
    """

    @abstractmethod
    async def stream_chat(
        self,
        messages,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Método base de streaming (debe ser implementado por el provider)"""
        pass

    async def cancellable_stream_chat(
        self,
        messages,
        model: str,
        cancel_event: asyncio.Event,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        cancel_check_interval: float = 0.1,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Versión cancelable del stream_chat

        Args:
            messages: Lista de mensajes
            model: Modelo a usar
            cancel_event: Evento de asyncio para cancelación
            temperature: Temperatura para generación
            max_tokens: Máximo número de tokens
            cancel_check_interval: Intervalo para verificar cancelación (segundos)
            **kwargs: Argumentos adicionales del provider

        Yields:
            Chunks de texto del stream

        Raises:
            asyncio.CancelledError: Si el stream es cancelado
        """
        logger.debug(f"Starting cancellable stream for model {model}")

        # Verificar cancelación inicial
        if cancel_event.is_set():
            logger.info(f"Stream already cancelled before start for model {model}")
            raise asyncio.CancelledError("Stream cancelled before start")

        try:
            # Iterar directamente sobre el stream con verificación de cancelación
            async for chunk in self.stream_chat(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            ):
                # Verificar cancelación entre chunks
                if cancel_event.is_set():
                    logger.info(f"Stream cancelled mid-generation for model {model}")
                    raise asyncio.CancelledError("Stream cancelled by user")

                yield chunk

        except asyncio.CancelledError:
            logger.info(f"Stream cancelled for model {model}")
            raise
        except Exception as e:
            logger.error(f"Error in cancellable stream for model {model}: {e}", exc_info=True)
            raise


class CancellableProviderMixin(CancellableStreamMixin):
    """
    Mixin más completo que incluye métodos de utilidad para providers
    """

    async def cancellable_stream_with_timeout(
        self,
        messages,
        model: str,
        cancel_event: asyncio.Event,
        timeout: Optional[float] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream con timeout adicional además de cancelación
        """
        if timeout is None:
            # Timeout por defecto: 5 minutos
            timeout = 300.0

        start_time = asyncio.get_event_loop().time()

        try:
            async for chunk in self.cancellable_stream_chat(
                messages=messages,
                model=model,
                cancel_event=cancel_event,
                **kwargs
            ):
                # Verificar timeout entre chunks
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout:
                    logger.warning(f"Stream timed out after {timeout}s for model {model}")
                    raise asyncio.CancelledError(f"Stream timed out after {timeout}s")

                yield chunk

        except asyncio.TimeoutError:
            logger.warning(f"Stream timed out after {timeout}s for model {model}")
            raise asyncio.CancelledError(f"Stream timed out after {timeout}s")

    def supports_cancellation(self) -> bool:
        """Indica si este provider soporta cancelación"""
        return True

    def get_cancellation_overhead(self) -> float:
        """
        Retorna el overhead estimado de cancelación en porcentaje
        0.0 = sin overhead, 1.0 = 100% más lento
        """
        return 0.02  # 2% overhead estimado por verificación de cancelación