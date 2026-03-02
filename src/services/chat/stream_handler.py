# =============================================================================
# src/services/chat/stream_handler.py
# Stream Handler - Streaming support (REFACTORED from chat_orchestrator.py)
# =============================================================================
"""
StreamHandler: Responsable de manejar streaming de respuestas.

Responsabilidades (SRP):
- Manejar streaming de LLM
- Manejar cancelación de streams
- Enviar chunks y metadata
- Manejar errores en streaming
- Validar contexto de ventana
- Formatear SSE
- Soporte para two-pass streaming (tool calling)
"""
import asyncio
import json
from typing import Dict, Any, AsyncGenerator, Optional, List

from src.providers.manager import ChatResponse
from src.providers.manager import provider_manager
from src.schemas.schemas import ConversationSettings
from src.services.utils.stream_cancel_manager import StreamCancelToken
from src.utils.logger import get_logger


class StreamHandler:
    """
    Responsable de manejar el streaming de respuestas.
    Separa la lógica de streaming de la orquestación.

    Migrado desde ChatOrchestrator para cumplir SRP.
    """

    def __init__(self):
        self.logger = get_logger(__name__)

    # =============================================================================
    # Streaming
    # =============================================================================

    async def stream_response(
        self,
        messages: list,
        settings: ConversationSettings,
        cancel_token: Optional[StreamCancelToken] = None,
        tool_definitions: list = None,
        tool_configs: Dict[str, Any] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream LLM response.

        Args:
            messages: List of ChatMessage objects
            settings: Conversation settings
            cancel_token: Optional cancellation token
            tool_definitions: Optional tool definitions for function calling
            tool_configs: Optional tool configurations

        Yields:
            Dict with chunk data
        """
        provider = provider_manager.get_provider(settings.provider)

        # Prepare kwargs
        kwargs = {
            "messages": messages,
            "model": settings.model,
            "temperature": settings.temperature,
            "max_tokens": settings.max_tokens,
            "top_p": settings.top_p,
        }

        if tool_definitions:
            kwargs["tools"] = tool_definitions
            kwargs["tool_choice"] = "auto"

        if tool_configs:
            kwargs.update(tool_configs)

        full_content = ""
        thinking_content = ""
        metadata = {}

        # Check cancellation before starting
        if cancel_token and cancel_token.is_cancelled():
            self.logger.warning(
                f"Stream already cancelled before LLM streaming started. "
                f"Reason: {cancel_token.cancelled_by or 'unknown'}"
            )
            yield {
                "type": "error",
                "message": "Stream was cancelled before response generation. Please try again."
            }
            return

        # Determine stream method
        if hasattr(provider, 'cancellable_stream_chat') and cancel_token:
            stream_method = provider.cancellable_stream_chat
            kwargs["cancel_event"] = cancel_token.cancel_event
            kwargs["cancel_check_interval"] = 0.1
        else:
            stream_method = provider.stream_chat

        try:
            async for chunk in stream_method(**kwargs):
                # Check cancellation
                if cancel_token and cancel_token.is_cancelled():
                    self.logger.info("Stream cancelled by token")
                    yield {
                        "type": "cancelled",
                        "message": "Stream cancelled"
                    }
                    return

                # Handle different chunk types
                if isinstance(chunk, dict):
                    if chunk.get("type") == "thinking":
                        thinking_chunk = chunk.get("content", "")
                        thinking_content += thinking_chunk
                        yield {
                            "type": "thinking",
                            "content": thinking_chunk
                        }
                    elif chunk.get("type") == "content":
                        content_chunk = chunk.get("chunk", "")
                        full_content += content_chunk
                        yield {
                            "type": "content",
                            "chunk": content_chunk
                        }
                    elif chunk.get("type") == "metadata":
                        metadata = chunk.get("data", {})
                else:
                    # Plain text chunk - handle JSON parsing
                    if isinstance(chunk, str):
                        try:
                            parsed_chunk = json.loads(chunk)
                            if isinstance(parsed_chunk, dict):
                                if parsed_chunk.get("type") == "thinking":
                                    thinking_chunk = parsed_chunk.get("content", "")
                                    thinking_content += thinking_chunk
                                    yield {"type": "thinking", "content": thinking_chunk}
                                elif parsed_chunk.get("type") == "content":
                                    content_chunk = parsed_chunk.get("chunk", "")
                                    full_content += content_chunk
                                    yield {"type": "content", "chunk": content_chunk}
                                else:
                                    full_content += chunk
                                    yield {"type": "content", "chunk": chunk}
                            else:
                                full_content += chunk
                                yield {"type": "content", "chunk": chunk}
                        except json.JSONDecodeError:
                            full_content += chunk
                            yield {"type": "content", "chunk": chunk}
                    else:
                        full_content += str(chunk)
                        yield {"type": "content", "chunk": chunk}

            # Send final metadata
            yield {
                "type": "metadata",
                "data": {
                    **metadata,
                    "model": settings.model,
                    "provider": settings.provider,
                    "thinking_content": thinking_content if thinking_content else None
                }
            }

        except asyncio.CancelledError:
            self.logger.info("Stream cancelled by client")
            yield {
                "type": "cancelled",
                "message": "Stream cancelled by client"
            }
            raise
        except Exception as e:
            self.logger.error(f"Error in stream: {e}")
            yield {
                "type": "error",
                "error": str(e),
                "message": "An error occurred during streaming"
            }
            raise

    # =============================================================================
    # Two-Pass Streaming (for tool calling)
    # =============================================================================

    async def stream_with_tools(
        self,
        messages: list,
        settings: ConversationSettings,
        tool_definitions: list,
        cancel_token: Optional[StreamCancelToken] = None,
        tool_configs: Dict[str, Any] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream with tool calling support (two-pass approach).

        Pass 1: Non-streaming call to get tool requests
        Pass 2: Stream final response after tool execution

        Args:
            messages: List of ChatMessage objects
            settings: Conversation settings
            tool_definitions: Tool definitions
            cancel_token: Optional cancellation token
            tool_configs: Optional tool configurations

        Yields:
            Dict with chunk data
        """
        provider = provider_manager.get_provider(settings.provider)

        # Pass 1: Get tool calls (non-streaming)
        self.logger.info("Agent mode: First pass to get tool requests...")

        kwargs = {
            "messages": messages,
            "model": settings.model,
            "temperature": settings.temperature,
            "max_tokens": settings.max_tokens,
            "tools": tool_definitions,
            "tool_choice": "auto"
        }

        if tool_configs:
            kwargs.update(tool_configs)

        response = await provider.chat(**kwargs)

        # Check if tools are needed
        if not response.tool_calls:
            # No tools needed, stream the response we got
            if response.content:
                yield {"type": "content", "chunk": response.content}

            yield {
                "type": "metadata",
                "data": {
                    "tools_executed": [],
                    "mode": "agent",
                    "provider": settings.provider,
                    "model": settings.model
                }
            }
            return

        # Yield tool calls for execution
        yield {
            "type": "tool_calls",
            "tool_calls": response.tool_calls
        }

    async def stream_after_tools(
        self,
        messages: list,
        settings: ConversationSettings,
        cancel_token: Optional[StreamCancelToken] = None,
        tool_configs: Dict[str, Any] = None,
        tool_results: List[Dict] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream LLM response after tool execution (Pass 2).

        Args:
            messages: List of ChatMessage objects (with tool results)
            settings: Conversation settings
            cancel_token: Optional cancellation token
            tool_configs: Optional tool configurations
            tool_results: Tool execution results

        Yields:
            Dict with chunk data
        """
        provider = provider_manager.get_provider(settings.provider)

        # Check cancellation before starting
        if cancel_token and cancel_token.is_cancelled():
            self.logger.warning(
                f"Stream already cancelled before LLM streaming started (second pass). "
                f"Reason: {cancel_token.cancelled_by or 'unknown'}"
            )
            yield {
                "type": "error",
                "message": "Stream was cancelled before response generation. Please try again."
            }
            return

        # Determine stream method
        if hasattr(provider, 'cancellable_stream_chat') and cancel_token:
            stream_method = provider.cancellable_stream_chat
            kwargs = {
                "cancel_event": cancel_token.cancel_event,
                "cancel_check_interval": 0.1
            }
        else:
            stream_method = provider.stream_chat
            kwargs = {}

        if tool_configs:
            kwargs.update(tool_configs)

        async for chunk in stream_method(
            messages=messages,
            model=settings.model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            **kwargs
        ):
            # Check cancellation
            if cancel_token and cancel_token.is_cancelled():
                self.logger.info("Stream cancelled by user")
                yield {
                    "type": "cancelled",
                    "message": "Stream cancelled"
                }
                return

            # Handle different chunk types
            if isinstance(chunk, dict) and chunk.get("type") == "thinking":
                yield {"type": "thinking", "content": chunk.get("content", "")}
            elif isinstance(chunk, str):
                try:
                    parsed_chunk = json.loads(chunk)
                    if isinstance(parsed_chunk, dict):
                        if parsed_chunk.get("type") == "thinking":
                            yield {"type": "thinking", "content": parsed_chunk.get("content", "")}
                        elif parsed_chunk.get("type") == "content":
                            yield {"type": "content", "chunk": parsed_chunk.get("chunk", "")}
                        else:
                            yield {"type": "content", "chunk": chunk}
                    else:
                        yield {"type": "content", "chunk": chunk}
                except json.JSONDecodeError:
                    yield {"type": "content", "chunk": chunk}
            else:
                yield {"type": "content", "chunk": chunk}

        # Send metadata
        yield {
            "type": "metadata",
            "data": {
                "tools_executed": [tr["tool_name"] for tr in tool_results] if tool_results else [],
                "tool_results": tool_results,
                "mode": "agent",
                "provider": settings.provider,
                "model": settings.model
            }
        }

    # =============================================================================
    # SSE Formatting
    # =============================================================================

    def format_sse(self, data: Dict[str, Any]) -> str:
        """
        Format data as Server-Sent Event.

        Args:
            data: Data dict to format

        Returns:
            SSE formatted string
        """
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    def format_sse_error(self, message: str) -> str:
        """
        Format error as SSE.

        Args:
            message: Error message

        Returns:
            SSE formatted error string
        """
        return f"event: error\ndata: {message}\n\n"

    def format_sse_done(self) -> str:
        """
        Format done signal as SSE.

        Returns:
            SSE formatted done string
        """
        return "data: [DONE]\n\n"

    # =============================================================================
    # Response Building
    # =============================================================================

    def build_response_from_stream(
        self,
        full_content: str,
        thinking_content: Optional[str],
        metadata: Dict[str, Any]
    ) -> ChatResponse:
        """
        Build ChatResponse from accumulated stream data.

        Args:
            full_content: Accumulated content
            thinking_content: Accumulated thinking content
            metadata: Final metadata

        Returns:
            ChatResponse object
        """
        return ChatResponse(
            content=full_content,
            model=metadata.get("model", ""),
            provider=metadata.get("provider", ""),
            tokens_used=metadata.get("tokens_used"),
            cost=metadata.get("cost"),
            thinking_content=thinking_content,
            metadata=metadata
        )

    # =============================================================================
    # Context Window Management
    # =============================================================================

    def validate_context_window(
        self,
        messages: list,
        provider_context_window: int
    ) -> bool:
        """
        Validate that messages fit within context window.

        Args:
            messages: List of messages
            provider_context_window: Provider's context window size

        Returns:
            True if within limits, False otherwise
        """
        total_chars = sum(
            len(msg.content) if hasattr(msg, 'content') and msg.content else 0
            for msg in messages
        )

        # Rough estimate: 1 token ≈ 3 characters
        estimated_tokens = total_chars // 3

        # Warn if using more than 90% of context
        if estimated_tokens > provider_context_window * 0.9:
            self.logger.warning(
                f"High context usage: {estimated_tokens} tokens "
                f"(window: {provider_context_window})"
            )
            return False

        return True

    def get_estimated_tokens(self, messages: list) -> int:
        """
        Get estimated token count for messages.

        Args:
            messages: List of messages

        Returns:
            Estimated token count
        """
        total_chars = sum(
            len(msg.content) if hasattr(msg, 'content') and msg.content else 0
            for msg in messages
        )
        return total_chars // 3

    # =============================================================================
    # Metadata Building
    # =============================================================================

    def build_stream_metadata(
        self,
        settings: ConversationSettings,
        tools_executed: List[str] = None,
        rag_metadata: Dict[str, Any] = None,
        mode: str = "agent"
    ) -> Dict[str, Any]:
        """
        Build metadata for stream completion.

        Args:
            settings: Conversation settings
            tools_executed: List of executed tools
            rag_metadata: RAG metadata
            mode: Processing mode

        Returns:
            Metadata dict
        """
        metadata = {
            "tools_executed": tools_executed or [],
            "mode": mode,
            "provider": settings.provider,
            "model": settings.model
        }

        if rag_metadata:
            metadata["rag_metadata"] = rag_metadata

        return metadata
