# =============================================================================
# src/services/chat/orchestrator.py
# Chat Orchestrator - Main coordinator (REFACTORED for SRP)
# =============================================================================
"""
ChatOrchestrator: Coordinador principal del chat (REFACTORED).

Este es un orchestrador DELGADO que DELEGA responsabilidades:
- ContextBuilder: Construcción de contexto (RAG, memoria, historial)
- ToolExecutor: Ejecución de herramientas
- ResponseFormatter: Formateo de respuestas
- StreamHandler: Manejo de streaming

El orchestrador solo COORDINA, no implementa lógica de negocio.
Esto cumple con el Single Responsibility Principle (SRP).

ANTES: 2725 líneas con 8+ responsabilidades
DESPUÉS: ~300 líneas como coordinador puro
"""
import json
from typing import List, Dict, Any, Optional, AsyncGenerator
from uuid import UUID

from src.models.models import Conversation, ToolMode
from src.providers.manager import ChatMessage, ChatResponse
from src.providers.manager import provider_manager
from src.repositories import (
    ConversationRepository,
    MessageRepository,
    FileRepository,
    CustomToolRepository
)
from src.schemas.schemas import ConversationSettings
# Import specialists (SRP compliance)
from src.services.chat.context_builder import ContextBuilder
from src.services.chat.response_formatter import ResponseFormatter
from src.services.chat.stream_handler import StreamHandler
from src.services.chat.tool_executor import ToolExecutor
from src.services.context.conversation_file_context import ConversationFileContext
# Import Sistema de Contexto Inteligente
from src.services.context.target_file_detector import TargetFileDetector
from src.services.intent.router import get_intent_router, ToolScore
from src.services.utils.stream_cancel_manager import StreamCancelToken
from src.tools.base_tool import tool_registry
from src.utils.logger import get_logger, set_conversation_context


class ChatOrchestrator:
    """
    Coordinador delgado para el chat.

    Delega responsabilidades a clases especializadas:
    - ContextBuilder: Contexto y mensajes
    - ToolExecutor: Herramientas
    - ResponseFormatter: Formateo
    - StreamHandler: Streaming

    CUMPLE SRP: Una sola razón para cambiar - coordinación del flujo de chat.
    """

    def __init__(
        self,
        conversation_repo: ConversationRepository,
        message_repo: MessageRepository,
        file_repo: FileRepository,
        custom_tool_repo: CustomToolRepository
    ):
        self.logger = get_logger(__name__)

        # Store repositories (for specialist initialization)
        self.conversation_repo = conversation_repo
        self.message_repo = message_repo
        self.file_repo = file_repo
        self.custom_tool_repo = custom_tool_repo

        # Initialize specialists (SRP: each has one responsibility)
        self.context_builder = ContextBuilder(
            conversation_repo=conversation_repo,
            message_repo=message_repo,
            file_repo=file_repo,
            custom_tool_repo=custom_tool_repo
        )

        self.tool_executor = ToolExecutor(
            file_repo=file_repo,
            custom_tool_repo=custom_tool_repo,
            message_repo=message_repo
        )

        self.response_formatter = ResponseFormatter()
        self.stream_handler = StreamHandler()

        # Sistema de Contexto Inteligente (NUEVO)
        self.target_file_detector = TargetFileDetector(file_repo=file_repo)
        self.conversation_file_context = ConversationFileContext()

        # IntentRouter — lazy init (loads embedding model on first use)
        self._intent_router = None

    # =============================================================================
    # Main Entry Points
    # =============================================================================

    async def process_message(
        self,
        conversation: Conversation,
        user_message: str,
        file_ids: Optional[List[UUID]] = None,
        collection_name: Optional[str] = None
    ) -> ChatResponse:
        """
        Process user message and generate response.

        Args:
            conversation: Conversation object
            user_message: User's message content
            file_ids: Optional list of attached file IDs
            collection_name: Optional collection name for RAG

        Returns:
            ChatResponse with generated content
        """
        set_conversation_context(str(conversation.id))
        self.logger.info("process_message started", extra={"conversation_id": str(conversation.id)})

        # NUEVO: Establecer conversation_id en el contexto de archivos
        self.conversation_file_context.set_conversation(str(conversation.id))

        # Parse settings (delegated to ContextBuilder)
        settings = self.context_builder.parse_settings(
            conversation.id,
            conversation.settings
        )

        self.logger.info(
            f"Processing: mode={settings.tool_mode.value}, "
            f"msg_len={len(user_message)}, files={len(file_ids) if file_ids else 0}"
        )

        try:
            # Get tool configurations
            tool_configs = await self._get_active_tool_configurations(conversation.id)

            # Route to appropriate mode
            if settings.tool_mode == ToolMode.AGENT:
                response = await self._agent_mode(
                    conversation, user_message, settings, file_ids, tool_configs, collection_name
                )
            else:
                response = await self._manual_mode(
                    conversation, user_message, settings, file_ids, tool_configs, collection_name
                )

            self.logger.info(
                "Message processed successfully",
                extra={
                    "response_length": len(response.content),
                    "tokens_used": response.tokens_used
                }
            )
            return response

        except Exception as e:
            self.logger.error(
                f"Error processing message: {str(e)}",
                exc_info=True,
                extra={"tool_mode": settings.tool_mode.value}
            )
            raise

    async def process_message_stream(
        self,
        conversation: Conversation,
        user_message: str,
        file_ids: Optional[List[UUID]] = None,
        cancel_token: Optional[StreamCancelToken] = None,
        collection_name: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process message with streaming support.

        Args:
            conversation: Conversation object
            user_message: User's message content
            file_ids: Optional list of attached file IDs
            cancel_token: Optional cancellation token
            collection_name: Optional collection name for RAG

        Yields:
            Dict with chunk data
        """
        set_conversation_context(str(conversation.id))
        self.logger.info("process_message_stream started", extra={"conversation_id": str(conversation.id)})

        # NUEVO: Establecer conversation_id en el contexto de archivos
        self.conversation_file_context.set_conversation(str(conversation.id))

        # Parse settings (delegated to ContextBuilder)
        settings = self.context_builder.parse_settings(
            conversation.id,
            conversation.settings
        )

        try:
            # Get tool configurations
            tool_configs = await self._get_active_tool_configurations(conversation.id)

            # Route to appropriate mode
            if settings.tool_mode == ToolMode.AGENT:
                async for chunk in self._agent_mode_stream(
                    conversation, user_message, settings, file_ids, cancel_token, tool_configs, collection_name
                ):
                    yield chunk
            else:
                async for chunk in self._manual_mode_stream(
                    conversation, user_message, settings, file_ids, cancel_token, tool_configs, collection_name
                ):
                    yield chunk

        except Exception as e:
            self.logger.error(
                f"Error in streaming message: {str(e)}",
                exc_info=True,
                extra={"tool_mode": settings.tool_mode.value}
            )
            raise

    # =============================================================================
    # Agent Mode (AI decides which tools to use)
    # =============================================================================

    async def _agent_mode(
        self,
        conversation: Conversation,
        user_message: str,
        settings: ConversationSettings,
        file_ids: Optional[List[UUID]],
        tool_configs: Optional[Dict[str, Any]],
        collection_name: Optional[str]
    ) -> ChatResponse:
        """Agent mode: LLM decides which tools to use via function calling."""

        # Build message history (delegated to ContextBuilder)
        messages = await self.context_builder.build_message_history(
            conversation, user_message, settings
        )

        # Build RAG context if files attached (delegated to ContextBuilder)
        context_parts = []
        rag_metadata = {}
        available_tools = self.tool_executor.get_available_tools(settings)

        if file_ids and "rag_search" in available_tools:
            rag_context, rag_data = await self.context_builder.build_rag_context(
                conversation, user_message, settings, file_ids, collection_name
            )
            if rag_context:
                context_parts.append(rag_context)
                rag_metadata = self.response_formatter.extract_rag_metadata(
                    rag_context, rag_data.get("chunks") if rag_data else None
                )

        # Add context to user message
        if context_parts:
            context_string = self.context_builder.build_context_string(context_parts)
            enhanced_message = f"{context_string}\n\n---\n\n{user_message}"
            # Replace last user message with enhanced version
            if messages and messages[-1].role == "user":
                messages[-1] = ChatMessage(role="user", content=enhanced_message)
            else:
                messages.append(ChatMessage(role="user", content=enhanced_message))

        # Inject code analysis hint if code files are present
        if file_ids and await self.context_builder.has_code_files(file_ids):
            messages.append(ChatMessage(
                role="system",
                content="HINT: Source code files have been uploaded. Use the 'codebase_tool' tool to perform structural analysis, find definitions, or understand the logic of these files if needed."
            ))

        # Get tool definitions (delegated to ToolExecutor)
        tool_definitions = None
        if available_tools:
            tool_definitions = self.tool_executor.get_tool_definitions(
                available_tools, settings.provider
            )

        # Get provider and make request
        provider = provider_manager.get_provider(settings.provider)

        # Check if provider supports function calling
        if settings.provider not in ["openai", "anthropic"]:
            # Fall back to manual mode for local providers
            return await self._manual_mode(
                conversation, user_message, settings, file_ids, tool_configs, collection_name
            )

        # Adjust max_tokens for short queries
        max_tokens = settings.max_tokens
        if len(user_message) < 100 and max_tokens > 500:
            max_tokens = 500

        # Prepare kwargs
        kwargs = {}
        if tool_configs:
            kwargs.update(tool_configs)

        response = await provider.chat(
            messages=messages,
            model=settings.model,
            temperature=settings.temperature,
            max_tokens=max_tokens,
            tools=tool_definitions,
            tool_choice="auto",
            **kwargs
        )

        # Handle tool calls if present
        if response.tool_calls:
            # Execute tools (delegated to ToolExecutor)
            tool_results = await self.tool_executor.execute_tools_batch(
                response.tool_calls, conversation, collection_name
            )

            # Add tool results to messages
            messages.append(ChatMessage(role="assistant", content=response.content or ""))
            messages.append(ChatMessage(
                role="tool",
                content=json.dumps(tool_results)
            ))

            # Second LLM call with tool results
            response = await provider.chat(
                messages=messages,
                model=settings.model,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens
            )

            # Add tool metadata
            response.metadata["tools_executed"] = [tr["tool_name"] for tr in tool_results]
            response.metadata["tool_results"] = tool_results

        # Format final response (delegated to ResponseFormatter)
        response = self.response_formatter.format_response(
            response,
            tools_executed=response.metadata.get("tools_executed"),
            rag_metadata=rag_metadata,
            mode="agent"
        )

        # Handle empty response
        if not response.content or not response.content.strip():
            response = self.response_formatter.format_empty_response(
                response, response.metadata.get("thinking_content")
            )

        return response

    async def _agent_mode_stream(
        self,
        conversation: Conversation,
        user_message: str,
        settings: ConversationSettings,
        file_ids: Optional[List[UUID]],
        cancel_token: Optional[StreamCancelToken],
        tool_configs: Optional[Dict[str, Any]],
        collection_name: Optional[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Agent mode with streaming support (two-pass approach)."""

        try:
            # Build message history (delegated to ContextBuilder)
            messages = await self.context_builder.build_message_history(
                conversation, user_message, settings
            )

            # Context window validation (delegated to StreamHandler)
            provider = provider_manager.get_provider(settings.provider)
            model_context_window = provider.context_window if hasattr(provider, 'context_window') else 8192
            self.stream_handler.validate_context_window(messages, model_context_window)

            # Build RAG context (delegated to ContextBuilder)
            context_parts = []
            rag_metadata = {}
            available_tools = self.tool_executor.get_available_tools(settings)

            if file_ids and "rag_search" in available_tools:
                rag_context, rag_data = await self.context_builder.build_rag_context(
                    conversation, user_message, settings, file_ids, collection_name
                )
                if rag_context:
                    context_parts.append(rag_context)
                    rag_metadata = self.response_formatter.extract_rag_metadata(
                        rag_context, rag_data.get("chunks") if rag_data else None
                    )

            # Add context to user message
            if context_parts:
                context_string = self.context_builder.build_context_string(context_parts)
                enhanced_message = f"{context_string}\n\n---\n\n{user_message}"
                if messages and messages[-1].role == "user":
                    messages[-1] = ChatMessage(role="user", content=enhanced_message)
                else:
                    messages.append(ChatMessage(role="user", content=enhanced_message))

            # Inject code analysis hint
            if file_ids and await self.context_builder.has_code_files(file_ids):
                messages.append(ChatMessage(
                    role="system",
                    content="HINT: Source code files have been uploaded. Use the 'codebase_analyzer' tool to perform structural analysis, find definitions, or understand the logic of these files if needed."
                ))

            # Get tool definitions
            tool_definitions = None
            if available_tools:
                tool_definitions = self.tool_executor.get_tool_definitions(
                    available_tools, settings.provider
                )

            # Check if provider supports function calling
            if settings.provider not in ["openai", "anthropic"]:
                # Fall back to manual mode streaming
                async for chunk in self._manual_mode_stream(
                    conversation, user_message, settings, file_ids, cancel_token, tool_configs, collection_name
                ):
                    yield chunk
                return

            # PASS 1: First LLM call (NON-streaming) to get tool requests
            self.logger.info("Agent mode: First pass to get tool requests...")
            response = await provider.chat(
                messages=messages,
                model=settings.model,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                tools=tool_definitions,
                tool_choice="auto"
            )

            # Check if LLM wants to use tools
            if not response.tool_calls:
                # No tools needed, stream the response we got
                if response.content:
                    yield {"type": "content", "chunk": response.content}

                yield {
                    "type": "metadata",
                    "data": self.stream_handler.build_stream_metadata(
                        settings, tools_executed=[], rag_metadata=rag_metadata, mode="agent"
                    )
                }
                return

            # Execute requested tools
            self.logger.info(f"Executing {len(response.tool_calls)} tool(s)...")
            tool_results = await self.tool_executor.execute_tools_batch(
                response.tool_calls, conversation, collection_name
            )

            # PASS 2: Second LLM call WITH streaming and tool results
            self.logger.info("Agent mode: Second pass with tool results (streaming)...")
            messages_with_tools = messages + [
                ChatMessage(role="assistant", content=response.content or ""),
                ChatMessage(role="tool", content=json.dumps(tool_results))
            ]

            # Stream final response (delegated to StreamHandler)
            async for chunk in self.stream_handler.stream_after_tools(
                messages=messages_with_tools,
                settings=settings,
                cancel_token=cancel_token,
                tool_configs=tool_configs,
                tool_results=tool_results
            ):
                yield chunk

        except Exception as e:
            self.logger.error(f"Error in agent_mode_stream: {e}", exc_info=True)
            yield {
                "type": "error",
                "error": str(e),
                "message": "An error occurred during streaming"
            }
            raise

    # =============================================================================
    # Manual Mode (User/System decides tools)
    # =============================================================================

    async def _manual_mode(
        self,
        conversation: Conversation,
        user_message: str,
        settings: ConversationSettings,
        file_ids: Optional[List[UUID]],
        tool_configs: Optional[Dict[str, Any]],
        collection_name: Optional[str]
    ) -> ChatResponse:
        """Manual mode: Execute tools directly before LLM call."""

        # =========================================================
        # NUEVO: Sistema de Contexto Inteligente - Detectar archivo objetivo
        # =========================================================
        target_file_id = None
        if file_ids:
            try:
                target_info = await self.target_file_detector.detect_target(
                    conversation_id=conversation.id,
                    attached_file_ids=file_ids,
                    user_message=user_message,
                    conversation_file_context=self.conversation_file_context
                )

                if target_info:
                    target_file_id = target_info.file_id
                    self.logger.info(
                        f"🎯 [Manual Mode] Target file: {target_info.filename} "
                        f"(source: {target_info.source})"
                    )

                    # Registrar archivo procesado
                    self.conversation_file_context.register_file_processed(
                        file_id=target_info.file_id,
                        filename=target_info.filename
                    )
            except Exception as e:
                self.logger.warning(f"Target detection failed in manual mode: {e}")

        # Execute RAG context (delegated to ContextBuilder)
        rag_context, rag_metadata, tools_executed = await self._execute_rag_context(
            conversation, user_message, settings, file_ids, collection_name,
            target_file_id=target_file_id
        )

        # Build context parts
        context_parts = []
        if rag_context:
            context_parts.append(rag_context)

        # Execute semantic memory if enabled
        if settings.memory_config.semantic_enabled:
            memory_context = await self.context_builder.build_memory_context(
                conversation, user_message, settings.memory_config.model_dump() if hasattr(settings.memory_config, 'model_dump') else settings.memory_config
            )
            if memory_context:
                context_parts.append(memory_context)
                self.logger.info("Semantic memory context added")

        # =========================================================
        # PASO 3: Ejecutar herramientas (codebase_tool, etc.)
        # =========================================================
        # Execute other tools (delegated to ToolExecutor)
        tool_context_parts, tool_tools_executed = await self._execute_tools_context(
            conversation, user_message, settings, rag_metadata,
            file_ids=file_ids,
            collection_name=collection_name,
            target_file_id=target_file_id,  # Bug 2 fix: ya detectado arriba
        )
        context_parts.extend(tool_context_parts)
        tools_executed.extend(tool_tools_executed)

        # Build tool context string with dynamic header
        tool_context_string = None
        if context_parts:
            # Build tool context string
            tool_context_string = None
            if context_parts:
                # NUEVO: Usar content_prompt dinámico si está disponible
                custom_header = None
                if hasattr(self, '_tool_content_prompts') and self._tool_content_prompts:
                    # Si hay content_prompts personalizados, usar el primero (o combinarlos)
                    # Por ahora usamos el primero encontrado
                    first_tool = list(self._tool_content_prompts.keys())[0]
                    custom_header = self._tool_content_prompts[first_tool]
                    self.logger.info(f"📝 Using custom content_prompt from tool '{first_tool}'")
                    # Limpiar para la próxima ejecución
                    self._tool_content_prompts = {}

                tool_context_string = self.context_builder.build_context_string(context_parts, custom_header)
                self.logger.info(f"Tool context built: {len(context_parts)} parts")

        # Build message history with tool context
        messages = await self.context_builder.build_message_history(
            conversation, user_message, settings, tool_context_string
        )

        # Get provider and make request
        provider = provider_manager.get_provider(settings.provider)

        # Dynamic max_tokens adjustment
        adjusted_max_tokens = settings.max_tokens
        if len(user_message) < 100 and settings.max_tokens > 500:
            adjusted_max_tokens = 500

        # Prepare kwargs
        kwargs = {}
        if tool_configs:
            kwargs.update(tool_configs)

        response = await provider.chat(
            messages=messages,
            model=settings.model,
            temperature=settings.temperature,
            max_tokens=adjusted_max_tokens,
            top_p=settings.top_p,
            **kwargs
        )

        # Format response (delegated to ResponseFormatter)
        response = self.response_formatter.format_response(
            response,
            tools_executed=tools_executed,
            rag_metadata=rag_metadata,
            mode="manual"
        )

        return response

    async def _manual_mode_stream(
        self,
        conversation: Conversation,
        user_message: str,
        settings: ConversationSettings,
        file_ids: Optional[List[UUID]],
        cancel_token: Optional[StreamCancelToken],
        tool_configs: Optional[Dict[str, Any]],
        collection_name: Optional[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Manual mode with streaming support."""

        self.logger.info("Starting MANUAL mode STREAMING processing")

        # =========================================================
        # NUEVO: Sistema de Contexto Inteligente - Detectar archivo objetivo
        # =========================================================
        target_file_id = None
        if file_ids:
            try:
                target_info = await self.target_file_detector.detect_target(
                    conversation_id=conversation.id,
                    attached_file_ids=file_ids,
                    user_message=user_message,
                    conversation_file_context=self.conversation_file_context
                )

                if target_info:
                    target_file_id = target_info.file_id
                    self.logger.info(
                        f"🎯 [Manual Stream] Target file: {target_info.filename} "
                        f"(source: {target_info.source})"
                    )

                    # Registrar archivo procesado
                    self.conversation_file_context.register_file_processed(
                        file_id=target_info.file_id,
                        filename=target_info.filename
                    )
            except Exception as e:
                self.logger.warning(f"Target detection failed in manual stream: {e}")

        # Execute RAG context (delegated to ContextBuilder)
        rag_context, rag_metadata, tools_executed = await self._execute_rag_context(
            conversation, user_message, settings, file_ids, collection_name,
            target_file_id=target_file_id
        )

        # Build context parts
        context_parts = []
        if rag_context:
            context_parts.append(rag_context)

        # Execute semantic memory if enabled
        if settings.memory_config.semantic_enabled:
            memory_context = await self.context_builder.build_memory_context(
                conversation, user_message, settings.memory_config.model_dump() if hasattr(settings.memory_config, 'model_dump') else settings.memory_config
            )
            if memory_context:
                context_parts.append(memory_context)
                self.logger.info("Semantic memory context added")

        # =========================================================
        # PASO 3: Ejecutar herramientas (codebase_tool, etc.)
        # =========================================================
        # Execute other tools
        tool_context_parts, tool_tools_executed = await self._execute_tools_context(
            conversation, user_message, settings, rag_metadata,
            file_ids=file_ids,
            collection_name=collection_name,
            target_file_id=target_file_id,  # Bug 2 fix: ya detectado arriba
        )
        context_parts.extend(tool_context_parts)
        tools_executed.extend(tool_tools_executed)

        # Build tool context string with dynamic header
        tool_context_string = None
        if context_parts:
            # Build tool context string
            tool_context_string = None
            if context_parts:
                # NUEVO: Usar content_prompt dinámico si está disponible
                custom_header = None
                if hasattr(self, '_tool_content_prompts') and self._tool_content_prompts:
                    # Si hay content_prompts personalizados, usar el primero (o combinarlos)
                    # Por ahora usamos el primero encontrado
                    first_tool = list(self._tool_content_prompts.keys())[0]
                    custom_header = self._tool_content_prompts[first_tool]
                    self.logger.info(f"📝 Using custom content_prompt from tool '{first_tool}'")
                    # Limpiar para la próxima ejecución
                    self._tool_content_prompts = {}

                tool_context_string = self.context_builder.build_context_string(context_parts, custom_header)
                self.logger.info(f"Tool context built: {len(context_parts)} parts")

        # Build message history with tool context
        messages = await self.context_builder.build_message_history(
            conversation, user_message, settings, tool_context_string
        )

        # Stream LLM response (delegated to StreamHandler)
        provider = provider_manager.get_provider(settings.provider)
        self.logger.info("Starting LLM streaming...")

        async for chunk in self.stream_handler.stream_response(
            messages=messages,
            settings=settings,
            cancel_token=cancel_token,
            tool_configs=tool_configs
        ):
            yield chunk

        # Yield RAG sources as final content chunk
        sources_text = self.response_formatter.format_sources_text(rag_metadata)
        if sources_text:
            yield {"type": "content", "chunk": sources_text}

        # Send metadata
        self.logger.info("MANUAL mode STREAMING processing completed")
        yield {
            "type": "metadata",
            "data": self.stream_handler.build_stream_metadata(
                settings, tools_executed=tools_executed, rag_metadata=rag_metadata, mode="manual"
            )
        }

    # =============================================================================
    # Shared Helper Methods (delegating to specialists)
    # =============================================================================

    async def _get_intent_router(self):
        """Lazy init del IntentRouter — carga el modelo de embeddings solo la primera vez."""
        if self._intent_router is None:
            self._intent_router = await get_intent_router()
        return self._intent_router

    async def _execute_rag_context(
        self,
        conversation: Conversation,
        user_message: str,
        settings: ConversationSettings,
        file_ids: Optional[List[UUID]],
        collection_name: Optional[str] = None,
        target_file_id: Optional[UUID] = None  # NUEVO: Filtrar RAG por archivo objetivo
    ) -> tuple:
        """
        Execute RAG context - shared by all modes.
        Delegates to ContextBuilder.

        NUEVO: Acepta target_file_id para filtrar chunks por archivo específico.
        """
        rag_context = None
        rag_metadata = {}
        tools_executed = []

        enabled_tools = settings.enabled_tools or settings.available_tools
        if 'rag_search' not in enabled_tools:
            return None, {}, []

        # If no file_ids attached, use conversation files
        if not file_ids:
            try:
                files = await self.file_repo.get_by_conversation(conversation.id)
                if files:
                    file_ids = [f.id for f in files]
                    self.logger.info(f"🔄 Using {len(file_ids)} conversation files for RAG")
            except Exception as e:
                self.logger.warning(f"Failed to fetch conversation files: {e}")

        if not file_ids:
            return None, {}, []

        try:
            # Execute RAG (delegated to ContextBuilder)
            # NUEVO: Pasar target_file_id para filtrar por archivo específico
            rag_context, rag_data = await self.context_builder.build_rag_context(
                conversation, user_message, settings, file_ids, collection_name,
                target_file_id=target_file_id  # NUEVO PARÁMETRO
            )

            if rag_context:
                tools_executed.append("rag_search")
                rag_metadata = self.response_formatter.extract_rag_metadata(
                    rag_context, rag_data.get('chunks') if rag_data else None
                )
                rag_metadata['rag_status'] = 'available'
                self.logger.info(f"RAG context added. Content preview: {str(rag_context)}...")
            else:
                self.logger.warning("⚠️ RAG returned empty - no relevant chunks found")
                rag_metadata = {
                    'files': [str(fid) for fid in file_ids],
                    'rag_status': 'empty_results'
                }

        except Exception as e:
            self.logger.error(f"❌ RAG failed: {e}", exc_info=True)
            rag_metadata = {
                'files': [str(fid) for fid in file_ids] if file_ids else [],
                'rag_status': 'error',
                'user_message': f'Error al buscar en el archivo: {str(e)}'
            }

        return rag_context, rag_metadata, tools_executed

    async def _execute_tools_context(
        self,
        conversation: Conversation,
        user_message: str,
        settings: ConversationSettings,
        rag_metadata: Dict[str, Any],
        file_ids: Optional[List[UUID]] = None,
        collection_name: Optional[str] = None,
        tools_to_exclude: Optional[List[str]] = None,
        target_file_id: Optional[UUID] = None,  # NUEVO: recibido desde el caller, sin re-detectar
    ) -> tuple:
        """
        Execute enabled tools (excluding RAG) - shared by all modes.
        Delegates to ToolExecutor.

        NUEVO: Integra Sistema de Contexto Inteligente para detectar archivo objetivo.
        """
        context_parts = []
        tools_executed = []
        tools_to_exclude = tools_to_exclude or []

        enabled_tools = settings.enabled_tools or []

        # =========================================================
        # NUEVO: Score tools via IntentRouter — sin LLM, solo embeddings
        # =========================================================
        tools_to_score = [
            t for t in enabled_tools
            if t not in ["rag_search"] + tools_to_exclude and tool_registry.get(t)
        ]

        if not tools_to_score:
            return [], []

        intent_router = await self._get_intent_router()

        intent_context = {
            "attached_files": file_ids or [],
            "file_names": [],
        }

        tool_scores: Dict[str, ToolScore] = await intent_router.score_tools_for_query(
            query=user_message,
            enabled_tool_names=tools_to_score,
            context=intent_context
        )

        # Filtrar solo las que superaron el umbral
        # No usar get_tools_above_threshold() aquí: rag fallback es responsabilidad de _execute_rag_context
        selected_tool_scores = [
            ts for ts in tool_scores.values() if ts.passes_threshold
        ]
        selected_tool_scores.sort(key=lambda x: x.score, reverse=True)

        if not selected_tool_scores:
            self.logger.info(
                f"No tools passed intent threshold | "
                f"scores: { {n: round(ts.score, 3) for n, ts in tool_scores.items()} }"
            )
            return [], []

        # =========================================================
        # target_file_id ya viene como parámetro — no re-detectar aquí
        # =========================================================
        if target_file_id:
            self.logger.info(f"🎯 Using pre-detected target_file_id: {target_file_id}")

        for tool_score in selected_tool_scores:
            tool_name = tool_score.tool_name
            tool = tool_registry.get(tool_name)
            if not tool:
                self.logger.warning(f"Tool '{tool_name}' not found in registry")
                continue

            try:
                self.logger.info(
                    f"Executing tool: {tool_name} | "
                    f"intent={tool_score.best_intent} action={tool_score.best_intent_action} "
                    f"score={tool_score.score:.3f} target={tool_score.target}"
                )

                # Setup CustomToolExecutor if needed
                from src.tools.custom_tool import CustomToolExecutor
                if isinstance(tool, CustomToolExecutor):
                    tool.file_repo = self.file_repo
                    tool.custom_tool_repo = self.custom_tool_repo
                    await tool._load_custom_tool_config()

                # Determine execution strategy (delegated to ToolExecutor)
                # NUEVO: Pasar target_file_id e intent para sticky context y acción correcta
                execution_strategy = await self.tool_executor.determine_execution_strategy(
                    tool, tool_name, user_message, rag_metadata,
                    file_ids=file_ids,
                    target_file_id=target_file_id,  # NUEVO: Permite sticky context
                    settings=settings  # ✅ NUEVO parámetro
                )

                # Inyectar intent info del router — override suave (no sobreescribe si ya existe)
                if tool_score.best_intent_action:
                    execution_strategy.setdefault("intent_action", tool_score.best_intent_action)
                if tool_score.target:
                    execution_strategy.setdefault("intent_target", tool_score.target)
                if tool_score.default_params:
                    execution_strategy.setdefault("intent_default_params", tool_score.default_params)

                # NUEVO: Si hay target_file_id, pasarlo en execution_strategy
                if target_file_id and tool_name in ("codebase_analyzer", "codebase_tool"):
                    execution_strategy["target_file_id"] = str(target_file_id)
                    self.logger.info(f"🎯 Passing target_file_id to {tool_name}: {target_file_id}")

                # NUEVO: Para Custom Tools de tipo RAG, pasar filtro file_id
                if target_file_id and isinstance(tool, CustomToolExecutor):
                    tool_type = tool.tool_type
                    self.logger.info(f"🔍 Custom Tool detected: {tool_name}, tool_type: {tool_type}")
                    if tool_type == "rag_search":
                        # Pasar el filtro file_id directamente a los parámetros extraídos
                        execution_strategy["filters"] = {"file_id": str(target_file_id)}
                        self.logger.info(
                            f"🎯 Passing file_id filter to Custom RAG tool: {target_file_id}",
                            extra={"filters": execution_strategy["filters"]}
                        )

                # Extract parameters (delegated to ToolExecutor)
                extracted_params = await self.tool_executor.extract_tool_parameters(
                    tool, tool_name, user_message, conversation, settings, execution_strategy
                )

                # Execute tool (delegated to ToolExecutor)
                result = await self.tool_executor.execute_tool_with_context(
                    tool, conversation, collection_name=collection_name, **extracted_params
                )

                if result.success and result.data:
                    self.logger.info(f"## Result from {tool_name}\n\n{result.data}\n\n")
                    tool_context = f"## Result from {tool_name}\n\n{result.data}\n\n"
                    context_parts.append(tool_context)
                    tools_executed.append(tool_name)
                    self.logger.info(f"Tool '{tool_name}' executed successfully")

                    # NUEVO: Capturar content_prompt de custom tools para header dinámico
                    if isinstance(tool, CustomToolExecutor) and tool.content_prompt:
                        # Pasar el content_prompt al context_builder via execution_strategy
                        if not hasattr(self, '_tool_content_prompts'):
                            self._tool_content_prompts = {}
                        self._tool_content_prompts[tool_name] = tool.content_prompt
                        self.logger.info(f"📝 Captured content_prompt for tool '{tool_name}'")
                else:
                    context_parts.append(f"## Result from {tool_name}\n\nNo information found.\n\n")
            except Exception as e:
                self.logger.error(f"Error executing tool '{tool_name}': {e}")
                context_parts.append(f"## Result from {tool_name}\n\nError: {str(e)}\n\n")

        return context_parts, tools_executed

    async def _get_active_tool_configurations(self, conversation_id: UUID) -> Dict[str, Any]:
        """Get all active tool configurations for a conversation."""
        try:
            conversation = await self.conversation_repo.get_with_details(conversation_id)
            if not conversation:
                return {}

            tool_configs = {}
            for config in conversation.tool_configurations:
                if config.is_active:
                    tool_configs[config.tool_name] = config.config

            return tool_configs
        except Exception as e:
            self.logger.warning(f"Failed to get tool configurations: {e}")
            return {}
