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
from src.schemas.schemas import ConversationSettings, ToolExecutionPlan, ToolScoreInfo
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
            if tool_configs is None:
                tool_configs = {}
            
            # Inject conversation object for providers that need it (e.g. Copilot365)
            tool_configs["conversation"] = conversation

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

        # =========================================================
        # NUEVO: Recuperar archivos del historial si no vienen en el request.
        # En mensajes de seguimiento el frontend no reenvía file_ids,
        # pero el archivo fue adjuntado en un turno anterior y está en BD.
        # TargetFileDetector lo recupera vía ConversationFileContext.
        # =========================================================
        try:
            target_info = await self.target_file_detector.detect_target(
                conversation_id=conversation.id,
                attached_file_ids=file_ids or [],
                user_message=user_message,
                conversation_file_context=self.conversation_file_context
            )
            if target_info:
                if not file_ids:
                    file_ids = [target_info.file_id]
                self.logger.info(
                    f"[Agent Mode] Target file: {target_info.filename} "
                    f"(source: {target_info.source})"
                )
                self.conversation_file_context.register_file_processed(
                    file_id=target_info.file_id,
                    filename=target_info.filename
                )
        except Exception as e:
            self.logger.warning(f"Target detection failed in agent mode: {e}")

        # =========================================================
        # FASE 1: Pre-análisis con IntentRouter
        # =========================================================
        available_tools = self.tool_executor.get_available_tools(settings)

        intent_hint = None
        execution_plan = None

        if available_tools:
            try:
                execution_plan = await self._plan_tool_execution(
                    user_message=user_message,
                    settings=settings,
                    file_ids=file_ids,
                    target_file_id=None
                )

                if execution_plan.has_valid_tools and execution_plan.tool_scores:
                    # Crear hint para el LLM con las top 3 tools más relevantes
                    top_tools = sorted(
                        execution_plan.tool_scores.items(),
                        key=lambda x: x[1].score,
                        reverse=True
                    )[:3]

                    top_tools_str = ", ".join([
                        f"{name}(score={info.score:.2f})"
                        for name, info in top_tools
                    ])

                    intent_hint = (
                        f"Based on semantic analysis, these tools are most likely needed: "
                        f"{top_tools_str}. Consider using them if relevant to the user's request."
                    )
                    self.logger.info(f"Agent Mode Intent Hint: {intent_hint}")
            except Exception as e:
                self.logger.warning(f"IntentRouter pre-analysis failed: {e}")

        # Build RAG context if files attached AND if plan indicates RAG is needed
        context_parts = []
        rag_metadata = {}

        if file_ids and "rag_search" in available_tools:
            if execution_plan is None or execution_plan.needs_rag:
                rag_context, rag_data = await self.context_builder.build_rag_context(
                    conversation, user_message, settings, file_ids, collection_name
                )
                if rag_context:
                    context_parts.append(rag_context)
                    rag_metadata = self.response_formatter.extract_rag_metadata(
                        rag_context, rag_data.get("chunks") if rag_data else None
                    )
                self.logger.info("Agent Mode: RAG executed per plan")
            else:
                self.logger.info("Agent Mode: RAG skipped per plan")

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

        # Inject IntentRouter hint if available
        if intent_hint:
            messages.append(ChatMessage(
                role="system",
                content=intent_hint
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
            model_context_window = provider.context_window if hasattr(provider,
                                                                      'context_window') else 8192
            self.stream_handler.validate_context_window(messages, model_context_window)

            # =========================================================
            # NUEVO: Recuperar archivos del historial si no vienen en el request.
            # Mismo principio que en _agent_mode: mensajes de seguimiento no
            # reenvían file_ids pero el archivo ya está en BD.
            # =========================================================
            try:
                target_info = await self.target_file_detector.detect_target(
                    conversation_id=conversation.id,
                    attached_file_ids=file_ids or [],
                    user_message=user_message,
                    conversation_file_context=self.conversation_file_context
                )
                if target_info:
                    if not file_ids:
                        file_ids = [target_info.file_id]
                    self.logger.info(
                        f"[Agent Stream] Target file: {target_info.filename} "
                        f"(source: {target_info.source})"
                    )
                    self.conversation_file_context.register_file_processed(
                        file_id=target_info.file_id,
                        filename=target_info.filename
                    )
            except Exception as e:
                self.logger.warning(f"Target detection failed in agent stream: {e}")

            # =========================================================
            # FASE 1: Pre-análisis con IntentRouter
            # =========================================================
            available_tools = self.tool_executor.get_available_tools(settings)

            # Usar IntentRouter para identificar tools relevantes
            intent_hint = None
            execution_plan = None

            if available_tools:
                try:
                    execution_plan = await self._plan_tool_execution(
                        user_message=user_message,
                        settings=settings,
                        file_ids=file_ids,
                        target_file_id=None
                    )

                    if execution_plan.has_valid_tools and execution_plan.tool_scores:
                        # Crear hint para el LLM con las top 3 tools más relevantes
                        top_tools = sorted(
                            execution_plan.tool_scores.items(),
                            key=lambda x: x[1].score,
                            reverse=True
                        )[:3]  # Top 3

                        top_tools_str = ", ".join([
                            f"{name}(score={info.score:.2f})"
                            for name, info in top_tools
                        ])

                        intent_hint = (
                            f"Based on semantic analysis, these tools are most likely needed: "
                            f"{top_tools_str}. Consider using them if relevant to the user's request."
                        )
                        self.logger.info(f"Agent Mode Stream Intent Hint: {intent_hint}")
                except Exception as e:
                    self.logger.warning(f"IntentRouter pre-analysis failed: {e}")

            # Build RAG context (delegated to ContextBuilder)
            # Solo ejecutar RAG si el plan lo indica
            context_parts = []
            rag_metadata = {}

            if file_ids and "rag_search" in available_tools:
                if execution_plan is None or execution_plan.needs_rag:
                    rag_context, rag_data = await self.context_builder.build_rag_context(
                        conversation, user_message, settings, file_ids, collection_name
                    )
                    if rag_context:
                        context_parts.append(rag_context)
                        rag_metadata = self.response_formatter.extract_rag_metadata(
                            rag_context, rag_data.get("chunks") if rag_data else None
                        )
                    self.logger.info("Agent Mode Stream: RAG executed per plan")
                else:
                    self.logger.info("Agent Mode Stream: RAG skipped per plan")

            # Add context to user message
            if context_parts:
                context_string = self.context_builder.build_context_string(context_parts)
                enhanced_message = f"{context_string}\n\n---\n\n{user_message}"
                if messages and messages[-1].role == "user":
                    messages[-1] = ChatMessage(role="user", content=enhanced_message)
                else:
                    messages.append(ChatMessage(role="user", content=enhanced_message))

            # Inject code analysis hint if code files are present
            # NOTA: nombre consistente con _agent_mode (codebase_tool, no codebase_analyzer)
            if file_ids and await self.context_builder.has_code_files(file_ids):
                messages.append(ChatMessage(
                    role="system",
                    content="HINT: Source code files have been uploaded. Use the 'codebase_tool' tool to perform structural analysis, find definitions, or understand the logic of these files if needed."
                ))

            # Inject IntentRouter hint if available
            if intent_hint:
                messages.append(ChatMessage(
                    role="system",
                    content=intent_hint
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
                    conversation, user_message, settings, file_ids, cancel_token, tool_configs,
                    collection_name
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
        # FASE 1: Detectar archivo objetivo
        # Siempre se ejecuta, con o sin file_ids en el request.
        # En mensajes de seguimiento el frontend no reenvía file_ids,
        # pero el archivo fue adjuntado en un turno anterior.
        # TargetFileDetector lo recupera desde ConversationFileContext / BD.
        # Si lo encuentra, reconstruimos file_ids para que el resto del flujo
        # (plan, is_relevant, RAG, codebase_tool) funcione correctamente.
        # =========================================================
        target_file_id = None
        try:
            target_info = await self.target_file_detector.detect_target(
                conversation_id=conversation.id,
                attached_file_ids=file_ids or [],
                user_message=user_message,
                conversation_file_context=self.conversation_file_context
            )
            if target_info:
                target_file_id = target_info.file_id
                # Reconstruir file_ids desde historial si el request llegó sin ellos
                if not file_ids:
                    file_ids = [target_info.file_id]
                self.logger.info(
                    f"[Manual Mode] Target file: {target_info.filename} "
                    f"(source: {target_info.source})"
                )
                # Registrar para que los siguientes turnos también puedan encontrarlo
                self.conversation_file_context.register_file_processed(
                    file_id=target_info.file_id,
                    filename=target_info.filename
                )
        except Exception as e:
            self.logger.warning(f"Target detection failed in manual mode: {e}")

        # =========================================================
        # FASE 2: PLANIFICACIÓN - Decidir qué tools ejecutar
        # El IntentRouter decide ANTES de ejecutar cualquier cosa.
        # Ahora file_ids está correctamente poblado (desde request o historial),
        # por lo que is_relevant() de cada tool recibe contexto real.
        # =========================================================
        execution_plan = await self._plan_tool_execution(
            user_message=user_message,
            settings=settings,
            file_ids=file_ids,
            target_file_id=target_file_id
        )

        # Usar el target_file_id del plan si el detector no lo encontró antes
        if execution_plan.target_file_id and not target_file_id:
            target_file_id = execution_plan.target_file_id

        # =========================================================
        # FASE 3: EJECUCIÓN según el plan - TODAS las tools incluyendo RAG
        # RAG ya no tiene tratamiento especial: entra por el mismo
        # _execute_tools_context que el resto de tools.
        # =========================================================
        context_parts = []
        tools_executed = []
        rag_metadata = {}

        tools_to_run = execution_plan.tools_to_execute  # Incluye rag_search si needs_rag=True

        if tools_to_run:
            self.logger.info("Executing tools per plan: {}".format(tools_to_run))
            tool_context_parts, tool_tools_executed = await self._execute_tools_context(
                conversation, user_message, settings,
                {},  # rag_metadata vacío — cada tool construye su propio contexto
                file_ids=file_ids,
                collection_name=collection_name,
                target_file_id=target_file_id,
                tools_to_execute=tools_to_run,
                tool_scores=execution_plan.tool_scores
            )
            context_parts.extend(tool_context_parts)
            tools_executed.extend(tool_tools_executed)
            # RAG metadata viene incluida en los tool_results si es relevante
            rag_metadata = {}
        else:
            self.logger.info("No hay tools a ejecutar según el plan")

        # ---- Semantic memory (siempre se ejecuta si está habilitada) ----
        if settings.memory_config.semantic_enabled:
            memory_context = await self.context_builder.build_memory_context(
                conversation, user_message,
                settings.memory_config.model_dump() if hasattr(settings.memory_config,
                                                               'model_dump') else settings.memory_config
            )
            if memory_context:
                context_parts.append(memory_context)
                self.logger.info("Semantic memory context added")

        # Build tool context string with dynamic header
        # Si alguna custom tool definió un content_prompt, se usa como header
        # personalizado en lugar del genérico.
        tool_context_string = None
        if context_parts:
            custom_header = None
            if hasattr(self, '_tool_content_prompts') and self._tool_content_prompts:
                first_tool = list(self._tool_content_prompts.keys())[0]
                custom_header = self._tool_content_prompts[first_tool]
                self.logger.info(f"Using custom content_prompt from tool '{first_tool}'")
                # Limpiar para la próxima ejecución
                self._tool_content_prompts = {}

            tool_context_string = self.context_builder.build_context_string(context_parts,
                                                                            custom_header)
            self.logger.info(f"Tool context built: {len(context_parts)} parts")

        # Build message history with tool context
        messages = await self.context_builder.build_message_history(
            conversation, user_message, settings, tool_context_string
        )

        # Get provider and make request
        provider = provider_manager.get_provider(settings.provider)

        # Dynamic max_tokens adjustment for short queries
        adjusted_max_tokens = settings.max_tokens
        if len(user_message) < 100 and settings.max_tokens > 500:
            adjusted_max_tokens = 500

        # Prepare kwargs
        kwargs = {}
        if tool_configs:
            kwargs.update(tool_configs)

        # NUEVO: Flatten messages if provider doesn't support history
        if not self._provider_supports_message_history(provider, settings.model):
            messages = self._format_single_message_for_provider(messages, user_message)

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
        # FASE 1: Detectar archivo objetivo
        # Siempre se ejecuta, con o sin file_ids en el request.
        # En mensajes de seguimiento el frontend no reenvía file_ids,
        # pero el archivo fue adjuntado en un turno anterior.
        # TargetFileDetector lo recupera desde ConversationFileContext / BD.
        # Si lo encuentra, reconstruimos file_ids para que el resto del flujo
        # (plan, is_relevant, RAG, codebase_tool) funcione correctamente.
        # =========================================================
        target_file_id = None
        try:
            target_info = await self.target_file_detector.detect_target(
                conversation_id=conversation.id,
                attached_file_ids=file_ids or [],
                user_message=user_message,
                conversation_file_context=self.conversation_file_context
            )
            if target_info:
                target_file_id = target_info.file_id
                # Reconstruir file_ids desde historial si el request llegó sin ellos
                if not file_ids:
                    file_ids = [target_info.file_id]
                self.logger.info(
                    f"[Manual Stream] Target file: {target_info.filename} "
                    f"(source: {target_info.source})"
                )
                # Registrar para que los siguientes turnos también puedan encontrarlo
                self.conversation_file_context.register_file_processed(
                    file_id=target_info.file_id,
                    filename=target_info.filename
                )
        except Exception as e:
            self.logger.warning(f"Target detection failed in manual stream: {e}")

        # =========================================================
        # FASE 2: PLANIFICACIÓN - Decidir qué tools ejecutar
        # El IntentRouter decide ANTES de ejecutar cualquier cosa.
        # Ahora file_ids está correctamente poblado (desde request o historial),
        # por lo que is_relevant() de cada tool recibe contexto real.
        # =========================================================
        execution_plan = await self._plan_tool_execution(
            user_message=user_message,
            settings=settings,
            file_ids=file_ids,
            target_file_id=target_file_id
        )

        # Usar el target_file_id del plan si el detector no lo encontró antes
        if execution_plan.target_file_id and not target_file_id:
            target_file_id = execution_plan.target_file_id

        # =========================================================
        # FASE 3: EJECUCIÓN según el plan - TODAS las tools incluyendo RAG
        # RAG ya no tiene tratamiento especial: entra por el mismo
        # _execute_tools_context que el resto de tools.
        # =========================================================
        context_parts = []
        tools_executed = []
        rag_metadata = {}

        tools_to_run = execution_plan.tools_to_execute  # Incluye rag_search si needs_rag=True

        if tools_to_run:
            self.logger.info("Executing tools per plan: {}".format(tools_to_run))
            tool_context_parts, tool_tools_executed = await self._execute_tools_context(
                conversation, user_message, settings,
                {},  # rag_metadata vacío — cada tool construye su propio contexto
                file_ids=file_ids,
                collection_name=collection_name,
                target_file_id=target_file_id,
                tools_to_execute=tools_to_run,
                tool_scores=execution_plan.tool_scores
            )
            context_parts.extend(tool_context_parts)
            tools_executed.extend(tool_tools_executed)
            # RAG metadata viene incluida en los tool_results si es relevante
            rag_metadata = {}
        else:
            self.logger.info("No hay tools a ejecutar según el plan")

        # ---- Semantic memory (siempre se ejecuta si está habilitada) ----
        if settings.memory_config.semantic_enabled:
            memory_context = await self.context_builder.build_memory_context(
                conversation, user_message,
                settings.memory_config.model_dump() if hasattr(settings.memory_config,
                                                               'model_dump') else settings.memory_config
            )
            if memory_context:
                context_parts.append(memory_context)
                self.logger.info("Semantic memory context added")

        # Build tool context string with dynamic header
        # Si alguna custom tool definió un content_prompt, se usa como header
        # personalizado en lugar del genérico.
        tool_context_string = None
        if context_parts:
            custom_header = None
            if hasattr(self, '_tool_content_prompts') and self._tool_content_prompts:
                first_tool = list(self._tool_content_prompts.keys())[0]
                custom_header = self._tool_content_prompts[first_tool]
                self.logger.info(f"Using custom content_prompt from tool '{first_tool}'")
                # Limpiar para la próxima ejecución
                self._tool_content_prompts = {}

            tool_context_string = self.context_builder.build_context_string(context_parts,
                                                                            custom_header)
            self.logger.info(f"Tool context built: {len(context_parts)} parts")

        # Build message history with tool context
        messages = await self.context_builder.build_message_history(
            conversation, user_message, settings, tool_context_string
        )

        # Stream LLM response (delegated to StreamHandler)
        provider = provider_manager.get_provider(settings.provider)
        self.logger.info("Starting LLM streaming...")

        if not self._provider_supports_message_history(provider, settings.model):
            messages = self._format_single_message_for_provider(messages, user_message)

        async for chunk in self.stream_handler.stream_response(
            messages=messages,
            settings=settings,
            cancel_token=cancel_token,
            tool_configs=tool_configs
        ):
            yield chunk

        # Yield RAG sources as final content chunk (si hay metadata de fuentes)
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

    async def _plan_tool_execution(
        self,
        user_message: str,
        settings: ConversationSettings,
        file_ids: Optional[List[UUID]] = None,
        target_file_id: Optional[UUID] = None
    ) -> ToolExecutionPlan:
        """
        FASE 1: Planificación - Decide qué tools ejecutar ANTES de ejecutar cualquier cosa.

        Flujo:
        1. IntentRouter.score_tools_for_query() - Scorea todas las tools
        2. Para cada tool con score passing threshold → tool.is_relevant(context)
        3. Solo ejecutar las que pasan ambas validaciones

        Args:
            user_message: Mensaje del usuario
            settings: Configuración de la conversación
            file_ids: Archivos adjuntos (opcional)
            target_file_id: Archivo objetivo ya detectado (opcional)

        Returns:
            ToolExecutionPlan con las tools validadas por IntentRouter + is_relevant()
        """
        self.logger.info("=== FASE 1: PLANIFICACIÓN (IntentRouter + is_relevant) ===")

        # Inicializar plan
        plan = ToolExecutionPlan(
            target_file_id=target_file_id,
            execution_context={"file_ids": file_ids or []}
        )

        # Obtener tools habilitadas (excluir RAG temporalmente para scoring)
        enabled_tools = settings.enabled_tools or settings.available_tools

        if not enabled_tools:
            plan.debug_message = "No hay tools habilitadas"
            self.logger.warning("No hay tools habilitadas en settings")
            return plan

        # Scorear tools con IntentRouter (incluyendo RAG para que el router decida)
        tools_to_score = [
            t for t in enabled_tools
            if tool_registry.get(t)
        ]

        if not tools_to_score:
            plan.debug_message = "No hay tools disponibles en registry"
            self.logger.warning("No hay tools en registry para scorear")
            return plan

        # Obtener IntentRouter
        intent_router = await self._get_intent_router()

        # Preparar contexto para is_relevant()
        from src.tools.base_tool import ExecutionContext
        execution_context = ExecutionContext(
            user_message=user_message,
            file_ids=file_ids,
            target_file_id=target_file_id,
            collection_name=None,
            provider=settings.provider,
            model=settings.model
        )

        # Scorear TODAS las tools (incluyendo RAG)
        self.logger.info(f"Scoreando {len(tools_to_score)} tools: {tools_to_score}")

        tool_scores: Dict[str, ToolScore] = await intent_router.score_tools_for_query(
            query=user_message,
            enabled_tool_names=tools_to_score,
            context={"attached_files": file_ids or [], "file_names": []},
            llm_provider=settings.provider,
            llm_model=settings.model
        )

        if not tool_scores:
            plan.debug_message = "IntentRouter no devolvió scores"
            self.logger.warning("IntentRouter no devolvió resultados")
            return plan

        # Convertir scores al formato del plan
        tool_scores_info = {}
        for tool_name, ts in tool_scores.items():
            tool_scores_info[tool_name] = ToolScoreInfo(
                tool_name=ts.tool_name,
                score=ts.score,
                best_intent=ts.best_intent,
                best_intent_action=ts.best_intent_action,
                passes_threshold=ts.passes_threshold,
                requires_target=ts.requires_target,
                target=ts.target,
                default_params=ts.default_params
            )

        plan.tool_scores = tool_scores_info

        # Loguear scores
        scores_summary = {
            name: f"{ts.score:.3f}(pass={ts.passes_threshold})"
            for name, ts in tool_scores.items()
        }
        self.logger.info(f"Tool scores: {scores_summary}")

        # Filtrar: 1) pasan threshold Y 2) is_relevant() devuelve True
        # Optimización: si una tool tiene score perfecto (≥0.95), ignorar las demás
        PERFECT_SCORE_THRESHOLD = 0.95
        selected_tools = []
        tools_not_relevant = []

        # Verificar si hay alguna tool con score perfecto
        has_perfect_score = any(
            ts.passes_threshold and ts.score >= PERFECT_SCORE_THRESHOLD
            for ts in tool_scores.values()
        )

        for ts in tool_scores.values():
            # Si hay una tool perfecta, solo incluirla a ella
            if has_perfect_score:
                if ts.score >= PERFECT_SCORE_THRESHOLD:
                    selected_tools.append(ts)
                    self.logger.info(
                        f"Tool {ts.tool_name}: score={ts.score:.3f} (PERFECT) → SELECTED (others ignored)"
                    )
                else:
                    self.logger.info(
                        f"Tool {ts.tool_name}: score={ts.score:.3f} → SKIPPED (perfect score present)"
                    )
                continue

            if not ts.passes_threshold:
                continue

            selected_tools.append(ts)

        selected_tools.sort(key=lambda x: x.score, reverse=True)

        plan.has_valid_tools = len(selected_tools) > 0

        if tools_not_relevant:
            self.logger.info(f"Tools filtradas por is_relevant(): {tools_not_relevant}")

        if not selected_tools:
            # No hay tools que pasen ambas validaciones
            # RAG como fallback GENERAL - si ninguna tool pasa, RAG siempre puede ayudar
            # ya que todo se indexa a Qdrant
            self.logger.info(
                "No hay tools que pasen threshold + is_relevant. "
                "Usando RAG como fallback general (todo está indexado en Qdrant)"
            )
            plan.needs_rag = True
            plan.rag_reason = "fallback_no_tools_passed_threshold"
            plan.debug_message = "RAG como fallback general"
            plan.tools_to_execute = ["rag_search"]
            self.logger.info(f"Plan: {plan.debug_message}")
            return plan

        # Determinar qué ejecutar
        tools_to_execute = []
        needs_rag = False
        rag_reason = None

        for ts in selected_tools:
            if ts.tool_name == "rag_search":
                needs_rag = True
                rag_reason = f"primary_intent: {ts.best_intent}"
                self.logger.info(
                    f"RAG seleccionado como tool primaria "
                    f"(score: {ts.score:.3f}, intent: {ts.best_intent})"
                )
            else:
                tools_to_execute.append(ts.tool_name)
                self.logger.info(
                    f"Tool seleccionada: {ts.tool_name} "
                    f"(score: {ts.score:.3f}, intent: {ts.best_intent})"
                )

        # RAG va primero como provider de contexto
        if needs_rag:
            plan.tools_to_execute = ["rag_search"] + tools_to_execute
        else:
            plan.tools_to_execute = tools_to_execute

        plan.needs_rag = needs_rag
        plan.rag_reason = rag_reason

        plan.debug_message = (
            f"Ejecutar: {plan.tools_to_execute} | "
            f"RAG: {needs_rag} ({rag_reason})"
        )

        self.logger.info(f"Plan final: {plan.debug_message}")
        self.logger.info("=== FIN FASE 1: PLANIFICACIÓN ===")

        return plan

    async def _execute_tools_context(
        self,
        conversation: Conversation,
        user_message: str,
        settings: ConversationSettings,
        rag_metadata: Dict[str, Any],
        file_ids: Optional[List[UUID]] = None,
        collection_name: Optional[str] = None,
        tools_to_exclude: Optional[List[str]] = None,
        target_file_id: Optional[UUID] = None,
        tools_to_execute: Optional[List[str]] = None,
        tool_scores: Optional[Dict[str, ToolScoreInfo]] = None,
    ) -> tuple:
        """
        Ejecuta las tools seleccionadas por el plan (o por scoring en tiempo real como fallback)
        y devuelve las partes de contexto generadas + nombres de tools ejecutadas.

        Flujo principal (cuando viene un plan precomputado de _plan_tool_execution):
            1. Convertir ToolScoreInfo → RouterToolScore para compatibilidad interna.
            2. Filtrar solo las tools que el plan decidió ejecutar (tools_to_execute).
            3. Para cada tool, resolver si debe usarse una CustomToolExecutor en lugar
               de la tool física directamente (lógica: custom > física).
            4. Inyectar el contexto de conversación (file_ids, collection_name,
               target_file_id, filters) en execution_strategy ANTES de extraer params.
               Este es el paso crítico que estaba ausente en el v2 original.
            5. Extraer parámetros finales y ejecutar.

        Flujo fallback (sin plan precomputado):
            - Se hace scoring en tiempo real via IntentRouter.
            - RAG se excluye aquí (no tiene tratamiento especial, el plan ya lo incluye).

        Args:
            conversation:      Objeto Conversation activo.
            user_message:      Mensaje del usuario (usado para scoring fallback).
            settings:          ConversationSettings con tools habilitadas y proveedor.
            rag_metadata:      Metadata RAG acumulada (puede llegar vacío desde el caller).
            file_ids:          IDs de archivos adjuntos a la conversación.
            collection_name:   Nombre de colección Qdrant (para RAG y custom RAG tools).
            tools_to_exclude:  Tools que el caller quiere forzar a omitir.
            target_file_id:    Archivo objetivo ya detectado por TargetFileDetector.
            tools_to_execute:  Lista ordenada de tools del plan (incluye rag_search si aplica).
            tool_scores:       Scores precalculados por _plan_tool_execution.

        Returns:
            tuple(context_parts: List[str], tools_executed: List[str])
        """
        from src.tools.custom_tool import CustomToolExecutor
        from src.services.intent.router import ToolScore as RouterToolScore

        context_parts: List[str] = []
        tools_executed: List[str] = []
        tools_to_exclude = tools_to_exclude or []

        # =========================================================
        # PASO 1: Obtener la lista de ToolScores a ejecutar
        # Dos caminos: plan precomputado (rápido) vs scoring en tiempo real (fallback)
        # =========================================================
        if tools_to_execute is not None and tool_scores is not None:
            # --- Camino principal: plan ya calculado por _plan_tool_execution ---
            self.logger.info(f"Using pre-computed plan: {tools_to_execute}")

            # Reconvertir ToolScoreInfo (schema) → RouterToolScore (dominio interno)
            # Necesario porque _plan_tool_execution serializa a ToolScoreInfo para el plan.
            precomputed_scores: Dict[str, RouterToolScore] = {}
            for tool_name, tsi in tool_scores.items():
                precomputed_scores[tool_name] = RouterToolScore(
                    tool_name=tsi.tool_name,
                    score=tsi.score,
                    best_intent=tsi.best_intent,
                    best_intent_action=tsi.best_intent_action,
                    passes_threshold=tsi.passes_threshold,
                    requires_target=tsi.requires_target,
                    confidence_threshold=0.65,  # default, ya no se usa para filtrar aquí
                    default_params=tsi.default_params,
                    target=tsi.target,
                    method="precomputed"
                )

            # Filtrar solo las tools que el plan indicó ejecutar.
            # IMPORTANTE: NO filtrar por passes_threshold aquí.
            # El plan puede incluir rag_search como fallback aunque su score no pasó
            # el umbral, y eso es una decisión válida ya tomada por _plan_tool_execution.
            selected_tool_scores = [
                ts for name, ts in precomputed_scores.items()
                if name in tools_to_execute
            ]
            # Respetar el orden del plan (ya está ordenado por score desc en _plan_tool_execution)
            selected_tool_scores.sort(key=lambda x: x.score, reverse=True)

            if not selected_tool_scores:
                self.logger.info(
                    "Pre-computed plan produced no executable tool scores. "
                    f"tools_to_execute={tools_to_execute} | "
                    f"available_scores={list(precomputed_scores.keys())}"
                )
                return [], []

        else:
            # --- Camino fallback: scoring en tiempo real ---
            # Se usa cuando _execute_tools_context se llama sin un plan previo
            # (compatibilidad con llamadas legacy o paths de Agent mode que no planifican).
            # RAG se excluye deliberadamente aquí: en el flujo moderno, RAG es
            # responsabilidad exclusiva del plan (_plan_tool_execution lo incluye).
            enabled_tools = settings.enabled_tools or []

            tools_to_score = [
                t for t in enabled_tools
                if t not in (["rag_search"] + tools_to_exclude) and tool_registry.get(t)
            ]

            if not tools_to_score:
                self.logger.info("No scoreable tools found for fallback path.")
                return [], []

            intent_router = await self._get_intent_router()

            tool_scores_raw: Dict[str, RouterToolScore] = await intent_router.score_tools_for_query(
                query=user_message,
                enabled_tool_names=tools_to_score,
                context={"attached_files": file_ids or [], "file_names": []},
                llm_provider=settings.provider,
                llm_model=settings.model
            )

            selected_tool_scores = [
                ts for ts in tool_scores_raw.values() if ts.passes_threshold
            ]
            selected_tool_scores.sort(key=lambda x: x.score, reverse=True)

            if not selected_tool_scores:
                self.logger.info(
                    f"No tools passed intent threshold (fallback scoring) | "
                    f"scores: { {n: round(ts.score, 3) for n, ts in tool_scores_raw.items()} }"
                )
                return [], []

        # =========================================================
        # PASO 2: Log del target file que se usará (ya detectado upstream)
        # =========================================================
        if target_file_id:
            self.logger.info(f"Using pre-detected target_file_id: {target_file_id}")

        # =========================================================
        # PASO 3: Loop de ejecución por tool
        # =========================================================
        for tool_score in selected_tool_scores:
            tool_name = tool_score.tool_name

            # --- 3a. Resolución: custom tool vs tool física ---
            # Regla: si existe una CustomToolExecutor configurada para este tool_type,
            # se usa en lugar de la tool física del registry.
            # Esto aplica a TODAS las tools (rag_search, codebase_tool, etc.),
            # no solo a RAG. Un usuario puede tener "Codebase Rápido" como custom tool
            # de tipo codebase_tool con parámetros preconfigurados.
            tool = await self._resolve_tool_instance(tool_name)

            if not tool:
                self.logger.warning(f"Tool '{tool_name}' not found in registry or custom tools")
                continue

            try:
                self.logger.info(
                    f"Executing tool: {tool_name} | "
                    f"intent={tool_score.best_intent} action={tool_score.best_intent_action} "
                    f"score={tool_score.score:.3f} target={tool_score.target} "
                    f"is_custom={isinstance(tool, CustomToolExecutor)}"
                )

                # Inyectar repos en CustomToolExecutor si es necesario
                # (puede haber sido instanciado sin ellos en _resolve_tool_instance)
                if isinstance(tool, CustomToolExecutor):
                    tool.file_repo = self.file_repo
                    tool.custom_tool_repo = self.custom_tool_repo
                    await tool._load_custom_tool_config()

                # DESPUÉS: construir el dict de hints ANTES de llamar, y pasarlo como rag_metadata:
                precomputed_hints = {}
                if tool_score.best_intent_action:
                    precomputed_hints["intent_action"] = tool_score.best_intent_action
                # intent_name: el nombre granular del intent ganador (ej: "count_methods")
                # Se usa como sub_action cuando intent_action es "basic_analyze_file",
                # permitiendo que llm_formatter._format_basic() responda con granularidad.
                if tool_score.best_intent:
                    precomputed_hints["intent_name"] = tool_score.best_intent
                if tool_score.target:
                    precomputed_hints["intent_target"] = tool_score.target
                if tool_score.default_params:
                    precomputed_hints["intent_default_params"] = tool_score.default_params

                # --- 3b. Determinar estrategia de ejecución base (delegado a ToolExecutor) ---
                execution_strategy = await self.tool_executor.determine_execution_strategy(
                    tool, tool_name, user_message, precomputed_hints,
                    file_ids=file_ids,
                    target_file_id=target_file_id,
                    settings=settings
                )

                # --- 3c. Inyectar hints del IntentRouter (override suave: setdefault) ---
                # No sobreescribir si determine_execution_strategy ya los resolvió.
                if tool_score.best_intent_action:
                    execution_strategy.setdefault("intent_action", tool_score.best_intent_action)
                if tool_score.best_intent:
                    execution_strategy.setdefault("intent_name", tool_score.best_intent)
                if tool_score.target:
                    execution_strategy.setdefault("intent_target", tool_score.target)
                if tool_score.default_params:
                    execution_strategy.setdefault("intent_default_params",
                                                  tool_score.default_params)

                # --- 3d. INYECCIÓN DE CONTEXTO DE CONVERSACIÓN ---
                # Este es el paso crítico ausente en el v2 original.
                # Cualquier tool de tipo RAG (física o custom) necesita saber:
                #   - Qué archivos buscar (file_ids → para construir colecciones o filtros)
                #   - En qué colección Qdrant buscar (collection_name)
                #   - Si hay un archivo objetivo específico (target_file_id → filtro Qdrant)
                #
                # Para CustomToolExecutor: estos valores llegan como kwargs a execute(),
                # donde se hace {**interpolated_config, **kwargs} con kwargs teniendo
                # PRIORIDAD. Así el contexto de conversación siempre supera la config
                # del template, pero el template sigue aplicando para parámetros como
                # search_mode, k, score_threshold, etc.
                #
                # Para RAGTool física: los parámetros deben estar en execution_strategy
                # para que extract_tool_parameters los incluya en extracted_params.
                tool_is_rag = (
                    tool_name == "rag_search" or
                    (isinstance(tool, CustomToolExecutor) and tool.tool_type == "rag_search")
                )

                if tool_is_rag:
                    # Propagar file_ids como strings (RAGTool los usa para construir
                    # collections si no se especifica collection_name explícitamente)
                    if file_ids:
                        execution_strategy["file_ids"] = [str(fid) for fid in file_ids]

                    # collection_name desde el contexto de la conversación.
                    # Tiene prioridad sobre cualquier colección hardcodeada en el template
                    # de la custom tool, porque refleja el contexto actual del usuario.
                    if collection_name:
                        execution_strategy["collection_name"] = collection_name

                    # Construir filtro Qdrant por archivo objetivo.
                    # Si el usuario pregunta sobre un archivo específico,
                    # RAG debe limitar su búsqueda a los chunks de ese archivo.
                    if target_file_id:
                        # Usar setdefault para no sobreescribir si el template ya tiene filters
                        # propios (e.g., {"method": "GET"}). En su lugar, añadir file_id al dict.
                        existing_filters = execution_strategy.get("filters", {})
                        existing_filters["file_id"] = str(target_file_id)
                        execution_strategy["filters"] = existing_filters
                        self.logger.info(
                            f"RAG filter by target file: {target_file_id} "
                            f"(tool={'custom' if isinstance(tool, CustomToolExecutor) else 'physical'})"
                        )

                # Para codebase_tool: propagar target_file_id directamente
                if target_file_id and tool_name in ("codebase_analyzer", "codebase_tool"):
                    execution_strategy["target_file_id"] = str(target_file_id)
                    self.logger.info(f"Passing target_file_id to {tool_name}: {target_file_id}")

                # --- 3e. Extraer parámetros finales (delegado a ToolExecutor) ---
                extracted_params = await self.tool_executor.extract_tool_parameters(
                    tool, tool_name, user_message, conversation, settings, execution_strategy,
                    collection_name=collection_name
                )

                # --- 3f. Ejecutar tool ---
                result = await self.tool_executor.execute_tool_with_context(
                    tool, conversation, collection_name=collection_name, **extracted_params
                )

                # --- 3g. Procesar resultado ---
                if result.success and result.data:
                    tool_context = f"## Result from {tool_name}\n\n{result.data}\n\n"
                    context_parts.append(tool_context)
                    tools_executed.append(tool_name)
                    self.logger.info(f"Tool '{tool_name}' executed successfully")

                    # Capturar content_prompt de custom tools para header dinámico en
                    # build_context_string (permite que el usuario personalice el encabezado
                    # del bloque de contexto que verá el LLM)
                    if isinstance(tool, CustomToolExecutor) and tool.content_prompt:
                        if not hasattr(self, '_tool_content_prompts'):
                            self._tool_content_prompts = {}
                        self._tool_content_prompts[tool_name] = tool.content_prompt
                        self.logger.info(f"Captured content_prompt for tool '{tool_name}'")
                else:
                    # Registrar resultado vacío igualmente para que el LLM sepa
                    # que la tool se intentó pero no encontró información.
                    context_parts.append(
                        f"## Result from {tool_name}\n\nNo information found.\n\n"
                    )

            except Exception as e:
                self.logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
                context_parts.append(
                    f"## Result from {tool_name}\n\nError: {str(e)}\n\n"
                )

        return context_parts, tools_executed

    async def _resolve_tool_instance(self, tool_name: str):
        """
        Resuelve qué instancia de tool usar para un tool_name dado.

        Lógica de resolución (prioridad descendente):
            1. CustomToolExecutor configurada por el usuario para este tool_type.
               El usuario puede tener "RAG Semántico", "RAG Rápido", etc. configurados.
               Se usa la primera custom tool activa que coincida con el tool_type.
            2. Tool física del registry (RAGTool, CodebaseTool, etc.).
            3. None si no se encuentra ninguna.

        Nota: La resolución por custom tool se hace aquí de forma centralizada,
        reemplazando el bloque hardcodeado que solo aplicaba para rag_search en
        el v2 original. Ahora funciona para cualquier tool_type.

        Args:
            tool_name: Nombre registrado de la tool (e.g., "rag_search", "codebase_tool")

        Returns:
            Instancia de BaseTool (física o CustomToolExecutor) o None.
        """
        from src.tools.custom_tool import CustomToolExecutor

        # Intentar primero con custom tools del usuario
        try:
            custom_tools = await self.context_builder._get_custom_tools_by_name(tool_name)
            if custom_tools:
                # Usar la primera custom tool activa (orden determinístico desde BD)
                custom_tool_record = custom_tools[0]
                tool = CustomToolExecutor(
                    custom_tool_id=custom_tool_record.id,
                    file_repo=self.file_repo,
                    custom_tool_repo=self.custom_tool_repo
                )
                await tool._load_custom_tool_config()
                self.logger.info(
                    f"Resolved '{tool_name}' → CustomToolExecutor '{custom_tool_record.name}'"
                )

                return tool
        except Exception as e:
            # Si la resolución de custom tools falla, continuar con la tool física.
            # No es un error bloqueante: la tool física puede ejecutarse igualmente.
            self.logger.warning(
                f"Custom tool resolution failed for '{tool_name}': {e}. "
                f"Falling back to physical tool."
            )

        # Fallback: tool física del registry
        physical_tool = tool_registry.get(tool_name)
        if physical_tool:
            self.logger.debug(f"Resolved '{tool_name}' → physical tool from registry")
        return physical_tool

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

    def _provider_supports_message_history(self, provider: Any, model_name: str) -> bool:
        """Check if provider supports message history."""
        try:
            models = provider.get_available_models()
            info = next((m for m in models if m.name == model_name), None)
            return info.supports_message_history if info else True
        except Exception:
            return True

    def _format_single_message_for_provider(self, messages: List[ChatMessage], user_message: str) -> List[ChatMessage]:
        """Format messages into a single flattext message for providers without history support."""
        system_prompt = next((m.content for m in messages if m.role == "system"), None)
        tool_context = next((m.content for m in messages if m.role == "tool"), None)
        
        parts = []
        if tool_context:
            parts.append(f"## Contexto disponible\n{tool_context}\n---")
        if system_prompt:
            parts.append(f"## Instrucciones\n{system_prompt}\n---")
            
        current_user_msg = next((m.content for m in reversed(messages) if m.role == "user"), user_message)
        parts.append(current_user_msg)
        
        flat_message = "\n\n".join(parts)
        return [ChatMessage(role="user", content=flat_message)]

