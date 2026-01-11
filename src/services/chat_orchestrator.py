# =============================================================================
# src/services/chat_orchestrator.py
# Main chat orchestration logic
# =============================================================================
"""
Orquestador principal que maneja ambos modos: Agent y Manual
"""
import json
import re
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.models import Conversation, Message, MessageRole, ToolMode, HallucinationMode, File, CustomTool
from src.providers.manager import ChatMessage
from src.providers.manager import provider_manager, ChatResponse
from src.schemas.schemas import ConversationSettings
from src.services.conversation_memory import ConversationMemoryService
from src.services.extraction_service import ParameterExtractionService
from src.services.stream_cancel_manager import StreamCancelToken
from src.tools.base_tool import tool_registry, BaseTool, ToolResult, ToolCategory
from src.tools.custom_tool import CustomToolExecutor
from src.utils.logger import get_logger, set_conversation_context
from src.services.query_intent_analyzer import QueryIntentAnalyzer, NavigationIntent
from src.document_loaders.obsidian_tree_navigator import ObsidianTreeNavigator
# =============================================================================
# Pre-compiled Regex Patterns (for performance)
# =============================================================================
_CONTEXTUAL_PATTERNS = [
    re.compile(r'\beste\s+documento', re.IGNORECASE),
    re.compile(r'\bel\s+documento', re.IGNORECASE),
    re.compile(r'\beste\s+archivo', re.IGNORECASE),
    re.compile(r'\bel\s+archivo', re.IGNORECASE),
    re.compile(r'\beste\s+adjunto', re.IGNORECASE),
    re.compile(r'\bel\s+adjunto', re.IGNORECASE),
    re.compile(r'\bel\s+fichero', re.IGNORECASE),
    re.compile(r'\beste\s+fichero', re.IGNORECASE),
    re.compile(
        r'\beste\s+(pdf|word|excel|csv|html|ppt|pptx|docx|xlsx|txt|md|markdown|imagen|foto|video|audio)',
        re.IGNORECASE),
    re.compile(
        r'\bel\s+(pdf|word|excel|csv|html|ppt|pptx|docx|xlsx|txt|md|markdown|imagen|foto|video|audio)',
        re.IGNORECASE),
]


class ChatOrchestrator:
    """
    Orchestrator for chat conversations
    Handles both agent and manual tool modes
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.logger = get_logger(__name__)
        self.memory_service = ConversationMemoryService(db)  # Memory service

        # Performance caches
        self._settings_cache = {}  # Cache parsed settings per conversation
        self._rag_tool = tool_registry.get("rag_search")  # Pre-cache RAG tool
        self._custom_tools_cache = {}  # Cache custom tools by ID
        self.extraction_service = ParameterExtractionService(db) # Extraction service
        
        # Nuevos componentes
        self._intent_analyzer = QueryIntentAnalyzer()
        self._obsidian_navigator: Optional[ObsidianTreeNavigator] = None


    async def process_message(
        self,
        conversation: Conversation,
        user_message: str,
        file_ids: Optional[List[UUID]] = None
    ) -> ChatResponse:
        """
        Process user message and generate response

        Args:
            conversation: Conversation object
            user_message: User's message content
            file_ids: Optional list of attached file IDs

        Returns:
            ChatResponse with generated content
        """
        # Set conversation context for logging
        set_conversation_context(str(conversation.id))
        self.logger.info("process_message started", extra={"conversation_id": str(conversation.id)})

        settings = self._parse_settings(conversation.id, conversation.settings)

        # Optimized logging - avoid dict construction in hot path
        self.logger.info(
            f"Processing: mode={settings.tool_mode.value}, msg_len={len(user_message)}, files={len(file_ids) if file_ids else 0}")

        try:
            # Obtener configuraciones actualizadas de herramientas
            tool_configs = await self._get_active_tool_configurations(conversation.id)

            # Determine mode
            if settings.tool_mode == ToolMode.AGENT:
                response = await self._agent_mode(conversation, user_message, settings, file_ids, tool_configs)
            else:
                response = await self._manual_mode(conversation, user_message, settings, file_ids, tool_configs)

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
        cancel_token: Optional[StreamCancelToken] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process message with streaming support

        PHASE 1: Pre-processing (NO streaming)
        - Execute RAG search
        - Execute memory service
        - Execute tools if needed
        - Build enriched context

        PHASE 2: Stream LLM response
        - Stream final response with context

        PHASE 3: Send metadata
        - Tools executed, model info, etc.

        Yields:
            Dict with:
            - {"type": "content", "chunk": "..."} → content chunks
            - {"type": "metadata", "data": {...}} → final metadata
        """
        set_conversation_context(str(conversation.id))
        self.logger.info("process_message_stream started",
                         extra={"conversation_id": str(conversation.id)})

        settings = self._parse_settings(conversation.id, conversation.settings)

        self.logger.info(
            "Processing streaming chat message",
            extra={
                "tool_mode": settings.tool_mode.value,
                "message_length": len(user_message),
                "file_count": len(file_ids) if file_ids else 0
            }
        )

        try:
            # Obtener configuraciones actualizadas de herramientas
            tool_configs = await self._get_active_tool_configurations(conversation.id)

            # Determine mode
            if settings.tool_mode == ToolMode.AGENT:
                async for chunk in self._agent_mode_stream(conversation, user_message, settings, file_ids, cancel_token, tool_configs):
                    yield chunk
            else:
                async for chunk in self._manual_mode_stream(conversation, user_message, settings, file_ids, cancel_token, tool_configs):
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
        tool_configs: Optional[Dict[str, Any]] = None
    ) -> ChatResponse:
        """
        Agent mode: LLM decides which tools to use via function calling
        """
        # Get conversation history
        messages = await self._build_message_history(conversation, user_message, settings)

        # ========================================================
        # PRE-EXECUTE RAG for attached files (proactive context)
        # This ensures LLM has document context before deciding tools
        # ========================================================
        context_parts = []
        available_tools = settings.available_tools or settings.enabled_tools

        if file_ids and "rag_search" in (available_tools or []):
            self.logger.info("Agent mode: Pre-executing RAG for attached files")
            rag_context = await self._execute_rag_tool(
                conversation,
                user_message,
                settings,
                file_ids
            )
            if rag_context:
                context_parts.append(rag_context)
                self.logger.info("Agent mode: RAG context added proactively")

        # Build context message if we have any
        if context_parts:
            context_string = self._build_context_string(context_parts)
            user_message = f"{context_string}\n\n---\n\n{user_message}"

        # Add current user message with context
        # Always add the enriched message (with context) to ensure LLM receives it
        # Remove any duplicate of the original message first
        if messages and len(messages) > 1:  # Keep system prompt
            # Remove last message if it's the original user message (without context)
            original_user_messages = [i for i, msg in enumerate(messages) if msg.role == "user"]
            if original_user_messages:
                # Remove all previous user messages to avoid duplication
                messages = [msg for msg in messages if msg.role != "user"]

        messages.append(ChatMessage(role="user", content=user_message))

        # Get available tools
        available_tools = settings.available_tools or settings.enabled_tools

        if not available_tools:
            # No tools, just chat
            return await self._simple_chat(messages, settings)

        # Get tool definitions for function calling
        provider = provider_manager.get_provider(settings.provider)

        # Convert tools to appropriate format
        if settings.provider == "openai":
            tool_definitions = tool_registry.to_openai_functions(available_tools)
        elif settings.provider == "anthropic":
            tool_definitions = tool_registry.to_anthropic_tools(available_tools)
        else:
            # Local provider doesn't support function calling well
            # Fall back to manual mode
            return await self._manual_mode(conversation, user_message, settings, file_ids,tool_configs)

        # Dynamic max_tokens adjustment for short queries
        adjusted_max_tokens = settings.max_tokens
        if len(user_message) < 100 and settings.max_tokens > 500:
            adjusted_max_tokens = 500
            self.logger.debug(
                f"Short message detected - reducing max_tokens to {adjusted_max_tokens}")

        # Preparar kwargs con configuraciones
        kwargs = {}
        if tool_configs:
            kwargs.update(tool_configs)

        response = await provider.chat(
            messages=messages,
            model=settings.model,
            temperature=settings.temperature,
            max_tokens=adjusted_max_tokens,
            tools=tool_definitions,
            tool_choice="auto",
            **kwargs
        )

        # Check if LLM wants to use tools
        if not response.tool_calls:
            # No tools needed, return response
            return response

        # Execute requested tools
        tool_results = []
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = json.loads(tool_call["arguments"]) if isinstance(tool_call["arguments"],
                                                                           str) else tool_call[
                "arguments"]

            tool = tool_registry.get(tool_name)
            if tool:
                try:
                    result = await self._execute_tool(tool, conversation, **tool_args)
                    tool_results.append({
                        "tool_call_id": tool_call["id"],
                        "tool_name": tool_name,
                        "result": tool.format_output(result),
                        "success": result.success
                    })
                except Exception as e:
                    tool_results.append({
                        "tool_call_id": tool_call["id"],
                        "tool_name": tool_name,
                        "result": f"Error: {str(e)}",
                        "success": False
                    })
            else:
                # Tool not found in registry - this should not happen if tool is properly registered
                tool_results.append({
                    "tool_call_id": tool_call["id"],
                    "tool_name": tool_name,
                    "result": f"Error: Tool '{tool_name}' not found in registry",
                    "success": False
                })

        # Add tool results to messages and make second LLM call
        messages_with_tools = messages + [
            ChatMessage(role="assistant", content=response.content or ""),
            ChatMessage(
                role="tool",
                content=json.dumps(tool_results)
            )
        ]

        # Second LLM call with tool results
        final_response = await provider.chat(
            messages=messages_with_tools,
            model=settings.model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens
        )

        # Add tool execution metadata
        final_response.metadata["tools_executed"] = [tr["tool_name"] for tr in tool_results]
        final_response.metadata["tool_results"] = tool_results

        return final_response

    async def _agent_mode_stream(
        self,
        conversation: Conversation,
        user_message: str,
        settings: ConversationSettings,
        file_ids: Optional[List[UUID]],
        cancel_token: Optional[StreamCancelToken] = None,
        tool_configs: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Agent mode with streaming support

        Uses 2-pass approach:
        PASS 1: Non-streaming call to get tool requests from LLM
        PASS 2: Stream final response after tool execution
        """
        # Get conversation history
        messages = await self._build_message_history(conversation, user_message, settings)

        # === CONTEXT WINDOW VALIDATION (Agent Mode) ===
        total_chars = sum(len(msg.content) for msg in messages if msg.content)
        estimated_tokens = total_chars // 3

        provider = provider_manager.get_provider(settings.provider)
        model_context_window = provider.context_window if hasattr(provider, 'context_window') else 8192

        if estimated_tokens > model_context_window * 0.9:
            self.logger.warning(
                "High context usage in agent mode - may affect performance",
                extra={
                    "estimated_tokens": estimated_tokens,
                    "context_window": model_context_window
                }
            )

        # ========================================================
        # PRE-EXECUTE RAG for attached files (proactive context)
        # ========================================================
        context_parts = []
        available_tools = settings.available_tools or settings.enabled_tools

        if file_ids and "rag_search" in (available_tools or []):
            self.logger.info("Agent mode stream: Pre-executing RAG for attached files")
            rag_context = await self._execute_rag_tool(
                conversation,
                user_message,
                settings,
                file_ids
            )
            if rag_context:
                context_parts.append(rag_context)
                self.logger.info("Agent mode stream: RAG context added proactively")

        # Build context message if we have any
        if context_parts:
            context_string = self._build_context_string(context_parts)
            user_message = f"{context_string}\n\n---\n\n{user_message}"

        # Add current user message with context
        # Always add the enriched message (with context) to ensure LLM receives it
        # Remove any duplicate of the original message first
        if messages and len(messages) > 1:  # Keep system prompt
            # Remove last message if it's the original user message (without context)
            original_user_messages = [i for i, msg in enumerate(messages) if msg.role == "user"]
            if original_user_messages:
                # Remove all previous user messages to avoid duplication
                messages = [msg for msg in messages if msg.role != "user"]

        messages.append(ChatMessage(role="user", content=user_message))

        # Get available tools
        available_tools = settings.available_tools or settings.enabled_tools

        if not available_tools:
            # No tools, just stream chat
            provider = provider_manager.get_provider(settings.provider)

            # Check if provider supports cancellation
            if hasattr(provider, 'cancellable_stream_chat') and cancel_token:
                stream_method = provider.cancellable_stream_chat
                kwargs = {
                    "cancel_event": cancel_token.cancel_event,
                    "cancel_check_interval": 0.1
                }
            else:
                stream_method = provider.stream_chat
                kwargs = {}

            # Agregar configuraciones de herramientas a kwargs
            if tool_configs:
                kwargs.update(tool_configs)

            async for chunk in stream_method(
                messages=messages,
                model=settings.model,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                **kwargs
            ):
                # Handle both string chunks (content) and dict chunks (thinking)
                if isinstance(chunk, dict) and chunk.get("type") == "thinking":
                    yield {"type": "thinking", "content": chunk.get("content", "")}
                else:
                    yield {"type": "content", "chunk": chunk}

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

        # Get tool definitions for function calling
        provider = provider_manager.get_provider(settings.provider)

        # Convert tools to appropriate format
        if settings.provider == "openai":
            tool_definitions = tool_registry.to_openai_functions(available_tools)
        elif settings.provider == "anthropic":
            tool_definitions = tool_registry.to_anthropic_tools(available_tools)
        else:
            # Local provider doesn't support function calling well
            # Fall back to manual mode streaming
            async for chunk in self._manual_mode_stream(conversation, user_message, settings, file_ids, cancel_token,tool_configs):
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
                "data": {
                    "tools_executed": [],
                    "mode": "agent",
                    "provider": settings.provider,
                    "model": settings.model
                }
            }
            return

        # Execute requested tools
        self.logger.info(f"Executing {len(response.tool_calls)} tool(s)...")
        tool_results = []
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = json.loads(tool_call["arguments"]) if isinstance(tool_call["arguments"],
                                                                           str) else tool_call[
                "arguments"]

            tool = tool_registry.get(tool_name)
            if tool:
                try:
                    result = await self._execute_tool(tool, conversation, **tool_args)
                    tool_results.append({
                        "tool_call_id": tool_call["id"],
                        "tool_name": tool_name,
                        "result": tool.format_output(result),
                        "success": result.success
                    })
                except Exception as e:
                    tool_results.append({
                        "tool_call_id": tool_call["id"],
                        "tool_name": tool_name,
                        "result": f"Error: {str(e)}",
                        "success": False
                    })
            else:
                # Tool not found in registry - this should not happen if tool is properly registered
                tool_results.append({
                    "tool_call_id": tool_call["id"],
                    "tool_name": tool_name,
                    "result": f"Error: Tool '{tool_name}' not found in registry",
                    "success": False
                })

        # PASS 2: Second LLM call WITH streaming and tool results
        self.logger.info("Agent mode: Second pass with tool results (streaming)...")
        messages_with_tools = messages + [
            ChatMessage(role="assistant", content=response.content or ""),
            ChatMessage(
                role="tool",
                content=json.dumps(tool_results)
            )
        ]

        # Stream final response
        # Check if provider supports cancellation
        if hasattr(provider, 'cancellable_stream_chat') and cancel_token:
            stream_method = provider.cancellable_stream_chat
            kwargs = {
                "cancel_event": cancel_token.cancel_event,
                "cancel_check_interval": 0.1
            }
        else:
            stream_method = provider.stream_chat
            kwargs = {}

        # Agregar configuraciones de herramientas a kwargs
        if tool_configs:
            kwargs.update(tool_configs)

        async for chunk in stream_method(
            messages=messages_with_tools,
            model=settings.model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            **kwargs
        ):
            # Handle both string chunks (content) and dict chunks (thinking)
            # Also handle JSON string chunks from LocalProvider
            if isinstance(chunk, dict) and chunk.get("type") == "thinking":
                yield {"type": "thinking", "content": chunk.get("content", "")}
            elif isinstance(chunk, str):
                # Check if chunk is a JSON string
                try:
                    parsed_chunk = json.loads(chunk)
                    if isinstance(parsed_chunk, dict):
                        if parsed_chunk.get("type") == "thinking":
                            # It's a thinking chunk
                            yield {"type": "thinking", "content": parsed_chunk.get("content", "")}
                        elif parsed_chunk.get("type") == "content":
                            # It's a content chunk
                            yield {"type": "content", "chunk": parsed_chunk.get("chunk", "")}
                        else:
                            # Regular content chunk (string)
                            yield {"type": "content", "chunk": chunk}
                    else:
                        # Regular content chunk (string)
                        yield {"type": "content", "chunk": chunk}
                except json.JSONDecodeError:
                    # Not a JSON string, treat as regular content
                    yield {"type": "content", "chunk": chunk}
            else:
                yield {"type": "content", "chunk": chunk}

        # Send metadata
        yield {
            "type": "metadata",
            "data": {
                "tools_executed": [tr["tool_name"] for tr in tool_results],
                "tool_results": tool_results,
                "mode": "agent",
                "provider": settings.provider,
                "model": settings.model
            }
        }

    # =============================================================================
    # Manual Mode (Programmatic tool orchestration)
    # =============================================================================

    async def _manual_mode(
        self,
        conversation: Conversation,
        user_message: str,
        settings: ConversationSettings,
        file_ids: Optional[List[UUID]],
        tool_configs: Optional[Dict[str, Any]] = None
    ) -> ChatResponse:
        """
        Manual mode: System decides which tools to execute based on configuration
        """
        t_start = time.perf_counter()
        context_parts = []
        tools_executed = []

        # Get conversation history
        messages = await self._build_message_history(conversation, user_message, settings)

        # Execute enabled tools based on heuristics
        enabled_tools = settings.enabled_tools
        self.logger.debug(f"Enabled tools: {enabled_tools}")

        # RAG Search Tool
        if "rag_search" in enabled_tools:
            t_rag_start = time.perf_counter()
            self.logger.info("RAG Search tool enabled - executing")
            rag_context = await self._execute_rag_tool(
                conversation,
                user_message,
                settings,
                file_ids  # Pass file_ids from current message
            )
            t_rag_end = time.perf_counter()
            self.logger.info(f"[PERF] _execute_rag_tool: {(t_rag_end - t_rag_start) * 1000:.2f}ms")

            if rag_context:
                context_parts.append(rag_context)
                tools_executed.append("rag_search")
                self.logger.info("RAG context added to context")
            else:
                self.logger.warning("RAG tool returned no context")
        else:
            self.logger.debug("RAG Search tool not enabled - skipped")

        # Semantic Memory (replaces deep_thinking tool)
        if settings.memory_config.semantic_enabled:
            t_memory_start = time.perf_counter()
            self.logger.info("Semantic memory enabled - retrieving relevant context")
            memory_context = await self.memory_service.retrieve_relevant_context(
                conversation=conversation,
                current_query=user_message,
                memory_config=settings.memory_config.model_dump() if hasattr(settings.memory_config,
                                                                              'model_dump') else settings.memory_config
            )
            t_memory_end = time.perf_counter()
            self.logger.info(
                f"[PERF] memory_service.retrieve_relevant_context: {(t_memory_end - t_memory_start) * 1000:.2f}ms")

            if memory_context:
                context_parts.append(memory_context)
                self.logger.info("Semantic memory context added")
            else:
                self.logger.debug("No relevant memory context found")

        # Custom Tools (execute all enabled custom tools)
        # Note: For custom tools in manual mode, we need to define how to call them
        # This is a simplified implementation - in practice, you might want more sophisticated logic
        for tool_name in enabled_tools:
            if tool_name not in ["rag_search"] and not settings.memory_config.semantic_enabled:
                # This is a custom tool - try to execute it
                tool = tool_registry.get(tool_name)
                if tool:
                    try:
                        self.logger.info(f"Custom tool '{tool_name}' enabled - executing")
                        
                        # Ensure custom tool config is loaded before getting parameters
                        if isinstance(tool, CustomToolExecutor):
                            tool.db = self.db
                            await tool._load_custom_tool_config()
                            
                        # Extraer parámetros de forma profesional usando el ExtractionService
                        params_to_extract = tool.get_parameters()
                        extracted_params = {}
                        
                        if params_to_extract:
                            from dataclasses import asdict
                            self.logger.info(f"Extracting parameters for custom tool '{tool_name}'")
                            extracted_params = await self.extraction_service.extract_parameters(
                                user_message=user_message,
                                parameters=[asdict(p) for p in params_to_extract],
                                conversation_id=conversation.id,
                                provider=settings.provider,
                                model=settings.model
                            )
                            self.logger.info(f"Extracted parameters: {extracted_params}")

                        # Ejecutar con los parámetros extraídos
                        result = await self._execute_tool(tool, conversation, **extracted_params)

                        if result.success and result.data:
                            # Format the result as context
                            tool_context = f"## Result from {tool_name}\n\n{result.data}\n\n"
                            context_parts.append(tool_context)
                            tools_executed.append(tool_name)
                            self.logger.info(f"Custom tool '{tool_name}' executed successfully")
                        else:
                            self.logger.warning(f"Custom tool '{tool_name}' returned no data")
                    except Exception as e:
                        self.logger.error(f"Error executing custom tool '{tool_name}': {e}")
                else:
                    self.logger.warning(f"Custom tool '{tool_name}' not found in registry")

        # Document Processor Tool - REMOVED (now automatic during file upload)

        # Build enriched context
        if context_parts:
            t_context_start = time.perf_counter()
            self.logger.info(
                "Building context string",
                extra={"context_parts_count": len(context_parts)}
            )
            context_string = self._build_context_string(context_parts)
            # user_message preserved (no longer pre-pended, injected via system message later)
            t_context_end = time.perf_counter()
            self.logger.info(
                f"[PERF] _build_context_string: {(t_context_end - t_context_start) * 1000:.2f}ms")
            self.logger.debug(f"Total messages to LLM (before user msg): {len(messages)}")
        else:
            self.logger.warning("No context parts - proceeding without tool context")

        if context_parts:
            # CONTEXT AUTHORITY: Inject tool results as a separate SYSTEM message for higher priority
            context_string = self._build_context_string(context_parts)
            
            # --- GREEDY HISTORY SANITIZATION ---
            # If we have tool context, we remove any previous turns (User+Assistant) 
            # that were unsuccessful attempts at the SAME query.
            i = len(messages) - 1
            while i >= 1:
                if (messages[i].role == "assistant" and 
                    messages[i-1].role == "user" and 
                    messages[i-1].content.strip() == user_message.strip()):
                    self.logger.info(f"Greedy Sanitization: Removing previous speculative attempt for '{user_message[:20]}...'")
                    messages.pop(i)
                    messages.pop(i-1)
                    i -= 2
                    continue
                i -= 1

            messages.append(ChatMessage(role="system", content=context_string))
        
        messages.append(ChatMessage(role="user", content=user_message))
        t_append_end = time.perf_counter()
        self.logger.debug(
            f"[PERF] Append user message: {(t_append_end - t_append_start) * 1000:.2f}ms")

        # Generate response with enriched context
        t_provider_start = time.perf_counter()
        provider = provider_manager.get_provider(settings.provider)
        t_provider_end = time.perf_counter()
        self.logger.info(
            f"[PERF] provider_manager.get_provider: {(t_provider_end - t_provider_start) * 1000:.2f}ms")

        # Get model info to determine context window
        t_model_info_start = time.perf_counter()
        try:
            model_info = await provider_manager.get_available_models()
            # Find the specific model's context window
            for model in model_info:
                if model.name == settings.model and model.provider.value == settings.provider:
                    provider.context_window = model.context_window
                    break
            else:
                # Default context window if not found
                provider.context_window = 8192
        except Exception as e:
            self.logger.warning(f"Could not fetch model info: {e}, using default context window")
            provider.context_window = 8192
        t_model_info_end = time.perf_counter()
        self.logger.debug(
            f"[PERF] Get model info: {(t_model_info_end - t_model_info_start) * 1000:.2f}ms")

        # Dynamic max_tokens adjustment for short queries to improve performance
        # Only apply if max_tokens is high (e.g. default 2000) and message is short
        adjusted_max_tokens = settings.max_tokens
        if len(user_message) < 100 and settings.max_tokens > 500:
            adjusted_max_tokens = 500
            self.logger.debug(
                f"Short message detected - reducing max_tokens to {adjusted_max_tokens}")

        # === DETAILED MESSAGE ANALYSIS BEFORE LLM CALL ===
        total_chars = sum(len(msg.content) for msg in messages if msg.content)
        self.logger.info(
            f"[PERF] LLM Input Analysis: {len(messages)} messages, {total_chars:,} total chars")

        # Log each message size for detailed analysis
        for idx, msg in enumerate(messages):
            msg_len = len(msg.content) if msg.content else 0
            msg_preview = (msg.content[:80] + "...") if msg.content and len(
                msg.content) > 80 else msg.content
            self.logger.debug(
                f"[PERF]   Message[{idx}] ({msg.role}): {msg_len:,} chars - Preview: {msg_preview}")

        # === CONTEXT WINDOW VALIDATION ===
        # Estimate tokens and check against model's context window
        # Rough approximation: 3-4 characters per token
        estimated_tokens = total_chars // 3
        model_context_window = provider.context_window if hasattr(provider, 'context_window') else 8192

        self.logger.info(
            f"[CONTEXT] Estimated tokens: {estimated_tokens:,}, "
            f"Model context window: {model_context_window:,}, "
            f"Ratio: {estimated_tokens / model_context_window:.1%}")

        # Warn if approaching or exceeding context window
        if estimated_tokens > model_context_window * 0.9:
            self.logger.warning(
                "High context usage detected - approaching model's context window limit",
                extra={
                    "estimated_tokens": estimated_tokens,
                    "context_window": model_context_window,
                    "usage_ratio": estimated_tokens / model_context_window
                }
            )

            # Truncate messages if exceeding context window
            if estimated_tokens > model_context_window:
                self.logger.warning(
                    "Input exceeds model context window - truncating messages",
                    extra={
                        "estimated_tokens": estimated_tokens,
                        "context_window": model_context_window,
                        "excess_tokens": estimated_tokens - model_context_window
                    }
                )

                # Calculate how much we need to truncate
                max_chars = (model_context_window * 0.8) // 3  # Keep at 80% capacity
                current_chars = total_chars
                chars_to_remove = current_chars - max_chars

                # Remove oldest messages first (preserve current user message)
                messages_to_keep = []
                chars_kept = 0

                # Keep system prompt
                if messages and messages[0].role == "system":
                    messages_to_keep.append(messages[0])
                    chars_kept += len(messages[0].content) if messages[0].content else 0

                # Keep newest messages, removing oldest ones
                for msg in reversed(messages[1:]):
                    msg_chars = len(msg.content) if msg.content else 0
                    if chars_kept + msg_chars <= max_chars:
                        messages_to_keep.insert(0, msg)
                        chars_kept += msg_chars
                    else:
                        break

                messages = messages_to_keep
                total_chars = chars_kept
                estimated_tokens = total_chars // 3

                self.logger.warning(
                    "Messages truncated to fit context window",
                    extra={
                        "original_messages": len(messages) + len(messages_to_keep) - len(messages_to_keep),
                        "kept_messages": len(messages),
                        "original_chars": current_chars,
                        "kept_chars": total_chars,
                        "removed_chars": chars_to_remove
                    }
                )

        t_llm_start = time.perf_counter()
        self.logger.info(
            f"[PERF] Calling LLM (model={settings.model}, max_tokens={adjusted_max_tokens})")
        response = await provider.chat(
            messages=messages,
            model=settings.model,
            temperature=settings.temperature,
            max_tokens=adjusted_max_tokens,
            top_p=settings.top_p
        )
        t_llm_end = time.perf_counter()
        llm_time_ms = (t_llm_end - t_llm_start) * 1000
        self.logger.info(
            f"[PERF] provider.chat (LLM call) completed: {llm_time_ms:.2f}ms (~{llm_time_ms / 1000:.1f}s)")

        # Calculate approximate processing speed
        if response.content:
            response_len = len(response.content)
            chars_per_sec = response_len / (llm_time_ms / 1000) if llm_time_ms > 0 else 0
            self.logger.info(
                f"[PERF] LLM Response: {response_len:,} chars, ~{chars_per_sec:.0f} chars/sec")
        else:
            # Log warning when content is empty - may indicate thinking-mode model issue
            self.logger.warning(
                "LLM returned empty content - possible thinking-mode model issue",
                extra={
                    "model": settings.model,
                    "thinking_content": response.metadata.get("thinking_content", "N/A"),
                    "finish_reason": response.finish_reason
                }
            )

        # Add execution metadata
        t_metadata_start = time.perf_counter()
        response.metadata["tools_executed"] = tools_executed
        response.metadata["mode"] = "manual"
        t_metadata_end = time.perf_counter()
        self.logger.debug(
            f"[PERF] Add metadata: {(t_metadata_end - t_metadata_start) * 1000:.2f}ms")

        # Validate response if strict mode
        if settings.hallucination_control.mode == HallucinationMode.STRICT:
            t_validate_start = time.perf_counter()
            response = self._validate_strict_response(response, context_parts)
            t_validate_end = time.perf_counter()
            self.logger.info(
                f"[PERF] _validate_strict_response: {(t_validate_end - t_validate_start) * 1000:.2f}ms")

        # === PERFORMANCE TRACKING: Total time ===
        t_end = time.perf_counter()
        total_time_ms = (t_end - t_start) * 1000
        self.logger.info(f"[PERF] TOTAL _manual_mode execution: {total_time_ms:.2f}ms")

        self.logger.info("MANUAL mode processing completed")
        return response

    async def _manual_mode_stream(
        self,
        conversation: Conversation,
        user_message: str,
        settings: ConversationSettings,
        file_ids: Optional[List[UUID]],
        cancel_token: Optional[StreamCancelToken] = None,
        tool_configs: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Manual mode with streaming support

        PHASE 1: Execute tools (RAG, Memory) WITHOUT streaming
        PHASE 2: Stream LLM final response
        """
        self.logger.info("Starting MANUAL mode STREAMING processing")

        # Get conversation history
        messages = await self._build_message_history(conversation, user_message, settings)

        # === CONTEXT WINDOW VALIDATION (Streaming Mode) ===
        total_chars = sum(len(msg.content) for msg in messages if msg.content)
        estimated_tokens = total_chars // 3

        provider = provider_manager.get_provider(settings.provider)
        model_context_window = provider.context_window if hasattr(provider, 'context_window') else 8192

        if estimated_tokens > model_context_window * 0.9:
            self.logger.warning(
                "High context usage in streaming mode - may affect performance",
                extra={
                    "estimated_tokens": estimated_tokens,
                    "context_window": model_context_window
                }
            )

        # Prepare context from tools
        context_parts = []
        tools_executed = []

        # Execute enabled tools based on heuristics
        enabled_tools = settings.enabled_tools
        self.logger.debug(f"Enabled tools: {enabled_tools}")

        # RAG Search Tool - Execute before streaming
        if "rag_search" in enabled_tools:
            self.logger.info("RAG Search tool enabled - executing before stream")
            rag_context = await self._execute_rag_tool(
                conversation,
                user_message,
                settings,
                file_ids
            )
            if rag_context:
                context_parts.append(rag_context)
                tools_executed.append("rag_search")
                self.logger.info("RAG context added to context")
            else:
                self.logger.warning("RAG tool returned no context")
        else:
            self.logger.debug("RAG Search tool not enabled - skipped")

        # Semantic Memory - Execute before streaming
        if settings.memory_config.semantic_enabled:
            self.logger.info("Semantic memory enabled - retrieving relevant context")
            memory_context = await self.memory_service.retrieve_relevant_context(
                conversation=conversation,
                current_query=user_message,
                memory_config=settings.memory_config.model_dump() if hasattr(settings.memory_config,
                                                                              'model_dump') else settings.memory_config
            )
            if memory_context:
                context_parts.append(memory_context)
                self.logger.info("Semantic memory context added")
            else:
                self.logger.debug("No relevant memory context found")

        # Custom Tools (execute all enabled custom tools)
        # Note: For custom tools in manual mode, we need to define how to call them
        # This is a simplified implementation - in practice, you might want more sophisticated logic
        for tool_name in enabled_tools:
            if tool_name not in ["rag_search"] and not settings.memory_config.semantic_enabled:
                # This is a custom tool - try to execute it
                tool = tool_registry.get(tool_name)
                if tool:
                    try:
                        self.logger.info(f"Custom tool '{tool_name}' enabled - executing")
                        
                        # Ensure custom tool config is loaded before getting parameters
                        if isinstance(tool, CustomToolExecutor):
                            tool.db = self.db
                            await tool._load_custom_tool_config()
                            
                        # Extraer parámetros de forma profesional usando el ExtractionService
                        params_to_extract = tool.get_parameters()
                        extracted_params = {}
                        
                        if params_to_extract:
                            from dataclasses import asdict
                            self.logger.info(f"Extracting parameters for custom tool '{tool_name}'")
                            extracted_params = await self.extraction_service.extract_parameters(
                                user_message=user_message,
                                parameters=[asdict(p) for p in params_to_extract],
                                conversation_id=conversation.id,
                                provider=settings.provider,
                                model=settings.model
                            )
                            self.logger.info(f"Extracted parameters: {extracted_params}")

                        # Ejecutar con los parámetros extraídos
                        result = await self._execute_tool(tool, conversation, **extracted_params)

                        if result.success and result.data:
                            # Format the result as context
                            tool_context = f"## Result from {tool_name}\n\n{result.data}\n\n"
                            context_parts.append(tool_context)
                            tools_executed.append(tool_name)
                            self.logger.info(f"Custom tool '{tool_name}' executed successfully")
                        else:
                            self.logger.warning(f"Custom tool '{tool_name}' returned no data")
                    except Exception as e:
                        self.logger.error(f"Error executing custom tool '{tool_name}': {e}")
                else:
                    self.logger.warning(f"Custom tool '{tool_name}' not found in registry")

        # Build enriched context
        if context_parts:
            self.logger.info(
                "Building context string",
                extra={"context_parts_count": len(context_parts)}
            )
            context_string = self._build_context_string(context_parts)
            # user_message preserved (no longer pre-pended, injected via system message later)
            self.logger.debug(f"Total messages to LLM (before user msg): {len(messages)}")
        else:
            self.logger.warning("No context parts - proceeding without tool context")

        if context_parts:
            # CONTEXT AUTHORITY: Inject tool results as a separate SYSTEM message for higher priority
            context_string = self._build_context_string(context_parts)
            
            # --- GREEDY HISTORY SANITIZATION ---
            # If we have tool context, we remove any previous turns (User+Assistant) 
            # that were unsuccessful attempts at the SAME query.
            i = len(messages) - 1
            while i >= 1:
                if (messages[i].role == "assistant" and 
                    messages[i-1].role == "user" and 
                    messages[i-1].content.strip() == user_message.strip()):
                    self.logger.info(f"Greedy Sanitization (stream): Removing previous speculative attempt for '{user_message[:20]}...'")
                    messages.pop(i)
                    messages.pop(i-1)
                    i -= 2
                    continue
                i -= 1

            messages.append(ChatMessage(role="system", content=context_string))

        messages.append(ChatMessage(role="user", content=user_message))

        # PHASE 2: Stream LLM response
        provider = provider_manager.get_provider(settings.provider)

        self.logger.info("Starting LLM streaming...")

        # Check if provider supports cancellation
        if hasattr(provider, 'cancellable_stream_chat') and cancel_token:
            stream_method = provider.cancellable_stream_chat
            kwargs = {
                "cancel_event": cancel_token.cancel_event,
                "cancel_check_interval": 0.1
            }
        else:
            stream_method = provider.stream_chat
            kwargs = {}

        # Agregar configuraciones de herramientas a kwargs
        if tool_configs:
            kwargs.update(tool_configs)

        async for chunk in stream_method(
            messages=messages,
            model=settings.model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            top_p=settings.top_p,
            **kwargs
        ):
            # Handle both string chunks (content) and dict chunks (thinking)
            # Also handle JSON string chunks from LocalProvider
            if isinstance(chunk, dict) and chunk.get("type") == "thinking":
                # Pass through thinking events to API layer
                yield {
                    "type": "thinking",
                    "content": chunk.get("content", "")
                }
            elif isinstance(chunk, str):
                # Check if chunk is a JSON string
                try:
                    parsed_chunk = json.loads(chunk)
                    if isinstance(parsed_chunk, dict):
                        if parsed_chunk.get("type") == "thinking":
                            # It's a thinking chunk
                            yield {
                                "type": "thinking",
                                "content": parsed_chunk.get("content", "")
                            }
                        elif parsed_chunk.get("type") == "content":
                            # It's a content chunk
                            yield {
                                "type": "content",
                                "chunk": parsed_chunk.get("chunk", "")
                            }
                        else:
                            # Regular content chunk (string)
                            yield {
                                "type": "content",
                                "chunk": chunk
                            }
                    else:
                        # Regular content chunk (string)
                        yield {
                            "type": "content",
                            "chunk": chunk
                        }
                except json.JSONDecodeError:
                    # Not a JSON string, treat as regular content
                    yield {
                        "type": "content",
                        "chunk": chunk
                    }
            else:
                # Regular content chunk (string)
                yield {
                    "type": "content",
                    "chunk": chunk
                }

        # PHASE 3: Send metadata
        self.logger.info("MANUAL mode STREAMING processing completed")
        yield {
            "type": "metadata",
            "data": {
                "tools_executed": tools_executed,
                "mode": "manual",
                "provider": settings.provider,
                "model": settings.model
            }
        }

    async def _get_default_collections(self, conversation: Conversation) -> List[str]:
        """Determine default collections for a conversation (project + chat)"""
        collections = []

        # 1. Project Collection
        if conversation.project_id:
            project_collection = f"project_{conversation.project_id}"
            collections.append(project_collection)

        # 2. Conversation Collection (if files exist)
        try:
            result = await self.db.execute(
                select(File).filter(
                    File.conversation_id == conversation.id,
                    File.processed == True
                )
            )
            files = result.scalars().all()

            if files:
                # Use temporary collection for this conversation
                temp_collection = f"chat_{conversation.id}"
                collections.append(temp_collection)
        except Exception as e:
            self.logger.warning(f"Error determining conversation collections: {e}")

        return collections

    async def _execute_tool(self, tool: BaseTool, conversation: Conversation, **kwargs) -> ToolResult:
        """
        centralized tool execution with context injection (db, collections for RAG)
        """
        # Inject DB session if not present
        if "db" not in kwargs:
            kwargs["db"] = self.db

        # Special handling for RAG-type tools (physical RAG or CustomTool instances of type RAG)
        is_rag = False
        if tool.name == "rag_search":
            is_rag = True
        elif hasattr(tool, 'tool_type') and tool.tool_type == "rag_search":
            is_rag = True
        elif tool.category == ToolCategory.RAG:
            is_rag = True

        # If it's a RAG tool and collections are not provided, inject default ones
        if is_rag and not kwargs.get("collections"):
            self.logger.debug(f"Injecting default collections for RAG tool: {tool.name}")
            kwargs["collections"] = await self._get_default_collections(conversation)

        # Execute the tool
        return await tool.execute(**kwargs)

    # =============================================================================
    # Tool Execution Helpers (Manual Mode)
    # =============================================================================

    async def _execute_rag_tool(
        self,
        conversation: Conversation,
        query: str,
        settings: ConversationSettings,
        file_ids: Optional[List[UUID]] = None
    ) -> Optional[str]:
        """Execute RAG search tool"""
        self.logger.info("Starting RAG search execution")
        self.logger.debug(f"Query: {query[:100]}...")

        # Get RAG tool configuration - choose custom instance if available, otherwise use physical tool
        tool_config = self._get_tool_config(conversation, "rag_search")

        # Check if there are custom tool instances for rag_search
        custom_rag_tools = await self._get_custom_rag_tools()
        if custom_rag_tools:
            # Use custom tool instances instead of physical tool
            self.logger.info(f"Found {len(custom_rag_tools)} custom RAG tool instances, using them instead of physical tool")
            # For now, use the first custom tool instance
            # In a more sophisticated implementation, we could choose based on configuration or user selection
            custom_tool = custom_rag_tools[0]
            return await self._execute_custom_rag_tool(custom_tool, conversation, query, settings, file_ids)

        # Detect if query references "this document" and we have attached files
        filters = tool_config.config.get("filters", {}) if tool_config else {}
        attached_file_ids = None  # Track for post-filtering if needed
        mentioned_file_name = None  # Track if user mentions a specific filename

        if file_ids and self._is_contextual_document_reference(query):
            # User is asking about attached files specifically
            # Filter RAG to search only in these files
            self.logger.info(
                "Contextual document reference detected",
                extra={
                    "pattern_detected": True,
                    "attached_files_count": len(file_ids)
                }
            )

            # Convert file_ids to strings
            file_id_strs = [str(fid) for fid in file_ids]
            attached_file_ids = file_id_strs  # Store for post-filtering

            # Enhance query with filenames for better semantic search
            try:
                result = await self.db.execute(
                    select(File.file_name).filter(File.id.in_(file_ids))
                )
                file_names = result.scalars().all()
                if file_names:
                    filenames_str = ", ".join(file_names)
                    original_query = query

                    # Check if filenames are already in query to avoid double concatenation
                    if filenames_str in query:
                        self.logger.debug(
                            f"Filenames already present in query - skipping enhancement")
                    else:
                        query = f"{query} ({filenames_str})"
                        self.logger.info(
                            f"Enhanced query with filenames",
                            extra={"original": original_query, "new": query}
                        )
            except Exception as e:
                self.logger.warning(f"Failed to fetch filenames for query enhancement: {e}")

            # If only one file, use direct equality filter
            if len(file_id_strs) == 1:
                filters["file_id"] = file_id_strs[0]
                self.logger.debug(f"Filtering by single file: {file_id_strs[0]}")
            else:
                # Multiple files: search in collection, then post-filter results
                # We'll filter the results after retrieval to include only these files
                self.logger.info(
                    f"Multiple files attached - will post-filter results",
                    extra={"file_ids": file_id_strs, "file_count": len(file_id_strs)}
                )
                # Don't set filter here - we'll filter results after retrieval
        else:
            # Check if user is asking about a specific document by name
            # Pattern: "de que se trata el documento "Solicitud de requerimiento - SR (03).docx""
            mentioned_file_name = self._extract_file_name_from_query(query)
            if mentioned_file_name:
                self.logger.info(
                    "Specific file name mentioned in query",
                    extra={
                        "mentioned_file": mentioned_file_name,
                        "original_query": query[:100]
                    }
                )

                # Try to find the file in the database
                try:
                    result = await self.db.execute(
                        select(File.id, File.file_name).filter(
                            File.file_name.ilike(f"%{mentioned_file_name}%")
                        )
                    )
                    files = result.all()

                    if files:
                        if len(files) == 1:
                            # Single match - use direct filter
                            file_id, file_name = files[0]
                            filters["file_id"] = str(file_id)
                            self.logger.info(
                                f"Found exact match for file: {file_name} (ID: {file_id})",
                                extra={"file_id": str(file_id)}
                            )
                        else:
                            # Multiple matches - enhance query with filename
                            file_names = [f[1] for f in files]
                            filenames_str = ", ".join(file_names)
                            query = f"{query} ({filenames_str})"
                            self.logger.info(
                                f"Multiple matches found, enhanced query with filenames",
                                extra={"file_count": len(files), "filenames": file_names}
                            )
                    else:
                        self.logger.debug(
                            f"No files found matching: {mentioned_file_name}",
                            extra={"search_term": mentioned_file_name}
                        )
                except Exception as e:
                    self.logger.warning(f"Failed to search for mentioned file: {e}")

            self.logger.debug(
                "No contextual reference or no attached files - searching all collections")

        # Determine collections to search
        collections = []

        # First priority: explicit configuration
        if tool_config and tool_config.config.get("collections"):
            collections = tool_config.config.get("collections", [])
            self.logger.debug(f"Using configured collections: {collections}")
        else:
            # Second priority: Context-aware collections (Project + Conversation)
            collections = await self._get_default_collections(conversation)
            self.logger.debug(f"Using default collections: {collections}")

        # If no collections available, return None
        if not collections:
            self.logger.warning("No collections available - RAG search skipped")
            return None

        # Use pre-cached RAG tool (from __init__)
        if not self._rag_tool:
            self.logger.error("RAG tool not found in registry")
            return None

        try:
            self.logger.debug(f"Executing RAG search... query : {query}")
            result = await self._rag_tool.execute(
                query=query,
                collections=collections,
                k=tool_config.config.get("k", 15) if tool_config else 15,
                score_threshold=tool_config.config.get("score_threshold",
                                                       0.3) if tool_config else 0.3,
                filters=filters,  # Use prepared filters (may include file_id)
                embedding_model=settings.embedding_model,
                db=self.db  # Pass database session
            )

            if result.success and result.data and result.data.get('chunks'):
                # Post-filter results if we have multiple attached files or a mentioned filename
                if (attached_file_ids and len(attached_file_ids) > 1) or mentioned_file_name:
                    chunks = result.data.get('chunks', [])
                    original_count = len(chunks)

                    # Filter chunks based on attached files or mentioned filename
                    filtered_chunks = []

                    if attached_file_ids and len(attached_file_ids) > 1:
                        # Filter by attached file IDs
                        filtered_chunks = [
                            chunk for chunk in chunks
                            if chunk.get('metadata', {}).get('file_id') in attached_file_ids
                           or chunk.get('file_id') in attached_file_ids  # Check both locations
                        ]
                        self.logger.info(
                            f"Post-filtered results for multiple files",
                            extra={
                                "original_count": original_count,
                                "filtered_count": len(filtered_chunks),
                                "target_files": attached_file_ids
                            }
                        )
                    elif mentioned_file_name:
                        # Filter by mentioned filename
                        filtered_chunks = [
                            chunk for chunk in chunks
                            if mentioned_file_name.lower() in chunk.get('file', '').lower()
                           or mentioned_file_name.lower() in str(chunk.get('metadata', {}).get('file', '')).lower()
                        ]
                        self.logger.info(
                            f"Post-filtered results for mentioned filename",
                            extra={
                                "original_count": original_count,
                                "filtered_count": len(filtered_chunks),
                                "mentioned_file": mentioned_file_name
                            }
                        )

                    result.data['chunks'] = filtered_chunks

                # === NUEVO: EXPANSIÓN OBSIDIAN ===
                expanded_context = await self._expand_obsidian_context(
                    main_results=result.data.get('chunks', []),
                    conversation=conversation,
                    query=query,
                    settings=settings
                )

                context = self._format_rag_context(result.data)
                chunks_count = len(result.data.get('chunks', []))
                self.logger.info(
                    "RAG search successful",
                    extra={
                        "chunks_found": chunks_count,
                        "context_length": len(context)
                    }
                )
                self.logger.debug(f"First 200 chars of context: {context[:200]}...")
                return context
            else:
                self.logger.warning("RAG search returned no results")

                # FALLBACK: If specific files were requested but no results found,
                # try to fetch content directly from DB/Disk
                if attached_file_ids or mentioned_file_name:
                    target_files = attached_file_ids if attached_file_ids else [mentioned_file_name]
                    self.logger.info(f"Attempting fallback for files: {target_files}")

                    if attached_file_ids:
                        fallback_context = await self._fetch_file_content_fallback(attached_file_ids)
                    else:
                        # For mentioned filename, search by filename
                        fallback_context = await self._fetch_file_by_name_fallback(mentioned_file_name)

                    if fallback_context:
                        self.logger.info(
                            f"Fallback successful. Context length: {len(fallback_context)}")
                        return fallback_context
                    else:
                        self.logger.warning("Fallback returned no content")

                return None
        except Exception as e:
            self.logger.error(
                f"RAG tool execution failed: {str(e)}",
                exc_info=True
            )
            # Try fallback on error too
            if attached_file_ids or mentioned_file_name:
                try:
                    self.logger.info("Attempting fallback after error")
                    if attached_file_ids:
                        return await self._fetch_file_content_fallback(attached_file_ids)
                    else:
                        return await self._fetch_file_by_name_fallback(mentioned_file_name)
                except Exception as fallback_error:
                    self.logger.error(f"Fallback failed: {fallback_error}")

        return None

    async def _execute_custom_rag_tool(
        self,
        custom_tool: CustomTool,
        conversation: Conversation,
        query: str,
        settings: ConversationSettings,
        file_ids: Optional[List[UUID]] = None
    ) -> Optional[str]:
        """Execute a custom RAG tool instance"""
        self.logger.info(f"Executing custom RAG tool: {custom_tool.name}")

        try:
            # Create custom tool executor
            executor = CustomToolExecutor(custom_tool.id, self.db)

            # Prepare parameters for custom tool execution
            # Get collections from custom tool configuration or determine them dynamically
            collections = []

            # Check if custom tool has collections in its configuration
            if custom_tool.configuration and custom_tool.configuration.get("collections"):
                collections = custom_tool.configuration.get("collections", [])
                self.logger.debug(f"Using collections from custom tool config: {collections}")
            else:
                # Use default collections helper
                collections = await self._get_default_collections(conversation)
                self.logger.debug(f"Using dynamic collections: {collections}")

            # If no collections available, return None
            if not collections:
                self.logger.warning(f"No collections available for custom tool {custom_tool.name} - RAG search skipped")
                return None

            # Execute the custom tool with the query and collections
            result = await executor.execute(
                query=query,
                collections=collections,
                db=self.db
            )

            if result.success and result.data:
                self.logger.info(f"Custom RAG tool {custom_tool.name} executed successfully")
                return str(result.data)
            else:
                self.logger.warning(f"Custom RAG tool {custom_tool.name} returned no data")
                return None

        except Exception as e:
            self.logger.error(f"Error executing custom RAG tool {custom_tool.name}: {e}", exc_info=True)
            return None

    async def _get_custom_rag_tools(self) -> List[CustomTool]:
        """Get all active custom RAG tool instances"""
        try:
            result = await self.db.execute(
                select(CustomTool).filter(
                    CustomTool.is_active == True,
                    CustomTool.is_template == False,  # Only instances, not templates
                    CustomTool.tool_type == "rag_search"
                )
            )
            custom_tools = result.scalars().all()
            return custom_tools
        except Exception as e:
            self.logger.error(f"Error fetching custom RAG tools: {e}")
            return []

    async def _fetch_file_content_fallback(self, file_ids: List[str]) -> Optional[str]:
        """Fetch file content directly from DB or disk as fallback"""
        try:
            # Convert strings back to UUIDs
            uuids = [UUID(fid) for fid in file_ids]

            result = await self.db.execute(
                select(File).filter(File.id.in_(uuids))
            )
            files = result.scalars().all()

            if not files:
                self.logger.warning(f"Fallback: No files found in DB for IDs {file_ids}")
                return None

            context = "## Relevant Documentation (Direct Retrieval)\n\n"

            for file_record in files:
                self.logger.debug(
                    f"Fallback processing file: {file_record.file_name} (ID: {file_record.id})")
                content = ""
                # Try extracted text from metadata first
                if file_record.extra_metadata and file_record.extra_metadata.get("extracted_text"):
                    content = file_record.extra_metadata.get("extracted_text")
                    self.logger.debug("Fallback: Retrieved content from metadata")

                # If no extracted text, try reading from disk (limit size)
                if not content and file_record.storage_path:
                    try:
                        self.logger.debug(
                            f"Fallback: Reading from disk: {file_record.storage_path}")
                        import aiofiles
                        async with aiofiles.open(file_record.storage_path, mode='r',
                                                 encoding='utf-8', errors='ignore') as f:
                            content = await f.read(2000)  # Limit to 2000 chars
                        self.logger.debug(f"Fallback: Read {len(content)} chars from disk")
                    except ImportError:
                        # Fallback to sync open if aiofiles not available
                        with open(file_record.storage_path, mode='r', encoding='utf-8',
                                  errors='ignore') as f:
                            content = f.read(2000)
                    except Exception as e:
                        self.logger.warning(f"Failed to read file {file_record.id} from disk: {e}")

                if content:
                    context += f"### File: {file_record.file_name}\n\n"
                    context += f"**Content Preview**\n{content[:2000]}\n\n"
                    context += "---\n\n"
                else:
                    self.logger.warning(
                        f"Fallback: No content found for file {file_record.file_name}")

            return context
        except Exception as e:
            self.logger.error(f"Error in fallback retrieval: {e}", exc_info=True)
            return None

        async def _fetch_file_by_name_fallback(self, file_name: str) -> Optional[str]:
            """Fetch file content by filename as fallback"""
            try:
                # Search for file by name
                result = await self.db.execute(
                    select(File).filter(
                        File.file_name.ilike(f"%{file_name}%")
                    )
                )
                files = result.scalars().all()

                if not files:
                    self.logger.warning(f"Fallback: No files found matching name: {file_name}")
                    return None

                context = f"## Relevant Documentation (Direct Retrieval - {file_name})\n\n"

                for file_record in files:
                    self.logger.debug(
                        f"Fallback processing file: {file_record.file_name} (ID: {file_record.id})")
                    content = ""
                    # Try extracted text from metadata first
                    if file_record.extra_metadata and file_record.extra_metadata.get("extracted_text"):
                        content = file_record.extra_metadata.get("extracted_text")
                        self.logger.debug("Fallback: Retrieved content from metadata")

                    # If no extracted text, try reading from disk (limit size)
                    if not content and file_record.storage_path:
                        try:
                            self.logger.debug(
                                f"Fallback: Reading from disk: {file_record.storage_path}")
                            import aiofiles
                            async with aiofiles.open(file_record.storage_path, mode='r',
                                                     encoding='utf-8', errors='ignore') as f:
                                content = await f.read(2000)  # Limit to 2000 chars
                            self.logger.debug(f"Fallback: Read {len(content)} chars from disk")
                        except ImportError:
                            # Fallback to sync open if aiofiles not available
                            with open(file_record.storage_path, mode='r', encoding='utf-8',
                                      errors='ignore') as f:
                                content = f.read(2000)
                        except Exception as e:
                            self.logger.warning(f"Failed to read file {file_record.id} from disk: {e}")

                    if content:
                        context += f"### File: {file_record.file_name}\n\n"
                        context += f"**Content Preview**\n{content[:2000]}\n\n"
                        context += "---\n\n"
                    else:
                        self.logger.warning(
                            f"Fallback: No content found for file {file_record.file_name}")

                return context
            except Exception as e:
                self.logger.error(f"Error in fallback retrieval by name: {e}", exc_info=True)
                return None

    # =============================================================================
    # Context Building
    # =============================================================================

    async def _build_message_history(
        self,
        conversation: Conversation,
        current_message: str,
        settings: ConversationSettings
    ) -> List[ChatMessage]:
        """Build message history for LLM"""
        t_total_start = time.perf_counter()
        messages = []

        # System prompt
        t_system_start = time.perf_counter()
        system_prompt = self._build_system_prompt(conversation, settings)
        messages.append(ChatMessage(role="system", content=system_prompt))
        t_system_end = time.perf_counter()
        self.logger.debug(
            f"[PERF]   Build system prompt: {(t_system_end - t_system_start) * 1000:.2f}ms")

        # Load previous messages (optimized query with only needed columns)
        # Use configurable limit from settings
        t_query_start = time.perf_counter()
        result = await self.db.execute(
            select(Message.role, Message.content)
            .filter(
                Message.conversation_id == conversation.id,
                Message.is_active == True
            )
            .order_by(Message.created_at.desc())
            .limit(settings.max_history_messages)
        )
        previous_messages = result.all()
        t_query_end = time.perf_counter()
        self.logger.info(
            f"[PERF]   DB query for {len(previous_messages)} messages: {(t_query_end - t_query_start) * 1000:.2f}ms")

        # Build message list in chronological order (optimized role conversion)
        t_process_start = time.perf_counter()
        for msg_role, msg_content in reversed(previous_messages):
            # Skip messages with empty content (can happen with thinking-mode model failures)
            if not msg_content or not msg_content.strip():
                self.logger.warning(f"Skipping message with empty content (role: {msg_role})")
                continue

            # Simplified role conversion - handle both enum and string
            if isinstance(msg_role, MessageRole):
                role = msg_role.value
            else:
                # Already a string or can be stringified
                role = str(msg_role).lower() if msg_role else "user"

            messages.append(ChatMessage(role=role, content=msg_content))
        t_process_end = time.perf_counter()
        self.logger.debug(
            f"[PERF]   Process/convert messages: {(t_process_end - t_process_start) * 1000:.2f}ms")

        t_total_end = time.perf_counter()
        self.logger.debug(
            f"[PERF]   TOTAL _build_message_history: {(t_total_end - t_total_start) * 1000:.2f}ms")

        return messages

    def _build_system_prompt(
        self,
        conversation: Conversation,
        settings: ConversationSettings
    ) -> str:
        """Build system prompt based on template and settings"""
        base_prompt = ""

        # Get from prompt template if exists
        if conversation.prompt_template:
            base_prompt = conversation.prompt_template.system_prompt
        else:
            base_prompt = "Eres un asistente de IA que responde siempre en español"

        # Add hallucination control instructions
        hallucination_mode = settings.hallucination_control.mode

        if hallucination_mode == HallucinationMode.STRICT:
            base_prompt += "\n\nIMPORTANT: You MUST NEVER invent information. Only respond with verifiable facts from the provided context. Always cite your sources. If you don't have the information, say so clearly."
        elif hallucination_mode == HallucinationMode.CREATIVE:
            base_prompt += "\n\nYou can make reasonable inferences and suggestions. If you're speculating, indicate it clearly."

        return base_prompt

    def _build_context_string(self, context_parts: List[str]) -> str:
        """Build context string from tool results with high authority and NL instructions"""
        context = "\n\n".join(context_parts)

        return (
            "## SOURCE OF TRUTH (CONTEXT FROM TOOLS)\n"
            "IMPORTANT: The following information is verified and ACTUAL.\n"
            "1. If this context contradicts any previous information in history, you MUST IGNORE the history and use ONLY this context.\n"
            "2. Answer the user's question in **NATURAL LANGUAGE**. Be conversational and helpful.\n"
            "3. **DO NOT** output raw JSON, dictionaries, or code snippets unless explicitly requested by the user.\n\n"
            f"{context}\n\n"
            "Use ONLY the verified information above to provide a friendly answer."
        )

    def _format_rag_context(self, rag_data: Dict) -> str:
        """Format RAG search results into context"""
        chunks = rag_data.get("chunks", [])

        if not chunks:
            return ""

        context = "## Relevant Documentation\n\n"

        # Group chunks by file
        files_content = {}
        for chunk in chunks:
            file_name = chunk.get('file', 'Unknown File')
            if file_name not in files_content:
                files_content[file_name] = []
            files_content[file_name].append(chunk)

        # Build context grouped by file
        for file_name, file_chunks in files_content.items():
            context += f"### File: {file_name}\n\n"
            for i, chunk in enumerate(file_chunks, 1):
                context += f"**Fragment {i} - {chunk['section']}**\n"
                context += f"{chunk['content']}\n\n"
            context += "---\n\n"

        return context

    # =============================================================================
    # Validation & Heuristics
    # =============================================================================

    def _validate_strict_response(
        self,
        response: ChatResponse,
        context_parts: List[str]
    ) -> ChatResponse:
        """Validate response in strict mode"""
        # Check if response has sources
        if not context_parts:
            response.metadata["warning"] = "No sources available for verification"
            response.metadata["confidence_score"] = 0.3
        else:
            # Simple check: does response reference the context?
            # More sophisticated validation could be added
            response.metadata["confidence_score"] = 0.8

        return response

    # =============================================================================
    # Simple Chat (No Tools)
    # =============================================================================

    async def _simple_chat(
        self,
        messages: List[ChatMessage],
        settings: ConversationSettings
    ) -> ChatResponse:
        """Simple chat without tools"""
        provider = provider_manager.get_provider(settings.provider)

        return await provider.chat(
            messages=messages,
            model=settings.model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens
        )

    # =============================================================================
    # Utilities
    # =============================================================================

    def _parse_settings(self, conversation_id: UUID, settings_dict: Dict) -> ConversationSettings:
        """Parse settings from dict to Pydantic model (with caching)"""
        # Cache settings per conversation to avoid repeated Pydantic validation
        if conversation_id not in self._settings_cache:
            self._settings_cache[conversation_id] = ConversationSettings(**settings_dict)
        return self._settings_cache[conversation_id]

    def _is_contextual_document_reference(self, query: str) -> bool:
        """
        Detecta si la consulta hace referencia contextual a un documento

        Patrones detectados:
        - "este documento"
        - "el documento"
        - "este archivo"
        - "el archivo"
        - "este fichero"
        - "el fichero"
        - "este pdf/word/excel/csv"
        - "el pdf/word/excel/csv"

        Returns:
            True si se detecta referencia contextual
        """
        # Use pre-compiled patterns (module-level) for performance
        for pattern in _CONTEXTUAL_PATTERNS:
            if pattern.search(query):
                return True

        return False

    def _extract_file_name_from_query(self, query: str) -> Optional[str]:
        """
        Extract file name from query when user mentions a specific document.

        Examples:
        - "de que se trata el documento 'Solicitud de requerimiento - SR (03).docx'"
        - "explica el archivo API TRON.docx"
        - "que dice el fichero requirements.txt"

        Returns:
            Extracted filename if found, None otherwise
        """
        # Common patterns for file references
        patterns = [
            # Pattern 1: "documento "filename""
            re.compile(r'documento\s+["][^"\']+["\']', re.IGNORECASE),
            # Pattern 2: "archivo "filename""
            re.compile(r'archivo\s+["][^"\']+["\']', re.IGNORECASE),
            # Pattern 3: "fichero "filename""
            re.compile(r'fichero\s+["][^"\']+["\']', re.IGNORECASE),
            # Pattern 4: "el documento "filename""
            re.compile(r'el\s+documento\s+["][^"\']+["\']', re.IGNORECASE),
            # Pattern 5: "el archivo "filename""
            re.compile(r'el\s+archivo\s+["][^"\']+["\']', re.IGNORECASE),
            # Pattern 6: "el fichero "filename""
            re.compile(r'el\s+fichero\s+["][^"\']+["\']', re.IGNORECASE),
            # Pattern 7: "este documento "filename""
            re.compile(r'este\s+documento\s+["][^"\']+["\']', re.IGNORECASE),
            # Pattern 8: "este archivo "filename""
            re.compile(r'este\s+archivo\s+["][^"\']+["\']', re.IGNORECASE),
            # Pattern 9: "este fichero "filename""
            re.compile(r'este\s+fichero\s+["][^"\']+["\']', re.IGNORECASE),
            # Pattern 10: "sobre "filename""
            re.compile(r'sobre\s+["][^"\']+["\']', re.IGNORECASE),
            # Pattern 11: "del documento "filename""
            re.compile(r'del\s+documento\s+["][^"\']+["\']', re.IGNORECASE),
            # Pattern 12: "del archivo "filename""
            re.compile(r'del\s+archivo\s+["][^"\']+["\']', re.IGNORECASE),
        ]

        for pattern in patterns:
            match = pattern.search(query)
            if match:
                # Extract the filename from quotes
                full_match = match.group(0)
                # Find the first quote
                first_quote = full_match.find('"')
                if first_quote != -1:
                    # Find the closing quote
                    second_quote = full_match.find('"', first_quote + 1)
                    if second_quote != -1:
                        filename = full_match[first_quote + 1:second_quote]
                        # Clean up the filename
                        filename = filename.strip()
                        # Remove common prefixes like "el ", "este ", etc.
                        filename = re.sub(r'^\s*(el|este|un|una|los|las|el\s+documento|el\s+archivo|el\s+fichero|este\s+documento|este\s+archivo|este\s+fichero|sobre|del\s+documento|del\s+archivo)\s+', '', filename, flags=re.IGNORECASE)
                        return filename

        # Also check for filename patterns without quotes
        # Look for common file extensions followed by quotes or end of sentence
        extension_patterns = [
            r'\.docx\s*["\']',
            r'\.doc\s*["\']',
            r'\.pdf\s*["\']',
            r'\.txt\s*["\']',
            r'\.md\s*["\']',
            r'\.xlsx\s*["\']',
            r'\.csv\s*["\']',
            r'\.pptx\s*["\']',
            r'\.json\s*["\']',
            r'\.py\s*["\']',
            r'\.js\s*["\']',
            r'\.ts\s*["\']',
            r'\.java\s*["\']',
            r'\.sql\s*["\']',
        ]

        for ext_pattern in extension_patterns:
            # Find the extension
            ext_match = re.search(ext_pattern, query, re.IGNORECASE)
            if ext_match:
                # Find the start of the filename (go backwards to find the beginning)
                match_pos = ext_match.start()
                # Go backwards to find the start of the filename
                start_pos = match_pos
                while start_pos > 0 and query[start_pos - 1] not in ['"', "'", '.', ' ', ',', ';', ':', '\n', '\t']:
                    start_pos -= 1

                # Extract filename
                filename = query[start_pos:ext_match.end()].strip()
                if filename:
                    return filename

        return None

    async def _get_active_tool_configurations(self, conversation_id: UUID) -> Dict[str, Any]:
        """Get all active tool configurations for a conversation"""
        from sqlalchemy import select
        from src.models.models import ToolConfiguration

        result = await self.db.execute(
            select(ToolConfiguration).filter(
                ToolConfiguration.conversation_id == conversation_id,
                ToolConfiguration.is_active == True
            )
        )
        configs = result.scalars().all()

        # Convert configurations to a dictionary for easy access
        tool_configs = {}
        for config in configs:
            tool_configs[config.tool_name] = config.config

        return tool_configs

    def _get_tool_config(self, conversation: Conversation, tool_name: str):
        """Get tool configuration for conversation"""
        for config in conversation.tool_configurations:
            if config.tool_name == tool_name and config.is_active:
                return config
        return None

    def _get_graph_builder(self):
        """
        Obtiene o crea el graph builder de Obsidian
        
        Returns:
            ObsidianGraphBuilder instance o None si no hay vault detectado
        """
        # Cache del graph builder para evitar recrearlo constantemente
        if not hasattr(self, '_graph_builder_cache'):
            self._graph_builder_cache = None
        
        if self._graph_builder_cache:
            return self._graph_builder_cache
        
        try:
            from src.document_loaders.obsidian_detector import ObsidianDetector
            from src.document_loaders.obsidian_graph import ObsidianGraphBuilder
            from pathlib import Path
            import os
            
            # Opción 1: Desde variable de entorno
            vault_path = settings.get_vault_path()

            if not vault_path:
                self.logger.warning("No OBSIDIAN_VAULT_PATH configured - Obsidian navigation disabled")
                return None

            vault_path = Path(vault_path)
            
            if not vault_path.exists():
                self.logger.warning(f"Obsidian vault path does not exist: {vault_path}")
                return None
            
            # Detectar vault
            detector = ObsidianDetector()
            context = detector.detect(vault_path)
            
            if not context.is_obsidian:
                self.logger.warning(f"Path is not an Obsidian vault: {vault_path}")
                return None
            
            # Construir grafo
            self.logger.info(f"Building Obsidian graph from vault: {context.vault_root}")
            graph_builder = ObsidianGraphBuilder()
            
            # Escanear vault (esto debe ejecutarse en contexto async)
            # Por ahora se cachea para evitar múltiples escaneos
            import asyncio
            try:
                # Intentar obtener loop existente
                loop = asyncio.get_running_loop()
                # Si llegamos aquí, ya estamos en async context
                # Crear task para no bloquear
                task = loop.create_task(graph_builder.scan_vault(context.vault_root))
                notes = loop.run_until_complete(task)
                graph_builder.build_bidirectional_graph()
            except RuntimeError:
                # No hay loop corriendo, crear uno nuevo
                notes = asyncio.run(graph_builder.scan_vault(context.vault_root))
                graph_builder.build_bidirectional_graph()
            
            self.logger.info(f"Graph built successfully: {len(notes)} notes indexed")
            
            # Cachear para próximas llamadas
            self._graph_builder_cache = graph_builder
            return graph_builder
            
        except ImportError as e:
            self.logger.error(f"Failed to import Obsidian modules: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error building Obsidian graph: {e}", exc_info=True)
            return None

async def _expand_obsidian_context(
    self,
    main_results: List[Dict],
    conversation: Conversation,
    query: str,
    settings: ConversationSettings
) -> Optional[str]:
    """
    Expande contexto Obsidian usando navegación inteligente y adaptativa
    Maneja ciclos y dependencias recursivas automáticamente
    
    Características:
    - Analiza intención de la query (¿busca contexto general o detalles?)
    - Navega direccionalmente (ascendente/descendente/bidireccional)
    - Detecta y evita ciclos infinitos
    - Prioriza notas según relevancia
    """
    if not main_results:
        return None
    
    top_result = main_results[0]
    note_name = top_result.get('file', '').replace('.md', '')
    
    if not note_name or not top_result.get('obsidian_outgoing_links'):
        return None
    
    # Inicializar navegador si es necesario
    if not self._obsidian_navigator:
        graph_builder = self._get_graph_builder()
        if not graph_builder:
            self.logger.debug("Graph builder not available - skipping Obsidian expansion")
            return None
        
        try:
            self._obsidian_navigator = ObsidianTreeNavigator(
                graph=graph_builder.graph,
                cache_enabled=True
            )
        except Exception as e:
            self.logger.error(f"Failed to create ObsidianTreeNavigator: {e}")
            return None
    
    # PASO 1: ANALIZAR INTENCIÓN DE LA QUERY
    note_metadata = {
        'is_hub': top_result.get('obsidian_is_hub', False),
        'is_index': top_result.get('obsidian_is_index', False),
        'note_type': top_result.get('obsidian_note_type', 'unknown')
    }
    
    intent = self._intent_analyzer.analyze(
        query=query,
        current_note=note_name,
        note_metadata=note_metadata
    )
    
    self.logger.info(
        "Navigation intent determined",
        extra={
            "query": query[:50],
            "note": note_name,
            "direction": intent.direction,
            "depth": intent.max_depth,
            "confidence": f"{intent.confidence:.2f}",
            "reasoning": intent.reasoning
        }
    )
    
    # PASO 2: NAVEGAR SEGÚN INTENCIÓN
    try:
        nav_result = self._obsidian_navigator.navigate_with_intent(
            start_note=note_name,
            intent=intent
        )
    except Exception as e:
        self.logger.error(f"Navigation failed: {e}")
        return None
    
    # Si no encontramos contexto relevante, retornar None
    if len(nav_result.visited_notes) <= 1:
        self.logger.debug("No additional context found")
        return None
    
    # PASO 3: RECUPERAR CONTENIDO DE NOTAS RELEVANTES
    collections = await self._get_default_collections(conversation)
    related_context_parts = []
    
    # Priorizar por capas según dirección
    layer_order = (
        sorted(nav_result.context_layers.keys(), reverse=True)  # Descendente: capas profundas primero
        if intent.direction == "down"
        else sorted(nav_result.context_layers.keys())  # Ascendente/Bidireccional: capas cercanas primero
    )
    
    for depth in layer_order:
        if depth == 0:
            continue  # Skip nota origen
        
        notes = nav_result.context_layers[depth]
        
        for related_note in notes[:5]:  # Top 5 por capa
            if len(related_context_parts) >= 8:  # Límite total
                break
            
            try:
                related_result = await self._rag_tool.execute(
                    query=f"{related_note} {query}",
                    collections=collections,
                    k=1,  # Solo 1 chunk por nota
                    score_threshold=0.25,  # Más permisivo para contexto
                    db=self.db
                )
                
                if related_result.success and related_result.data:
                    chunks = related_result.data.get('chunks', [])
                    if chunks:
                        best_chunk = chunks[0]
                        direction_emoji = "↑" if intent.direction == "up" else "↓" if intent.direction == "down" else "↔"
                        related_context_parts.append(
                            f"**{direction_emoji} [Nivel {depth}] {related_note}**\n"
                            f"{best_chunk['content'][:350]}\n"
                        )
            except Exception as e:
                self.logger.warning(f"Failed to fetch content for {related_note}: {e}")
    
    if not related_context_parts:
        return None
    
    # PASO 4: CONSTRUIR CONTEXTO EXPANDIDO
    summary = self._obsidian_navigator.get_context_summary(nav_result)
    
    direction_label = {
        "up": "⬆️ Contexto General (Ascendente)",
        "down": "⬇️ Detalles Específicos (Descendente)",
        "bidirectional": "↔️ Contexto Completo (Bidireccional)"
    }[intent.direction]
    
    expanded_context = (
        f"## 🔗 {direction_label}\n\n"
        f"**Estrategia:** {intent.reasoning}\n"
        f"**Confianza:** {intent.confidence:.0%}\n\n"
        f"{summary}\n\n"
        f"---\n\n"
        f"### Contenido Relacionado:\n\n"
        + "\n---\n\n".join(related_context_parts)
    )
    
    self.logger.info(
        f"Context expanded successfully",
        extra={
            "origin": note_name,
            "direction": intent.direction,
            "visited_notes": len(nav_result.visited_notes),
            "context_chunks": len(related_context_parts),
            "cycles_detected": len(nav_result.cycles_detected),
            "execution_time_ms": f"{nav_result.execution_time_ms:.2f}"
        }
    )
    
    return expanded_context