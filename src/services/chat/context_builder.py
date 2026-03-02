# =============================================================================
# src/services/chat/context_builder.py
# Context Builder - RAG context construction (REFACTORED from chat_orchestrator.py)
# =============================================================================
"""
ContextBuilder: Responsable de construir todo el contexto necesario para el chat.

Responsabilidades (SRP):
- Construir historial de mensajes
- Ejecutar RAG y construir contexto RAG
- Expandir contexto Obsidian
- Construir contexto de memoria conversacional
- Parsear settings con cache
- Construir prompts de sistema
"""
import hashlib
import os
import re
import time
from collections import OrderedDict
from datetime import timedelta
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID

from src.config.prompts import SystemPrompts
from src.config.settings import settings
from src.document_loaders import DocumentLoaderFactory
from src.document_loaders.obsidian_tree_navigator import ObsidianTreeNavigator
from src.models.models import Conversation, MessageRole, HallucinationMode
from src.providers.manager import ChatMessage
from src.repositories import ConversationRepository, MessageRepository, FileRepository, \
    CustomToolRepository
from src.schemas.schemas import ConversationSettings
from src.services.context.conversation_memory import ConversationMemoryService, get_qdrant_client
from src.tools.base_tool import tool_registry
from src.utils.date_utils import get_current_utc
from src.utils.logger import get_logger


class ContextBuilder:
    """
    Responsable de construir todo el contexto necesario para el chat.
    Separa la lógica de construcción de contexto de la orquestación.

    Migrado desde ChatOrchestrator para cumplir SRP.
    """

    def __init__(
        self,
        conversation_repo: ConversationRepository,
        message_repo: MessageRepository,
        file_repo: FileRepository,
        custom_tool_repo: Optional[CustomToolRepository] = None
    ):
        self.conversation_repo = conversation_repo
        self.message_repo = message_repo
        self.file_repo = file_repo
        self.custom_tool_repo = custom_tool_repo
        self.logger = get_logger(__name__)

        # Memory service for semantic search
        self.memory_service = ConversationMemoryService(message_repo)

        # Settings cache (LRU + TTL)
        self._settings_cache = OrderedDict()
        self._settings_cache_max_size = 1000
        self._settings_cache_ttl = timedelta(hours=1)
        self._cache_timestamps = {}

        # RAG tool reference
        self._rag_tool = tool_registry.get("rag_search")

        # Intent analyzer for Obsidian navigation
        self._obsidian_navigator: Optional[ObsidianTreeNavigator] = None

        # File size limits for fallback
        self._max_file_size = 1024 * 1024  # 1MB per file
        self._max_total_size = 5 * 1024 * 1024  # 5MB total

    # =============================================================================
    # Settings Parsing (with LRU cache + TTL)
    # =============================================================================

    def parse_settings(
        self,
        conversation_id: UUID,
        settings_dict: Dict[str, Any]
    ) -> ConversationSettings:
        """
        Parse conversation settings with LRU cache + TTL.

        Args:
            conversation_id: Conversation UUID
            settings_dict: Raw settings dict from conversation

        Returns:
            Parsed ConversationSettings object
        """
        now = get_current_utc()

        # Limpiar entradas expiradas
        if conversation_id in self._cache_timestamps:
            last_access = self._cache_timestamps[conversation_id]
            if now - last_access > self._settings_cache_ttl:
                self._settings_cache.pop(conversation_id, None)
                self._cache_timestamps.pop(conversation_id, None)
                self.logger.debug(f"Cache TTL expired for conversation {conversation_id}")

        # Obtener del cache o parsear
        if conversation_id not in self._settings_cache:
            # Si cache está lleno, eliminar el más antiguo (LRU)
            if len(self._settings_cache) >= self._settings_cache_max_size:
                oldest_key = next(iter(self._settings_cache))
                self._settings_cache.pop(oldest_key)
                self._cache_timestamps.pop(oldest_key, None)
                self.logger.debug(f"Cache eviction: {oldest_key}")

            # Parsear y agregar al cache
            self._settings_cache[conversation_id] = ConversationSettings(**settings_dict)
            self.logger.debug(f"Settings cached for conversation {conversation_id}")

        # Actualizar timestamp de acceso (LRU)
        self._cache_timestamps[conversation_id] = now

        # Mover al final (más reciente)
        self._settings_cache.move_to_end(conversation_id)

        return self._settings_cache[conversation_id]

    # =============================================================================
    # Message History Building
    # =============================================================================

    async def build_message_history(
        self,
        conversation: Conversation,
        current_message: str,
        settings: ConversationSettings,
        tool_context: Optional[str] = None
    ) -> List[ChatMessage]:
        """
        Build message history for LLM (Optimized & Bug-Free).

        Args:
            conversation: Conversation object
            current_message: Current user message
            settings: Conversation settings
            tool_context: Optional tool context to inject

        Returns:
            List of ChatMessage objects
        """
        t_total_start = time.perf_counter()

        # 1. System prompt
        final_messages = [ChatMessage(role="system", content=self._build_system_prompt(conversation, settings))]

        # 2. History con filtro/exclusión
        t_query_start = time.perf_counter()
        history_models = await self.message_repo.get_last_n_messages(
            conversation.id, n=settings.max_history_messages
        )
        t_query_end = time.perf_counter()
        self.logger.debug(f"[PERF] DB query: {(t_query_end - t_query_start)*1000:.2f}ms")
        self.logger.info(f"[HISTORY] Retrieved {len(history_models)} messages from DB (max_history_messages={settings.max_history_messages})")

        # 3. Process history + dedup robusto
        t_process_start = time.perf_counter()
        user_hash = hashlib.md5(current_message.strip().encode()).hexdigest()[:8]
        history_added = 0
        skipped_empty = 0
        skipped_duplicate = 0
        for msg in history_models:  # Cronológico (Ya vienen ordenados del repo)
            if not msg.content or not msg.content.strip():
                skipped_empty += 1
                continue
            msg_hash = hashlib.md5(msg.content.strip().encode()).hexdigest()[:8]
            if msg.role == MessageRole.USER.value and msg_hash == user_hash:
                self.logger.debug("Skipping DB duplicate of current_message")
                skipped_duplicate += 1
                continue
            role = msg.role.value if isinstance(msg.role, MessageRole) else str(msg.role).lower()
            final_messages.append(ChatMessage(role=role, content=msg.content))
            history_added += 1

        # 4. SIEMPRE agregar current_message al final (evita inconsistencias)
        final_messages.append(ChatMessage(role="user", content=current_message))

        # 5. Agregar contexto de herramientas si está disponible
        if tool_context:
            final_messages.append(ChatMessage(role="tool", content=tool_context))
            self.logger.info("Tool context added to message history")

        t_process_end = time.perf_counter()
        self.logger.debug(f"[PERF] Process: {(t_process_end - t_process_start)*1000:.2f}ms")
        self.logger.info(f"[HISTORY] Built: {history_added} history msgs + current (skipped: {skipped_empty} empty, {skipped_duplicate} duplicates)")

        return final_messages

    def _build_system_prompt(
        self,
        conversation: Conversation,
        settings: ConversationSettings
    ) -> str:
        """Build system prompt based on template and settings."""
        base_prompt = ""

        # Get from prompt template if exists
        if conversation.prompt_template:
            base_prompt = conversation.prompt_template.system_prompt
        else:
            base_prompt = SystemPrompts.DEFAULT_SYSTEM_PROMPT

        # Add hallucination control instructions
        hallucination_mode = settings.hallucination_control.mode

        if hallucination_mode == HallucinationMode.STRICT:
            base_prompt += SystemPrompts.HALLUCINATION_STRICT
        elif hallucination_mode == HallucinationMode.CREATIVE:
            base_prompt += SystemPrompts.HALLUCINATION_CREATIVE

        # Add information about attached files if any
        base_prompt += SystemPrompts.FILE_REFERENCE_INSTRUCTIONS
        base_prompt += SystemPrompts.SOURCE_CITATION_INSTRUCTIONS

        return base_prompt

    # =============================================================================
    # RAG Context Building
    # =============================================================================

    async def build_rag_context(
        self,
        conversation: Conversation,
        user_message: str,
        settings: ConversationSettings,
        file_ids: Optional[List[UUID]] = None,
        collection_name: Optional[str] = None,
        target_file_id: Optional[UUID] = None  # NUEVO: Filtrar por archivo específico
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Build RAG context for the query.

        Args:
            conversation: Conversation object
            user_message: User query
            settings: Conversation settings
            file_ids: Optional file IDs to search
            collection_name: Optional collection name
            target_file_id: NUEVO - Filtrar chunks solo para este archivo específico.
                           Esto evita que se mezclen chunks de archivos anteriores.

        Returns:
            Tuple of (context_string, rag_data)
        """
        if not self._rag_tool:
            self.logger.warning("RAG tool not available")
            return None, {}

        self.logger.info("Starting RAG search execution")
        self.logger.debug(f"Query: {user_message[:100]}...")

        # Get RAG tool configuration
        tool_config = self._get_tool_config(conversation, "rag_search")

        # Check for custom RAG tools
        custom_rag_tools = await self._get_custom_rag_tools()
        if custom_rag_tools:
            self.logger.info(f"Found {len(custom_rag_tools)} custom RAG tool instances")
            custom_tool = custom_rag_tools[0]
            return await self._execute_custom_rag_tool(
                custom_tool, conversation, user_message, settings, file_ids, collection_name
            )

        # Consolidar lógica de filtros
        filters = tool_config.config.get("filters", {}) if tool_config else {}
        attached_file_ids = None

        # NUEVO: Si hay target_file_id, filtrar directamente por ese archivo
        if target_file_id:
            filters["file_id"] = str(target_file_id)
            self.logger.info(f"🎯 RAG filtered by target file: {target_file_id}")
        else:
            # ✅ NUEVO: Obtener historial para resolver referencias contextuales
            try:
                conversation_history = await self.message_repo.get_last_n_messages(
                    conversation.id,
                    n=20  # Últimos 20 mensajes suficiente para contexto
                )
                self.logger.debug(
                    f"Retrieved {len(conversation_history)} messages for file context"
                )
            except Exception as e:
                self.logger.warning(f"Failed to retrieve conversation history: {e}")
                conversation_history = None

            # Detectar archivo mencionado (CON soporte contextual)
            mentioned_filename = self._extract_file_name_from_query(
                user_message,
                conversation_history=conversation_history  # ← NUEVO parámetro
            )

            if file_ids:
                self.logger.debug(
                    f"RAG execution with {len(file_ids)} files",
                    extra={"file_ids": [str(f) for f in file_ids[:3]]}
                )

            # Caso 1: Archivos adjuntos
            if file_ids:
                file_id_strs = [str(fid) for fid in file_ids]

                if mentioned_filename and len(file_ids) > 1:
                    try:
                        files = await self.file_repo.get_by_ids(file_ids)
                        all_files = [(f.id, f.filename) for f in files]

                        matched = [
                            (fid, fname) for fid, fname in all_files
                            if mentioned_filename.lower() in fname.lower()
                        ]

                        if matched:
                            filters["file_id"] = str(matched[0][0])
                            self.logger.info(f"🎯 Filtered to mentioned file: {matched[0][1]}")
                        else:
                            attached_file_ids = file_id_strs
                            self.logger.info(f"📎 Searching all {len(file_ids)} attached files")

                    except Exception as e:
                        self.logger.warning(f"Error filtering files: {e}")
                        attached_file_ids = file_id_strs
                else:
                    if len(file_ids) == 1:
                        filters["file_id"] = file_id_strs[0]
                        self.logger.info(f"🎯 Filtering by single attached file: {file_ids[0]}")
                    else:
                        attached_file_ids = file_id_strs
                        self.logger.info(f"📎 Multiple files attached ({len(file_ids)}), will post-filter")

            # Caso 2: No hay adjuntos pero menciona filename
            elif mentioned_filename:
                try:
                    files = await self.file_repo.search_by_filename(mentioned_filename, limit=10)
                    matched_files = [(f.id, f.filename) for f in files]

                    if matched_files:
                        if len(matched_files) == 1:
                            filters["file_id"] = str(matched_files[0][0])
                            self.logger.info(f"🎯 Found exact match: {matched_files[0][1]}")
                        else:
                            attached_file_ids = [str(f[0]) for f in matched_files]
                            self.logger.info(f"📎 Multiple matches for '{mentioned_filename}': {len(matched_files)} files")

                except Exception as e:
                    self.logger.warning(f"Error searching filename: {e}")

        # Vague query check
        if file_ids and self._is_vague_query(user_message):
            self.logger.info("Vague query detected with attached files. Injecting direct file content.")
            fallback_context = await self._fetch_file_content_fallback([str(fid) for fid in file_ids])
            if fallback_context:
                return fallback_context, {"chunks": [], "fallback": True}

        # Determine collections
        collections = []
        if collection_name:
            collections = [collection_name]
            self.logger.debug(f"Using explicitly requested collection: {collections}")
        elif tool_config and tool_config.config.get("collections"):
            collections = tool_config.config.get("collections", [])
            self.logger.debug(f"Using configured collections: {collections}")
        else:
            collections = await self._get_default_collections(conversation)
            self.logger.debug(f"Using default collections: {collections}")

        if not collections:
            self.logger.warning("No collections available - RAG search skipped")
            return None, {}

        try:
            result = await self._rag_tool.execute(
                query=user_message,
                collections=collections,
                k=tool_config.config.get("k", 15) if tool_config else 15,
                score_threshold=tool_config.config.get("score_threshold", 0.3) if tool_config else 0.3,
                filters=filters,
                embedding_model=settings.embedding_model,
                file_repo=self.file_repo
            )

            if result.success and result.data and result.data.get('chunks'):
                # Post-filter si es necesario
                if attached_file_ids:
                    chunks = result.data.get('chunks', [])
                    original_count = len(chunks)

                    filtered_chunks = [
                        chunk for chunk in chunks
                        if chunk.get('metadata', {}).get('file_id') in attached_file_ids
                        or chunk.get('file_id') in attached_file_ids
                    ]
                    self.logger.info(
                        f"Post-filtered: {original_count} → {len(filtered_chunks)} chunks",
                        extra={"target_files": attached_file_ids}
                    )
                    result.data['chunks'] = filtered_chunks

                # Expansión Obsidian (si aplica)
                expanded_context = await self._expand_obsidian_context(
                    main_results=result.data.get('chunks', []),
                    conversation=conversation,
                    query=user_message,
                    settings=settings
                )

                context = self._format_rag_context(result.data)
                chunks_count = len(result.data.get('chunks', []))
                self.logger.info("RAG search successful", extra={"chunks_found": chunks_count})
                return context, result.data
            else:
                self.logger.warning("RAG search returned no results")
                if file_ids:
                    fallback_context = await self._fetch_file_content_fallback([str(fid) for fid in file_ids])
                    if fallback_context:
                        return fallback_context, {"chunks": [], "fallback": True}
                return None, {}

        except Exception as e:
            self.logger.error(f"RAG tool execution failed: {str(e)}", exc_info=True)
            if attached_file_ids:
                try:
                    fallback_context = await self._fetch_file_content_fallback(attached_file_ids)
                    if fallback_context:
                        return fallback_context, {"chunks": [], "fallback": True}
                except Exception as fallback_error:
                    self.logger.error(f"Fallback failed: {fallback_error}")
            return None, {}

    async def _get_default_collections(self, conversation: Conversation) -> List[str]:
        """Determine default collections for a conversation (project + chat)."""
        collections = []

        # 0. Metadata Collection (Explicit override)
        if conversation.extra_metadata:
            if "collection_name" in conversation.extra_metadata:
                col_name = conversation.extra_metadata["collection_name"]
                if col_name and col_name not in collections:
                    collections.append(col_name)

            if "collections" in conversation.extra_metadata:
                cols = conversation.extra_metadata["collections"]
                if isinstance(cols, list):
                    for col in cols:
                        if col and col not in collections:
                            collections.append(col)
                elif isinstance(cols, str):
                    if cols and cols not in collections:
                        collections.append(cols)

        # 1. Project Collection
        if conversation.project_id:
            project_collection = f"project_{conversation.project_id}"
            if project_collection not in collections:
                collections.append(project_collection)

        # 2. Conversation Collection (if files exist)
        try:
            files = await self.file_repo.get_conversation_files(
                conversation.id,
                skip=0,
                limit=1
            )

            if files:
                temp_collection = f"chat_{conversation.id}"
                if temp_collection not in collections:
                    collections.append(temp_collection)
        except Exception as e:
            self.logger.warning(f"Error determining conversation collections: {e}")

        return collections

    async def _check_collection_exists(self, collection_name: str) -> bool:
        """Verifica si una colección de Qdrant existe."""
        try:
            qdrant_client = get_qdrant_client()
            collections = qdrant_client.get_collections()
            return collection_name in [c.name for c in collections.collections]
        except Exception as e:
            self.logger.warning(f"Failed to check collection existence: {e}")
            return False

    def _format_rag_context(self, rag_data: Dict) -> str:
        """Format RAG search results into context."""
        chunks = rag_data.get("chunks", [])

        if not chunks:
            return ""

        # ── Anti-Hallucination: Detectar si los chunks contienen código fuente ──
        # El patrón y el mensaje están centralizados en SystemPrompts (prompts.py).
        # Para añadir soporte a un nuevo lenguaje basta con editar ese archivo.
        _code_pattern = re.compile(SystemPrompts.CODE_SIGNATURE_PATTERN)
        _sample = " ".join(c.get('content', '') for c in chunks[:5])
        _has_code = bool(_code_pattern.search(_sample))

        context = "## Relevant Documentation\n\n"

        if _has_code:
            context += SystemPrompts.CODE_ANTI_HALLUCINATION_INSTRUCTION

        # Group chunks by file
        files_content = {}
        for chunk in chunks:
            file_name = chunk.get('file') or chunk.get('metadata', {}).get('file') or 'Unknown'
            if file_name not in files_content:
                files_content[file_name] = []
            files_content[file_name].append(chunk)

        # Format grouped content
        for file_name, file_chunks in files_content.items():
            context += f"\n### File: {file_name}\n"
            for i, chunk in enumerate(file_chunks, start=1):
                context += f"\n[Chunk {i}]\n"
                context += f"Content: {chunk.get('content', '')}\n"

        return context

    # =============================================================================
    # Custom RAG Tools
    # =============================================================================

    async def _get_custom_rag_tools(self) -> List[Any]:
        """Get all active custom RAG tool instances."""
        if not self.custom_tool_repo:
            return []
        try:
            custom_tools = await self.custom_tool_repo.get_rag_instances()
            return custom_tools
        except Exception as e:
            self.logger.error(f"Error fetching custom RAG tools: {e}")
            return []

    async def _execute_custom_rag_tool(
        self,
        custom_tool: Any,
        conversation: Conversation,
        query: str,
        settings: ConversationSettings,
        file_ids: Optional[List[UUID]] = None,
        collection_name: Optional[str] = None,
        **kwargs
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Execute a custom RAG tool instance."""
        from src.tools.custom_tool import CustomToolExecutor

        self.logger.info(f"Executing custom RAG tool: {custom_tool.name}")

        try:
            executor = CustomToolExecutor(
                custom_tool.id,
                file_repo=self.file_repo,
                custom_tool_repo=self.custom_tool_repo
            )

            collections = []

            if collection_name:
                collections = [collection_name]
                self.logger.info(f"Using explicit collection: {collection_name}")
            else:
                if custom_tool.configuration and custom_tool.configuration.get("collections"):
                    collections = custom_tool.configuration.get("collections", [])
                else:
                    collections = await self._get_default_collections(conversation)

            if not collections:
                self.logger.warning(f"No collections available for custom tool {custom_tool.name}")
                return None, None

            result = await executor.execute(
                query=query,
                collections=collections,
                file_repo=self.file_repo,
                **kwargs
            )

            if result.success and result.data:
                self.logger.info(f"Custom RAG tool {custom_tool.name} executed successfully")
                formatted_context = self._format_rag_context(result.data)
                return formatted_context, result.data
            else:
                self.logger.warning(f"Custom RAG tool {custom_tool.name} returned no data")
                return None, None

        except Exception as e:
            self.logger.error(f"Error executing custom RAG tool {custom_tool.name}: {e}", exc_info=True)
            return None, None

    def _get_tool_config(self, conversation: Conversation, tool_name: str):
        """Get tool configuration for conversation."""
        for config in conversation.tool_configurations:
            if config.tool_name == tool_name and config.is_active:
                return config
        return None

    # =============================================================================
    # Fallback Content Retrieval
    # =============================================================================

    async def _fetch_file_content_fallback(self, file_ids: List[str]) -> Optional[str]:
        """
        Fetch file content directly from DB or disk as fallback.
        Size limits para prevenir OOM.
        """
        try:
            uuids = [UUID(fid) for fid in file_ids]
            files = await self.file_repo.get_by_ids(uuids)

            if not files:
                self.logger.warning(f"Fallback: No files found in DB for IDs {file_ids}")
                return None

            context = "## Relevant Documentation (Direct Retrieval)\n\n"
            total_size = 0

            for file_record in files:
                self.logger.debug(f"Fallback processing file: {file_record.file_name}")
                content = ""

                # Try extracted text from metadata first
                if file_record.extra_metadata and file_record.extra_metadata.get("extracted_text"):
                    content = file_record.extra_metadata.get("extracted_text")
                    self.logger.debug("Fallback: Retrieved content from metadata")

                # If no extracted text, try reading from disk with size limit
                if not content and file_record.storage_path:
                    try:
                        file_size = os.path.getsize(file_record.storage_path)

                        if file_size > self._max_file_size:
                            self.logger.warning(
                                f"File {file_record.file_name} too large ({file_size} bytes), truncating"
                            )
                            try:
                                import aiofiles
                                async with aiofiles.open(file_record.storage_path, mode='r',
                                                        encoding='utf-8', errors='ignore') as f:
                                    content = await f.read(self._max_file_size)
                                    content += "\n\n[... File truncated due to size ...]"
                            except ImportError:
                                with open(file_record.storage_path, mode='r', encoding='utf-8',
                                        errors='ignore') as f:
                                    content = f.read(self._max_file_size)
                                    content += "\n\n[... File truncated due to size ...]"
                        else:
                            try:
                                import aiofiles
                                async with aiofiles.open(file_record.storage_path, mode='r',
                                                        encoding='utf-8', errors='ignore') as f:
                                    content = await f.read()
                            except ImportError:
                                with open(file_record.storage_path, mode='r', encoding='utf-8',
                                        errors='ignore') as f:
                                    content = f.read()

                        self.logger.debug(f"Fallback: Read {len(content)} chars from disk")

                    except Exception as e:
                        self.logger.warning(f"Failed to read file {file_record.id} from disk: {e}")

                if content:
                    total_size += len(content)

                    if total_size > self._max_total_size:
                        self.logger.warning(
                            f"Total content size exceeded {self._max_total_size} bytes, stopping"
                        )
                        context += "\n\n[... Additional files omitted due to size limits ...]"
                        break

                    context += f"### File: {file_record.file_name}\n\n"
                    context += f"**Content**\n{content}\n\n"
                    context += "---\n\n"
                else:
                    self.logger.warning(f"Fallback: No content found for file {file_record.file_name}")

            return context if total_size > 0 else None

        except Exception as e:
            self.logger.error(f"Error in fallback retrieval: {e}", exc_info=True)
            return None

    # =============================================================================
    # Memory Context
    # =============================================================================

    async def build_memory_context(
        self,
        conversation: Conversation,
        user_message: str,
        memory_config: Dict[str, Any]
    ) -> Optional[str]:
        """
        Build semantic memory context from past conversations.

        Args:
            conversation: Current conversation
            user_message: User query
            memory_config: Memory configuration

        Returns:
            Memory context string or None
        """
        if not memory_config.get("semantic_enabled", False):
            return None

        try:
            return await self.memory_service.retrieve_relevant_context(
                conversation=conversation,
                current_query=user_message,
                memory_config=memory_config
            )
        except Exception as e:
            self.logger.error(f"Error building memory context: {e}")
            return None

    # =============================================================================
    # Context String Building
    # =============================================================================

    def build_context_string(self, context_parts: List[str], custom_header: Optional[str] = None) -> str:
        """
        Build context string from tool results with high authority and NL instructions.

        Args:
            context_parts: List of context strings
            custom_header: Optional custom header (e.g. for specific intents)

        Returns:
            Combined context string
        """
        if not context_parts:
            return ""

        context = "\n\n".join(context_parts)

        header = custom_header if custom_header else SystemPrompts.SOURCE_OF_TRUTH_HEADER

        return (
            header +
            f"{context}" +
            SystemPrompts.SOURCE_OF_TRUTH_FOOTER
        )

    # =============================================================================
    # Obsidian Context Expansion
    # =============================================================================

    async def _expand_obsidian_context(
        self,
        main_results: List[Dict],
        conversation: Conversation,
        query: str,
        settings: ConversationSettings
    ) -> Optional[str]:
        """
        Expande contexto Obsidian usando navegación inteligente y adaptativa.
        Maneja ciclos y dependencias recursivas automáticamente.
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

        intent = self._analyze_navigation_intent(query, note_name, note_metadata)

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

        if len(nav_result.visited_notes) <= 1:
            self.logger.debug("No additional context found")
            return None

        # PASO 3: RECUPERAR CONTENIDO DE NOTAS RELEVANTES
        collections = await self._get_default_collections(conversation)
        related_context_parts = []

        layer_order = (
            sorted(nav_result.context_layers.keys(), reverse=True)
            if intent.direction == "down"
            else sorted(nav_result.context_layers.keys())
        )

        for depth in layer_order:
            if depth == 0:
                continue

            notes = nav_result.context_layers[depth]

            for related_note in notes[:5]:
                if len(related_context_parts) >= 8:
                    break

                try:
                    related_result = await self._rag_tool.execute(
                        query=f"{related_note} {query}",
                        collections=collections,
                        k=1,
                        score_threshold=0.25,
                        file_repo=self.file_repo
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

    def _get_graph_builder(self):
        """Obtiene o crea el graph builder de Obsidian."""
        if not hasattr(self, '_graph_builder_cache'):
            self._graph_builder_cache = None

        if self._graph_builder_cache:
            return self._graph_builder_cache

        try:
            from src.document_loaders.obsidian_detector import ObsidianDetector
            from src.document_loaders.obsidian_graph import ObsidianGraphBuilder
            from pathlib import Path
            import asyncio

            vault_path = settings.get_vault_path()

            if not vault_path:
                self.logger.warning("No OBSIDIAN_VAULT_PATH configured - Obsidian navigation disabled")
                return None

            vault_path = Path(vault_path)

            if not vault_path.exists():
                self.logger.warning(f"Obsidian vault path does not exist: {vault_path}")
                return None

            detector = ObsidianDetector()
            context = detector.detect(vault_path)

            if not context.is_obsidian:
                self.logger.warning(f"Path is not an Obsidian vault: {vault_path}")
                return None

            self.logger.info(f"Building Obsidian graph from vault: {context.vault_root}")
            graph_builder = ObsidianGraphBuilder()

            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(graph_builder.scan_vault(context.vault_root))
                notes = loop.run_until_complete(task)
                graph_builder.build_bidirectional_graph()
            except RuntimeError:
                notes = asyncio.run(graph_builder.scan_vault(context.vault_root))
                graph_builder.build_bidirectional_graph()

            self.logger.info(f"Graph built successfully: {len(notes)} notes indexed")

            self._graph_builder_cache = graph_builder
            return graph_builder

        except ImportError as e:
            self.logger.error(f"Failed to import Obsidian modules: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error building Obsidian graph: {e}", exc_info=True)
            return None

    # =============================================================================
    # Utility Methods
    # =============================================================================

    def _is_vague_query(self, query: str) -> bool:
        """Determina si una consulta es vaga o demasiado genérica."""
        query_strip = query.strip().lower()

        if len(query_strip) < 20:
            return True

        vague_patterns = [
            r'^qu[eé]\s+dice(\s+esto)?$',
            r'^analiza(\s+esto)?$',
            r'^resumen$',
            r'^de\s+qu[eé]\s+trata(\s+esto)?$',
            r'^explica(\s+esto)?$',
            r'^dime\s+m[aá]s$',
            r'^qu[eé]\s+opinas(\s+de\s+esto)?$'
        ]

        for pattern in vague_patterns:
            if re.match(pattern, query_strip):
                return True

        return False

    # =============================================================================
    # MÉTODOS PARA AGREGAR EN context_builder.py
    # Ubicación: Después de _extract_file_name_from_query() (línea ~1340)
    # =============================================================================

    def _extract_file_name_from_query(
        self,
        query: str,
        conversation_history: Optional[List[ChatMessage]] = None
    ) -> Optional[str]:
        """
        Extract file name from query when user mentions a specific document.

        NUEVO: Soporta referencias contextuales usando historial de conversación.

        Args:
            query: User query
            conversation_history: Optional list of previous messages for context

        Returns:
            Filename if found, None otherwise

        Examples:
            - "documento code_graph.py" → "code_graph.py"
            - "el archivo collection_service.py" → "collection_service.py"
            - "el otro archivo" → busca en historial → "code_graph.py"
            - "y en el anterior?" → busca en historial → archivo previo
        """
        # 1. PRIMERO: Buscar nombre explícito (lógica original)
        explicit_filename = self._extract_explicit_filename(query)
        if explicit_filename:
            self.logger.debug(f"Explicit filename detected: {explicit_filename}")
            return explicit_filename

        # 2. SEGUNDO: Si no hay nombre explícito, resolver referencias contextuales
        if conversation_history:
            contextual_filename = self._resolve_file_reference(query, conversation_history)
            if contextual_filename:
                self.logger.debug(
                    f"Contextual filename resolved: {contextual_filename} "
                    f"from reference in query"
                )
                return contextual_filename

        return None

    def _extract_explicit_filename(self, query: str) -> Optional[str]:
        """
        Extrae nombres de archivos explícitos del query (lógica original).

        Detecta patrones como:
        - "el documento X"
        - "archivo Y"
        - "fichero Z"
        - Nombres con extensión directamente

        Args:
            query: User query

        Returns:
            Filename if found explicitly
        """
        # Patrones con comillas o explícitos
        patterns = [
            re.compile(r'documento\s+["][^"\']+["\']', re.IGNORECASE),
            re.compile(r'archivo\s+["][^"\']+["\']', re.IGNORECASE),
            re.compile(r'fichero\s+["][^"\']+["\']', re.IGNORECASE),
            re.compile(r'el\s+documento\s+["][^"\']+["\']', re.IGNORECASE),
            re.compile(r'el\s+archivo\s+["][^"\']+["\']', re.IGNORECASE),
            re.compile(r'el\s+fichero\s+["][^"\']+["\']', re.IGNORECASE),
            re.compile(r'este\s+documento\s+["][^"\']+["\']', re.IGNORECASE),
            re.compile(r'este\s+archivo\s+["][^"\']+["\']', re.IGNORECASE),
            re.compile(r'este\s+fichero\s+["][^"\']+["\']', re.IGNORECASE),
            re.compile(r'sobre\s+["][^"\']+["\']', re.IGNORECASE),
            re.compile(r'del\s+documento\s+["][^"\']+["\']', re.IGNORECASE),
            re.compile(r'del\s+archivo\s+["][^"\']+["\']', re.IGNORECASE),
        ]

        for pattern in patterns:
            match = pattern.search(query)
            if match:
                full_match = match.group(0)
                first_quote = full_match.find('"')
                if first_quote != -1:
                    second_quote = full_match.find('"', first_quote + 1)
                    if second_quote != -1:
                        filename = full_match[first_quote + 1:second_quote]
                        filename = filename.strip()
                        # Limpiar prefijos comunes
                        filename = re.sub(
                            r'^\\s*(el|este|un|una|los|las|el\\s+documento|el\\s+archivo|el\\s+fichero|este\\s+documento|este\\s+archivo|este\\s+fichero|sobre|del\\s+documento|del\\s+archivo)\\s+',
                            '', filename, flags=re.IGNORECASE
                        )
                        return filename

        # Patrones de extensión directa (sin comillas)
        extension_patterns = [
            r'\\.docx\\s*["\\\']',
            r'\\.doc\\s*["\\\']',
            r'\\.pdf\\s*["\\\']',
            r'\\.txt\\s*["\\\']',
            r'\\.md\\s*["\\\']',
            r'\\.xlsx\\s*["\\\']',
            r'\\.csv\\s*["\\\']',
            r'\\.pptx\\s*["\\\']',
            r'\\.json\\s*["\\\']',
            r'\\.py\\s*["\\\']',
            r'\\.js\\s*["\\\']',
            r'\\.ts\\s*["\\\']',
            r'\\.java\\s*["\\\']',
            r'\\.sql\\s*["\\\']',
        ]

        for ext_pattern in extension_patterns:
            ext_match = re.search(ext_pattern, query, re.IGNORECASE)
            if ext_match:
                match_pos = ext_match.start()
                start_pos = match_pos
                while start_pos > 0 and query[start_pos - 1] not in ['"', "'", '.', ' ', ',', ';',
                                                                     ':',
                                                                     '\\n', '\\t']:
                    start_pos -= 1
                filename = query[start_pos:ext_match.end()].strip()
                if filename:
                    return filename

        return None

    def _resolve_file_reference(
        self,
        query: str,
        history: List[ChatMessage]
    ) -> Optional[str]:
        """
        Resuelve referencias pronominales/contextuales a archivos usando historial.

        Soporta patrones como:
        - "el otro archivo" → penúltimo archivo mencionado
        - "ese archivo" → último archivo mencionado
        - "el anterior" → archivo anterior al actual
        - "el primero" → primer archivo de la conversación
        - "y en el otro?" → mismo que "el otro archivo"

        Args:
            query: User query con referencia contextual
            history: Lista de mensajes previos de la conversación

        Returns:
            Filename resuelto desde historial, None si no se puede resolver
        """
        query_lower = query.lower().strip()

        # Detectar patrones de referencia contextual
        reference_patterns = {
            "otro": [
                "otro archivo", "el otro", "otro documento",
                "y en el otro", "en otro", "del otro archivo"
            ],
            "anterior": [
                "archivo anterior", "el anterior", "documento anterior",
                "y el anterior", "en el anterior"
            ],
            "ese": [
                "ese archivo", "ese documento", "aquel archivo",
                "ese", "aquel"
            ],
            "este": [
                "este archivo", "este documento", "este"
            ],
            "primero": [
                "el primero", "primer archivo", "primera"
            ],
            "último": [
                "el último", "último archivo", "última"
            ]
        }

        detected_reference = None
        for ref_type, patterns in reference_patterns.items():
            if any(p in query_lower for p in patterns):
                detected_reference = ref_type
                self.logger.debug(f"Detected file reference type: {ref_type}")
                break

        if not detected_reference:
            return None

        # Extraer archivos mencionados del historial (cronológico)
        files_in_history = self._extract_files_from_history(history)

        if not files_in_history:
            self.logger.debug("No files found in conversation history")
            return None

        self.logger.debug(
            f"Files in history (chronological): {files_in_history}"
        )

        # Resolver según tipo de referencia
        resolved_filename = None

        if detected_reference == "otro":
            # "Otro" = diferente al último mencionado (penúltimo)
            if len(files_in_history) >= 2:
                resolved_filename = files_in_history[-2]
            else:
                # Si solo hay 1 archivo, "otro" no tiene sentido, retornar el único
                resolved_filename = files_in_history[-1] if files_in_history else None

        elif detected_reference == "anterior":
            # "Anterior" = penúltimo archivo
            if len(files_in_history) >= 2:
                resolved_filename = files_in_history[-2]
            else:
                resolved_filename = files_in_history[0] if files_in_history else None

        elif detected_reference == "ese" or detected_reference == "este":
            # "Ese/Este" = último mencionado
            resolved_filename = files_in_history[-1]

        elif detected_reference == "primero":
            # "Primero" = primer archivo de la conversación
            resolved_filename = files_in_history[0]

        elif detected_reference == "último":
            # "Último" = archivo más reciente
            resolved_filename = files_in_history[-1]

        if resolved_filename:
            self.logger.info(
                f"Resolved contextual reference '{detected_reference}' → {resolved_filename}"
            )

        return resolved_filename

    def _extract_files_from_history(
        self,
        history: List[ChatMessage]
    ) -> List[str]:
        """
        Extrae nombres de archivos mencionados en el historial de conversación.

        Busca en:
        - Mensajes USER que mencionen archivos
        - Mensajes ASSISTANT que hablen de archivos
        - Detecta archivos por extensión común

        Args:
            history: Lista de mensajes de la conversación (ordenados cronológicamente)

        Returns:
            Lista de filenames únicos en orden cronológico de primera mención
        """
        files = []  # Mantener orden de primera aparición
        seen = set()  # Para evitar duplicados

        # Patrón para detectar archivos por extensión
        # Soporta: .py, .js, .ts, .java, .cpp, .txt, .md, .json, .yaml, .yml,
        #          .pdf, .docx, .xlsx, .csv, .sql, .sh, .bat, .html, .css
        file_pattern = re.compile(
            r'\b([\w\-]+\.(py|js|ts|jsx|tsx|java|cpp|c|h|hpp|txt|md|json|yaml|yml|'
            r'pdf|docx|doc|xlsx|xls|csv|sql|sh|bat|html|css|xml|go|rs|rb|php))\b',
            re.IGNORECASE
        )

        for msg in history:
            if not msg.content:
                continue

            content = msg.content

            # Buscar archivos en el contenido
            matches = file_pattern.findall(content)

            for match, _ in matches:
                # match[0] es el filename completo (con extensión)
                filename = match

                # Normalizar (lowercase para comparación)
                filename_normalized = filename.lower()

                if filename_normalized not in seen:
                    seen.add(filename_normalized)
                    files.append(filename)  # Mantener caso original
                    self.logger.debug(
                        f"Found file in history: {filename} "
                        f"(from {msg.role} message)"
                    )

        return files

    async def has_code_files(self, file_ids: List[UUID]) -> bool:
        """Check if any of the attached files are source code files."""
        if not file_ids:
            return False

        try:
            files = await self.file_repo.get_by_ids(file_ids)
            file_names = [f.filename for f in files]

            code_extensions = DocumentLoaderFactory.get_code_extensions()
            for name in file_names:
                if any(name.lower().endswith(ext) for ext in code_extensions):
                    return True
            return False
        except Exception as e:
            self.logger.warning(f"Error checking for code files: {e}")
            return False

    def _analyze_navigation_intent(
        self,
        query: str,
        current_note: str,
        note_metadata: Dict[str, Any]
    ) -> "NavigationIntent":
        """
        Analiza la intención de navegación para Obsidian.

        Determina dirección (up/down/bidirectional) basado en patrones de query.

        ✅ NUEVO: Reemplaza QueryIntentAnalyzer con lógica simple basada en patrones.

        Args:
            query: Query del usuario
            current_note: Nota actual
            note_metadata: Metadata de la nota (links, etc)

        Returns:
            NavigationIntent con dirección y parámetros
        """
        from src.document_loaders.obsidian_tree_navigator import NavigationIntent

        query_lower = query.lower().strip()

        # Patrones para dirección UP (contexto general)
        up_patterns = [
            "qué es", "que es", "explica", "define",
            "contexto", "overview", "resumen",
            "para qué", "para que sirve", "propósito"
        ]

        # Patrones para dirección DOWN (detalles, componentes)
        down_patterns = [
            "componentes", "partes", "contiene",
            "qué tiene", "que tiene", "lista",
            "detalles", "implementación", "cómo funciona",
            "aplicaciones", "servicios", "archivos"
        ]

        # Detectar dirección
        direction = "bidirectional"  # Default
        reasoning = "Query general - navegación bidireccional"
        confidence = 0.6

        if any(pattern in query_lower for pattern in up_patterns):
            direction = "up"
            reasoning = "Query solicita contexto general - navegando hacia arriba"
            confidence = 0.8
        elif any(pattern in query_lower for pattern in down_patterns):
            direction = "down"
            reasoning = "Query solicita detalles específicos - navegando hacia abajo"
            confidence = 0.8

        # Determinar profundidad según complejidad de query
        max_depth = 2  # Default
        if len(query_lower) < 20:
            max_depth = 1  # Query simple
        elif any(
            word in query_lower for word in ["completa", "toda", "arquitectura", "estructura"]):
            max_depth = 3  # Query exhaustiva

        # Ajustar max_nodes según dirección
        max_nodes = 15
        if direction == "down":
            max_nodes = 20  # Más nodos para detalles
        elif direction == "up":
            max_nodes = 10  # Menos nodos para contexto

        return NavigationIntent(
            direction=direction,
            max_depth=max_depth,
            max_nodes=max_nodes,
            confidence=confidence,
            reasoning=reasoning
        )
