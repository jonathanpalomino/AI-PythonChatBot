# =============================================================================
# src/services/file_processor.py
# File Processing Service - FIXED VERSION with Dynamic Batching
# =============================================================================
"""
Service for processing uploaded files:
- Extract content using document loaders
- Chunk content for embedding
- Generate embeddings via Ollama
- Index to Qdrant vector database with DYNAMIC BATCHING
- Update file processing status
- Analyze code files automatically

ROOT CAUSE FIXES APPLIED:
1. ✅ Pre-validation batching (add ONLY if fits)
2. ✅ Smart metadata truncation (lists, strings, nested dicts)
3. ✅ Real byte-accurate size estimation
4. ✅ Conservative limits (28MB safe margin)
5. ✅ Preserves ALL existing functionality
"""
import asyncio
import hashlib
import json
import re
from asyncio import Semaphore
from collections import OrderedDict
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any
from uuid import UUID

import redis
from ollama import AsyncClient as OllamaClient
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.config.settings import settings, get_qdrant_config
from src.document_loaders import DocumentLoaderFactory
from src.document_loaders.obsidian_detector import ObsidianContext, ObsidianDetector
from src.document_loaders.obsidian_graph import ObsidianGraphBuilder
from src.models.models import File as FileModel, ProcessingStatus, VisibilityType
from src.repositories import (
    FileRepository,
    ConversationRepository,
    QdrantCollectionRepository,
)
from src.services.analysis.codebase_analyzer import LANGUAGE_BY_EXTENSION, CodebaseAnalyzer
from src.services.embedding.embedding_service import get_embedding_service
from src.services.search.hybrid_search import BM25Index
from src.utils.date_utils import get_current_utc, get_current_utc_iso
from src.utils.logger import get_logger, set_conversation_context


class ChunkingStrategy(str, Enum):
    """Estrategias de chunking disponibles"""
    FIXED_SIZE = "fixed"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"


class FileProcessingError(Exception):
    """Custom exception for file processing errors"""
    pass


# =============================================================================
# METADATA POLICY: Smart truncation by type
# =============================================================================

class MetadataPolicy:
    """
    Intelligent metadata filtering and truncation.

    Prevents payload explosion while preserving semantic value.
    """

    # Keys always allowed (core metadata)
    CORE_KEYS = {
        'file', 'file_id', 'conversation_id', 'section', 'content',
        'parent_id', 'chunk_index', 'total_chunks', 'context'
    }

    # Keys allowed by document type
    TYPE_SPECIFIC = {
        'markdown': {'heading_level', 'breadcrumb', 'is_list_item'},
        'word': {'doc_author', 'doc_title', 'breadcrumb', 'heading_level', 'is_table'},
        'code': {
            'language', 'functions', 'classes', 'complexity_estimate',
            # Python semantic metadata
            'type', 'name', 'full_name', 'signature', 'docstring',
            'parameters', 'return_type', 'decorators', 'is_async',
            'parent_class', 'start_line', 'end_line', 'line_count',
            'complexity_hints', 'base_classes', 'methods', 'attributes',
            'imports', 'global_variables', 'module_docstring'
        },
        'sql': {'language', 'object_type', 'schema'},
        'default': {'language', 'created_at'}
    }

    # Max sizes for different types
    MAX_STRING_LENGTH = 2000  # Increased for docstrings and signatures
    MAX_LIST_ITEMS = 10  # Increased for parameters and decorators
    MAX_DICT_ITEMS = 15  # Increased for parameter details

    @classmethod
    def detect_type(cls, file_path: Path) -> str:
        """Detect document type from extension"""
        ext = file_path.suffix.lower()
        if ext in {'.md', '.markdown'}:
            return 'markdown'
        if ext in {'.docx', '.doc'}:
            return 'word'
        from src.document_loaders import CodeLoader, SqlPlsqlLoader

        # SQL/PLSQL extensions
        sql_extensions = set()
        for loader in DocumentLoaderFactory._loaders:
            if isinstance(loader, SqlPlsqlLoader):
                sql_extensions.update(loader.supported_extensions)

        if ext in sql_extensions:
            return 'sql'

        # Other code extensions
        code_extensions = set()
        for loader in DocumentLoaderFactory._loaders:
            if isinstance(loader, CodeLoader):
                code_extensions.update(loader.supported_extensions)

        if ext in code_extensions:
            return 'code'

        return 'default'

    @classmethod
    def truncate_value(cls, value: Any) -> Any:
        """
        Truncate a value to prevent payload explosion.

        Handles: strings, lists, dicts, nested structures
        """
        if isinstance(value, str):
            return value[:cls.MAX_STRING_LENGTH]

        elif isinstance(value, list):
            # Keep first N items, truncate each
            truncated = []
            for item in value[:cls.MAX_LIST_ITEMS]:
                truncated.append(cls.truncate_value(item))
            return truncated

        elif isinstance(value, dict):
            # Keep first N items, truncate each
            truncated = {}
            for i, (k, v) in enumerate(value.items()):
                if i >= cls.MAX_DICT_ITEMS:
                    break
                truncated[k] = cls.truncate_value(v)
            return truncated

        elif isinstance(value, (int, float, bool, type(None))):
            return value

        else:
            # Unknown type, convert to string and truncate
            return str(value)[:cls.MAX_STRING_LENGTH]

    @classmethod
    def filter_and_truncate(
        cls,
        metadata: Dict[str, Any],
        doc_type: str,
        file_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Filter and truncate metadata intelligently.

        Args:
            metadata: Raw metadata dict
            doc_type: Document type ('word', 'code', etc.)
            file_path: Optional file path for type detection

        Returns:
            Filtered and truncated metadata dict
        """
        if not metadata:
            return {}

        # Detect type from path if provided
        if file_path:
            doc_type = cls.detect_type(file_path)

        # Get allowed keys
        allowed_keys = cls.CORE_KEYS | cls.TYPE_SPECIFIC.get(doc_type, cls.TYPE_SPECIFIC['default'])

        # Filter and truncate
        result = {}
        for key, value in metadata.items():
            if key in allowed_keys:
                result[key] = cls.truncate_value(value)

        return result


# =============================================================================
# DYNAMIC BATCHER: Conservative & Incremental (VERSIÓN FINAL)
# =============================================================================

class DynamicQdrantBatcher:
    """
    Acumula puntos hasta alcanzar un límite seguro de tamaño o un máximo de puntos.

    Esta versión utiliza:
    1. Estimación conservadora (ensure_ascii=True) para caracteres Unicode escapados.
    2. Límite de 12MB (Ultra-conservador) para bytes.
    3. Límite estricto de 150 puntos.
    """

    # Límite Qdrant = 33,554,432 bytes (32MB).
    # Límite de bytes ultra-conservador para compensar overhead de HTTP y JSON escaping.
    # Optimizado para documentos Word más limpios: aumentado a 16MB
    MAX_BATCH_BYTES = 16 * 1024 * 1024  # 16 MB

    # Límite estricto de puntos optimizado para documentos Word
    # Aumentado a 200 para mejor throughput con payloads más pequeños
    MAX_BATCH_POINTS = 200

    def __init__(self, logger=None):
        self.points: List[PointStruct] = []
        self.current_batch_size = 0
        self.logger = logger
        self._batch_count = 0

    def _get_point_size(self, point: PointStruct) -> int:
        """
        Calcula el tamaño JSON de un solo punto de forma conservadora.
        Usa ensure_ascii=True para simular el peor caso de tamaño (caracteres escapados).
        """
        payload = {
            "id": point.id,
            "vector": point.vector,
            "payload": point.payload
        }
        # +2 bytes por la coma y espacio en la lista JSON
        return len(json.dumps(payload, ensure_ascii=True).encode("utf-8")) + 2

    def add(self, point: PointStruct) -> Optional[List[PointStruct]]:
        """
        Intenta agregar un punto. Si excede el límite de bytes O el límite de puntos,
        devuelve el lote anterior para enviar.
        """
        point_size = self._get_point_size(point)
        current_points_count = len(self.points)

        # ---------------------------------------------------------------------
        # Condición para forzar el envío del lote actual
        # ---------------------------------------------------------------------
        should_flush = (
            self.points and
            (
                (self.current_batch_size + point_size > self.MAX_BATCH_BYTES) or
                (current_points_count >= self.MAX_BATCH_POINTS)
            )
        )
        # ---------------------------------------------------------------------

        if should_flush:
            # ... devolvemos el lote actual para que se envíe.
            batch_to_flush = self.points

            if self.logger:
                # Determinar la razón del vaciado para el log
                reason = "Size Limit" if self.current_batch_size + point_size > self.MAX_BATCH_BYTES else "Point Count Limit"
                self.logger.info(
                    f"Batch full ({reason}). Flushing {len(batch_to_flush)} points.",
                    extra={
                        "batch_size_mb": round(self.current_batch_size / 1024 / 1024, 2),
                        "next_point_size_kb": round(point_size / 1024, 2),
                        "batch_number": self._batch_count + 1,
                        "flush_reason": reason
                    }
                )

            # Comenzamos un nuevo lote con el punto actual
            self.points = [point]
            self.current_batch_size = point_size
            self._batch_count += 1

            return batch_to_flush
        else:
            # Cabe en el lote actual, lo agregamos
            self.points.append(point)
            self.current_batch_size += point_size
            return None

    def flush(self) -> Optional[List[PointStruct]]:
        """Devuelve los puntos restantes al finalizar."""
        if not self.points:
            return None

        batch = self.points
        if self.logger:
            self.logger.info(
                f"Final flush of {len(batch)} points",
                extra={
                    "batch_size_mb": round(self.current_batch_size / 1024 / 1024, 2),
                    "batch_number": self._batch_count + 1
                }
            )

        self.points = []
        self.current_batch_size = 0
        self._batch_count += 1
        return batch

    def get_stats(self) -> dict:
        return {
            "total_batches": self._batch_count,
            "points_in_current": len(self.points),
            "current_size_mb": round(self.current_batch_size / 1024 / 1024, 2)
        }

# =============================================================================
# FILE PROCESSOR: Main class (ALL FUNCTIONALITY PRESERVED)
# =============================================================================

class FileProcessor:
    """
    Handles file content extraction and indexing to Qdrant.

    REPOSITORY PATTERN COMPLIANCE:
    This service ONLY interacts with Repositories, NEVER with the database session directly.
    All database operations are encapsulated in the Repository layer:
        Service → Repository → Database
    """

    def __init__(
        self,
        file_repo: FileRepository,
        conversation_repo: ConversationRepository,
        qdrant_collection_repo: QdrantCollectionRepository
    ):
        self.file_repo = file_repo
        self.conversation_repo = conversation_repo
        self.qdrant_collection_repo = qdrant_collection_repo
        # NOTE: We do NOT store self.db - all DB operations go through repositories
        self.logger = get_logger(__name__)

        # Initialize clients
        qdrant_config = get_qdrant_config()
        self.qdrant = AsyncQdrantClient(**qdrant_config)
        self.ollama = OllamaClient(host=settings.OLLAMA_BASE_URL)
        # Initialize Redis client for progress tracking
        try:
            self.redis_client = redis.Redis.from_url(settings.REDIS_URL)
        except Exception as e:
            self.logger.warning(f"Failed to initialize Redis client: {e}")
            self.redis_client = None

        # === BUG #1 FIX: Obsidian vault cache con LRU + TTL ===
        self._vault_graphs_cache = OrderedDict()  # LRU cache
        self._vault_cache_max_size = 50  # Máximo 50 vaults en cache
        self._vault_cache_ttl = timedelta(hours=1)  # TTL 1 hora
        self._vault_cache_timestamps = {}  # Timestamp de último acceso

        # === BUG #3 FIX: Async locks para vault cache ===
        self._vault_locks = {}  # Lock per vault
        self._cache_lock = asyncio.Lock()  # Lock para locks dict

        # Obsidian detector
        self.obsidian_detector = ObsidianDetector()

    async def __aenter__(self):
        """Context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources"""
        await self.close()
        return False

    async def close(self):
        """Cleanup all resources"""
        try:
            # Close Qdrant client
            if hasattr(self, 'qdrant') and self.qdrant:
                try:
                    await self.qdrant.close()
                    self.logger.debug("Qdrant client closed")
                except Exception as e:
                    self.logger.warning(f"Error closing Qdrant client: {e}")

            # Close Redis client
            if hasattr(self, 'redis_client') and self.redis_client:
                try:
                    self.redis_client.close()
                    self.logger.debug("Redis client closed")
                except Exception as e:
                    self.logger.warning(f"Error closing Redis client: {e}")

            # Clear caches to free memory
            if hasattr(self, '_vault_graphs_cache'):
                self._vault_graphs_cache.clear()
                self._vault_cache_timestamps.clear()
                self._vault_locks.clear()
                self.logger.debug("Vault caches cleared")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}", exc_info=True)

    def _set_progress(self, file_id: UUID, progress: int):
        """
        Set file processing progress in Redis with TTL.

        BUG #7 FIX: Adds 1-hour TTL to prevent Redis key leaks.
        """
        if not self.redis_client:
            return

        try:
            # Use setex to set value with expiration (atomic operation)
            self.redis_client.setex(
                name=f"processing:{file_id}",
                time=3600,  # 1 hour TTL
                value=progress
            )
        except Exception as e:
            self.logger.warning(f"Failed to set progress in Redis: {e}")

    async def delete_file_chunks(self, file_id: UUID) -> bool:
        """
        Delete all chunks associated with a file from Qdrant
        """
        try:
            # Get file record to determine collection
            file_record = await self.file_repo.get_by_id_with_conversation(file_id)

            if not file_record:
                self.logger.warning(f"File {file_id} not found, cannot delete chunks")
                return False

            collection_name = await self._get_collection_name(file_record)

            # Check if collection exists
            try:
                await self.qdrant.get_collection(collection_name)
            except Exception:
                self.logger.warning(f"Collection {collection_name} not found, skipping chunk deletion")
                return True

            self.logger.info(f"Deleting chunks for file {file_id} from collection {collection_name}")

            # Delete points where file_id matches
            # Qdrant filter based on payload
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            await self.qdrant.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="file_id",
                            match=MatchValue(value=str(file_id))
                        )
                    ]
                )
            )

            self.logger.info(f"Successfully deleted chunks for file {file_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete file chunks: {e}", exc_info=True)
            return False

    def _generate_point_id(self, file_id: UUID, section_index: int) -> str:
        """Generate unique and deterministic ID for a vector point"""
        hash_input = f"{file_id}:{section_index}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def _split_long_section(self, section, max_chars: int = 2000, overlap: int = 0, previous_context: str = "") -> List[dict]:
        """
        Split a long section into smaller chunks with parent document tracking.

        Args:
            section: DocumentSection to split
            max_chars: Maximum characters per chunk
            overlap: Number of characters to overlap between chunks
            previous_context: Context from the end of the previous section to prepend

        Returns:
            List of dicts with title, content, metadata, and parent references
        """
        # Prepend previous context if available
        # This ensures the first chunk of this section contains the tail of the previous section
        if previous_context:
            content = previous_context + section.content
        else:
            content = section.content

        # Generate unique parent_id based on section title and content hash
        parent_id = hashlib.md5(f"{section.title}:{content[:100]}".encode()).hexdigest()

        # If content fits in one chunk, still add parent metadata for consistency
        if len(content) <= max_chars:
            return [{
                'title': section.title,
                'content': content,
                'metadata': section.metadata,
                'parent_id': parent_id,
                'parent_title': section.title,
                'chunk_index': 0,
                'total_chunks': 1
            }]

        chunks = []
        current_pos = 0
        chunk_num = 0

        while current_pos < len(content):
            # Extract chunk
            chunk_end = current_pos + max_chars
            chunk_text = content[current_pos:chunk_end]

            # Variable to track where the actual content ends for this chunk
            # (excluding skipped delimiters for the next iteration's calculation, if any)
            actual_chunk_length = len(chunk_text)
            next_start_pos = current_pos + actual_chunk_length

            # Try to cut at natural point
            if chunk_end < len(content):
                # 1. Look for last double newline (Paragraphs)
                last_double_newline = chunk_text.rfind('\n\n')
                if last_double_newline > max_chars * 0.5:
                    chunk_text = chunk_text[:last_double_newline]
                    actual_chunk_length = last_double_newline
                    next_start_pos = current_pos + last_double_newline + 2

                # 2. Look for last single newline (Code lines)
                else:
                    last_newline = chunk_text.rfind('\n')
                    if last_newline > max_chars * 0.6: # Relaxed for lines
                        chunk_text = chunk_text[:last_newline]
                        actual_chunk_length = last_newline
                        next_start_pos = current_pos + last_newline + 1

                    # 3. Look for last period (Sentences - avoid for code if possible)
                    else:
                        is_code = section.metadata.get('language') or section.metadata.get('symbol_type')
                        last_period = chunk_text.rfind('. ')

                        # Only split at period if NOT code or if period is very late
                        if not is_code and last_period > max_chars * 0.7:
                            chunk_text = chunk_text[:last_period + 1]
                            actual_chunk_length = last_period + 1
                            next_start_pos = current_pos + last_period + 2
                        else:
                            # 4. Last space (Last resort)
                            last_space = chunk_text.rfind(' ')
                            if last_space > max_chars * 0.5:
                                chunk_text = chunk_text[:last_space]
                                actual_chunk_length = last_space
                                next_start_pos = current_pos + last_space + 1
                            else:
                                # Hard cut
                                next_start_pos = chunk_end
            else:
                next_start_pos = len(content)

            chunk_data = {
                'title': f"{section.title} (parte {chunk_num + 1})",
                'content': chunk_text.strip(),
                'metadata': section.metadata,
                'parent_id': parent_id,
                'parent_title': section.title,
                'chunk_index': chunk_num,
                'total_chunks': None
            }

            # TEMPORAL LOGGING: Print chunk content before it goes to embedding
            self.logger.info(f"📄 CHUNK TEMPORAL - Sección: {section.title} - Parte: {chunk_num + 1}")
            self.logger.info(f"📏 Tamaño: {len(chunk_text.strip())} caracteres")
            self.logger.info(f"📝 Contenido completo del chunk:")
            self.logger.info(f"{'='*80}")
            self.logger.info(chunk_text.strip())
            self.logger.info(f"{'='*80}")

            chunks.append(chunk_data)
            chunk_num += 1

            # Apply overlap
            # We want to start the next chunk 'overlap' chars before the current chunk ended (logically)
            # strictly speaking, we step forward by (length - overlap)
            # We must use next_start_pos as the baseline "end of this chunk's unique progression"

            if overlap > 0 and next_start_pos < len(content):
                # Calculate potential new start
                potential_start = next_start_pos - overlap

                # Ensure we always progress forward at least 1 character to avoid infinite loops
                if potential_start <= current_pos:
                    potential_start = current_pos + 1

                current_pos = potential_start
            else:
                current_pos = next_start_pos

        # Set total_chunks
        total = len(chunks)
        for chunk in chunks:
            chunk['total_chunks'] = total

        return chunks

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Ollama with handling for long texts"""
        embedding_service = await get_embedding_service()
        return await embedding_service.generate_embedding(text)

    async def _generate_chunk_context(
        self,
        chunk_content: str,
        document_title: str,
        document_preview: str,
        section_title: str
    ) -> str:
        """
        Generate contextual description for a chunk using LLM.

        Uses fast, small model to create 1-2 sentence context explaining
        chunk's role within the larger document.

        Returns: Brief context description
        """
        if not settings.ENABLE_CONTEXTUAL_RETRIEVAL:
            return ""

        try:
            # Create concise prompt
            prompt = f"""Documento: "{document_title}"

            Vista previa:
            {document_preview[:1500]}

            Sección: "{section_title}"

            Fragmento:
            {chunk_content[:400]}

            Escribe UNA frase corta (máximo 20 palabras) explicando qué información contiene este fragmento."""

            # Call LLM with fast model
            response = await self.ollama.chat(
                model=settings.CONTEXT_GENERATION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={
                    'num_predict': settings.CONTEXT_MAX_TOKENS,
                    'temperature': 0.3,  # Low temp for consistency
                    'num_gpu': 99
                }
            )

            context = response['message']['content'].strip()

            # Clean up common prefixes
            context = context.replace("Este fragmento ", "").replace("Esta sección ", "")
            context = context.strip('"').strip()

            return context

        except Exception as e:
            self.logger.warning(f"Failed to generate chunk context: {e}")
            return ""

    async def _ensure_collection_exists(self, collection_name: str, model: Optional[str] = None) -> None:
        """Ensure Qdrant collection exists, create if not"""
        try:
            self.logger.debug(f"Checking if collection '{collection_name}' exists in Qdrant")
            await self.qdrant.get_collection(collection_name)
            self.logger.debug(f"Collection '{collection_name}' already exists")
            return  # Collection exists, we're done
        except Exception as e:
            # Collection doesn't exist or error checking, try to create it
            self.logger.info(f"Creating new collection '{collection_name}'")
            try:
                # Dynamically determine vector size
                embedding_service = await get_embedding_service()
                vector_size = await embedding_service.get_embedding_dimension(model=model)

                await self.qdrant.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                self.logger.info(f"Collection '{collection_name}' created successfully with dimension {vector_size}")
            except Exception as qdrant_error:
                # Check if error is 409 Conflict (collection already exists)
                error_msg = str(qdrant_error)
                if "409" in error_msg or "already exists" in error_msg.lower():
                    self.logger.debug(
                        f"Collection '{collection_name}' already exists (409 Conflict)")
                    # Collection exists, this is fine
                else:
                    self.logger.error(
                        f"Failed to create collection in Qdrant: {qdrant_error}",
                        exc_info=True,
                        extra={"collection_name": collection_name}
                    )
                    raise

            # Register in PostgreSQL if chat collection
            if collection_name.startswith("chat_"):
                self.logger.debug("Registering collection in PostgreSQL",
                                  extra={"collection_name": collection_name})
                # Check if already registered
                try:
                    conv_id_str = collection_name.replace("chat_", "")
                    conversation_id = conv_id_str

                    # Use repository to create or get collection
                    collection_record, created = await self.qdrant_collection_repo.get_or_create(
                        name=collection_name,
                        display_name=f"Chat Collection - {conversation_id[:8]}",
                        description=f"Temporary collection for conversation {conversation_id}",
                        category="chat",
                        visibility=VisibilityType.PRIVATE,
                        extra_metadata={
                            "conversation_id": conversation_id,
                            "created_by": "file_processor",
                            "type": "temporary"
                        }
                    )
                    if created:
                        self.logger.info(
                            "Collection registered in database",
                            extra={"collection_name": collection_name,
                                   "conversation_id": conversation_id}
                        )
                except Exception as db_error:
                    self.logger.error(
                        f"Failed to register collection in database: {db_error}",
                        exc_info=True,
                        extra={"collection_name": collection_name}
                    )
                    # Rollback via repository
                    await self.qdrant_collection_repo.rollback()
                    self.logger.warning("Database registration failed but Qdrant collection exists")

            elif collection_name.startswith("project_"):
                self.logger.debug("Registering project collection in PostgreSQL",
                                  extra={"collection_name": collection_name})
                try:
                    project_id = collection_name.replace("project_", "")

                    # Use repository to create or get collection
                    collection_record, created = await self.qdrant_collection_repo.get_or_create(
                        name=collection_name,
                        display_name=f"Project Collection - {project_id[:8]}",
                        description=f"Shared collection for project {project_id}",
                        category="project",
                        visibility=VisibilityType.PRIVATE,
                        extra_metadata={
                            "project_id": project_id,
                            "created_by": "file_processor",
                            "type": "shared"
                        }
                    )
                    if created:
                        self.logger.info(
                            "Project collection registered in database",
                            extra={"collection_name": collection_name,
                                   "project_id": project_id}
                        )
                except Exception as db_error:
                    self.logger.error(
                        f"Failed to register project collection: {db_error}",
                        exc_info=True
                    )
                    await self.qdrant_collection_repo.rollback()

    async def _get_collection_name(self, file_record: FileModel) -> str:
        """Determine the collection name for a file"""
        # If file is associated with a project (PRIORITY)
        if file_record.project_id:
            return f"project_{file_record.project_id}"

        # If file is associated with a conversation
        if file_record.conversation_id:
            # Fetch conversation explicitly to avoid async lazy-load issues
            conversation = await self.conversation_repo.get_by_id(file_record.conversation_id)
            if conversation and conversation.settings:
                settings_dict = conversation.settings
                if hasattr(settings_dict, 'model_dump'):
                    settings_dict = settings_dict.model_dump()
                if settings_dict.get("rag_collection"):
                    return settings_dict["rag_collection"]

            # Create temporary collection for this conversation
            # Format: chat_{conversation_id}
            return f"chat_{file_record.conversation_id}"

        # Use default collection for files not associated with a conversation
        return settings.COLLECTION_NAME

    async def _get_embedding_model(self, file_record: FileModel) -> Optional[str]:
        """Get embedding model from conversation settings if available"""
        if file_record.conversation_id:
            conversation = await self.conversation_repo.get_by_id(file_record.conversation_id)
            if conversation and conversation.settings:
                settings_dict = conversation.settings
                if hasattr(settings_dict, 'model_dump'):
                    settings_dict = settings_dict.model_dump()
                return settings_dict.get('embedding_model')

        return None

    # =========================================================================
    # Code Analysis Methods (PRESERVED)
    # =========================================================================

    def _is_code_file(self, file_path: Path) -> bool:
        """Check if file is a code file based on extension"""
        code_extensions = DocumentLoaderFactory.get_code_extensions()
        return file_path.suffix.lower() in code_extensions

    def _detect_language(self, code: str, file_extension: str) -> str:
        """Detect programming language from code content and file extension"""

        # First try by extension
        if file_extension.lower() in LANGUAGE_BY_EXTENSION:
            return LANGUAGE_BY_EXTENSION[file_extension.lower()]

        # Fallback to content-based detection
        code_lower = code.lower()

        # JavaScript / TypeScript patterns
        if any(
            p in code for p in ["function", "const ", "let ", "var ", "=>", "import ", "export "]):
            if "interface " in code or ": string" in code or ": number" in code:
                return "typescript"
            return "javascript"

        # Python patterns
        if any(p in code_lower for p in ["def ", "import ", "class ", "self.", "print("]):
            return "python"

        # Java patterns
        if any(p in code for p in ["public class", "private ", "protected ", "void ", "static "]):
            return "java"

        # SQL / PL/SQL patterns
        if any(p in code_lower for p in ["select ", "create ", "insert ", "update ", "delete "]):
            if any(p in code_lower for p in ["procedure", "begin", "end;", "declare"]):
                return "plsql"
            return "sql"

        return "unknown"

    def _calculate_code_metrics(self, code: str, language: str) -> Dict[str, Any]:
        """Calculate code metrics"""
        lines = code.split("\n")
        metrics = {
            "total_lines": len(lines),
            "code_lines": self._count_code_lines(lines),
            "comment_lines": self._count_comment_lines(lines, language),
            "blank_lines": self._count_blank_lines(lines),
            "functions": self._count_functions(code, language),
            "classes": self._count_classes(code, language),
            "complexity_estimate": self._estimate_complexity(code, language),
        }
        return metrics

    def _count_code_lines(self, lines: list[str]) -> int:
        """Count non-blank, non-comment lines"""
        count = 0
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("//") and not stripped.startswith("#"):
                count += 1
        return count

    def _count_comment_lines(self, lines: list[str], language: str) -> int:
        """Count comment lines"""
        count = 0
        comment_prefixes = {
            "javascript": ["//", "/*", "*"],
            "typescript": ["//", "/*", "*"],
            "python": ["#"],
            "java": ["//", "/*", "*"],
            "sql": ["--", "/*"],
            "plsql": ["--", "/*"],
        }
        prefixes = comment_prefixes.get(language, ["//", "#"])
        for line in lines:
            stripped = line.strip()
            if any(stripped.startswith(p) for p in prefixes):
                count += 1
        return count

    def _count_blank_lines(self, lines: list[str]) -> int:
        """Count blank lines"""
        return sum(1 for line in lines if not line.strip())

    def _count_functions(self, code: str, language: str) -> int:
        """Count functions/methods"""
        patterns = {
            "javascript": r"\bfunction\s+\w+",
            "typescript": r"\bfunction\s+\w+",
            "python": r"\bdef\s+\w+",
            "java": r"\b(public|private|protected)\s+\w+\s+\w+\s*\(",
            "plsql": r"\bPROCEDURE\s+\w+|\bFUNCTION\s+\w+",
        }
        pattern = patterns.get(language)
        if not pattern:
            return 0
        return len(re.findall(pattern, code, re.IGNORECASE))

    def _count_classes(self, code: str, language: str) -> int:
        """Count classes"""
        patterns = {
            "javascript": r"\bclass\s+\w+",
            "typescript": r"\bclass\s+\w+",
            "python": r"\bclass\s+\w+",
            "java": r"\bclass\s+\w+",
        }
        pattern = patterns.get(language)
        if not pattern:
            return 0
        return len(re.findall(pattern, code))

    def _estimate_complexity(self, code: str, language: str) -> str:
        """Estimate code complexity"""
        complexity_indicators = [
            r"\bif\b",
            r"\bfor\b",
            r"\bwhile\b",
            r"\bswitch\b",
            r"\bcase\b",
            r"\btry\b",
            r"\bcatch\b",
            r"\b&&\b",
            r"\b\|\|\b",
        ]
        count = sum(len(re.findall(p, code, re.IGNORECASE)) for p in complexity_indicators)
        if count < 5:
            return "low"
        elif count < 15:
            return "medium"
        else:
            return "high"

    def _generate_code_suggestions(self, code: str, language: str, metrics: Dict) -> list[str]:
        """Generate code improvement suggestions"""
        suggestions = []

        if metrics["complexity_estimate"] == "high":
            suggestions.append("Consider breaking down complex functions into smaller ones")

        comment_ratio = metrics["comment_lines"] / max(metrics["code_lines"], 1)
        if comment_ratio < 0.1:
            suggestions.append("Add more comments to improve code readability")

        if metrics["functions"] == 0 and metrics["code_lines"] > 20:
            suggestions.append("Consider organizing code into functions for better modularity")

        if language in ("javascript", "typescript"):
            if "var " in code:
                suggestions.append("Replace 'var' with 'const' or 'let' for better scoping")
            if "== " in code or "!= " in code:
                suggestions.append("Use strict equality (=== and !==) instead of loose equality")

        if language == "python":
            if "\t" in code:
                suggestions.append("Use spaces instead of tabs for indentation (PEP 8)")

        if language in ("sql", "plsql"):
            if "SELECT *" in code.upper():
                suggestions.append("Avoid SELECT *, specify column names explicitly")

        return suggestions

    async def _analyze_code_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Analyze a code file and return analysis results"""
        try:
            # Read file content
            try:
                import aiofiles
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    code = await f.read()
            except ImportError:
                # Fallback to blocking
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()

            # Detect language
            language = self._detect_language(code, file_path.suffix)
            self.logger.debug(f"Detected language: {language} for file: {file_path.name}")

            # USE CODEBASE ANALYZER FOR ALL LANGUAGES (Deduped & Multi-language)
            try:
                analyzer = CodebaseAnalyzer()
                analysis_result = analyzer.analyze_file(code, file_path.name)

                # Check if analysis was useful (has symbols or explicitly successful)
                if "error" not in analysis_result:
                    symbols = analysis_result.get("symbols", [])
                    complexity_val = analysis_result.get("complexity", 1)
                    analyzer_metrics = analysis_result.get("metrics", {})

                    # Map to FileProcessor metrics structure
                    metrics = {
                        "code_lines": code.count('\n'),
                        "comment_lines": len([l for l in code.splitlines() if l.strip().startswith(('#', '//', '--'))]),
                        "classes": analyzer_metrics.get("classes", len([s for s in symbols if s.type == "class"])),
                        "functions": analyzer_metrics.get("functions", len([s for s in symbols if s.type in ("function", "method")])),
                        "complexity_estimate": "high" if complexity_val > 15 else ("medium" if complexity_val > 5 else "low"),
                        "imports": len(analysis_result.get("imports", []))
                    }

                    # Generate suggestions
                    suggestions = self._generate_code_suggestions(code, language, metrics)

                    # Convert CodeSymbol objects to dicts for JSON serialization
                    serializable_symbols = [s.to_dict() if hasattr(s, 'to_dict') else s for s in symbols]

                    return {
                        "language": language,
                        "metrics": metrics,
                        "suggestions": suggestions,
                        "analyzed": True,
                        "details": {
                            "file_path": analysis_result.get("file_path"),
                            "language": analysis_result.get("language"),
                            "imports": analysis_result.get("imports", []),
                            "symbols": serializable_symbols,
                            "complexity": complexity_val,
                            "metrics": analyzer_metrics
                        }
                    }
            except Exception as e:
                self.logger.warning(f"Universal analysis failed for {file_path}, falling back to legacy: {e}")

            # Legacy Fallback (only if Analyzer crashes entirely)
            metrics = self._calculate_code_metrics(code, language)

            # ... rest of legacy fallback ...
            suggestions = self._generate_code_suggestions(code, language, metrics)
            analysis = {
                "language": language,
                "metrics": metrics,
                "suggestions": suggestions,
                "analyzed": True
            }
            return analysis

        except Exception as e:
            self.logger.error(
                f"Code analysis failed: {e}",
                exc_info=True,
                extra={"file_path": str(file_path)}
            )
            return None

    # =========================================================================
    # MAIN PROCESS METHOD (FIXED WITH DYNAMIC BATCHING & PARALLELISM)
    # =========================================================================

    async def process_file(self, file_id: UUID) -> dict:
        """
        Process a file: extract content, generate embeddings, index to Qdrant

        Args:
            file_id: UUID of the file to process

        Returns:
            dict with processing statistics

        Raises:
            FileProcessingError: if processing fails
        Process file con TRUE BATCHING optimizado para GPU.

        OPTIMIZACIONES APLICADAS:
        1. ✅ Contextos en batch (no secuencial)
        2. ✅ Embeddings en batch optimizado
        3. ✅ Points building paralelo (20 workers)
        4. ✅ DynamicQdrantBatcher para evitar payload explosions
        5. ✅ Preserva TODA la funcionalidad original
        """
        # =========================================================================
        # CONFIGURACIÓN OPTIMIZADA
        # =========================================================================
        CONTEXT_BATCH_SIZE = 10  # Contextos en paralelo
        EMBED_BATCH_SIZE = 16  # Seguro para mxbai-embed-large 4GB
        CONCURRENCY_LIMIT = 20  # Points building paralelo

        # Get file record
        file_record = await self.file_repo.get_by_id_with_conversation(file_id)

        if not file_record:
            raise FileProcessingError(f"File {file_id} not found")

        # Update status to processing via repository
        await self.file_repo.update_processing_status(
            file_id,
            ProcessingStatus.PROCESSING
        )

        # Set conversation context for logging if available
        if file_record.conversation_id:
            set_conversation_context(str(file_record.conversation_id))

        try:
            self.logger.info(
                "Processing file",
                extra={
                    "file_id": str(file_id),
                    "file_name": file_record.file_name,
                    "embed_batch": EMBED_BATCH_SIZE,
                    "concurrency": CONCURRENCY_LIMIT,
                    "file_type": file_record.file_type,
                    "conversation_id": str(file_record.conversation_id)
                    if file_record.conversation_id else None
                }
            )

            # Get file path
            file_path = Path(file_record.storage_path) if file_record.storage_path else None

            if not file_path or not file_path.exists() or file_path.is_dir():
                # If storage_path is invalid, we might be in a race condition or have a broken record
                self.logger.error(f"Invalid storage path: {file_path}", extra={"storage_path": file_record.storage_path})
                raise FileProcessingError(f"Invalid storage path: {file_record.storage_path}")

            # === DETECCIÓN OBSIDIAN ===
            obsidian_context = None
            graph = None

            if file_path.suffix == '.md':
                obsidian_context = self.obsidian_detector.detect(file_path)

                self.logger.info(
                    "Obsidian detection result",
                    extra={
                        "type": obsidian_context.type,
                        "is_obsidian": obsidian_context.is_obsidian,
                        "vault_root": str(
                            obsidian_context.vault_root) if obsidian_context.vault_root else None,
                        "files_count": len(obsidian_context.files)
                    }
                )

                # Si es Obsidian, construir/obtener grafo
                if obsidian_context.is_obsidian:
                    graph = await self._get_or_build_vault_graph(obsidian_context)

            # Load document
            loader = DocumentLoaderFactory.get_loader(file_path)

            # Robust loader selection: try with original filename or file_type if extension is missing on disk path
            if not loader:
                # Try using the original filename from the database record
                self.logger.warning(f"No loader found for storage path extension {file_path.suffix}, trying original name...")
                loader = DocumentLoaderFactory.get_loader(Path(file_record.file_name))

            if not loader and file_record.file_type:
                # Try using the file_type from the database record (e.g., '.plsql')
                self.logger.warning(f"Still no loader, trying database file_type: {file_record.file_type}")
                # Ensure it starts with a dot for the loader factory
                ext = file_record.file_type if file_record.file_type.startswith('.') else f".{file_record.file_type}"
                loader = DocumentLoaderFactory.get_loader(Path(f"dummy{ext}"))

            if not loader:
                raise FileProcessingError(f"No loader for type: {file_record.file_name or file_record.file_type}")

            # Get original filename from metadata
            original_filename = file_record.extra_metadata.get("original_name") if file_record.extra_metadata else None

            # Cargar documento (con o sin contexto Obsidian)
            if graph and hasattr(loader, 'load_with_obsidian_context'):
                doc = loader.load_with_obsidian_context(file_path, graph, original_filename)
                self.logger.info(f"Loaded with Obsidian context: {len(doc.sections)} sections")
            else:
                doc = loader.load(file_path, original_filename)
                self.logger.info(f"Loaded {len(doc.sections)} sections")

            # Get embedding model
            embedding_model_name = await self._get_embedding_model(file_record)
            if embedding_model_name:
                self.logger.info(f"Using conversation embedding model: {embedding_model_name}")

            # Determine collection
            collection_name = await self._get_collection_name(file_record)
            await self._ensure_collection_exists(collection_name, model=embedding_model_name)
            # =====================================================================
            # PREPARE CHUNKS FOR PARALLEL PROCESSING
            # =====================================================================
            chunk_tasks_data = []
            point_ids = []
            global_chunk_index = 0

            # Use loader-recommended chunk size if available (loaders know their content best)
            # Otherwise fall back to default settings
            if hasattr(doc, 'recommended_chunk_size') and doc.recommended_chunk_size:
                chunk_size = doc.recommended_chunk_size
                chunk_overlap = int(chunk_size * 0.05)  # 5% overlap
                self.logger.info(f"Using loader-recommended chunking: {chunk_size} chars (overlap: {chunk_overlap})")
            else:
                chunk_size = settings.DEFAULT_CHUNK_SIZE
                chunk_overlap = settings.DEFAULT_CHUNK_OVERLAP

            previous_section_tail = ""

            for section in doc.sections:
                chunks = self._split_long_section(
                    section,
                    max_chars=chunk_size,
                    overlap=chunk_overlap,
                    previous_context=previous_section_tail
                )

                # Capture tail for next section before processing chunks
                # We need the tail from the *original* content of this section, not the combined one
                if len(section.content) >= chunk_overlap:
                    previous_section_tail = section.content[-chunk_overlap:]
                else:
                    # If section is very short, append it to existing tail (up to a limit if needed,
                    # but simple append is usually safe enough for this purpose)
                    previous_section_tail += section.content
                    # Ensure we don't carry too much history if multiple small sections accumulate
                    if len(previous_section_tail) > chunk_overlap:
                        previous_section_tail = previous_section_tail[-chunk_overlap:]
                for chunk in chunks:
                    # Inject section level into chunk map for the helper to use
                    chunk['section_level'] = section.level

                    point_id = self._generate_point_id(file_record.id, global_chunk_index)
                    point_ids.append(point_id)

                    chunk_tasks_data.append({
                        "chunk": chunk,
                        "point_id": point_id,
                        "section_index": global_chunk_index
                    })
                    global_chunk_index += 1

            total_chunks = len(chunk_tasks_data)
            self.logger.info(f"Prepared {total_chunks} chunks for processing")

            # =====================================================================
            # 4. GENERATE CONTEXTS IN BATCH (OPTIMIZACIÓN CLAVE #1)
            # =====================================================================
            doc_title = getattr(doc.metadata, "get", lambda k: None)("doc_title") or doc.file_name
            document_preview = doc.content[:2000] if hasattr(doc, "content") else ""

            contexts = []
            if settings.ENABLE_CONTEXTUAL_RETRIEVAL:
                self.logger.info("Generating contexts in batch...")

                # Crear tasks de contexto en batch
                context_tasks = []
                for data in chunk_tasks_data:
                    chunk = data["chunk"]
                    context_tasks.append(
                        self._generate_chunk_context(
                            chunk_content=chunk["content"],
                            document_title=doc_title,
                            document_preview=document_preview,
                            section_title=chunk["title"]
                        )
                    )

                # Ejecutar en batches con límite de concurrencia
                semaphore = Semaphore(CONTEXT_BATCH_SIZE)

                async def run_with_semaphore(task):
                    async with semaphore:
                        return await task

                # Procesar todos en paralelo (limitado por semáforo)
                contexts = await asyncio.gather(*[
                    run_with_semaphore(task) for task in context_tasks
                ])

                self.logger.info(f"Generated {len(contexts)} contexts in batch")
            else:
                contexts = [""] * total_chunks

            # =====================================================================
            # 5. PREPARE TEXTS FOR EMBEDDING (CON ENRIQUECIMIENTO OBSIDIAN)
            # =====================================================================
            texts_for_embedding = []
            for idx, data in enumerate(chunk_tasks_data):
                chunk = data["chunk"]
                context = contexts[idx]

                # === ENRIQUECIMIENTO OBSIDIAN ===
                text_parts = []

                # Agregar contexto si existe
                if context:
                    text_parts.append(context)

                # Agregar información de links si es nota Obsidian
                if graph:
                    note_name = file_path.stem
                    if note_name in graph:
                        note_meta = graph[note_name]["metadata"]

                        # Agregar contexto de relaciones
                        if note_meta["outgoing"]:
                            related_str = ", ".join(note_meta["outgoing"][:5])  # Top 5
                            text_parts.append(f"[Relacionado con: {related_str}]")

                        if note_meta["is_hub"]:
                            text_parts.append("[Concepto central]")
                        elif note_meta["is_index"]:
                            text_parts.append("[Nota índice]")

                # Agregar título y contenido
                text_parts.append(chunk['title'])
                text_parts.append(chunk['content'])

                text = "\n\n".join(text_parts)
                # Agregar información de links si es nota Obsidian
                if graph:
                    note_name = file_path.stem
                    if note_name in graph:
                        note_meta = graph[note_name]["metadata"]

                        # Agregar contexto de relaciones
                        if note_meta["outgoing"]:
                            related_str = ", ".join(note_meta["outgoing"][:5])  # Top 5
                            text_parts.append(f"[Relacionado con: {related_str}]")

                        if note_meta["is_hub"]:
                            text_parts.append("[Concepto central]")
                        elif note_meta["is_index"]:
                            text_parts.append("[Nota índice]")

                # Agregar título y contenido
                text_parts.append(chunk['title'])
                text_parts.append(chunk['content'])

                text = "\n\n".join(text_parts)
                texts_for_embedding.append(text)
            # =====================================================================
            # 6. GENERATE EMBEDDINGS IN TRUE BATCH (OPTIMIZACIÓN CLAVE #2)
            # =====================================================================
            self.logger.info(f"Generating embeddings in batches of {EMBED_BATCH_SIZE}")

            # Detect if this is a code file (skip truncation for code)
            # Use LANGUAGE_BY_EXTENSION from codebase_analyzer to avoid duplication
            file_ext = file_path.suffix.lower()
            is_code_file = file_ext in LANGUAGE_BY_EXTENSION
            disable_truncation = is_code_file

            if disable_truncation:
                self.logger.info(f"Code file detected ({file_ext}), disabling truncation for embeddings")

            # Generate embeddings in batch
            embedding_service = await get_embedding_service()
            all_embeddings = []

            # Procesar embeddings en batches del tamaño óptimo para GPU
            for i in range(0, len(texts_for_embedding), EMBED_BATCH_SIZE):
                batch_texts = texts_for_embedding[i:i + EMBED_BATCH_SIZE]

                batch_embeddings = await embedding_service.generate_embeddings_batch(
                    batch_texts,
                    model=embedding_model_name,
                    disable_truncation=disable_truncation
                )

                all_embeddings.extend(batch_embeddings)

                # Update progress
                if self.redis_client:
                    progress = int((len(all_embeddings) / total_chunks) * 50)  # 0-50%
                    self._set_progress(file_id, progress)  # BUG #7 FIX

                self.logger.info(
                    f"Embedded batch {i // EMBED_BATCH_SIZE + 1}/{(total_chunks + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE}"
                )

            # =====================================================================
            # 7. BUILD POINTS IN PARALLEL (OPTIMIZACIÓN CLAVE #3)
            # =====================================================================
            self.logger.info(f"Building points with concurrency {CONCURRENCY_LIMIT}")

            semaphore = Semaphore(CONCURRENCY_LIMIT)

            async def build_point(idx: int, data: dict, embedding: List[float]) -> Optional[
                PointStruct]:
                """Build a single point (fast, mainly CPU work)"""
                async with semaphore:
                    try:
                        chunk = data["chunk"]
                        context = contexts[idx]

                        # Build payload with smart metadata
                        display_filename = getattr(doc, 'original_filename', None) or doc.file_name

                        # Fix: Use global chunk index from data['section_index'] instead of local chunk['chunk_index']
                        # ensuring sequentiality across the whole file (0, 1, 2...)
                        global_index = data.get('section_index', 0)

                        payload = {
                            'file': display_filename,
                            'file_id': str(file_record.id),
                            'conversation_id': str(
                                file_record.conversation_id) if file_record.conversation_id else None,
                            'section': chunk['title'],
                            'sectionLevel': chunk.get('section_level', 0),
                            'content': chunk['content'],
                            'context': context if context else None,
                            'parent_id': chunk.get('parent_id'),
                            'parent_title': chunk.get('parent_title'),
                            'chunk_index': global_index,
                            'total_chunks': total_chunks,  # Use the total calculated before parallel processing
                            'indexed_at': get_current_utc_iso(),
                        }

                        # === AGREGAR METADATA OBSIDIAN AL PAYLOAD ===
                        if graph:
                            note_name = file_path.stem
                            if note_name in graph:
                                note_meta = graph[note_name]["metadata"]

                                # Metadata filtrada para Qdrant
                                payload['obsidian_outgoing_links'] = note_meta["outgoing"][
                                    :10]  # Top 10
                                payload['obsidian_incoming_links'] = note_meta["incoming"][
                                    :10]  # Top 10
                                payload['obsidian_link_count'] = note_meta["link_count"]
                                payload['obsidian_is_hub'] = note_meta["is_hub"]
                                payload['obsidian_is_index'] = note_meta["is_index"]
                                payload['obsidian_note_type'] = note_meta["note_type"]
                                payload['obsidian_tags'] = note_meta["tags"][:5]  # Top 5 tags

                        # Add filtered metadata
                        doc_type = MetadataPolicy.detect_type(file_path)
                        filtered_doc_metadata = MetadataPolicy.filter_and_truncate(
                            doc.metadata or {}, doc_type, file_path
                        )
                        filtered_chunk_metadata = MetadataPolicy.filter_and_truncate(
                            chunk.get('metadata') or {}, doc_type, file_path
                        )

                        payload.update(filtered_doc_metadata)
                        payload.update(filtered_chunk_metadata)

                        return PointStruct(
                            id=data["point_id"],
                            vector=embedding,
                            payload=payload
                        )
                    except Exception as e:
                        self.logger.error(f"Error building point {idx}: {e}")
                        return None

            # Build all points in parallel
            point_tasks = [
                build_point(idx, data, embedding)
                for idx, (data, embedding) in enumerate(zip(chunk_tasks_data, all_embeddings))
            ]
            points_results = await asyncio.gather(*point_tasks)
            # Filter out None results (failures)
            valid_points = [p for p in points_results if p is not None]

            self.logger.info(
                f"Parallel processing finished. {len(valid_points)}/{len(chunk_tasks_data)} chunks successfully processed.")


            # =====================================================================
            # 8. UPSERT TO QDRANT WITH DYNAMIC BATCHER (CRÍTICO)
            # =====================================================================
            self.logger.info("Upserting to Qdrant with DynamicQdrantBatcher")

            # Inicializar el batcher inteligente
            batcher = DynamicQdrantBatcher(logger=self.logger)

            # Alimentar puntos al batcher
            for point in valid_points:
                # El batcher decide cuándo enviar
                batch_to_flush = batcher.add(point)

                if batch_to_flush:
                    # Enviar batch que excedió límites
                    await self.qdrant.upsert(
                        collection_name=collection_name,
                        points=batch_to_flush,
                        wait=False  # No-blocking
                    )
                    self.logger.info(f"Upserted intermediate batch of size {len(batch_to_flush)}")

            # Update progress
            processed_chunks = len(valid_points)
            total_chunks = len(chunk_tasks_data)

            if self.redis_client and total_chunks > 0:
                processed = len([p for p in valid_points if
                                 p == point or valid_points.index(p) < valid_points.index(point)])
                progress = 50 + int((processed / len(valid_points)) * 50)  # 50-100%
                # Use a specific key for processing progress
                self._set_progress(file_id, progress)  # BUG #7 FIX
                # Also ensure the main file status key reflects 'processing' or 100 if we want to track it there
                # But for now, let's keep it separate to avoid overwriting upload progress if they are same key

                # =====================================================================
                # FLUSH REMAINING POINTS
                # =====================================================================
                final_batch = batcher.flush()
                if final_batch:
                    await self.qdrant.upsert(
                        collection_name=collection_name,
                        points=final_batch,
                        wait=True  # Wait for final batch
                    )
                    self.logger.info(
                        f"Upserted final batch",
                        extra={
                            "points": len(final_batch),
                            "collection": collection_name
                        }
                    )

            # Get batching stats
            stats = batcher.get_stats()
            self.logger.info(
                "Dynamic batching completed",
                extra={
                    "total_batches": stats["total_batches"],
                    "total_points": len(point_ids)
                }
            )

            # =====================================================================
            # BUILD BM25 INDEX (PRESERVED)
            # =====================================================================
            try:
                self.logger.info("Building BM25 index")

                # Collect all points for BM25
                # Note: We just inserted them, so we scroll from Qdrant to be sure
                # Or since we have valid_points, we could build it directly from memory?
                # Building from memory saves a scroll call.

                bm25_documents = []
                for point in valid_points:
                    bm25_documents.append({
                        "content": point.payload.get("content", ""),
                        "file": point.payload.get("file", ""),
                        "section": point.payload.get("section", ""),
                        "collection": collection_name,
                        "metadata": {
                            k: v for k, v in point.payload.items()
                            if k not in ["content", "file", "section"]
                        }
                    })

                # Create BM25 index
                bm25_index = BM25Index(
                    collection_name=collection_name,
                    k1=settings.BM25_K1,
                    b=settings.BM25_B
                )
                bm25_index.build_index(bm25_documents)

                # Save to disk
                bm25_index.save(directory=str(settings.BM25_INDEX_DIR))

                self.logger.info("BM25 index built successfully from memory points")

            except Exception as bm25_error:
                self.logger.error(f"BM25 index failed: {bm25_error}")

            # =====================================================================
            # UPDATE FILE RECORD (PRESERVED)
            # =====================================================================
            file_record.processed = True
            file_record.processing_status = ProcessingStatus.COMPLETED

            metadata = getattr(file_record, 'extra_metadata', {}) or {}
            metadata.update({
                'extracted_text': doc.content if hasattr(doc, 'content') else "",
                'sections_count': len(doc.sections),
                'chunks_count': len(point_ids),
                'collection_name': collection_name,
                'point_ids': point_ids,
                'batching_stats': stats,
                'optimization_stats': {
                    'context_batch_size': CONTEXT_BATCH_SIZE,
                    'embed_batch_size': EMBED_BATCH_SIZE,
                    'concurrency_limit': CONCURRENCY_LIMIT
                }
            })

            # === AGREGAR METADATA OBSIDIAN AL FILE RECORD ===
            if obsidian_context and obsidian_context.is_obsidian:
                metadata['obsidian'] = {
                    'type': obsidian_context.type,
                    'vault_root': str(obsidian_context.vault_root),
                    'vault_config': obsidian_context.vault_config
                }

                if graph:
                    note_name = file_path.stem
                    if note_name in graph:
                        note_meta = graph[note_name]["metadata"]
                        metadata['obsidian']['note_metadata'] = {
                            'outgoing_links': note_meta["outgoing"],
                            'incoming_links': note_meta["incoming"],
                            'link_count': note_meta["link_count"],
                            'is_hub': note_meta["is_hub"],
                            'is_index': note_meta["is_index"],
                            'note_type': note_meta["note_type"],
                            'tags': note_meta["tags"]
                        }

            # AUTO-ANALYZE CODE FILES (PRESERVED)
            if self._is_code_file(file_path):
                self.logger.info("Running code analysis")
                code_analysis = await self._analyze_code_file(file_path)
                if code_analysis:
                    metadata['code_analysis'] = code_analysis

            # Update file metadata and status via repository
            await self.file_repo.update_processing_status(
                file_id,
                ProcessingStatus.COMPLETED,
                processed=True,
                extra_metadata=metadata
            )

            self.logger.info(
                "Processing completed",
                extra={
                    "file_id": str(file_record.id),
                "chunks": len(point_ids),
                "collection": collection_name
            }
        )

            # Final progress update
            if self.redis_client:
                self._set_progress(file_id, 100)  # BUG #7 FIX

            return {
                'file_id': str(file_record.id),
                'file_name': file_record.file_name,
                'sections': len(doc.sections),
                'chunks': len(point_ids),
                'collection': collection_name,
                'status': 'completed'
            }

        except Exception as e:
            # Update status to failed via repository
            await self.file_repo.update_processing_status(
                file_id,
                ProcessingStatus.ERROR,
                extra_metadata={'error': str(e)}
            )

            self.logger.error(f"Processing failed: {e}", exc_info=True)
            raise FileProcessingError(f"Failed to process file: {str(e)}")

    # =========================================================================
    # NUEVO: MÉTODOS PARA MANEJO DE OBSIDIAN
    # =========================================================================

    async def _get_or_build_vault_graph(self, context: ObsidianContext) -> Dict:
        """
        Get or build vault graph with LRU cache, TTL, and race condition protection.

        BUG FIXES:
        - #1: LRU eviction when cache full (max 50 vaults)
        - #1: TTL expiration (1 hour)
        - #3: Async locks to prevent duplicate graph building
        - #5: Single definition (eliminates duplicate)
        """
        vault_key = str(context.vault_root)

        # === BUG #3 FIX: Get or create lock for this vault ===
        async with self._cache_lock:
            if vault_key not in self._vault_locks:
                self._vault_locks[vault_key] = asyncio.Lock()
            vault_lock = self._vault_locks[vault_key]

        # === BUG #3 FIX: Use vault-specific lock (prevents race condition) ===
        async with vault_lock:
            now = get_current_utc()

            # === BUG #1 FIX: Check TTL expiration ===
            if vault_key in self._vault_cache_timestamps:
                last_access = self._vault_cache_timestamps[vault_key]
                if now - last_access > self._vault_cache_ttl:
                    # TTL expired, remove from cache
                    self._vault_graphs_cache.pop(vault_key, None)
                    self._vault_cache_timestamps.pop(vault_key, None)
                    self.logger.debug(f"Vault cache TTL expired: {vault_key}")

            # === BUG #3 FIX: Double-check pattern (after acquiring lock) ===
            if vault_key in self._vault_graphs_cache:
                # Update access time and move to end (LRU)
                self._vault_cache_timestamps[vault_key] = now
                self._vault_graphs_cache.move_to_end(vault_key)
                self.logger.debug(f"Vault graph cache hit: {vault_key}")
                return self._vault_graphs_cache[vault_key]

            # === BUG #1 FIX: LRU eviction if cache is full ===
            if len(self._vault_graphs_cache) >= self._vault_cache_max_size:
                # Remove oldest (first item in OrderedDict)
                oldest_key = next(iter(self._vault_graphs_cache))
                self._vault_graphs_cache.pop(oldest_key)
                self._vault_cache_timestamps.pop(oldest_key, None)
                self.logger.info(
                    f"Vault cache eviction (LRU): {oldest_key}",
                    extra={
                        "cache_size": len(self._vault_graphs_cache),
                        "max_size": self._vault_cache_max_size
                    }
                )

            # Build graph (only one coroutine reaches here per vault)
            self.logger.info(f"Building vault graph: {vault_key}")
            builder = ObsidianGraphBuilder()

            # Process specific files or entire vault
            if context.type in ("single_file", "subset"):
                await builder.scan_files(context.files, context.vault_root)
            else:
                await builder.scan_vault(context.vault_root)

            graph = builder.build_bidirectional_graph()

            # Store in cache with timestamp
            self._vault_graphs_cache[vault_key] = graph
            self._vault_cache_timestamps[vault_key] = now

            self.logger.info(
                f"Vault graph cached: {vault_key}",
                extra={
                    "cache_size": len(self._vault_graphs_cache),
                    "graph_nodes": len(graph)
                }
            )

            return graph

    async def get_file_content_by_id(self, file_id: UUID) -> Optional[Dict[str, Any]]:
        """
        🆕 Obtiene contenido de archivo adjunto por UUID para análisis de CodebaseTool.
        Returns: {"filename": str, "content": str} o None
        """
        try:
            # 🆕 DEBUG: Verificar tipo de sesión
            self.logger.info(f"File id: {file_id}")
            filerecord = await self.file_repo.get_by_id(file_id)

            if not filerecord:
                self.logger.warning(f"File {file_id} not found")
                return None

            # Prioridad 1: extracted_text del metadata
            if filerecord.extra_metadata and filerecord.extra_metadata.get("extracted_text"):
                content = filerecord.extra_metadata["extracted_text"]
            # Prioridad 2: leer desde storage_path
            elif filerecord.storage_path:
                try:
                    import aiofiles
                    async with aiofiles.open(filerecord.storage_path, mode='r', encoding='utf-8', errors='ignore') as f:
                        content = await f.read()
                except ImportError:
                    # Fallback sync
                    with open(filerecord.storage_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
            else:
                self.logger.warning(f"No content available for file {file_id}")
                return None

            filename = filerecord.extra_metadata.get("original_name", filerecord.file_name) if filerecord.extra_metadata else filerecord.file_name

            self.logger.info(f"Retrieved content for analysis: {filename} ({len(content)} chars)")
            return {
                "file_id": str(file_id),
                "file_name": filename,
                "content": content,
                "file_type": filerecord.file_type
            }

        except Exception as e:
            self.logger.error(f"Error getting file content {file_id}: {e}", exc_info=True)
            return None
