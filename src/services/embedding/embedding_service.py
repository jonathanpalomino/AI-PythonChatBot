# =============================================================================
# src/services/embedding_service.py
# Embedding Service - Unified embedding generation (REFACTORED)
# =============================================================================
"""
Service for generating embeddings using Ollama.

CHANGES:
- ✅ Singleton pattern con lazy loading
- ✅ Eliminado self.db del constructor
- ✅ Eliminado código zombie de _dimension_cache
- ✅ @cached decorator funcional con cache_manager
- ✅ Una sola instancia de AsyncClient compartida
"""

import asyncio
from typing import List, Optional, Tuple

from ollama import AsyncClient, Client as OllamaClientSync

from src.config.settings import settings
from src.utils.cache import cache_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Service for generating embeddings using Ollama (Singleton)"""

    # Class constants
    MAX_CHARS = 2000
    MAX_CHARS_RETRY = 1000
    MAX_CHARS_AGGRESSIVE = 500
    MAX_CHARS_CODE = 8000  # Higher limit for source code
    BATCH_SIZE = 5
    CACHE_NAMESPACE = 'embedding_dimensions'
    STRICT_SPLIT_CHARS = 1000  # ~250 tokens, used for forced splitting when context length is exceeded

    # Context length limits by model (in tokens, approximate chars)
    # Source: Ollama model documentation
    MODEL_CONTEXT_LIMITS = {
        'nomic-embed-text': {'tokens': 8192, 'chars_estimate': 30000},
        'mxbai-embed-large': {'tokens': 32768, 'chars_estimate': 120000},
        'all-minilm': {'tokens': 8192, 'chars_estimate': 30000},
        'minilm': {'tokens': 8192, 'chars_estimate': 30000},
        'e5-large-v2': {'tokens': 512, 'chars_estimate': 2000},  # Very limited!
        'bge-large': {'tokens': 8192, 'chars_estimate': 30000},
        'bge-base': {'tokens': 512, 'chars_estimate': 2000},
        'gte-large': {'tokens': 8192, 'chars_estimate': 30000},
        'gte-base': {'tokens': 512, 'chars_estimate': 2000},
        'snowflake-arctic-embed': {'tokens': 8192, 'chars_estimate': 30000},
        'llama2': {'tokens': 4096, 'chars_estimate': 15000},
        'mistral': {'tokens': 8192, 'chars_estimate': 30000},
    }

    # Default safe limit for unknown models
    DEFAULT_CONTEXT_LIMIT = 8000  # ~8K tokens safe estimate

    def __init__(self, num_gpu: Optional[int] = None):
        """
        Constructor sin DB dependency.
        DB se pasa como parámetro en métodos que la necesitan.
        """
        self.logger = logger
        self.ollama = AsyncClient(host=settings.OLLAMA_BASE_URL)
        self.ollama_sync = OllamaClientSync(host=settings.OLLAMA_BASE_URL)
        self.num_gpu = num_gpu if num_gpu is not None else getattr(settings, 'OLLAMA_NUM_GPU', 1)

        # ✅ Crear namespace de cache para dimensiones
        cache_manager.create_cache(
            namespace=self.CACHE_NAMESPACE,
            max_size=100,
            ttl=3600  # 1 hora
        )

    def get_model_context_limit(self, model: str) -> int:
        """
        Get the context limit (in characters) for a specific embedding model.

        Args:
            model: Model name (e.g., 'nomic-embed-text', 'mxbai-embed-large')

        Returns:
            Maximum characters supported by the model
        """
        # Normalize model name (remove :latest, v1, etc.)
        normalized_model = model.lower().split(':')[0]

        # Check exact match first
        if normalized_model in self.MODEL_CONTEXT_LIMITS:
            return self.MODEL_CONTEXT_LIMITS[normalized_model]['chars_estimate']

        # Check partial match
        for known_model, limit in self.MODEL_CONTEXT_LIMITS.items():
            if known_model in normalized_model or normalized_model in known_model:
                return limit['chars_estimate']

        # Use default for unknown models
        self.logger.warning(
            f"Unknown embedding model '{model}', using default context limit",
            extra={"model": model, "default_limit": self.DEFAULT_CONTEXT_LIMIT}
        )
        return self.DEFAULT_CONTEXT_LIMIT

    def _is_context_length_error(self, error: Exception) -> bool:
        """Check if error is a context length error"""
        return "exceeds the context length" in str(error)

    def _truncate_text(self, text: str, max_chars: int = None) -> Tuple[str, int]:
        """
        Centralized text truncation logic.

        Args:
            text: Text to truncate
            max_chars: Maximum characters (defaults to self.MAX_CHARS)

        Returns:
            Tuple of (truncated_text, original_length)
        """
        max_chars = max_chars or self.MAX_CHARS
        original_length = len(text)

        if original_length > max_chars:
            truncated_text = text[:max_chars] + "..."
            self.logger.warning(
                "Text truncated for embedding",
                extra={
                    "max_chars": max_chars,
                    "original_length": original_length,
                    "truncated_length": len(truncated_text)
                }
            )
            return truncated_text, original_length

        return text, original_length

    def _split_for_embedding(
        self,
        text: str,
        max_chars: int = None,
        max_splits: int = 5,
        model: str = None
    ) -> List[str]:
        """
        Split text into chunks that fit within embedding model's context limit.

        This preserves all content by splitting rather than truncating.
        Uses smart splitting at natural boundaries (newlines, paragraphs).

        Args:
            text: Text to split
            max_chars: Maximum characters per chunk (auto-detected from model if not provided)
            max_splits: Maximum number of splits to avoid infinite recursion
            model: Embedding model name (used to auto-detect limit)

        Returns:
            List of text chunks, each within max_chars limit
        """
        # Auto-detect model context limit if not provided
        if max_chars is None and model:
            max_chars = self.get_model_context_limit(model)
        max_chars = max_chars or self.DEFAULT_CONTEXT_LIMIT  # Fallback

        if len(text) <= max_chars:
            return [text]

        if max_splits <= 0:
            # Force truncate if we've split too many times
            return [text[:max_chars] + "..."]

        # Try to split at natural boundaries
        # Priority: double newlines, single newlines, sentences, words

        # Try splitting by paragraphs (double newlines)
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            result = []
            current_chunk = ""

            for para in paragraphs:
                if len(current_chunk) + len(para) + 2 <= max_chars:
                    current_chunk += ("\n\n" if current_chunk else "") + para
                else:
                    if current_chunk:
                        result.append(current_chunk)
                    # Start new chunk, potentially split paragraph if too long
                    if len(para) > max_chars:
                        result.extend(self._split_for_embedding(
                            para, max_chars, max_splits - 1, model
                        ))
                    else:
                        current_chunk = para

            if current_chunk:
                result.append(current_chunk)

            if all(len(chunk) <= max_chars for chunk in result):
                return result

        # Try splitting by single newlines
        lines = text.split('\n')
        if len(lines) > 1:
            result = []
            current_chunk = ""

            for line in lines:
                if len(current_chunk) + len(line) + 1 <= max_chars:
                    current_chunk += ("\n" if current_chunk else "") + line
                else:
                    if current_chunk:
                        result.append(current_chunk)
                    if len(line) > max_chars:
                        result.extend(self._split_for_embedding(
                            line, max_chars, max_splits - 1, model
                        ))
                    else:
                        current_chunk = line

            if current_chunk:
                result.append(current_chunk)

            if all(len(chunk) <= max_chars for chunk in result):
                return result

        # Last resort: split by characters with overlap consideration
        chunk_count = (len(text) // max_chars) + 1
        chunk_size = len(text) // chunk_count

        result = []
        for i in range(chunk_count):
            start = i * chunk_size
            end = start + max_chars
            if end > len(text):
                end = len(text)
            chunk = text[start:end]

            # Try to end at a word boundary
            if end < len(text) and chunk[-1] not in [' ', '\n', '.', ',', ';', ')']:
                last_space = chunk.rfind(' ')
                if last_space > max_chars * 0.5:  # Only split if we're past half the chunk
                    chunk = chunk[:last_space]
                    end = start + last_space

            result.append(chunk)

        self.logger.info(
            f"Split text into {len(result)} chunks (max_chars={max_chars})",
            extra={"original_length": len(text), "chunks": len(result)}
        )

        return result

    async def generate_embedding(
        self,
        text: str,
        model: Optional[str] = None,
        disable_truncation: bool = False
    ) -> List[float]:
        """
        Generate embedding using Ollama with handling for long texts.

        Args:
            text: Text to embed
            model: Embedding model to use
            disable_truncation: If True, skip truncation (use for code/source files)
        """
        # Input validation
        if not text or not isinstance(text, str):
            raise ValueError(f"Text must be a non-empty string, got: {type(text)}")

        original_length = len(text)

        # Smart truncation - skip for code if disabled
        if disable_truncation:
            # Don't truncate for source code - use model's full context
            pass
        elif original_length > 5000:
            text, _ = self._truncate_text(text, self.MAX_CHARS_AGGRESSIVE)
        else:
            text, _ = self._truncate_text(text, self.MAX_CHARS)

        # Get embedding model
        embedding_model = model
        if not embedding_model:
            embedding_model = model or settings.EMBEDDING_MODEL

        try:
            response = await self.ollama.embeddings(
                model=embedding_model,
                prompt=text,
                options={'num_gpu': self.num_gpu}
            )
            return response['embedding']

        except Exception as e:
            # Retry with shorter text if context length exceeded
            # This applies even when disable_truncation=True, as the model has limits
            if self._is_context_length_error(e):
                self.logger.warning(
                    f"Context length exceeded, retrying with truncated text (disable_truncation={disable_truncation})",
                    extra={"original_length": original_length, "retry_length": self.MAX_CHARS_RETRY}
                )
                text, _ = self._truncate_text(text, self.MAX_CHARS_RETRY)
                response = await self.ollama.embeddings(
                    model=embedding_model,
                    prompt=text,
                    options={'num_gpu': self.num_gpu}
                )
                return response['embedding']
            else:
                raise

    async def generate_embeddings_batch(
        self,
        texts: List[str],
        model: Optional[str] = None,
        disable_truncation: bool = False
    ) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts using Ollama with controlled concurrency.
        Proper error handling with return_exceptions=True to avoid fail-all-if-one-fails.

        Args:
            texts: List of texts to embed
            model: Embedding model to use
            disable_truncation: If True, skip truncation (use for code/source files)
        """
        # Input validation
        if not texts:
            self.logger.warning("Empty texts list provided to generate_embeddings_batch")
            return []

        if not isinstance(texts, list):
            raise TypeError(f"Expected list of strings, got {type(texts)}")

        # Truncate texts if too long (skip for code)
        truncated_texts = []
        for text in texts:
            if not isinstance(text, str):
                self.logger.warning(f"Non-string item in batch: {type(text)}, converting to string")
                text = str(text)

            if disable_truncation:
                truncated_texts.append(text)
            else:
                truncated_text, _ = self._truncate_text(text, self.MAX_CHARS)
                truncated_texts.append(truncated_text)

        # Get embedding model
        embedding_model = model
        if not embedding_model:
            embedding_model = settings.EMBEDDING_MODEL

        try:
            all_embeddings = []

            # Process texts in smaller batches to control concurrency
            for i in range(0, len(truncated_texts), self.BATCH_SIZE):
                batch_texts = truncated_texts[i:i + self.BATCH_SIZE]
                self.logger.debug(
                    f"Processing embedding batch {i // self.BATCH_SIZE + 1} with {len(batch_texts)} texts"
                )

                # Generate embeddings with return_exceptions=True
                responses = await asyncio.gather(*[
                    self.ollama.embeddings(
                        model=embedding_model,
                        prompt=text,
                        options={'num_gpu': self.num_gpu}
                    )
                    for text in batch_texts
                ], return_exceptions=True)

                # Process responses and handle errors individually
                for idx, response in enumerate(responses):
                    global_idx = i + idx
                    original_text = truncated_texts[global_idx]

                    if isinstance(response, Exception):
                        # Check if it's a context length error and split into sub-chunks
                        if self._is_context_length_error(response):
                            self.logger.warning(
                                f"Context length exceeded for text at index {global_idx}, splitting into sub-chunks",
                                extra={
                                    "text_preview": batch_texts[idx][:100],
                                    "original_length": len(original_text)
                                }
                            )

                            # Split the text into sub-chunks using STRICT character limit
                            # We force split because the model already rejected the text
                            # Use a strict limit to ensure it fits (1000 chars = ~250 tokens)

                            sub_chunks = self._split_for_embedding(
                                original_text,
                                max_chars=self.STRICT_SPLIT_CHARS,
                                max_splits=10,
                                model=None  # Force strict split, don't use model limits
                            )

                            if len(sub_chunks) == 1:
                                # Force split by characters
                                self.logger.warning(
                                    f"Could not split text, forcing character-based split"
                                )
                                # Simple forced split into multiple chunks
                                chunk_count = (len(original_text) // self.STRICT_SPLIT_CHARS) + 1
                                sub_chunks = []
                                for i in range(chunk_count):
                                    start = i * self.STRICT_SPLIT_CHARS
                                    end = min(start + self.STRICT_SPLIT_CHARS, len(original_text))
                                    sub_chunks.append(original_text[start:end])

                            # Generate embeddings for each sub-chunk
                            sub_embeddings = []
                            for sub_chunk in sub_chunks:
                                try:
                                    sub_response = await self.ollama.embeddings(
                                        model=embedding_model,
                                        prompt=sub_chunk,
                                        options={'num_gpu': self.num_gpu}
                                    )
                                    sub_embeddings.append(sub_response['embedding'])
                                except Exception as sub_error:
                                    self.logger.error(
                                        f"Failed to embed sub-chunk: {sub_error}"
                                    )

                            if sub_embeddings:
                                # Average all sub-chunk embeddings
                                dimension = len(sub_embeddings[0])
                                averaged = [
                                    sum(emb[d] for emb in sub_embeddings) / len(sub_embeddings)
                                    for d in range(dimension)
                                ]
                                all_embeddings.append(averaged)
                                self.logger.info(
                                    f"Successfully embedded {len(sub_chunks)} sub-chunks "
                                    f"(averaged) for text at index {global_idx}"
                                )
                            else:
                                # Fall back to zero embedding if all sub-chunks failed
                                raise response  # Re-raise to trigger outer error handling

                            continue  # Skip to next item

                        # For other errors, log and use zero embedding as last resort
                        self.logger.error(
                            f"Failed to generate embedding for text at index {global_idx}",
                            extra={
                                "error": str(response),
                                "text_preview": batch_texts[idx][:100],
                                "batch_index": i // self.BATCH_SIZE + 1
                            }
                        )

                        # Try to get dimension for zero embedding
                        try:
                            dimension = await self.get_embedding_dimension(
                                model=embedding_model
                            )
                            zero_embedding = [0.0] * dimension
                            all_embeddings.append(zero_embedding)
                            self.logger.info(
                                f"Using zero embedding for failed text at index {global_idx}"
                            )
                        except:
                            # If we can't get dimension, use a common default
                            all_embeddings.append([0.0] * 768)
                            self.logger.warning(
                                f"Using default 768-dim zero embedding for index {global_idx}"
                            )
                    else:
                        all_embeddings.append(response['embedding'])

            return all_embeddings

        except Exception as e:
            self.logger.error(f"Error generating batch embeddings: {e}")
            raise

    def generate_embedding_sync(
        self,
        text: str,
        model: Optional[str] = None,
        disable_truncation: bool = False
    ) -> List[float]:
        """
        Generate embedding synchronously using Ollama with handling for long texts.

        Args:
            text: Text to embed
            model: Embedding model to use
            disable_truncation: If True, skip truncation (use for code/source files)
        """
        # Input validation
        if not text or not isinstance(text, str):
            raise ValueError(f"Text must be a non-empty string, got: {type(text)}")

        embedding_model = model or settings.EMBEDDING_MODEL

        # Use centralized truncation (skip for code if disabled)
        if disable_truncation:
            original_length = len(text)
        else:
            text, original_length = self._truncate_text(text, self.MAX_CHARS)

        try:
            response = self.ollama_sync.embeddings(
                model=embedding_model,
                prompt=text,
                options={'num_gpu': self.num_gpu}
            )
            return response['embedding']

        except Exception as e:
            # Retry with splitting if context length exceeded
            # This applies even when disable_truncation=True, as the model has limits
            if self._is_context_length_error(e):
                self.logger.warning(
                    f"Context length exceeded, splitting text into sub-chunks (disable_truncation={disable_truncation})",
                    extra={"original_length": original_length}
                )

                # Split the text into sub-chunks using STRICT character limit
                # We force split because the model already rejected the text
                # Use a strict limit to ensure it fits (1000 chars = ~250 tokens)

                sub_chunks = self._split_for_embedding(
                    text,
                    max_chars=self.STRICT_SPLIT_CHARS,
                    max_splits=10,
                    model=None  # Force strict split, don't use model limits
                )

                if len(sub_chunks) == 1:
                    # Force split by characters
                    self.logger.warning(
                        f"Could not split text, forcing character-based split"
                    )
                    # Simple forced split into multiple chunks
                    chunk_count = (len(text) // self.STRICT_SPLIT_CHARS) + 1
                    sub_chunks = []
                    for i in range(chunk_count):
                        start = i * self.STRICT_SPLIT_CHARS
                        end = min(start + self.STRICT_SPLIT_CHARS, len(text))
                        sub_chunks.append(text[start:end])

                # Generate embeddings for each sub-chunk and average
                sub_embeddings = []
                for sub_chunk in sub_chunks:
                    sub_response = self.ollama_sync.embeddings(
                        model=embedding_model,
                        prompt=sub_chunk,
                        options={'num_gpu': self.num_gpu}
                    )
                    sub_embeddings.append(sub_response['embedding'])

                # Average all sub-chunk embeddings
                dimension = len(sub_embeddings[0])
                averaged = [
                    sum(emb[d] for emb in sub_embeddings) / len(sub_embeddings)
                    for d in range(dimension)
                ]
                return averaged
            else:
                raise

    async def get_embedding_dimension(
        self,
        model: Optional[str] = None
    ) -> int:
        """
        Get the dimension (vector size) of the specified or default embedding model.
        Uses cache_manager for caching (thread-safe).
        """
        # Determine the model to use
        embedding_model = model
        if not embedding_model:
            embedding_model = model or settings.EMBEDDING_MODEL

        # ✅ FIX: Usar cache_manager en lugar de self._dimension_cache
        cache_key = f"dim_{embedding_model}"
        cached_dimension = cache_manager.get(self.CACHE_NAMESPACE, cache_key)

        if cached_dimension is not None:
            return cached_dimension

        try:
            self.logger.info(f"Detecting dimension for model: {embedding_model}")

            # Generate a small embedding to check dimension
            test_embedding = await self.generate_embedding("test",  model=embedding_model)
            dimension = len(test_embedding)

            # ✅ FIX: Store in cache_manager
            cache_manager.set(self.CACHE_NAMESPACE, cache_key, dimension)

            self.logger.info(f"Detected dimension for {embedding_model}: {dimension}")
            return dimension

        except Exception as e:
            self.logger.error(f"Failed to detect embedding dimension: {e}")

            # Fallback to settings or common default if detection fails
            fallback_dimension = getattr(settings, 'VECTOR_SIZE', 768)
            self.logger.warning(f"Using fallback dimension: {fallback_dimension}")
            return fallback_dimension


# =============================================================================
# ✅ SINGLETON FACTORY (Lazy Loading)
# =============================================================================
_embedding_service: Optional[EmbeddingService] = None
_embedding_service_lock = asyncio.Lock()


async def get_embedding_service(num_gpu: Optional[int] = None) -> EmbeddingService:
    """
    Lazy singleton factory para EmbeddingService.
    Thread-safe con asyncio.Lock + double-checked locking.

    Args:
        num_gpu: GPU count override (solo se usa en primera inicialización)

    Returns:
        La única instancia de EmbeddingService.
    """
    global _embedding_service

    # Fast path: already initialized
    if _embedding_service is not None:
        return _embedding_service

    # Slow path: need to initialize with lock
    async with _embedding_service_lock:
        # Double-check after acquiring lock
        if _embedding_service is None:
            _embedding_service = EmbeddingService(num_gpu=num_gpu)
            logger.info("EmbeddingService singleton initialized")

        return _embedding_service
