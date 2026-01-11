# =============================================================================
# src/services/embedding_service.py
# Embedding Service - Unified embedding generation
# =============================================================================
"""
Service for generating embeddings using Ollama.
This service unifies all embedding generation logic to avoid code duplication.
"""
from typing import List, Optional
import asyncio
from ollama import AsyncClient, Client as OllamaClientSync
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import settings
from src.services.model_service import model_service
from src.utils.logger import get_logger


class EmbeddingService:
    """Service for generating embeddings using Ollama"""

    def __init__(self, db: Optional[AsyncSession] = None):
        self.db = db
        self.logger = get_logger(__name__)
        self.ollama = AsyncClient(host=settings.OLLAMA_BASE_URL)
        self.ollama_sync = OllamaClientSync(host=settings.OLLAMA_BASE_URL)
        self._dimension_cache = {}

    async def generate_embedding(self, text: str, db: Optional[AsyncSession] = None, model: Optional[str] = None) -> List[float]:
        """Generate embedding using Ollama with handling for long texts"""
        MAX_CHARS = 2000

        # Truncate if too long
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS] + "..."
            self.logger.warning(
                "Text truncated for embedding",
                extra={"max_chars": MAX_CHARS, "original_length": len(text)}
            )

        # Get embedding model
        embedding_model = model
        if not embedding_model:
            if db:
                embedding_model = await model_service.get_embedding_model(db)
            else:
                embedding_model = settings.EMBEDDING_MODEL

        try:
            response = await self.ollama.embeddings(
                model=embedding_model,
                prompt=text,
                options={'num_gpu': 99}
            )
            return response['embedding']
        except Exception as e:
            # If still fails, try with even shorter text
            if "exceeds the context length" in str(e) and len(text) > 1000:
                self.logger.warning("Retrying embedding with shorter text")
                text = text[:1000] + "..."
                response = await self.ollama.embeddings(
                    model=embedding_model,
                    prompt=text,
                    options={'num_gpu': 99}
                )
                return response['embedding']
            else:
                raise

    async def generate_embeddings_batch(self, texts: List[str], db: Optional[AsyncSession] = None, model: Optional[str] = None) -> List[List[float]]:
        """Generate embeddings for a batch of texts using Ollama with controlled concurrency"""
        MAX_CHARS = 2000
        BATCH_SIZE = 5  # Process in batches of 5 concurrent requests to reduce RPC overhead

        # Truncate texts if too long
        truncated_texts = []
        for text in texts:
            if len(text) > MAX_CHARS:
                truncated_text = text[:MAX_CHARS] + "..."
                truncated_texts.append(truncated_text)
                self.logger.warning(
                    "Text truncated for embedding",
                    extra={"max_chars": MAX_CHARS, "original_length": len(text)}
                )
            else:
                truncated_texts.append(text)

        # Get embedding model
        embedding_model = model
        if not embedding_model:
            if db:
                embedding_model = await model_service.get_embedding_model(db)
            else:
                embedding_model = settings.EMBEDDING_MODEL

        try:
            embeddings = []
            # Process texts in smaller batches to control concurrency
            for i in range(0, len(truncated_texts), BATCH_SIZE):
                batch_texts = truncated_texts[i:i + BATCH_SIZE]
                self.logger.debug(f"Processing embedding batch {i//BATCH_SIZE + 1} with {len(batch_texts)} texts")

                # Generate embeddings for this sub-batch concurrently
                responses = await asyncio.gather(*[
                    self.ollama.embeddings(
                        model=embedding_model,
                        prompt=text,
                        options={'num_gpu': 99}
                    )
                    for text in batch_texts
                ])
                embeddings.extend([response['embedding'] for response in responses])

            return embeddings
        except Exception as e:
            self.logger.error(f"Error generating batch embeddings: {e}")
            raise

    def generate_embedding_sync(self, text: str, model: Optional[str] = None) -> List[float]:
        """Generate embedding synchronously using Ollama with handling for long texts"""
        MAX_CHARS = 2000
        
        embedding_model = model or settings.EMBEDDING_MODEL

        # Truncate if too long
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS] + "..."
            self.logger.warning(
                "Text truncated for embedding",
                extra={"max_chars": MAX_CHARS, "original_length": len(text)}
            )

        try:
            response = self.ollama_sync.embeddings(
                model=embedding_model,
                prompt=text,
                options={'num_gpu': 1}
            )
            return response['embedding']
        except Exception as e:
            # If still fails, try with even shorter text
            if "exceeds the context length" in str(e) and len(text) > 1000:
                self.logger.warning("Retrying embedding with shorter text")
                text = text[:1000] + "..."
                response = self.ollama_sync.embeddings(
                    model=embedding_model,
                    prompt=text,
                    options={'num_gpu': 1}
                )
                return response['embedding']
            else:
                raise

    async def get_embedding_dimension(self, model: Optional[str] = None, db: Optional[AsyncSession] = None) -> int:
        """
        Get the dimension (vector size) of the specified or default embedding model.
        Uses a local cache to avoid redundant calls to Ollama.
        """
        # Determine the model to use
        embedding_model = model
        if not embedding_model:
            if db:
                embedding_model = await model_service.get_embedding_model(db)
            else:
                embedding_model = settings.EMBEDDING_MODEL

        # Check cache
        if embedding_model in self._dimension_cache:
            return self._dimension_cache[embedding_model]

        try:
            self.logger.info(f"Detecting dimension for model: {embedding_model}")
            # Generate a small embedding to check dimension
            test_embedding = await self.generate_embedding("test", db=db, model=embedding_model)
            dimension = len(test_embedding)
            
            # Store in cache
            self._dimension_cache[embedding_model] = dimension
            self.logger.info(f"Detected dimension for {embedding_model}: {dimension}")
            
            return dimension
        except Exception as e:
            self.logger.error(f"Failed to detect embedding dimension: {e}")
            # Fallback to settings or common default if detection fails
            return settings.VECTOR_SIZE