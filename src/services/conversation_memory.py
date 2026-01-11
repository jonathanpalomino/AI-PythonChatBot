# =============================================================================
# src/services/conversation_memory.py
# Conversation Memory Service - Semantic search over conversation history
# =============================================================================
"""
Service for managing conversation memory using semantic search.
Automatically indexes messages and retrieves relevant context from past conversation.
"""
from typing import List, Dict, Any, Optional
from uuid import UUID

from ollama import Client as OllamaClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import settings, get_qdrant_config
from src.models.models import Conversation, Message
from src.services.embedding_service import EmbeddingService
from src.services.model_service import model_service
from src.utils.logger import get_logger

# =============================================================================
# Global Client Singletons (for performance)
# =============================================================================
_qdrant_client = None
_ollama_client = None


def get_qdrant_client() -> QdrantClient:
    """Get or create singleton Qdrant client"""
    global _qdrant_client
    if _qdrant_client is None:
        qdrant_config = get_qdrant_config()
        _qdrant_client = QdrantClient(**qdrant_config)
    return _qdrant_client


def get_ollama_client() -> OllamaClient:
    """Get or create singleton Ollama client"""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient(host=settings.OLLAMA_BASE_URL)
    return _ollama_client


class ConversationMemoryService:
    """Manages semantic memory for conversations"""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.logger = get_logger(__name__)

        # Lazy loading
        self._qdrant = None
        self._ollama = None

    @property
    def qdrant(self):
        """Lazy-loaded Qdrant client"""
        if self._qdrant is None:
            self._qdrant = get_qdrant_client()
        return self._qdrant

    @property
    def ollama(self):
        """Lazy-loaded Ollama client"""
        if self._ollama is None:
            self._ollama = get_ollama_client()
        return self._ollama

    # =============================================================================
    # Main Public Methods
    # =============================================================================

    async def retrieve_relevant_context(
        self,
        conversation: Conversation,
        current_query: str,
        memory_config: dict
    ) -> Optional[str]:
        """
        Retrieve semantically relevant past messages

        Args:
            conversation: Conversation object
            current_query: Current user message to find relevant context for
            memory_config: Memory configuration dict

        Returns:
            Formatted context string or None
        """
        if not memory_config.get('semantic_enabled', False):
            return None

        self.logger.info(
            "Retrieving semantic memory",
            extra={"conversation_id": str(conversation.id), "query_length": len(current_query)}
        )

        try:
            # Ensure conversation messages are indexed
            collection_name = self._get_collection_name(conversation.id)
            await self._ensure_conversation_indexed(conversation, collection_name)

            # Perform semantic search
            search_k = memory_config.get('search_k', 5)
            score_threshold = memory_config.get('score_threshold', 0.3)

            messages = await self._semantic_search(
                collection_name=collection_name,
                query=current_query,
                k=search_k,
                score_threshold=score_threshold
            )

            if not messages:
                self.logger.info("No relevant past messages found")
                return None

            # Format context
            context = self._format_memory_context(messages)
            self.logger.info(
                "Memory context retrieved",
                extra={"messages_found": len(messages), "context_length": len(context)}
            )

            return context

        except Exception as e:
            self.logger.error(
                f"Memory retrieval failed: {e}",
                exc_info=True,
                extra={"conversation_id": str(conversation.id)}
            )
            return None

    async def index_message(
        self,
        conversation_id: UUID,
        message: Message
    ):
        """
        Index a new message for future retrieval

        Args:
            conversation_id: UUID of conversation
            message: Message object to index
        """
        self.logger.debug(f"Indexing message {message.id} for conversation {conversation_id}")

        try:
            collection_name = self._get_collection_name(conversation_id)

            # Ensure collection exists
            await self._ensure_collection_exists(collection_name)

            # Generate embedding
            embedding = await self._generate_embedding(message.content)

            # Create point
            point = PointStruct(
                id=str(message.id),
                vector=embedding,
                payload={
                    "message_id": str(message.id),
                    "conversation_id": str(conversation_id),
                    "role": message.role.value,
                    "content": message.content,
                    "timestamp": message.created_at.isoformat()
                }
            )

            # Upsert to Qdrant
            self.qdrant.upsert(
                collection_name=collection_name,
                points=[point]
            )

            self.logger.debug(f"Message {message.id} indexed successfully")

        except Exception as e:
            self.logger.error(
                f"Failed to index message: {e}",
                exc_info=True,
                extra={"message_id": str(message.id)}
            )

    # =============================================================================
    # Semantic Search
    # =============================================================================

    async def _semantic_search(
        self,
        collection_name: str,
        query: str,
        k: int,
        score_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Perform semantic search over conversation history"""
        self.logger.debug(
            f"Semantic search on {collection_name}",
            extra={"k": k, "threshold": score_threshold}
        )

        # Generate query embedding
        query_vector = await self._generate_embedding(query)

        # Search in Qdrant
        try:
            search_results = self.qdrant.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=k,
                score_threshold=score_threshold
            )
        except Exception as e:
            self.logger.warning(f"Qdrant search failed: {e}")
            return []

        # Format results
        messages = []
        for hit in search_results:
            messages.append({
                "role": hit.payload.get("role", "unknown"),
                "content": hit.payload.get("content", ""),
                "timestamp": hit.payload.get("timestamp", ""),
                "score": hit.score,
                "message_id": hit.payload.get("message_id", "")
            })

        self.logger.debug(f"Found {len(messages)} relevant messages")
        return messages

    # =============================================================================
    # Collection Management
    # =============================================================================

    def _get_collection_name(self, conversation_id: UUID) -> str:
        """Get collection name for conversation memory"""
        return f"conversation_memory_{str(conversation_id).replace('-', '_')}"

    async def _ensure_collection_exists(self, collection_name: str):
        """Ensure collection exists in Qdrant"""
        try:
            self.qdrant.get_collection(collection_name)
            return
        except:
            # Collection doesn't exist, create it
            self.logger.info(f"Creating memory collection: {collection_name}")
            
            # Dynamically determine vector size
            embedding_service = EmbeddingService(self.db)
            vector_size = await embedding_service.get_embedding_dimension(db=self.db)
            
            self.qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            self.logger.info(f"Memory collection created with dimension {vector_size}")

    async def _ensure_conversation_indexed(
        self,
        conversation: Conversation,
        collection_name: str
    ):
        """Ensure all conversation messages are indexed"""
        self.logger.debug("Ensuring conversation is indexed")

        # Ensure collection exists
        await self._ensure_collection_exists(collection_name)

        # Get all messages
        # ASYNC CHANGE: Use select and execute
        result = await self.db.execute(
            select(Message)
            .filter(
                Message.conversation_id == conversation.id,
                Message.is_active == True
            )
            .order_by(Message.created_at)
        )
        messages = result.scalars().all()

        if not messages:
            return

        # Get already indexed message IDs
        try:
            existing_points = self.qdrant.scroll(
                collection_name=collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=False
            )[0]
            indexed_message_ids = {
                point.payload.get("message_id") for point in existing_points
            }
        except:
            indexed_message_ids = set()

        # Index new messages
        points = []
        for msg in messages:
            msg_id_str = str(msg.id)

            # Skip if already indexed
            if msg_id_str in indexed_message_ids:
                continue

            try:
                embedding = await self._generate_embedding(msg.content)

                points.append(PointStruct(
                    id=msg_id_str,
                    vector=embedding,
                    payload={
                        "message_id": msg_id_str,
                        "conversation_id": str(conversation.id),
                        "role": msg.role.value,
                        "content": msg.content,
                        "timestamp": msg.created_at.isoformat()
                    }
                ))
            except Exception as e:
                self.logger.warning(f"Failed to generate embedding for message {msg.id}: {e}")
                continue

        # Upload points if any
        if points:
            self.logger.info(f"Indexing {len(points)} new messages")
            self.qdrant.upsert(
                collection_name=collection_name,
                points=points
            )

    # =============================================================================
    # Helper Methods
    # =============================================================================

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        embedding_service = EmbeddingService(self.db)
        return await embedding_service.generate_embedding(text, self.db)

    def _format_memory_context(self, messages: List[Dict[str, Any]]) -> str:
        """Format memory search results into context string"""
        if not messages:
            return ""

        context = "## Relevant Past Conversation Context\n\n"

        for i, msg in enumerate(messages, 1):
            role = msg['role'].upper()
            score = msg.get('score', 0.0)
            timestamp = msg.get('timestamp', '')
            content = msg['content']

            # Truncate very long messages for context
            if len(content) > 300:
                content = content[:297] + "..."

            context += f"[{i}] **{role}** (relevance: {score:.2f}, {timestamp}):\n"
            context += f"{content}\n\n"

        return context
