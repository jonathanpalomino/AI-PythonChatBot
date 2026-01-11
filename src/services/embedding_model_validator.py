# =============================================================================
# src/services/embedding_model_validator.py
# Embedding Model Validation Service
# =============================================================================
"""
Service for validating and managing embedding model selection at conversation level.
Ensures that embedding models cannot be changed after initial selection to maintain
consistency with existing embeddings in the vector database.
"""
from typing import Optional, Dict, Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.models import Conversation, Message
from src.utils.logger import get_logger


class EmbeddingModelValidator:
    """Service for validating embedding model changes"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    async def validate_embedding_model_change(
        self, 
        db: AsyncSession, 
        conversation_id: UUID, 
        new_embedding_model: Optional[str],
        current_settings: Dict[str, Any]
    ) -> Optional[str]:
        """
        Validate if an embedding model change is allowed.
        
        Rules:
        1. If no embedding model is currently set, allow the change
        2. If embedding model is already set, prevent changes to maintain consistency
        3. If the new model is the same as current, allow it
        
        Args:
            db: Database session
            conversation_id: Conversation ID
            new_embedding_model: Proposed new embedding model
            current_settings: Current conversation settings
            
        Returns:
            None if change is allowed, error message if not allowed
        """
        current_embedding_model = current_settings.get('embedding_model')
        
        # If no current embedding model is set, allow the change
        if not current_embedding_model:
            self.logger.info(
                f"No current embedding model set for conversation {conversation_id}, allowing change to {new_embedding_model}"
            )
            return None
        
        # If the new model is the same as current, allow it
        if new_embedding_model == current_embedding_model:
            self.logger.debug(
                f"Embedding model unchanged for conversation {conversation_id}, allowing: {new_embedding_model}"
            )
            return None
        
        # If we have an existing embedding model and it's different, prevent the change
        if current_embedding_model and new_embedding_model and new_embedding_model != current_embedding_model:
            self.logger.warning(
                f"Attempt to change embedding model for conversation {conversation_id} from {current_embedding_model} to {new_embedding_model} - BLOCKED"
            )
            return (f"Cannot change embedding model for this conversation. "
                   f"Current model '{current_embedding_model}' is already in use and "
                   f"changing it would make existing embeddings incompatible.")
        
        return None
    
    async def has_existing_embeddings(
        self, 
        db: AsyncSession, 
        conversation_id: UUID
    ) -> bool:
        """
        Check if conversation has existing messages that used embeddings.
        
        Args:
            db: Database session
            conversation_id: Conversation ID
            
        Returns:
            True if conversation has messages that likely used embeddings
        """
        # Check if conversation has any messages
        result = await db.execute(
            select(Message).filter(
                Message.conversation_id == conversation_id
            ).limit(1)
        )
        
        existing_message = result.scalar_one_or_none()
        return existing_message is not None


# Singleton instance
embedding_model_validator = EmbeddingModelValidator()