# =============================================================================
# src/services/message_service.py
# Message Service - Business Logic
# =============================================================================
"""
Business logic for message operations

REFACTORED: Service now receives Repositories directly, not UnitOfWork.
This follows the Repository pattern correctly:
    Service → Repository → Session
"""
from typing import Optional
from uuid import UUID
from src.models.models import Message
from src.utils.logger import get_logger
from src.utils.transactional import transactional

logger = get_logger(__name__)


class MessageService:
    """
    Service for message business logic.
    
    REFACTORED: Receives Repositories directly, not UnitOfWork.
    This follows the Repository pattern correctly.
    """
    
    def __init__(self, message_repo):
        """
        Initialize MessageService with repository.
        
        Args:
            message_repo: MessageRepository instance
        """
        self.message_repo = message_repo

    async def commit(self):
        """Commit current transaction using repository's session."""
        await self.message_repo.commit()

    async def rollback(self):
        """Rollback current transaction using repository's session."""
        await self.message_repo.rollback()

    async def get_message(self, message_id: UUID) -> Optional[Message]:
        """Get message by ID (solo lectura)"""
        logger.debug(f"Retrieving message: {message_id}")
        return await self.message_repo.get_by_id(message_id)
    
    @transactional
    async def delete_message(self, message_id: UUID) -> bool:
        """
        Delete message
        @transactional usa commit/rollback automáticamente
        """
        logger.info(f"Deleting message: {message_id}")
        message = await self.message_repo.get_by_id(message_id)
        if not message:
            logger.warning(f"Message not found for deletion: {message_id}")
            return False
        
        await self.message_repo.delete(message_id)
        logger.info(f"Message deleted successfully: {message_id}")
        # @transactional hace commit automático aquí
        return True
