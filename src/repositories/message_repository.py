# =============================================================================
# src/repositories/message_repository.py
# Message Repository
# =============================================================================
"""
Repository for Message entity operations.
"""

from typing import List
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.models import Message, MessageRole
from src.repositories.base_repository import BaseRepository


class MessageRepository(BaseRepository[Message]):
    """Repository for managing chat messages."""

    def __init__(self, db: AsyncSession):
        super().__init__(Message, db)

    async def get_conversation_messages(
        self,
        conversation_id: UUID,
        limit: int = 50,
        skip: int = 0,
        include_inactive: bool = False
    ) -> List[Message]:
        """
        Get messages for a conversation.

        Args:
            conversation_id: Conversation UUID
            limit: Max messages to return
            skip: Pagination offset
            include_inactive: Include deleted/inactive messages

        Returns:
            List of messages ordered by creation time
        """
        try:
            stmt = (
                select(Message)
                .where(Message.conversation_id == conversation_id)
            )

            if not include_inactive:
                stmt = stmt.where(Message.is_active == True)

            stmt = (
                stmt
                .order_by(Message.created_at.asc())
                .offset(skip)
                .limit(limit)
            )

            result = await self.db.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            self.logger.error(f"Error getting conversation messages: {e}")
            raise

    async def has_messages(
        self,
        conversation_id: UUID,
        include_inactive: bool = False
    ) -> bool:
        """
        Check if conversation has any messages.

        Args:
            conversation_id: Conversation UUID
            include_inactive: Include deleted/inactive messages

        Returns:
            True if conversation has messages, False otherwise
        """
        try:
            stmt = (
                select(Message)
                .where(Message.conversation_id == conversation_id)
            )

            if not include_inactive:
                stmt = stmt.where(Message.is_active == True)

            stmt = stmt.limit(1)

            result = await self.db.execute(stmt)
            return result.scalar_one_or_none() is not None
        except Exception as e:
            self.logger.error(f"Error checking if conversation has messages: {e}")
            raise

    async def get_last_n_messages(
        self,
        conversation_id: UUID,
        n: int = 10
    ) -> List[Message]:
        """
        Get last N messages from conversation.

        Args:
            conversation_id: Conversation UUID
            n: Number of messages

        Returns:
            List of last N messages (ordered oldest to newest)
        """
        try:
            stmt = (
                select(Message)
                .where(
                    Message.conversation_id == conversation_id,
                    Message.is_active == True
                )
                .order_by(Message.created_at.desc())
                .limit(n)
            )
            result = await self.db.execute(stmt)
            messages = result.scalars().all()
            # Reverse to get chronological order
            return list(reversed(messages))
        except Exception as e:
            self.logger.error(f"Error getting last N messages: {e}")
            raise

    async def create_message(
        self,
        conversation_id: UUID,
        role: MessageRole,
        content: str,
        **kwargs
    ) -> Message:
        """
        Create a new message.

        Args:
            conversation_id: Conversation UUID
            role: Message role (user/assistant/system)
            content: Message content
            **kwargs: Additional fields

        Returns:
            Created message
        """
        return await self.create(
            conversation_id=conversation_id,
            role=role,
            content=content,
            **kwargs
        )

    async def soft_delete(self, message_id: UUID) -> bool:
        """
        Soft delete a message (set is_active=False).

        Args:
            message_id: Message UUID

        Returns:
            True if deleted, False if not found
        """
        try:
            updated = await self.update(message_id, is_active=False)
            return updated is not None
        except Exception as e:
            self.logger.error(f"Error soft deleting message: {e}")
            raise
