# =============================================================================
# src/repositories/conversation_repository.py
# Conversation Repository
# =============================================================================
"""
Repository for Conversation entity operations.
Handles all database access for conversations.
"""

from typing import List, Optional, Tuple
from uuid import UUID

from sqlalchemy import func, desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.models.models import Conversation, Message
from src.repositories.base_repository import BaseRepository


class ConversationRepository(BaseRepository[Conversation]):
    """Repository for managing conversations."""

    def __init__(self, db: AsyncSession):
        super().__init__(Conversation, db)

    async def get_by_id_with_messages(self, conversation_id: UUID) -> Optional[Conversation]:
        """
        Get conversation with eager-loaded messages.

        Args:
            conversation_id: Conversation UUID

        Returns:
            Conversation with messages loaded, or None
        """
        try:
            stmt = (
                select(Conversation)
                .where(Conversation.id == conversation_id)
                .options(selectinload(Conversation.messages))
            )
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            self.logger.error(f"Error getting conversation with messages: {e}")
            raise

    async def get_user_conversations(
        self,
        user_id: UUID,
        skip: int = 0,
        limit: int = 50
    ) -> List[Conversation]:
        """
        Get all conversations for a user.

        Args:
            user_id: User UUID
            skip: Pagination offset
            limit: Max results

        Returns:
            List of conversations
        """
        try:
            stmt = (
                select(Conversation)
                .where(Conversation.user_id == user_id)
                .order_by(Conversation.updated_at.desc())
                .offset(skip)
                .limit(limit)
            )
            result = await self.db.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            self.logger.error(f"Error getting user conversations: {e}")
            raise

    async def update_settings(
        self,
        conversation_id: UUID,
        settings: dict
    ) -> Optional[Conversation]:
        """
        Update conversation settings.

        Args:
            conversation_id: Conversation UUID
            settings: New settings dict

        Returns:
            Updated conversation or None
        """
        return await self.update(conversation_id, settings=settings)

    async def get_active_conversations(self, limit: int = 100) -> List[Conversation]:
        """
        Get recently active conversations.

        Args:
            limit: Max results

        Returns:
            List of active conversations
        """
        try:
            stmt = (
                select(Conversation)
                .where(Conversation.is_active == True)
                .order_by(Conversation.updated_at.desc())
                .limit(limit)
            )
            result = await self.db.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            self.logger.error(f"Error getting active conversations: {e}")
            raise

    async def get_with_details(self, conversation_id: UUID) -> Optional[Conversation]:
        """Get conversation with tool_configurations and prompt_template loaded."""
        try:
            stmt = (
                select(Conversation)
                .options(
                    selectinload(Conversation.tool_configurations),
                    selectinload(Conversation.prompt_template)
                )
                .where(Conversation.id == conversation_id)
            )
            result = await self.db.execute(stmt)
            return result.scalars().first()
        except Exception as e:
            self.logger.error(f"Error getting conversation {conversation_id} with details: {e}")
            raise

    async def list_optimized(self, skip: int = 0, limit: int = 50) -> List[dict]:
        """List conversations with message counts and last message timestamp (optimized)."""
        try:
            from src.models.models import Message

            message_count_subquery = (
                select(
                    Message.conversation_id,
                    func.count(Message.id).label('msg_count'),
                    func.max(Message.created_at).label('last_msg_at')
                )
                .group_by(Message.conversation_id)
                .subquery()
            )

            stmt = (
                select(
                    Conversation.id,
                    Conversation.title,
                    Conversation.created_at,
                    Conversation.updated_at,
                    func.coalesce(message_count_subquery.c.msg_count, 0).label('message_count'),
                    message_count_subquery.c.last_msg_at.label('last_message_at'),
                    Conversation.settings['provider'].astext.label('provider'),
                    Conversation.settings['model'].astext.label('model')
                )
                .outerjoin(message_count_subquery, Conversation.id == message_count_subquery.c.conversation_id)
                .order_by(desc(Conversation.updated_at))
                .offset(skip)
                .limit(limit)
            )

            result = await self.db.execute(stmt)
            rows = result.all()

            return [
                {
                    "id": str(row.id),
                    "title": row.title,
                    "message_count": row.message_count,
                    "last_message_at": row.last_message_at,
                    "created_at": row.created_at,
                    "updated_at": row.updated_at,
                    "provider": row.provider,
                    "model": row.model
                } for row in rows
            ]
        except Exception as e:
            self.logger.error(f"Error listing optimized conversations: {e}")
            raise

    async def get_with_relations(
        self,
        conversation_id: UUID,
        load_tools: bool = True,
        load_prompt: bool = True
    ) -> Optional[Conversation]:
        """Get conversation with eager-loaded relationships."""
        try:
            query = select(Conversation).where(Conversation.id == conversation_id)

            if load_tools:
                query = query.options(selectinload(Conversation.tool_configurations))
            if load_prompt:
                query = query.options(selectinload(Conversation.prompt_template))

            result = await self.db.execute(query)
            return result.scalars().first()
        except Exception as e:
            self.logger.error(f"Error getting conversation with relations: {e}")
            raise

    async def list_with_message_counts(
        self,
        skip: int = 0,
        limit: int = 20
    ) -> Tuple[List[dict], int]:
        """
        List conversations with message counts (optimized - headers only).
        Returns tuple of (conversations_data, total_count)
        """
        try:
            # Get total count
            total = await self.db.scalar(select(func.count(Conversation.id)))

            # Subquery for message counts
            message_count_subquery = (
                select(
                    Message.conversation_id,
                    func.count(Message.id).label('msg_count'),
                    func.max(Message.created_at).label('last_msg_at')
                )
                .group_by(Message.conversation_id)
                .subquery()
            )

            # Main query with join
            conversations_query = (
                select(
                    Conversation.id,
                    Conversation.title,
                    Conversation.created_at,
                    Conversation.updated_at,
                    func.coalesce(message_count_subquery.c.msg_count, 0).label('message_count'),
                    message_count_subquery.c.last_msg_at.label('last_message_at'),
                    Conversation.settings['provider'].astext.label('provider'),
                    Conversation.settings['model'].astext.label('model')
                )
                .outerjoin(
                    message_count_subquery,
                    Conversation.id == message_count_subquery.c.conversation_id
                )
                .order_by(desc(Conversation.updated_at))
                .offset(skip)
                .limit(limit)
            )

            result = await self.db.execute(conversations_query)
            conversations = result.all()

            # Build response
            items = []
            for conv in conversations:
                items.append({
                    "id": str(conv.id),
                    "title": conv.title,
                    "message_count": conv.message_count or 0,
                    "last_message_at": conv.last_message_at,
                    "created_at": conv.created_at,
                    "updated_at": conv.updated_at,
                    "provider": conv.provider,
                    "model": conv.model
                })

            return items, total or 0

        except Exception as e:
            self.logger.error(f"Error listing conversations with counts: {e}")
            raise
