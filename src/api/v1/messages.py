# =============================================================================
# api/v1/messages.py
# Messages API endpoints (standalone)
# =============================================================================
"""
API endpoints independientes para mensajes
"""
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.connection import get_async_db
from src.models.models import Message
from src.schemas.schemas import MessageResponse
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


# =============================================================================
# Message Operations
# =============================================================================

@router.get("/{message_id}", response_model=MessageResponse)
async def get_message(
    message_id: UUID,
    db: AsyncSession = Depends(get_async_db)
):
    """Get a specific message by ID"""
    logger.debug(f"Retrieving message: {message_id}")
    result = await db.execute(
        select(Message).filter(Message.id == message_id)
    )
    message = result.scalars().first()

    if not message:
        logger.warning(f"Message not found: {message_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found"
        )

    return message


@router.delete("/{message_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_message(
    message_id: UUID,
    db: AsyncSession = Depends(get_async_db)
):
    """Delete a message"""
    logger.info(f"Deleting message: {message_id}")
    result = await db.execute(
        select(Message).filter(Message.id == message_id)
    )
    message = result.scalars().first()

    if not message:
        logger.warning(f"Message not found for deletion: {message_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found"
        )

    await db.delete(message)
    await db.commit()
    logger.info(f"Message deleted successfully: {message_id}")

    return None
