# =============================================================================
# api/v1/messages.py
# Messages API endpoints (standalone)
# =============================================================================
"""
API endpoints independientes para mensajes
"""
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from src.dependencies import get_message_service
from src.schemas.schemas import MessageResponse
from src.services.message_service import MessageService
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


# =============================================================================
# Message Operations
# =============================================================================

@router.get("/{message_id}", response_model=MessageResponse)
async def get_message(
    message_id: UUID,
    service: MessageService = Depends(get_message_service)
):
    """Get a specific message by ID"""
    message = await service.get_message(message_id)
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found"
        )
    return message


@router.delete("/{message_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_message(
    message_id: UUID,
    service: MessageService = Depends(get_message_service)
):
    """Delete a message"""
    deleted = await service.delete_message(message_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found"
        )
    return None
