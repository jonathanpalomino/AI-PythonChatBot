# =============================================================================
# api/v1/conversations.py
# Conversations API endpoints
# =============================================================================
"""
API endpoints para gestión de conversaciones y chat
"""
from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status, Response
from fastapi.responses import StreamingResponse
from src.dependencies import get_conversation_service
from src.schemas.schemas import (
    ConversationCreate, ConversationUpdate, ConversationResponse,
    ChatRequest, ChatResponse, MessageResponse, PaginationParams, ListResponse
)
from src.services.conversation_service import ConversationService
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


# =============================================================================
# Conversations CRUD
# =============================================================================

@router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    data: ConversationCreate,
    service: ConversationService = Depends(get_conversation_service)
):
    """Create a new conversation"""
    try:
        conversation = await service.create_conversation(data)
        return conversation
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@router.get("", response_model=ListResponse)
async def list_conversations(
    pagination: PaginationParams = Depends(),
    service: ConversationService = Depends(get_conversation_service)
):
    """List all conversations with pagination (optimized - headers only)"""
    return await service.list_conversations(
        skip=pagination.skip,
        limit=pagination.limit
    )


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: UUID,
    service: ConversationService = Depends(get_conversation_service)
):
    """Get a specific conversation"""
    conversation = await service.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    return conversation


@router.patch("/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: UUID,
    data: ConversationUpdate,
    service: ConversationService = Depends(get_conversation_service)
):
    """Update a conversation"""
    try:
        conversation = await service.update_conversation(conversation_id, data)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        return conversation
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: UUID,
    service: ConversationService = Depends(get_conversation_service)
):
    """Delete a conversation"""
    deleted = await service.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    return None


# =============================================================================
# Messages
# =============================================================================

@router.get("/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_conversation_messages(
    conversation_id: UUID,
    limit: int = 50,
    service: ConversationService = Depends(get_conversation_service)
):
    """Get messages for a conversation"""
    try:
        messages = await service.get_conversation_messages(conversation_id, limit)
        return messages
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


# =============================================================================
# Chat Endpoint (Main)
# =============================================================================

@router.post("/{conversation_id}/chat", response_model=ChatResponse)
async def chat(
    conversation_id: UUID,
    request: ChatRequest,
    service: ConversationService = Depends(get_conversation_service)
):
    """
    Send a message and get AI response
    Main chat endpoint that orchestrates the entire flow
    """
    try:
        return await service.chat(conversation_id, request)
    except ValueError as e:
        status_code = status.HTTP_404_NOT_FOUND if "not found" in str(e).lower() else status.HTTP_400_BAD_REQUEST
        raise HTTPException(status_code=status_code, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing chat: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {str(e)}"
        )


@router.post("/{conversation_id}/chat/stream")
async def stream_chat(
    conversation_id: UUID,
    request: ChatRequest,
    service: ConversationService = Depends(get_conversation_service)
):
    """
    Stream chat response using Server-Sent Events (SSE)
    """
    return StreamingResponse(
        service.stream_chat(conversation_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/{conversation_id}/chat/cancel", response_model=dict)
async def cancel_stream(
    conversation_id: UUID,
    service: ConversationService = Depends(get_conversation_service)
):
    """Cancel an active streaming chat session"""
    return await service.cancel_stream(conversation_id)


@router.post("/chat", response_model=ChatResponse)
async def quick_chat(
    request: ChatRequest,
    service: ConversationService = Depends(get_conversation_service)
):
    """Quick chat without existing conversation (creates temporary)"""
    try:
        return await service.quick_chat(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# =============================================================================
# Regenerate & Fork
# =============================================================================

@router.post("/{conversation_id}/regenerate", response_model=ChatResponse)
async def regenerate_last_message(
    conversation_id: UUID,
    service: ConversationService = Depends(get_conversation_service)
):
    """Regenerate the last assistant message"""
    try:
        return await service.regenerate_last_message(conversation_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


# =============================================================================
# Export
# =============================================================================

@router.get("/{conversation_id}/export/pdf")
async def export_conversation_pdf(
    conversation_id: UUID,
    service: ConversationService = Depends(get_conversation_service)
):
    """Export a conversation as PDF"""
    try:
        pdf_content = await service.export_conversation_pdf(conversation_id)
        filename = f"conversation_{str(conversation_id)[:8]}.pdf"
        return Response(
            content=pdf_content,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    except ValueError as e:
        status_code = status.HTTP_404_NOT_FOUND if "not found" in str(e).lower() else status.HTTP_400_BAD_REQUEST
        raise HTTPException(status_code=status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
