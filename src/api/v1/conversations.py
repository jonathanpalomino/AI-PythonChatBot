# =============================================================================
# api/v1/conversations.py
# Conversations API endpoints
# =============================================================================
"""
API endpoints para gestión de conversaciones y chat
"""
import json
import asyncio
from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Response
from fastapi.responses import StreamingResponse
from sqlalchemy import func, desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.config.settings import settings
from src.database.connection import get_async_db
from src.models.models import (
    Conversation, Message, MessageRole,
    ToolConfiguration, PromptTemplate, Project
)
from src.schemas.schemas import (
    ConversationCreate, ConversationUpdate, ConversationResponse,
    ChatRequest, ChatResponse,
    MessageResponse, PaginationParams, ListResponse
)
from src.services.chat_orchestrator import ChatOrchestrator
from src.services.stream_cancel_manager import stream_cancel_manager
from src.services.pdf_service import pdf_service
from src.utils.logger import get_logger, set_conversation_context

logger = get_logger(__name__)
router = APIRouter()


# =============================================================================
# Conversations CRUD
# =============================================================================

@router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    data: ConversationCreate,
    db: AsyncSession = Depends(get_async_db)
):
    """Create a new conversation"""
    logger.info("Creating new conversation", extra={"title": data.title})

    # Validate prompt template if provided
    if data.prompt_template_id:
        result = await db.execute(
            select(PromptTemplate).filter(PromptTemplate.id == data.prompt_template_id)
        )
        template = result.scalars().first()
        if not template:
            logger.warning(f"Prompt template not found: {data.prompt_template_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Prompt template not found"
            )

    # Validate project if provided
    if data.project_id:
        result = await db.execute(
            select(Project).filter(Project.id == data.project_id)
        )
        if not result.scalars().first():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )

    # Create conversation
    conversation = Conversation(
        title=data.title,
        project_id=data.project_id,
        prompt_template_id=data.prompt_template_id,
        settings=data.settings.model_dump(),
        metadata=data.metadata
    )

    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)

    # Create tool configurations if tools are enabled
    enabled_tools = data.settings.enabled_tools
    if enabled_tools:
        for tool_name in enabled_tools:
            tool_config = ToolConfiguration(
                conversation_id=conversation.id,
                tool_name=tool_name,
                config={},
                is_active=True
            )
            db.add(tool_config)
        await db.commit()

    logger.info(
        "Conversation created successfully",
        extra={"conversation_id": str(conversation.id), "enabled_tools": enabled_tools}
    )
    return conversation


@router.get("", response_model=ListResponse)
async def list_conversations(
    pagination: PaginationParams = Depends(),
    db: AsyncSession = Depends(get_async_db)
):
    """List all conversations with pagination (optimized - headers only)"""
    logger.info("Listing conversations", extra={"skip": pagination.skip, "limit": pagination.limit})

    # Get total count
    total = await db.scalar(select(func.count(Conversation.id)))

    # OPTIMIZED: Single query with message count (no N+1)
    # Use subquery to count messages per conversation
    # Use subquery to count messages per conversation

    message_count_subquery = (
        select(
            Message.conversation_id,
            func.count(Message.id).label('msg_count'),
            func.max(Message.created_at).label('last_msg_at')
        )
        .group_by(Message.conversation_id)
        .subquery()
    )

    # Main query with join to get counts
    conversations_query = (
        select(
            Conversation.id,
            Conversation.title,
            Conversation.created_at,
            Conversation.updated_at,
            func.coalesce(message_count_subquery.c.msg_count, 0).label('message_count'),
            message_count_subquery.c.last_msg_at.label('last_message_at'),
            # Extract provider and model from settings JSON for preview
            Conversation.settings['provider'].astext.label('provider'),
            Conversation.settings['model'].astext.label('model')
        )
        .outerjoin(message_count_subquery,
                   Conversation.id == message_count_subquery.c.conversation_id)
        .order_by(desc(Conversation.updated_at))
        .offset(pagination.skip)
        .limit(pagination.limit)
    )

    result = await db.execute(conversations_query)
    conversations = result.all()

    # Build lightweight response (no settings, no metadata)
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

    logger.debug(f"Retrieved {len(items)} conversations (optimized)")
    return ListResponse(
        items=items,
        total=total,
        skip=pagination.skip,
        limit=pagination.limit
    )


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: UUID,
    db: AsyncSession = Depends(get_async_db)
):
    """Get a specific conversation"""
    result = await db.execute(
        select(Conversation).filter(Conversation.id == conversation_id)
    )
    conversation = result.scalars().first()

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
    db: AsyncSession = Depends(get_async_db)
):
    """Update a conversation"""
    result = await db.execute(
        select(Conversation).filter(Conversation.id == conversation_id)
    )
    conversation = result.scalars().first()

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    # Import embedding model validator
    from src.services.embedding_model_validator import embedding_model_validator

    # Validate embedding model change if settings are being updated
    if data.settings is not None:
        new_settings = data.settings.model_dump()
        new_embedding_model = new_settings.get('embedding_model')
        
        # Validate the embedding model change
        validation_error = await embedding_model_validator.validate_embedding_model_change(
            db=db,
            conversation_id=conversation_id,
            new_embedding_model=new_embedding_model,
            current_settings=conversation.settings
        )
        
        if validation_error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=validation_error
            )

    # Update fields
    if data.title is not None:
        conversation.title = data.title
    if data.settings is not None:
        conversation.settings = data.settings.model_dump()
    if data.metadata is not None:
        conversation.metadata = data.metadata

    await db.commit()
    await db.refresh(conversation)

    return conversation


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: UUID,
    db: AsyncSession = Depends(get_async_db)
):
    """Delete a conversation"""
    result = await db.execute(
        select(Conversation).filter(Conversation.id == conversation_id)
    )
    conversation = result.scalars().first()

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    # Delete Qdrant collection associated with the conversation
    from src.services.file_processor import FileProcessor
    file_processor = FileProcessor(db)
    collection_name = f"chat_{conversation_id}"
    
    try:
        # Check if collection exists in Qdrant
        try:
            await file_processor.qdrant.get_collection(collection_name)
            # Delete the collection
            await file_processor.qdrant.delete_collection(collection_name)
            logger.info(f"Deleted Qdrant collection: {collection_name}")
        except Exception:
            # Collection doesn't exist, which is fine
            logger.debug(f"Qdrant collection {collection_name} not found, skipping deletion")
    except Exception as e:
        logger.error(f"Failed to delete Qdrant collection: {e}", exc_info=True)

    await db.delete(conversation)
    await db.commit()

    return None


# =============================================================================
# Messages
# =============================================================================

@router.get("/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_conversation_messages(
    conversation_id: UUID,
    limit: int = 50,
    db: AsyncSession = Depends(get_async_db)
):
    """Get messages for a conversation"""
    result = await db.execute(
        select(Conversation).filter(Conversation.id == conversation_id)
    )
    conversation = result.scalars().first()

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    result = await db.execute(
        select(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
        .limit(limit)
    )
    messages = result.scalars().all()

    return messages


# =============================================================================
# Chat Endpoint (Main)
# =============================================================================

@router.post("/{conversation_id}/chat", response_model=ChatResponse)
async def chat(
    conversation_id: UUID,
    request: ChatRequest,
    db: AsyncSession = Depends(get_async_db)
):
    """
    Send a message and get AI response
    Main chat endpoint that orchestrates the entire flow
    """
    logger.info("Processing chat message", extra={"conversation_id": str(conversation_id)})
    set_conversation_context(str(conversation_id))

    # Get conversation
    result = await db.execute(
        select(Conversation)
        .options(selectinload(Conversation.tool_configurations))
        .options(selectinload(Conversation.prompt_template))
        .filter(Conversation.id == conversation_id)
    )
    conversation = result.scalars().first()

    if not conversation:
        logger.warning(f"Conversation not found: {conversation_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    # =========================================================================
    # DEBUG: Print request parameters (only in dev/debug mode)
    # =========================================================================
    if settings.DEBUG or settings.ENVIRONMENT != "production":
        print("\n" + "=" * 80)
        print("[DEBUG] Chat Request Parameters")
        print("=" * 80)
        print(f"Conversation ID: {conversation_id}")
        print(f"Message: {request.message[:200]}..." if len(
            request.message) > 200 else f"Message: {request.message}")
        print(f"Message Length: {len(request.message)} chars")
        print(f"File IDs: {request.file_ids if request.file_ids else 'None'}")
        print(f"File Count: {len(request.file_ids) if request.file_ids else 0}")
        print(f"Request Model: {request.model_dump() if hasattr(request, 'model_dump') else 'N/A'}")
        print(f"Conversation Title: {conversation.title}")
        print(f"Conversation Settings: {conversation.settings}")
        print("=" * 80 + "\n")
    # =========================================================================

    # Prepare user message (don't commit yet - batch at end)
    user_message = Message(
        conversation_id=conversation_id,
        role=MessageRole.USER,
        content=request.message,
        extra_metadata={
            "model": conversation.settings.get("model"),
            "provider": conversation.settings.get("provider")
        },
        attachments=[
            {"file_id": str(fid)} for fid in request.file_ids
        ] if request.file_ids else []
    )
    db.add(user_message)
    # No commit here - defer to batch commit at end for better performance

    try:
        # Process with orchestrator
        orchestrator = ChatOrchestrator(db)

        response = await orchestrator.process_message(
            conversation=conversation,
            user_message=request.message,
            file_ids=request.file_ids
        )

        # Validate response content - handle thinking-mode models that may return empty content
        response_content = response.content
        if not response_content or not response_content.strip():
            # Check if there was thinking content
            thinking = response.metadata.get("thinking_content")
            if thinking:
                logger.warning(
                    f"LLM returned only thinking content without final response (model: {conversation.settings.get('model')})")
                response_content = "[El modelo solo generó razonamiento interno sin respuesta final. Por favor, intenta reformular tu pregunta.]"
            else:
                logger.warning("LLM returned empty response")
                response_content = "[El modelo no generó una respuesta. Por favor, intenta de nuevo.]"
            # Update response content
            response.content = response_content

        # Save assistant message
        assistant_message = Message(
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT,
            content=response.content,
            thinking_content=response.metadata.get("thinking_content") if response.metadata else None,
            extra_metadata={
                **(response.metadata or {}),
                "model": response.model,
                "provider": response.provider,
                "tokens_used": response.tokens_used,
                "cost": response.cost
            },
            attachments=[]
        )
        db.add(assistant_message)

        # Update conversation timestamp
        conversation.updated_at = func.now()

        # Single batch commit for user message + assistant message + timestamp update
        await db.commit()
        await db.refresh(assistant_message)

        # Optimized logging - avoid dict creation in hot path
        logger.info(
            f"Chat processed: {len(response.content)} chars, {len(response.metadata.get('tools_executed', []))} tools")

        # Build response
        return ChatResponse(
            conversation_id=conversation_id,
            message=MessageResponse.model_validate(assistant_message, from_attributes=True),
            sources=response.metadata.get("rag_sources", []),
            tools_executed=response.metadata.get("tools_executed", []),
            confidence_score=response.metadata.get("confidence_score"),
            thinking_content=response.metadata.get("thinking_content")  # NEW: Expose to frontend
        )

    except Exception as e:
        # Rollback on error
        await db.rollback()
        logger.error(
            f"Error processing chat message: {e}",
            exc_info=True,
            extra={"conversation_id": str(conversation_id)}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {str(e)}"
        )


@router.post("/{conversation_id}/chat/stream")
async def stream_chat(
    conversation_id: UUID,
    request: ChatRequest,
    db: AsyncSession = Depends(get_async_db)
):
    """
    Stream chat response using Server-Sent Events (SSE) with cancellation support

    Returns real-time chunks as the LLM generates them.
    Supports cancellation via client disconnect or explicit cancel endpoint.
    Maintains full feature parity with /chat endpoint:
    - RAG search execution
    - Memory service
    - Tool orchestration
    - Message persistence
    - Metadata

    SSE Format:
    - data: {"chunk": "..."} -> content chunks
    - data: {"type": "metadata", "data": {...}} -> final metadata
    - data: [DONE] -> end of stream
    """
    logger.info("Processing streaming chat message",
                extra={"conversation_id": str(conversation_id)})
    set_conversation_context(str(conversation_id))

    # Register stream for cancellation tracking
    cancel_token = await stream_cancel_manager.register_stream(conversation_id)

    async def generate():
        try:
            # 1. Get conversation
            result = await db.execute(
                select(Conversation)
                .options(selectinload(Conversation.tool_configurations))
                .options(selectinload(Conversation.prompt_template))
                .filter(Conversation.id == conversation_id)
            )
            conversation = result.scalars().first()

            if not conversation:
                logger.warning(f"Conversation not found: {conversation_id}")
                yield f"event: error\ndata: Conversation not found\n\n"
                return

            # 2. Save user message
            user_message = Message(
                conversation_id=conversation_id,
                role=MessageRole.USER,
                content=request.message,
                extra_metadata={
                    "model": conversation.settings.get("model"),
                    "provider": conversation.settings.get("provider")
                },
                attachments=[
                    {"file_id": str(fid)} for fid in request.file_ids
                ] if request.file_ids else []
            )
            db.add(user_message)
            await db.commit()
            logger.debug("User message saved for streaming",
                         extra={"message_length": len(request.message)})

            # 3. Process with orchestrator (STREAMING MODE with cancellation)
            orchestrator = ChatOrchestrator(db)

            full_response = ""
            thinking_content = ""  # NEW: Accumulate thinking from model
            metadata = {}

            try:
                async for chunk_data in orchestrator.process_message_stream(
                    conversation=conversation,
                    user_message=request.message,
                    file_ids=request.file_ids,
                    cancel_token=cancel_token
                ):
                    # Check if cancelled before yielding
                    if cancel_token.is_cancelled():
                        logger.info(f"Stream cancelled during processing: {conversation_id}")
                        break

                    if chunk_data["type"] == "thinking":
                        # NEW: Stream thinking content to frontend
                        thinking_chunk = chunk_data["content"]
                        thinking_content += thinking_chunk
                        yield f"data: {json.dumps({'type': 'thinking', 'content': thinking_chunk})}\n\n"

                    elif chunk_data["type"] == "content":
                        # Stream content chunk
                        chunk = chunk_data["chunk"]
                        full_response += chunk
                        yield f"data: {json.dumps({'chunk': chunk})}\n\n"

                    elif chunk_data["type"] == "metadata":
                        # Final metadata
                        metadata = chunk_data["data"]

            except asyncio.CancelledError:
                logger.info(f"Stream cancelled by client disconnect: {conversation_id}")
                cancel_token.cancel("disconnect")
                # Don't yield anything more - client has disconnected
                return

            # 4. Validate and save assistant message
            # Handle thinking-mode models that may return empty content
            if not full_response or not full_response.strip():
                logger.warning(
                    f"Streaming response was empty (model: {conversation.settings.get('model')})")
                full_response = "[El modelo no generó una respuesta. Por favor, intenta de nuevo.]"
                # Send fallback message to client (if not cancelled)
                if not cancel_token.is_cancelled():
                    yield f"data: {json.dumps({'chunk': full_response})}\n\n"

            # Only save assistant message if not cancelled
            if not cancel_token.is_cancelled():
                assistant_message = Message(
                    conversation_id=conversation_id,
                    role=MessageRole.ASSISTANT,
                    content=full_response,
                    thinking_content=thinking_content if thinking_content else None,
                    extra_metadata={
                        **metadata,
                        "model": metadata.get("model", conversation.settings.get("model")),
                        "provider": metadata.get("provider", conversation.settings.get("provider"))
                    },
                    attachments=[]
                )
                db.add(assistant_message)

                # Update conversation timestamp
                conversation.updated_at = func.now()

                await db.commit()
                await db.refresh(assistant_message)

                logger.info(
                    "Streaming chat completed",
                    extra={
                        "response_length": len(full_response),
                        "tools_executed": len(metadata.get("tools_executed", []))
                    }
                )

                # 5. Send final metadata and done signal
                yield f"data: {json.dumps({'type': 'metadata', 'data': metadata})}\n\n"
                yield f"data: [DONE]\n\n"
            else:
                # Stream was cancelled - rollback user message and log
                await db.rollback()
                logger.info(f"Stream cancelled, rolling back user message: {conversation_id}")

        except Exception as e:
            # Rollback on error
            await db.rollback()
            
            # Check for specific provider errors (like Groq decommissioned models)
            error_msg = str(e)
            is_decommissioned = "model_decommissioned" in error_msg or "decommissioned" in error_msg.lower()
            
            is_rate_limit = "429" in error_msg or "RateLimitError" in error_msg
            
            if is_decommissioned:
                logger.warning(
                    f"Model decommissioned error detected: {error_msg}",
                    extra={"conversation_id": str(conversation_id)}
                )
                friendly_message = "El modelo seleccionado ya no está disponible (descontinuado por el proveedor). Por favor, selecciona un modelo diferente en la configuración."
                if not cancel_token.is_cancelled():
                    yield f"event: error\ndata: {friendly_message}\n\n"
            elif is_rate_limit:
                logger.warning(
                    f"Rate limit error detected: {error_msg}",
                    extra={"conversation_id": str(conversation_id)}
                )
                friendly_message = "El proveedor de IA está saturado temporalmente (Rate Limit 429). Por favor, espera un momento o intenta con otro modelo."
                if not cancel_token.is_cancelled():
                    yield f"event: error\ndata: {friendly_message}\n\n"
            else:
                logger.error(
                    f"Error in streaming chat: {e}",
                    exc_info=True,
                    extra={"conversation_id": str(conversation_id)}
                )
                if not cancel_token.is_cancelled():
                    yield f"event: error\ndata: {str(e)}\n\n"
        finally:
            # Always unregister the stream
            await stream_cancel_manager.unregister_stream(conversation_id)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@router.post("/{conversation_id}/chat/cancel", response_model=dict)
async def cancel_stream(
    conversation_id: UUID,
    db: AsyncSession = Depends(get_async_db)
):
    """
    Cancel an active streaming chat session

    This endpoint allows explicit cancellation of streaming chats.
    Useful for UI buttons or programmatic cancellation.
    """
    logger.info("Cancelling stream", extra={"conversation_id": str(conversation_id)})
    set_conversation_context(str(conversation_id))

    # Attempt to cancel the stream
    cancelled = await stream_cancel_manager.cancel_stream(conversation_id, "user")

    if cancelled:
        logger.info(f"Stream cancelled successfully: {conversation_id}")
        return {
            "success": True,
            "message": "Stream cancelled successfully",
            "conversation_id": str(conversation_id)
        }
    else:
        logger.warning(f"No active stream found to cancel: {conversation_id}")
        return {
            "success": False,
            "message": "No active stream found for this conversation",
            "conversation_id": str(conversation_id)
        }


@router.post("/chat", response_model=ChatResponse)
async def quick_chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_async_db)
):
    """
    Quick chat without existing conversation
    Creates a temporary conversation
    """
    # Create temporary conversation
    conversation = Conversation(
        title=request.message[:50] + "...",
        settings={
            "provider": "local",
            "model": "mistral",
            "temperature": 0.7,
            "tool_mode": "manual",
            "enabled_tools": []
        }
    )

    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)

    # Process message
    return await chat(conversation.id, request, db)


# =============================================================================
# Regenerate & Fork
# =============================================================================

@router.post("/{conversation_id}/regenerate", response_model=ChatResponse)
async def regenerate_last_message(
    conversation_id: UUID,
    db: AsyncSession = Depends(get_async_db)
):
    """Regenerate the last assistant message"""
    # Get last two messages (user + assistant)
    result = await db.execute(
        select(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(desc(Message.created_at))
        .limit(2)
    )
    messages = result.scalars().all()

    if len(messages) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Not enough messages to regenerate"
        )

    assistant_msg = messages[0]
    user_msg = messages[1]

    if assistant_msg.role != MessageRole.ASSISTANT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Last message is not from assistant"
        )

    # Deactivate old assistant message instead of deleting (allows versioning)
    assistant_msg.is_active = False
    await db.commit()

    # Regenerate
    request = ChatRequest(message=user_msg.content)
    response = await chat(conversation_id, request, db)

    # Also update the thinking content in the response if available
    if response.thinking_content:
        # The chat endpoint already saves thinking_content, but we need to ensure it's in the response
        response.message.thinking_content = response.thinking_content

    return response


# =============================================================================
# Export
# =============================================================================

@router.get("/{conversation_id}/export/pdf")
async def export_conversation_pdf(
    conversation_id: UUID,
    db: AsyncSession = Depends(get_async_db)
):
    """Export a conversation as PDF"""
    logger.info(f"Exporting conversation {conversation_id} to PDF")

    # 1. Get conversation
    result = await db.execute(
        select(Conversation).filter(Conversation.id == conversation_id)
    )
    conversation = result.scalars().first()

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    # 2. Get all messages (ordered)
    result = await db.execute(
        select(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )
    messages = result.scalars().all()

    if not messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Conversation has no messages to export"
        )

    # 3. Generate PDF
    try:
        pdf_buffer = pdf_service.generate_conversation_pdf(
            title=conversation.title,
            messages=messages
        )
        
        filename = f"conversation_{str(conversation_id)[:8]}.pdf"
        
        return Response(
            content=pdf_buffer.getvalue(),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    except Exception as e:
        logger.error(f"Failed to generate PDF: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate PDF: {str(e)}"
        )
