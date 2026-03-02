# =============================================================================
# src/services/conversation_service.py
# Conversation Service - Business Logic
# =============================================================================
"""
Business logic for conversation operations

REFACTORED: Service now receives Repositories directly, not UnitOfWork.
This follows the Repository pattern correctly:
    Service → Repository → Session
"""
import asyncio
import json
from typing import List, Optional
from uuid import UUID

from sqlalchemy import func

from src.models.models import (
    Conversation, Message, MessageRole
)
# Repository imports
from src.repositories.conversation_repository import ConversationRepository
from src.repositories.custom_tool_repository import CustomToolRepository
from src.repositories.file_repository import FileRepository
from src.repositories.message_repository import MessageRepository
from src.repositories.project_repository import ProjectRepository
from src.repositories.prompt_template_repository import PromptTemplateRepository
from src.repositories.qdrant_collection_repository import QdrantCollectionRepository
from src.repositories.tool_configuration_repository import ToolConfigurationRepository
from src.schemas.schemas import (
    ConversationCreate, ConversationUpdate, ChatRequest, ChatResponse, MessageResponse, ListResponse
)
# SRP Refactored: Import from new location
from src.services.chat.orchestrator import ChatOrchestrator
from src.services.embedding.embedding_model_validator import embedding_model_validator
from src.services.processing.file_processor import FileProcessor
from src.services.utils.pdf_service import pdf_service
from src.services.utils.stream_cancel_manager import stream_cancel_manager
from src.utils.logger import get_logger, set_conversation_context
from src.utils.transactional import transactional

logger = get_logger(__name__)


class ConversationService:
    """
    Service for conversation business logic.

    Receives Repositories directly (not UnitOfWork) following proper Repository pattern.
    Each repository has its own session reference for transaction management.
    """

    def __init__(
        self,
        conversation_repository: ConversationRepository,
        message_repository: MessageRepository,
        file_repository: FileRepository,
        custom_tool_repository: CustomToolRepository,
        prompt_template_repository: PromptTemplateRepository,
        tool_configuration_repository: ToolConfigurationRepository,
        qdrant_collection_repository: QdrantCollectionRepository,
        project_repository: ProjectRepository,
    ):
        """
        Initialize service with injected repositories.

        Args:
            conversation_repository: Repository for Conversation entities
            message_repository: Repository for Message entities
            file_repository: Repository for File entities
            custom_tool_repository: Repository for CustomTool entities
            prompt_template_repository: Repository for PromptTemplate entities
            tool_configuration_repository: Repository for ToolConfiguration entities
            qdrant_collection_repository: Repository for QdrantCollection entities
            project_repository: Repository for Project entities
        """
        self.conversations = conversation_repository
        self.messages = message_repository
        self.files = file_repository
        self.custom_tools = custom_tool_repository
        self.prompt_templates = prompt_template_repository
        self.tool_configurations = tool_configuration_repository
        self.qdrant_collections = qdrant_collection_repository
        self.projects = project_repository

    async def commit(self):
        """Commit current transaction using repository's session."""
        await self.conversations.commit()

    async def rollback(self):
        """Rollback current transaction using repository's session."""
        await self.conversations.rollback()

    async def flush(self):
        """Flush pending changes using repository's session."""
        await self.conversations.flush()

    async def refresh(self, instance):
        """Refresh instance from database using repository's session."""
        await self.conversations.refresh(instance)

    @transactional
    async def create_conversation(self, data: ConversationCreate) -> Conversation:
        """Create new conversation"""
        logger.info("Creating new conversation", extra={"title": data.title})

        # Validate prompt template if provided
        if data.prompt_template_id:
            template = await self.prompt_templates.get_by_id(data.prompt_template_id)
            if not template:
                raise ValueError("Prompt template not found")

        # Validate project if provided
        if data.project_id:
            project = await self.projects.get_by_id(data.project_id)
            if not project:
                raise ValueError("Project not found")

        # Create conversation
        conversation = await self.conversations.create(
            title=data.title,
            project_id=data.project_id,
            prompt_template_id=data.prompt_template_id,
            settings=data.settings.model_dump(),
            metadata=data.metadata
        )

        # Create tool configurations if tools are enabled
        enabled_tools = data.settings.enabled_tools
        if enabled_tools:
            for tool_name in enabled_tools:
                await self.tool_configurations.create(
                    conversation_id=conversation.id,
                    tool_name=tool_name,
                    config={},
                    is_active=True
                )

        await self.refresh(conversation)
        logger.info(
            "Conversation created successfully",
            extra={"conversation_id": str(conversation.id), "enabled_tools": enabled_tools}
        )

        return conversation

    async def list_conversations(
        self,
        skip: int = 0,
        limit: int = 20
    ) -> ListResponse:
        """List all conversations with pagination (optimized - headers only)"""
        logger.info("Listing conversations", extra={"skip": skip, "limit": limit})

        items, total = await self.conversations.list_with_message_counts(skip, limit)

        logger.debug(f"Retrieved {len(items)} conversations (optimized)")
        return ListResponse(
            items=items,
            total=total,
            skip=skip,
            limit=limit
        )

    async def get_conversation(self, conversation_id: UUID) -> Optional[Conversation]:
        """Get conversation by ID"""
        return await self.conversations.get_by_id(conversation_id)

    @transactional
    async def update_conversation(
        self,
        conversation_id: UUID,
        data: ConversationUpdate
    ) -> Optional[Conversation]:
        """Update conversation"""
        logger.info(f"Updating conversation: {conversation_id}")
        conversation = await self.conversations.get_by_id(conversation_id)
        if not conversation:
            return None

        # Validate embedding model change if settings are being updated
        if data.settings is not None:
            new_settings = data.settings.model_dump()
            new_embedding_model = new_settings.get('embedding_model')

            validation_error = await embedding_model_validator.validate_embedding_model_change(
                message_repository=self.messages,
                conversation_id=conversation_id,
                new_embedding_model=new_embedding_model,
                current_settings=conversation.settings
            )

            if validation_error:
                raise ValueError(validation_error)

        # Update fields
        if data.title is not None:
            conversation.title = data.title
        if data.settings is not None:
            conversation.settings = data.settings.model_dump()
        if data.metadata is not None:
            conversation.metadata = data.metadata

        await self.flush()
        await self.refresh(conversation)
        logger.info(f"Conversation updated: {conversation_id}")
        return conversation

    @transactional
    async def delete_conversation(self, conversation_id: UUID) -> bool:
        """Delete conversation"""
        logger.info(f"Deleting conversation: {conversation_id}")
        conversation = await self.conversations.get_by_id(conversation_id)
        if not conversation:
            return False

        # Delete Qdrant collection associated with the conversation
        file_processor = FileProcessor(
            self.files,
            self.conversations,
            self.qdrant_collections
        )
        collection_name = f"chat_{conversation_id}"

        try:
            try:
                await file_processor.qdrant.get_collection(collection_name)
                await file_processor.qdrant.delete_collection(collection_name)
                logger.info(f"Deleted Qdrant collection: {collection_name}")
            except Exception:
                logger.debug(f"Qdrant collection {collection_name} not found, skipping deletion")
        except Exception as e:
            logger.error(f"Failed to delete Qdrant collection: {e}", exc_info=True)

        await self.conversations.delete(conversation_id)
        logger.info(f"Conversation deleted: {conversation_id}")
        return True

    async def get_conversation_messages(
        self,
        conversation_id: UUID,
        limit: int = 50
    ) -> List[Message]:
        """Get messages for a conversation"""
        conversation = await self.conversations.get_by_id(conversation_id)
        if not conversation:
            raise ValueError("Conversation not found")

        return await self.messages.get_conversation_messages(
            conversation_id=conversation_id,
            limit=limit
        )

    @transactional
    async def chat(
        self,
        conversation_id: UUID,
        request: ChatRequest
    ) -> ChatResponse:
        """
        Send a message and get AI response
        Main chat endpoint that orchestrates the entire flow
        """
        logger.info("Processing chat message", extra={"conversation_id": str(conversation_id)})
        set_conversation_context(str(conversation_id))

        # Get conversation with relations
        conversation = await self.conversations.get_with_relations(conversation_id)
        if not conversation:
            raise ValueError("Conversation not found")

        # Update generic conversation metadata if provided
        if request.extra_metadata:
            current_metadata = dict(conversation.extra_metadata or {})
            current_metadata.update(request.extra_metadata)
            conversation.extra_metadata = current_metadata
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(conversation, "extra_metadata")

        # Log request parameters
        logger.info(
            "Chat request parameters",
            extra={
                "conversation_id": str(conversation_id),
                "conversation_title": conversation.title,
                "conversation_settings": conversation.settings,
                "message_len": len(request.message),
                "message_preview": request.message[:200] if request.message else "",
                "file_ids": [str(f) for f in request.file_ids] if request.file_ids else [],
                "collection_name": request.collection_name,
                "extra_metadata": request.extra_metadata
            }
        )

        # Create user message via Repository
        user_message = await self.messages.create(
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

        try:
            # Process with orchestrator
            orchestrator = ChatOrchestrator(
                conversation_repo=self.conversations,
                message_repo=self.messages,
                file_repo=self.files,
                custom_tool_repo=self.custom_tools
            )

            response = await orchestrator.process_message(
                conversation=conversation,
                user_message=request.message,
                file_ids=request.file_ids,
                collection_name=request.collection_name
            )

            # Validate response content
            response_content = response.content
            if not response_content or not response_content.strip():
                thinking = response.metadata.get("thinking_content")
                if thinking:
                    logger.warning(
                        f"LLM returned only thinking content without final response (model: {conversation.settings.get('model')})"
                    )
                    response_content = "[El modelo solo generó razonamiento interno sin respuesta final. Por favor, intenta reformular tu pregunta.]"
                else:
                    logger.warning("LLM returned empty response")
                    response_content = "[El modelo no generó una respuesta. Por favor, intenta de nuevo.]"
                response.content = response_content

            # Create assistant message via Repository
            assistant_message = await self.messages.create(
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

            # Update conversation timestamp
            conversation.updated_at = func.now()

            # Commit handled by @transactional
            await self.refresh(assistant_message)

            logger.info(
                f"Chat processed: {len(response.content)} chars, {len(response.metadata.get('tools_executed', []))} tools"
            )

            return ChatResponse(
                conversation_id=conversation_id,
                message=MessageResponse.model_validate(assistant_message, from_attributes=True),
                sources=response.metadata.get("rag_sources", []),
                tools_executed=response.metadata.get("tools_executed", []),
                confidence_score=response.metadata.get("confidence_score"),
                thinking_content=response.metadata.get("thinking_content")
            )

        except Exception as e:
            logger.error(
                f"Error processing chat message: {e}",
                exc_info=True,
                extra={"conversation_id": str(conversation_id)}
            )
            raise

    async def stream_chat(
        self,
        conversation_id: UUID,
        request: ChatRequest
    ):
        """
        Stream chat response using Server-Sent Events (SSE)
        Generator function for streaming
        """
        logger.info("Processing streaming chat message", extra={"conversation_id": str(conversation_id)})
        set_conversation_context(str(conversation_id))

        # Register stream for cancellation tracking
        cancel_token = await stream_cancel_manager.register_stream(conversation_id)

        try:
            # Get conversation
            conversation = await self.conversations.get_with_relations(conversation_id)
            if not conversation:
                yield f"event: error\ndata: Conversation not found\n\n"
                return

            # Update metadata if provided
            if request.extra_metadata:
                current_metadata = dict(conversation.extra_metadata or {})
                current_metadata.update(request.extra_metadata)
                conversation.extra_metadata = current_metadata
                from sqlalchemy.orm.attributes import flag_modified
                flag_modified(conversation, "extra_metadata")

            # Log request
            logger.info(
                "Stream chat request parameters",
                extra={
                    "conversation_id": str(conversation_id),
                    "conversation_title": conversation.title,
                    "message_len": len(request.message),
                    "file_ids": [str(f) for f in request.file_ids] if request.file_ids else []
                }
            )

            # Create user message via Repository
            user_message = await self.messages.create(
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
            await self.commit()

            logger.debug("User message saved for streaming")

            # Process with orchestrator (STREAMING MODE)
            orchestrator = ChatOrchestrator(
                conversation_repo=self.conversations,
                message_repo=self.messages,
                file_repo=self.files,
                custom_tool_repo=self.custom_tools
            )

            full_response = ""
            thinking_content = ""
            metadata = {}

            try:
                async for chunk_data in orchestrator.process_message_stream(
                    conversation=conversation,
                    user_message=request.message,
                    file_ids=request.file_ids,
                    cancel_token=cancel_token,
                    collection_name=request.collection_name
                ):
                    # Check if cancelled
                    if cancel_token.is_cancelled():
                        logger.info(f"Stream cancelled during processing: {conversation_id}")
                        break

                    if chunk_data["type"] == "thinking":
                        thinking_chunk = chunk_data["content"]
                        thinking_content += thinking_chunk
                        yield f"data: {json.dumps({'type': 'thinking', 'content': thinking_chunk})}\n\n"

                    elif chunk_data["type"] == "content":
                        chunk = chunk_data["chunk"]
                        full_response += chunk
                        yield f"data: {json.dumps({'chunk': chunk})}\n\n"

                    elif chunk_data["type"] == "metadata":
                        metadata = chunk_data["data"]

            except asyncio.CancelledError:
                logger.info(f"Stream cancelled by client disconnect: {conversation_id}")
                cancel_token.cancel("disconnect")
                return

            # Validate and save assistant message
            if not full_response or not full_response.strip():
                logger.warning(f"Streaming response was empty (model: {conversation.settings.get('model')})")
                full_response = "[El modelo no generó una respuesta. Por favor, intenta de nuevo.]"
                if not cancel_token.is_cancelled():
                    yield f"data: {json.dumps({'chunk': full_response})}\n\n"

            # Only save if not cancelled
            if not cancel_token.is_cancelled():
                # Create assistant message via Repository
                assistant_message = await self.messages.create(
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
                conversation.updated_at = func.now()
                await self.commit()
                await self.refresh(assistant_message)

                logger.info(
                    "Streaming chat completed",
                    extra={
                        "response_length": len(full_response),
                        "tools_executed": len(metadata.get("tools_executed", []))
                    }
                )

                # Send final metadata and done signal
                yield f"data: {json.dumps({'type': 'metadata', 'data': metadata})}\n\n"
                yield f"data: [DONE]\n\n"
            else:
                # Stream was cancelled - rollback
                await self.rollback()
                logger.info(f"Stream cancelled, rolling back: {conversation_id}")

        except Exception as e:
            # Handle errors
            await self.rollback()
            error_msg = str(e)
            is_decommissioned = "model_decommissioned" in error_msg or "decommissioned" in error_msg.lower()
            is_rate_limit = "429" in error_msg or "RateLimitError" in error_msg

            if is_decommissioned:
                logger.warning(f"Model decommissioned error: {error_msg}")
                friendly_message = "El modelo seleccionado ya no está disponible (descontinuado por el proveedor)."
                if not cancel_token.is_cancelled():
                    yield f"event: error\ndata: {friendly_message}\n\n"
            elif is_rate_limit:
                logger.warning(f"Rate limit error: {error_msg}")
                friendly_message = "El proveedor de IA está saturado temporalmente (Rate Limit 429)."
                if not cancel_token.is_cancelled():
                    yield f"event: error\ndata: {friendly_message}\n\n"
            else:
                logger.error(f"Error in streaming chat: {e}", exc_info=True)
                if not cancel_token.is_cancelled():
                    yield f"event: error\ndata: {str(e)}\n\n"

        finally:
            await stream_cancel_manager.unregister_stream(conversation_id)

    async def cancel_stream(self, conversation_id: UUID) -> dict:
        """Cancel an active streaming chat session"""
        logger.info("Cancelling stream", extra={"conversation_id": str(conversation_id)})
        set_conversation_context(str(conversation_id))

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

    @transactional
    async def quick_chat(self, request: ChatRequest) -> ChatResponse:
        """Quick chat without existing conversation (creates temporary)"""
        # Create temporary conversation
        conversation = await self.conversations.create(
            title=request.message[:50] + "...",
            settings={
                "provider": "local",
                "model": "mistral",
                "temperature": None,
                "max_tokens": None,
                "tool_mode": "manual",
                "enabled_tools": []
            }
        )
        await self.refresh(conversation)

        # Process message
        return await self.chat(conversation.id, request)

    @transactional
    async def regenerate_last_message(self, conversation_id: UUID) -> ChatResponse:
        """Regenerate the last assistant message"""
        logger.info(f"Regenerating last message: {conversation_id}")

        # Get last two messages
        messages = await self.messages.get_last_n_messages(conversation_id, n=2)

        if len(messages) < 2:
            raise ValueError("Not enough messages to regenerate")

        assistant_msg = messages[-1]
        user_msg = messages[-2]

        if assistant_msg.role != MessageRole.ASSISTANT:
            raise ValueError("Last message is not from assistant")

        # Deactivate old assistant message
        assistant_msg.is_active = False

        # Regenerate
        request = ChatRequest(message=user_msg.content)
        response = await self.chat(conversation_id, request)

        # Update thinking content in response if available
        if response.thinking_content:
            response.message.thinking_content = response.thinking_content

        logger.info(f"Message regenerated: {conversation_id}")
        return response

    async def export_conversation_pdf(self, conversation_id: UUID) -> bytes:
        """Export a conversation as PDF"""
        logger.info(f"Exporting conversation {conversation_id} to PDF")

        # Get conversation
        conversation = await self.conversations.get_by_id(conversation_id)
        if not conversation:
            raise ValueError("Conversation not found")

        # Get all messages
        messages = await self.messages.get_conversation_messages(
            conversation_id=conversation_id,
            limit=10000
        )

        if not messages:
            raise ValueError("Conversation has no messages to export")

        # Generate PDF
        try:
            pdf_buffer = pdf_service.generate_conversation_pdf(
                title=conversation.title,
                messages=messages
            )
            return pdf_buffer.getvalue()
        except Exception as e:
            logger.error(f"Failed to generate PDF: {e}", exc_info=True)
            raise ValueError(f"Failed to generate PDF: {str(e)}")
