# =============================================================================
# src/dependencies.py
# Dependency Injection for Repositories and Services
# =============================================================================
"""
Centralized dependency injection for FastAPI.

This module provides factory functions for:
- Database sessions (via connection.py)
- Repositories
- Services (with injected repositories)

REFACTORED: Services now receive Repositories directly, not UnitOfWork.
This follows the Repository pattern correctly:
    Service → Repository → Session

Usage in FastAPI endpoints:
    @router.get("/conversations/{id}")
    async def get_conversation(
        id: UUID,
        service: ConversationService = Depends(get_conversation_service)
    ):
        return await service.get_conversation(id)
"""

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

# ✅ Import correcto desde connection.py
from src.database.connection import get_async_db
from src.database.unit_of_work import UnitOfWork
from src.repositories import (
    ConversationRepository,
    MessageRepository,
    FileRepository,
    CustomToolRepository,
    PromptTemplateRepository,
    ProjectRepository,
    ToolConfigurationRepository,
    QdrantCollectionRepository,
)
# SRP Refactored: Import from new location
from src.services.chat.orchestrator import ChatOrchestrator
from src.services.chat.context_builder import ContextBuilder
from src.services.chat.tool_executor import ToolExecutor
from src.services.chat.response_formatter import ResponseFormatter
from src.services.chat.stream_handler import StreamHandler
from src.services.collection_service import CollectionService
from src.services.conversation_service import ConversationService
from src.services.file_service import FileService
from src.services.message_service import MessageService
from src.services.processing.file_processor import FileProcessor
from src.services.project_service import ProjectService
from src.services.prompt_template_service import PromptTemplateService
from src.services.tool_service import ToolService
from src.services.collection_ingest_service import CollectionIngestService


# =============================================================================
# DATABASE SESSION DEPENDENCY
# =============================================================================
# Ya existe en connection.py como get_async_db()
# Se usa directamente en los repository factories


# =============================================================================
# REPOSITORY DEPENDENCIES
# =============================================================================

def create_repository_factory(repository_class):
    """
    Generic factory function for creating repository dependencies.
    
    Args:
        repository_class: The repository class to instantiate
        
    Returns:
        A dependency function that creates the repository instance
        
    Example:
        get_conversation_repository = create_repository_factory(ConversationRepository)
    """
    def factory(db: AsyncSession = Depends(get_async_db)):
        return repository_class(db)
    factory.__name__ = f"get_{repository_class.__name__.lower().replace('repository', '')}_repository"
    factory.__doc__ = f"Dependency for {repository_class.__name__}."
    return factory


# Repository factories using the generic factory
get_conversation_repository = create_repository_factory(ConversationRepository)
get_message_repository = create_repository_factory(MessageRepository)
get_file_repository = create_repository_factory(FileRepository)
get_custom_tool_repository = create_repository_factory(CustomToolRepository)
get_prompt_template_repository = create_repository_factory(PromptTemplateRepository)
get_project_repository = create_repository_factory(ProjectRepository)
get_tool_configuration_repository = create_repository_factory(ToolConfigurationRepository)
get_qdrant_collection_repository = create_repository_factory(QdrantCollectionRepository)


# =============================================================================
# SERVICE DEPENDENCIES (REFACTORED - Repositories injected directly)
# =============================================================================

def get_conversation_service(
    conversation_repo: ConversationRepository = Depends(get_conversation_repository),
    message_repo: MessageRepository = Depends(get_message_repository),
    file_repo: FileRepository = Depends(get_file_repository),
    custom_tool_repo: CustomToolRepository = Depends(get_custom_tool_repository),
    prompt_template_repo: PromptTemplateRepository = Depends(get_prompt_template_repository),
    tool_configuration_repo: ToolConfigurationRepository = Depends(get_tool_configuration_repository),
    qdrant_collection_repo: QdrantCollectionRepository = Depends(get_qdrant_collection_repository),
    project_repo: ProjectRepository = Depends(get_project_repository),
) -> ConversationService:
    """
    Factory for ConversationService.
    Injects all required repositories directly (not UnitOfWork).
    
    This follows the Repository pattern correctly:
        Service → Repository → Session
    """
    return ConversationService(
        conversation_repository=conversation_repo,
        message_repository=message_repo,
        file_repository=file_repo,
        custom_tool_repository=custom_tool_repo,
        prompt_template_repository=prompt_template_repo,
        tool_configuration_repository=tool_configuration_repo,
        qdrant_collection_repository=qdrant_collection_repo,
        project_repository=project_repo,
    )


def get_chat_orchestrator(
    conversation_repo: ConversationRepository = Depends(get_conversation_repository),
    message_repo: MessageRepository = Depends(get_message_repository),
    file_repo: FileRepository = Depends(get_file_repository),
    custom_tool_repo: CustomToolRepository = Depends(get_custom_tool_repository),
) -> "ChatOrchestrator":
    """
    Factory for ChatOrchestrator service.
    
    SRP Refactored: ChatOrchestrator now delegates to specialists:
    - ContextBuilder: RAG, memory, history
    - ToolExecutor: Tool execution
    - ResponseFormatter: Response formatting
    - StreamHandler: Streaming support
    
    Injects all required repositories.
    """
    return ChatOrchestrator(
        conversation_repo=conversation_repo,
        message_repo=message_repo,
        file_repo=file_repo,
        custom_tool_repo=custom_tool_repo,
    )


# =============================================================================
# SRP SPECIALIST DEPENDENCIES (for advanced use cases)
# =============================================================================

def get_context_builder(
    conversation_repo: ConversationRepository = Depends(get_conversation_repository),
    message_repo: MessageRepository = Depends(get_message_repository),
    file_repo: FileRepository = Depends(get_file_repository),
    custom_tool_repo: CustomToolRepository = Depends(get_custom_tool_repository),
) -> "ContextBuilder":
    """
    Factory for ContextBuilder (SRP specialist).
    Use for direct context building without full orchestration.
    """
    return ContextBuilder(
        conversation_repo=conversation_repo,
        message_repo=message_repo,
        file_repo=file_repo,
        custom_tool_repo=custom_tool_repo,
    )


def get_tool_executor(
    file_repo: FileRepository = Depends(get_file_repository),
    custom_tool_repo: CustomToolRepository = Depends(get_custom_tool_repository),
    message_repo: MessageRepository = Depends(get_message_repository),
) -> "ToolExecutor":
    """
    Factory for ToolExecutor (SRP specialist).
    Use for direct tool execution without full orchestration.
    """
    return ToolExecutor(
        file_repo=file_repo,
        custom_tool_repo=custom_tool_repo,
        message_repo=message_repo,
    )


def get_response_formatter() -> "ResponseFormatter":
    """
    Factory for ResponseFormatter (SRP specialist).
    Use for direct response formatting without full orchestration.
    """
    return ResponseFormatter()


def get_stream_handler() -> "StreamHandler":
    """
    Factory for StreamHandler (SRP specialist).
    Use for direct streaming without full orchestration.
    """
    return StreamHandler()


def get_file_processor(
    file_repo: FileRepository = Depends(get_file_repository),
    conversation_repo: ConversationRepository = Depends(get_conversation_repository),
    qdrant_repo: QdrantCollectionRepository = Depends(get_qdrant_collection_repository),
) -> "FileProcessor":
    """
    Factory for FileProcessor service.
    """
    from src.services.processing.file_processor import FileProcessor
    return FileProcessor(file_repo, conversation_repo, qdrant_repo)


# =============================================================================
# UTILITY: Context manager for background tasks
# =============================================================================

async def get_db_for_background_task() -> AsyncSession:
    """
    Get database session for background tasks (Celery, cron jobs, etc.).

    Unlike get_async_db(), this returns the session directly.
    Caller is responsible for closing it.

    Usage:
        async def background_job():
            db = await get_db_for_background_task()
            try:
                repo = FileRepository(db)
                await repo.update(...)
                await db.commit()
            finally:
                await db.close()

    Returns:
        AsyncSession: Database session (caller must close it)
    """
    from src.database.connection import AsyncSessionLocal
    return AsyncSessionLocal()


# =============================================================================
# SERVICE DEPENDENCIES (Legacy - using UnitOfWork)
# =============================================================================

def get_project_service(
    project_repo: ProjectRepository = Depends(get_project_repository),
    file_repo: FileRepository = Depends(get_file_repository),
    conversation_repo: ConversationRepository = Depends(get_conversation_repository),
    file_processor: FileProcessor = Depends(get_file_processor),
) -> "ProjectService":
    """
    Factory for ProjectService.
    Injects all required repositories and services.

    Returns:
        Configured ProjectService instance
    """
    from src.services.project_service import ProjectService

    return ProjectService(
        project_repo=project_repo,
        file_repo=file_repo,
        conversation_repo=conversation_repo,
        file_processor=file_processor,
    )


async def get_redis_client():
    """
    Get Redis client for progress tracking.

    Returns:
        Redis client or None if unavailable
    """
    try:
        import redis.asyncio as redis
        from src.config.settings import settings

        return redis.from_url(settings.REDIS_URL)
    except Exception:
        return None


async def get_unit_of_work(
    db: AsyncSession = Depends(get_async_db)
) -> UnitOfWork:
    """Get UnitOfWork instance (for legacy services)"""
    return UnitOfWork(db)


# =============================================================================
# SERVICE DEPENDENCIES (REFACTORED - All services now use Repositories)
# =============================================================================

def get_message_service(
    message_repo: MessageRepository = Depends(get_message_repository)
) -> MessageService:
    """
    Factory for MessageService.
    
    REFACTORED: Service now receives Repository directly, not UnitOfWork.
    This follows the Repository pattern correctly:
        Service → Repository → Session
    """
    return MessageService(message_repo)


def get_prompt_template_service(
    prompt_template_repo: PromptTemplateRepository = Depends(get_prompt_template_repository)
) -> PromptTemplateService:
    """
    Factory for PromptTemplateService.
    
    REFACTORED: Service now receives Repository directly, not UnitOfWork.
    This follows the Repository pattern correctly:
        Service → Repository → Session
    """
    return PromptTemplateService(prompt_template_repo)


def get_collection_service(
    qdrant_collection_repo: QdrantCollectionRepository = Depends(get_qdrant_collection_repository)
) -> CollectionService:
    """
    Factory for CollectionService.
    
    REFACTORED: Service now receives Repository directly, not UnitOfWork.
    This follows the Repository pattern correctly:
        Service → Repository → Session
    """
    return CollectionService(qdrant_collection_repo)


async def get_file_service(
    file_repo: FileRepository = Depends(get_file_repository),
    conversation_repo: ConversationRepository = Depends(get_conversation_repository),
    qdrant_collection_repo: QdrantCollectionRepository = Depends(get_qdrant_collection_repository),
) -> FileService:
    """
    Factory for FileService.
    
    REFACTORED: Service now receives Repositories directly, not UnitOfWork.
    This follows the Repository pattern correctly:
        Service → Repository → Session
    """
    return FileService(
        file_repo=file_repo,
        conversation_repo=conversation_repo,
        qdrant_collection_repo=qdrant_collection_repo,
    )


def get_tool_service(
    custom_tool_repo: CustomToolRepository = Depends(get_custom_tool_repository),
    tool_configuration_repo: ToolConfigurationRepository = Depends(get_tool_configuration_repository),
    conversation_repo: ConversationRepository = Depends(get_conversation_repository),
    file_repo: FileRepository = Depends(get_file_repository),
) -> ToolService:
    """
    Factory for ToolService.
    
    REFACTORED: Service now receives Repositories directly, not UnitOfWork.
    This follows the Repository pattern correctly:
        Service → Repository → Session
    """
    return ToolService(
        custom_tool_repo=custom_tool_repo,
        tool_configuration_repo=tool_configuration_repo,
        conversation_repo=conversation_repo,
        file_repo=file_repo,
    )


def get_collection_ingest_service(
    file_repo: FileRepository = Depends(get_file_repository),
    collection_repo: QdrantCollectionRepository = Depends(get_qdrant_collection_repository),
    file_processor: FileProcessor = Depends(get_file_processor),
    redis_client=Depends(get_redis_client),
) -> CollectionIngestService:
    """Factory for CollectionIngestService."""
    return CollectionIngestService(
        file_repo=file_repo,
        collection_repo=collection_repo,
        file_processor=file_processor,
        redis_client=redis_client
    )


# =============================================================================
# GENERIC SERVICE FACTORY (Legacy)
# =============================================================================

def create_service_factory(service_class):
    """
    Generic factory function for creating service dependencies that use UnitOfWork.
    
    NOTE: This is legacy pattern. New services should receive repositories directly.
    
    Args:
        service_class: The service class to instantiate (must accept UnitOfWork)
        
    Returns:
        An async dependency function that creates the service instance
        
    Example:
        get_message_service = create_service_factory(MessageService)
    """
    async def factory(uow: UnitOfWork = Depends(get_unit_of_work)):
        return service_class(uow)
    factory.__name__ = f"get_{service_class.__name__.lower().replace('service', '')}_service"
    factory.__doc__ = f"Get {service_class.__name__} instance"
    return factory
