# =============================================================================
# src/utils/service_factory.py
# Service Factory - Provides configured services without exposing database session
# =============================================================================
"""
Factory for creating configured services.
This module provides services without exposing database sessions to callers.
Follows the Repository pattern by hiding database access details.

REFACTORED: Services now receive Repositories directly, not UnitOfWork.
This follows the Repository pattern correctly:
    Service → Repository → Session
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator, TypeVar, Callable

from src.database.connection import get_async_db_session
from src.repositories import (
    CustomToolRepository,
    ToolConfigurationRepository,
    ConversationRepository,
    FileRepository,
)
from src.services.tool_service import ToolService
from src.services.utils.model_service import ModelService
from src.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


@asynccontextmanager
async def get_tool_service() -> AsyncGenerator[ToolService, None]:
    """
    Get a configured ToolService with Repositories.

    REFACTORED: Service now receives Repositories directly, not UnitOfWork.

    Usage:
        async with get_tool_service() as tool_service:
            tools = await tool_service.list_available_tools()

    Yields:
        ToolService: Configured tool service instance
    """
    async with get_async_db_session() as session:
        # Create repositories with the session
        custom_tool_repo = CustomToolRepository(session)
        tool_configuration_repo = ToolConfigurationRepository(session)
        conversation_repo = ConversationRepository(session)
        file_repo = FileRepository(session)

        yield ToolService(
            custom_tool_repo=custom_tool_repo,
            tool_configuration_repo=tool_configuration_repo,
            conversation_repo=conversation_repo,
            file_repo=file_repo,
        )


@asynccontextmanager
async def get_model_service() -> AsyncGenerator[ModelService, None]:
    """
    Get a configured ModelService with Repositories.

    Usage:
        async with get_model_service() as model_service:
            models = await model_service.get_all_active_models()

    Yields:
        ModelService: Configured model service instance
    """
    from src.repositories import LLMModelRepository

    async with get_async_db_session() as session:
        llm_model_repo = LLMModelRepository(session)
        yield ModelService(llm_model_repo=llm_model_repo)


@asynccontextmanager
async def get_session_for_provider_sync():
    """
    Get database session for provider model synchronization.
    This is a special case for provider_manager.sync_available_models.

    Yields:
        AsyncSession: Database session
    """
    async with get_async_db_session() as session:
        yield session


async def with_tool_service(func: Callable[[ToolService], T]) -> T:
    """
    Execute a function with a configured ToolService.

    Usage:
        result = await with_tool_service(
            lambda ts: ts.list_available_tools()
        )

    Args:
        func: Function that receives a ToolService

    Returns:
        Result of the function call
    """
    async with get_tool_service() as tool_service:
        return await func(tool_service)


async def with_model_service(func: Callable[[ModelService], T]) -> T:
    """
    Execute a function with a configured ModelService.

    Usage:
        result = await with_model_service(
            lambda ms: ms.get_all_active_models()
        )

    Args:
        func: Function that receives a ModelService

    Returns:
        Result of the function call
    """
    async with get_model_service() as model_service:
        return await func(model_service)
