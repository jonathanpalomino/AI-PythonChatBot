# =============================================================================
# src/repositories/__init__.py
# Repository Exports
# =============================================================================
"""
Repository pattern implementation for data access layer.

Usage:
    from src.repositories import ConversationRepository

    # In your service/endpoint:
    async def my_function(db: AsyncSession):
        repo = ConversationRepository(db)
        conversation = await repo.get_by_id(conversation_id)
"""

from src.repositories.base_repository import BaseRepository
from src.repositories.conversation_repository import ConversationRepository
from src.repositories.message_repository import MessageRepository
from src.repositories.file_repository import FileRepository
from src.repositories.custom_tool_repository import CustomToolRepository
from src.repositories.prompt_template_repository import PromptTemplateRepository
from src.repositories.project_repository import ProjectRepository
from src.repositories.tool_configuration_repository import ToolConfigurationRepository
from src.repositories.qdrant_collection_repository import QdrantCollectionRepository
from src.repositories.llm_model_repository import LLMModelRepository

__all__ = [
    "BaseRepository",
    "ConversationRepository",
    "MessageRepository",
    "FileRepository",
    "CustomToolRepository",
    "PromptTemplateRepository",
    "ProjectRepository",
    "ToolConfigurationRepository",
    "QdrantCollectionRepository",
    "LLMModelRepository",
]
