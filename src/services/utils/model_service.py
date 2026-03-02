# =============================================================================
# src/services/utils/model_service.py
# Service for querying LLM model metadata from database
# =============================================================================
"""
Centralized service for accessing model information from llm_models table.

REPOSITORY PATTERN COMPLIANCE:
This service ONLY interacts with Repositories, NEVER with the database session directly.
All database operations are encapsulated in the Repository layer:
    Service → Repository → Database
"""
from typing import Optional, List, Dict, Any

from src.config.settings import settings
from src.models.llm_models import LLMModel
from src.repositories.llm_model_repository import LLMModelRepository
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelService:
    """
    Service for querying model metadata.
    
    REPOSITORY PATTERN: Uses LLMModelRepository for all database operations.
    """

    def __init__(self, llm_model_repo: LLMModelRepository = None):
        """
        Initialize ModelService.
        
        Args:
            llm_model_repo: LLMModelRepository instance for database operations
        """
        self._llm_model_repo = llm_model_repo

    @property
    def llm_model_repo(self) -> LLMModelRepository:
        """Get LLMModelRepository instance"""
        return self._llm_model_repo

    async def get_embedding_model(self) -> str:
        """
        Get the active embedding model name.
        Returns the first active embedding model from the database.
        Falls back to settings.EMBEDDING_MODEL if none found.
        
        Returns:
            Name of the embedding model to use
        """
        if not self._llm_model_repo:
            return settings.EMBEDDING_MODEL
        
        try:
            models = await self._llm_model_repo.get_by_type('embedding')
            for model in models:
                if model.model_name == 'mxbai-embed-large':
                    return model.model_name
        except Exception as e:
            logger.warning(f"Error getting embedding model from database: {e}")
        
        # Fallback to settings if DB not populated yet
        return settings.EMBEDDING_MODEL

    async def get_models_by_type(self, model_type: str) -> List[LLMModel]:
        """
        Get all active models of a specific type.

        Args:
            model_type: Type of model (chat, embedding, vision, reasoning, etc.)

        Returns:
            List of LLMModel instances
        """
        if not self._llm_model_repo:
            raise ValueError("LLMModelRepository not initialized")
        
        return await self._llm_model_repo.get_by_type(model_type)

    async def get_model_info(self, model_name: str, provider: str = 'local') -> Optional[LLMModel]:
        """
        Get detailed information about a specific model.

        Args:
            model_name: Name of the model
            provider: Provider name (default: 'local')

        Returns:
            LLMModel instance or None if not found
        """
        if not self._llm_model_repo:
            raise ValueError("LLMModelRepository not initialized")
        
        return await self._llm_model_repo.get_by_provider_and_name(provider, model_name)

    async def get_all_active_models(self) -> List[LLMModel]:
        """
        Get all active models from database.

        Returns:
            List of all active LLMModel instances
        """
        if not self._llm_model_repo:
            raise ValueError("LLMModelRepository not initialized")
        
        return await self._llm_model_repo.get_all_active()

    async def get_models_grouped_by_provider(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all active models grouped by provider.
        Used by the /providers endpoint.

        Returns:
            Dictionary with provider keys and lists of model info
        """
        if not self._llm_model_repo:
            raise ValueError("LLMModelRepository not initialized")
        
        return await self._llm_model_repo.get_grouped_by_provider()


# Singleton instance (for backward compatibility)
model_service = ModelService()
