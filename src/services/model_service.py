# =============================================================================
# src/services/model_service.py
# Service for querying LLM model metadata from database
# =============================================================================
"""
Centralized service for accessing model information from llm_models table
"""
from typing import Optional, List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import settings
from src.models.llm_models import LLMModel


class ModelService:
    """Service for querying model metadata"""

    @staticmethod
    async def get_embedding_model(db: AsyncSession) -> str:
        """
        Get the active embedding model name
        Returns the first active embedding model from the database
        Falls back to settings.EMBEDDING_MODEL if none found
        """
        stmt = select(LLMModel).where(
            LLMModel.model_type == 'embedding',
            LLMModel.is_active == True,
            LLMModel.model_name == 'mxbai-embed-large'
        ).limit(1)

        result = await db.execute(stmt)
        model = result.scalar_one_or_none()

        if model:
            return model.model_name

        # Fallback to settings if DB not populated yet
        return settings.EMBEDDING_MODEL

    @staticmethod
    async def get_models_by_type(db: AsyncSession, model_type: str) -> List[LLMModel]:
        """
        Get all active models of a specific type

        Args:
            db: Database session
            model_type: Type of model (chat, embedding, vision, reasoning, etc.)

        Returns:
            List of LLMModel instances
        """
        stmt = select(LLMModel).where(
            LLMModel.model_type == model_type,
            LLMModel.is_active == True
        )

        result = await db.execute(stmt)
        return result.scalars().all()

    @staticmethod
    async def get_model_info(db: AsyncSession, model_name: str, provider: str = 'local') -> \
        Optional[LLMModel]:
        """
        Get detailed information about a specific model

        Args:
            db: Database session
            model_name: Name of the model
            provider: Provider name (default: 'local')

        Returns:
            LLMModel instance or None if not found
        """
        stmt = select(LLMModel).where(
            LLMModel.model_name == model_name,
            LLMModel.provider == provider
        )

        result = await db.execute(stmt)
        return result.scalar_one_or_none()


# Singleton instance
model_service = ModelService()
