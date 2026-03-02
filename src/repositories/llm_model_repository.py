# =============================================================================
# src/repositories/llm_model_repository.py
# LLMModel Repository
# =============================================================================
"""
Repository for LLMModel entity operations.

REPOSITORY PATTERN:
This repository encapsulates ALL database operations for LLMModel.
Services should NEVER access the database session directly.
"""
from typing import List, Optional, Dict, Any
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.llm_models import LLMModel
from src.repositories.base_repository import BaseRepository


class LLMModelRepository(BaseRepository[LLMModel]):
    """Repository for managing LLM model configurations."""
    
    def __init__(self, db: AsyncSession):
        super().__init__(LLMModel, db)
    
    async def get_by_provider_and_name(
        self,
        provider: str,
        model_name: str
    ) -> Optional[LLMModel]:
        """
        Get a model by provider and model name.
        
        Args:
            provider: Provider name (e.g., 'local', 'openai')
            model_name: Model name (e.g., 'deepseek-r1')
            
        Returns:
            LLMModel instance or None if not found
        """
        try:
            stmt = select(LLMModel).where(
                LLMModel.provider == provider,
                LLMModel.model_name == model_name
            )
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            self.logger.error(f"Error getting model by provider/name: {e}")
            raise
    
    async def get_all_active(self) -> List[LLMModel]:
        """
        Get all active models.
        
        Returns:
            List of all active LLMModel instances
        """
        try:
            stmt = select(LLMModel).where(LLMModel.is_active == True)
            result = await self.db.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            self.logger.error(f"Error getting active models: {e}")
            raise
    
    async def get_by_type(self, model_type: str) -> List[LLMModel]:
        """
        Get all active models of a specific type.
        
        Args:
            model_type: Model type (e.g., 'chat', 'reasoning', 'embedding')
            
        Returns:
            List of matching LLMModel instances
        """
        try:
            stmt = select(LLMModel).where(
                LLMModel.is_active == True,
                LLMModel.model_type == model_type
            )
            result = await self.db.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            self.logger.error(f"Error getting models by type: {e}")
            raise
    
    async def get_by_provider(self, provider: str) -> List[LLMModel]:
        """
        Get all active models for a provider.
        
        Args:
            provider: Provider name
            
        Returns:
            List of matching LLMModel instances
        """
        try:
            stmt = select(LLMModel).where(
                LLMModel.is_active == True,
                LLMModel.provider == provider
            )
            result = await self.db.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            self.logger.error(f"Error getting models by provider: {e}")
            raise
    
    async def get_grouped_by_provider(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all active models grouped by provider.
        
        Returns:
            Dictionary with provider keys and lists of model info dicts
        """
        models = await self.get_all_active()
        
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for model in models:
            if model.provider not in grouped:
                grouped[model.provider] = []
            
            grouped[model.provider].append({
                "id": str(model.id),
                "name": model.model_name,
                "type": model.model_type,
                "context_window": model.context_window,
                "supports_streaming": model.supports_streaming,
                "supports_function_calling": model.supports_function_calling,
                "supports_thinking": model.supports_thinking,
                "is_free": model.is_free,
                "cost_per_1k_input": model.cost_per_1k_input,
                "cost_per_1k_output": model.cost_per_1k_output
            })
        
        return grouped
    
    async def upsert_model(
        self,
        provider: str,
        model_name: str,
        **kwargs
    ) -> LLMModel:
        """
        Create or update a model configuration.
        
        Args:
            provider: Provider name
            model_name: Model name
            **kwargs: Additional model attributes
            
        Returns:
            Created or updated LLMModel instance
        """
        existing = await self.get_by_provider_and_name(provider, model_name)
        
        if existing:
            # Update existing
            for key, value in kwargs.items():
                if hasattr(existing, key):
                    setattr(existing, key, value)
            await self.db.flush()
            await self.db.refresh(existing)
            return existing
        else:
            # Create new
            return await self.create(
                provider=provider,
                model_name=model_name,
                **kwargs
            )
