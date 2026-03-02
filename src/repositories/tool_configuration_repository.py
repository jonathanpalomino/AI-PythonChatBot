# =============================================================================
# src/repositories/tool_configuration_repository.py
# ToolConfiguration Repository
# =============================================================================
"""
Repository for ToolConfiguration entity operations.
"""
from typing import List, Optional
from uuid import UUID
from sqlalchemy import select
from src.models.models import ToolConfiguration
from src.repositories.base_repository import BaseRepository
from sqlalchemy.ext.asyncio import AsyncSession


class ToolConfigurationRepository(BaseRepository[ToolConfiguration]):
    """Repository for managing tool configurations."""
    
    def __init__(self, db: AsyncSession):
        super().__init__(ToolConfiguration, db)
    
    async def get_by_conversation(
        self,
        conversation_id: UUID,
        active_only: bool = True
    ) -> List[ToolConfiguration]:
        """Get all tool configurations for a conversation."""
        try:
            stmt = select(ToolConfiguration).where(
                ToolConfiguration.conversation_id == conversation_id
            )
            if active_only:
                stmt = stmt.where(ToolConfiguration.is_active == True)
            
            result = await self.db.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            self.logger.error(f"Error getting tool configs for conversation {conversation_id}: {e}")
            raise
    
    async def get_by_conversation_and_tool(
        self,
        conversation_id: UUID,
        tool_name: str
    ) -> Optional[ToolConfiguration]:
        """Get specific tool configuration for a conversation."""
        try:
            stmt = select(ToolConfiguration).where(
                ToolConfiguration.conversation_id == conversation_id,
                ToolConfiguration.tool_name == tool_name
            )
            result = await self.db.execute(stmt)
            return result.scalars().first()
        except Exception as e:
            self.logger.error(f"Error getting tool config: {e}")
            raise
