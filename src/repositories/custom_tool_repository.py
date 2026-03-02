# =============================================================================
# src/repositories/custom_tool_repository.py
# Custom Tool Repository
# =============================================================================
"""
Repository for CustomTool entity operations.
"""
from typing import List, Optional
from uuid import UUID
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src.models.models import CustomTool
from src.repositories.base_repository import BaseRepository


class CustomToolRepository(BaseRepository[CustomTool]):
    """Repository for managing custom tools."""
    
    def __init__(self, db: AsyncSession):
        super().__init__(CustomTool, db)
    
    async def get_conversation_tools(
        self,
        conversation_id: UUID,
        active_only: bool = True,
        is_template: bool = False
    ) -> List[CustomTool]:
        """
        Get all custom tools for a conversation.
        
        Args:
            conversation_id: Conversation UUID
            active_only: Only return active tools
            
        Returns:
            List of custom tools
        """
        try:
            stmt = select(CustomTool).where(
                CustomTool.conversation_id == conversation_id,
                CustomTool.is_template == is_template,
            )
            
            if active_only:
                stmt = stmt.where(CustomTool.is_active == True)
            
            stmt = stmt.order_by(CustomTool.created_at.desc())
            result = await self.db.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            self.logger.error(f"Error getting conversation tools: {e}")
            raise
    
    async def get_by_name(
        self,
        conversation_id: UUID,
        tool_name: str
    ) -> Optional[CustomTool]:
        """
        Get custom tool by name within a conversation.
        
        Args:
            conversation_id: Conversation UUID
            tool_name: Tool name
            
        Returns:
            Custom tool or None
        """
        try:
            stmt = select(CustomTool).where(
                CustomTool.conversation_id == conversation_id,
                CustomTool.name == tool_name,
                CustomTool.is_active == True
            )
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            self.logger.error(f"Error getting tool by name: {e}")
            raise
    
    async def get_rag_instances(self) -> List[CustomTool]:
        """
        Get all active RAG tool instances (not templates).
        
        Returns:
            List of active RAG tool instances
        """
        try:
            stmt = select(CustomTool).where(
                CustomTool.is_active == True,
                CustomTool.is_template == False,
                CustomTool.tool_type == "rag_search"
            )
            result = await self.db.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            self.logger.error(f"Error getting RAG instances: {e}")
            raise
    
    async def get_active_tools(self, is_template: Optional[bool] = None) -> List[CustomTool]:
        """
        Get all active custom tools.
        
        Args:
            is_template: Filter by template status (None = all)
        
        Returns:
            List of active CustomTool objects
        """
        try:
            stmt = select(CustomTool).where(CustomTool.is_active == True)
            
            if is_template is not None:
                stmt = stmt.where(CustomTool.is_template == is_template)
                
            result = await self.db.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            self.logger.error(f"Error getting active custom tools: {e}")
            raise
    
    # ========================================================================
    # Métodos adicionales útiles (opcionales, no en el original)
    # ========================================================================
    
    async def get_templates(self) -> List[CustomTool]:
        """Get all active tool templates."""
        try:
            stmt = select(CustomTool).where(
                CustomTool.is_active == True,
                CustomTool.is_template == True
            )
            result = await self.db.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            self.logger.error(f"Error getting templates: {e}")
            raise
    
    async def get_template_by_tool_type(self, tool_type: str) -> Optional[CustomTool]:
        """
        Get a tool template by its tool_type.
        
        Args:
            tool_type: The tool type identifier (e.g., 'rag_search', 'http_request')
            
        Returns:
            CustomTool template or None if not found
        """
        try:
            stmt = select(CustomTool).where(
                CustomTool.is_template == True,
                CustomTool.tool_type == tool_type
            )
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            self.logger.error(f"Error getting template by tool_type: {e}")
            raise
    
    async def upsert_template(
        self,
        tool_type: str,
        name: str,
        description: str,
        config_schema: dict,
        example: dict,
        configuration: dict = None
    ) -> CustomTool:
        """
        Create or update a tool template.
        
        Args:
            tool_type: The tool type identifier
            name: Display name for the template
            description: Template description
            config_schema: JSON schema for configuration
            example: Example configuration values
            configuration: Default configuration (optional)
            
        Returns:
            Created or updated CustomTool template
        """
        try:
            # Try to find existing template
            existing = await self.get_template_by_tool_type(tool_type)
            
            if existing:
                # Update existing template
                existing.name = name
                existing.description = description
                existing.config_schema = config_schema
                existing.example = example
                if configuration is not None:
                    existing.configuration = configuration
                await self.db.flush()
                await self.db.refresh(existing)
                return existing
            else:
                # Create new template
                new_template = CustomTool(
                    name=name,
                    description=description,
                    tool_type=tool_type,
                    is_template=True,
                    is_active=True,
                    configuration=configuration or {},
                    config_schema=config_schema,
                    example=example
                )
                self.db.add(new_template)
                await self.db.flush()
                await self.db.refresh(new_template)
                return new_template
        except Exception as e:
            self.logger.error(f"Error upserting template: {e}")
            raise

    async def commit(self):
        """Commit transaction."""
        await self.db.commit()
