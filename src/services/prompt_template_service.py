# =============================================================================
# src/services/prompt_template_service.py
# Prompt Template Service - Business Logic
# =============================================================================
"""
Business logic for prompt template operations

REFACTORED: Service now receives Repositories directly, not UnitOfWork.
This follows the Repository pattern correctly:
    Service → Repository → Session
"""
from typing import List, Optional
from uuid import UUID
from src.models.models import PromptTemplate, VisibilityType
from src.schemas.schemas import (
    PromptTemplateCreate,
    PromptTemplateUpdate,
    PromptTemplateResponse,
    ListResponse
)
from src.utils.logger import get_logger
from src.utils.transactional import transactional

logger = get_logger(__name__)


class PromptTemplateService:
    """
    Service for prompt template business logic.
    
    REFACTORED: Receives Repositories directly, not UnitOfWork.
    This follows the Repository pattern correctly.
    """
    
    def __init__(self, prompt_template_repo):
        """
        Initialize PromptTemplateService with repository.
        
        Args:
            prompt_template_repo: PromptTemplateRepository instance
        """
        self.prompt_template_repo = prompt_template_repo
    
    async def commit(self):
        """Commit current transaction using repository's session."""
        await self.prompt_template_repo.commit()
    
    async def rollback(self):
        """Rollback current transaction using repository's session."""
        await self.prompt_template_repo.rollback()
    
    async def flush(self):
        """Flush pending changes using repository's session."""
        await self.prompt_template_repo.flush()
    
    @transactional
    async def create_template(self, data: PromptTemplateCreate) -> PromptTemplate:
        """Create new prompt template"""
        logger.info(f"Creating prompt template: {data.name}")
        
        # Validación de negocio
        existing = await self.prompt_template_repo.get_by_name(data.name)
        if existing:
            logger.warning(f"Prompt template name already exists: {data.name}")
            raise ValueError(f"Prompt template with name '{data.name}' already exists")
        
        # Repository operation - handles flush/refresh internally
        template = await self.prompt_template_repo.create(
            name=data.name,
            description=data.description,
            category=data.category,
            visibility=data.visibility,
            system_prompt=data.system_prompt,
            user_prompt_template=data.user_prompt_template,
            variables=[v.model_dump() for v in data.variables],
            settings=data.settings.model_dump(),
            created_by=data.created_by
        )
        
        logger.info(f"Prompt template created: {template.id}")
        # @transactional hace commit automático
        return template
    
    async def list_templates(
        self,
        category: Optional[str] = None,
        visibility: Optional[VisibilityType] = None,
        search: Optional[str] = None,
        is_active: bool = True,
        skip: int = 0,
        limit: int = 20
    ) -> ListResponse:
        """List prompt templates"""
        templates, total = await self.prompt_template_repo.list_with_filters(
            category=category,
            visibility=visibility,
            search=search,
            is_active=is_active,
            skip=skip,
            limit=limit
        )
        
        logger.debug(f"Retrieved {len(templates)} prompt templates")
        
        return ListResponse(
            items=[PromptTemplateResponse.model_validate(t) for t in templates],
            total=total,
            skip=skip,
            limit=limit
        )
    
    async def get_categories(self) -> List[str]:
        """Get categories"""
        return await self.prompt_template_repo.get_categories()
    
    async def get_template(self, template_id: UUID) -> Optional[PromptTemplate]:
        """Get template by ID"""
        return await self.prompt_template_repo.get_by_id(template_id)
    
    @transactional
    async def update_template(
        self,
        template_id: UUID,
        data: PromptTemplateUpdate
    ) -> Optional[PromptTemplate]:
        """Update prompt template"""
        logger.info(f"Updating prompt template: {template_id}")
        template = await self.prompt_template_repo.get_by_id(template_id)
        if not template:
            return None
        
        # Validación
        if data.name and data.name != template.name:
            existing = await self.prompt_template_repo.get_by_name(data.name)
            if existing and existing.id != template_id:
                raise ValueError(f"Prompt template with name '{data.name}' already exists")
        
        # Update
        update_data = data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(template, field, value)
        
        # Repository handles flush/refresh internally
        template = await self.prompt_template_repo.save(template)
        logger.info(f"Prompt template updated: {template_id}")
        # @transactional hace commit automático
        return template
    
    @transactional
    async def delete_template(self, template_id: UUID) -> bool:
        """Delete prompt template"""
        logger.info(f"Deleting prompt template: {template_id}")
        template = await self.prompt_template_repo.get_by_id(template_id)
        if not template:
            return False
        
        await self.prompt_template_repo.delete(template_id)
        logger.info(f"Prompt template deleted: {template_id}")
        # @transactional hace commit automático
        return True
