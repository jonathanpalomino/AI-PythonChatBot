# =============================================================================
# api/v1/prompts.py
# Prompt Templates API endpoints
# =============================================================================
"""
API endpoints para gestión de plantillas de prompts
"""
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, status
from src.dependencies import get_prompt_template_service
from src.models.models import VisibilityType
from src.schemas.schemas import (
    PromptTemplateCreate,
    PromptTemplateUpdate,
    PromptTemplateResponse,
    PaginationParams,
    ListResponse
)
from src.services.prompt_template_service import PromptTemplateService
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


# =============================================================================
# CRUD Endpoints
# =============================================================================

@router.post("", response_model=PromptTemplateResponse, status_code=status.HTTP_201_CREATED)
async def create_prompt_template(
    data: PromptTemplateCreate,
    service: PromptTemplateService = Depends(get_prompt_template_service)
):
    """Create a new prompt template"""
    try:
        template = await service.create_template(data)
        return template
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("", response_model=ListResponse)
async def list_prompt_templates(
    category: Optional[str] = Query(None, description="Filter by category"),
    visibility: Optional[VisibilityType] = Query(None, description="Filter by visibility"),
    search: Optional[str] = Query(None, description="Search in name/description"),
    is_active: bool = Query(True, description="Filter by active status"),
    pagination: PaginationParams = Depends(),
    service: PromptTemplateService = Depends(get_prompt_template_service)
):
    """List prompt templates with filters"""
    return await service.list_templates(
        category=category,
        visibility=visibility,
        search=search,
        is_active=is_active,
        skip=pagination.skip,
        limit=pagination.limit
    )


@router.get("/categories", response_model=List[str])
async def get_categories(
    service: PromptTemplateService = Depends(get_prompt_template_service)
):
    """Get list of all categories"""
    return await service.get_categories()


@router.get("/{template_id}", response_model=PromptTemplateResponse)
async def get_prompt_template(
    template_id: UUID,
    service: PromptTemplateService = Depends(get_prompt_template_service)
):
    """Get a specific prompt template"""
    template = await service.get_template(template_id)
    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prompt template not found"
        )
    return template


@router.put("/{template_id}", response_model=PromptTemplateResponse)
async def update_prompt_template(
    template_id: UUID,
    data: PromptTemplateUpdate,
    service: PromptTemplateService = Depends(get_prompt_template_service)
):
    """Update a prompt template"""
    try:
        template = await service.update_template(template_id, data)
        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Prompt template not found"
            )
        return template
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.delete("/{template_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_prompt_template(
    template_id: UUID,
    service: PromptTemplateService = Depends(get_prompt_template_service)
):
    """Delete a prompt template"""
    deleted = await service.delete_template(template_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prompt template not found"
        )
    return None
