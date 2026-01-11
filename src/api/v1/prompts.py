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
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.connection import get_async_db
from src.models.models import PromptTemplate, VisibilityType
from src.schemas.schemas import (
    PromptTemplateCreate,
    PromptTemplateUpdate,
    PromptTemplateResponse,
    PaginationParams,
    ListResponse
)
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


# =============================================================================
# CRUD Endpoints
# =============================================================================

@router.post("", response_model=PromptTemplateResponse, status_code=status.HTTP_201_CREATED)
async def create_prompt_template(
    data: PromptTemplateCreate,
    db: AsyncSession = Depends(get_async_db)
):
    """Create a new prompt template"""
    logger.info(f"Creating prompt template: {data.name}")
    # Check if name already exists
    result = await db.execute(
        select(PromptTemplate).filter(PromptTemplate.name == data.name)
    )
    existing = result.scalars().first()

    if existing:
        logger.warning(f"Prompt template name already exists: {data.name}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prompt template with name '{data.name}' already exists"
        )

    # Create template
    template = PromptTemplate(
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

    db.add(template)
    await db.commit()
    await db.refresh(template)

    logger.info(f"Prompt template created: {template.id}")
    return template


@router.get("", response_model=ListResponse)
async def list_prompt_templates(
    category: Optional[str] = Query(None, description="Filter by category"),
    visibility: Optional[VisibilityType] = Query(None, description="Filter by visibility"),
    search: Optional[str] = Query(None, description="Search in name/description"),
    is_active: bool = Query(True, description="Filter by active status"),
    pagination: PaginationParams = Depends(),
    db: AsyncSession = Depends(get_async_db)
):
    """List prompt templates with filters"""
    query = select(PromptTemplate)

    # Apply filters
    if category:
        query = query.filter(PromptTemplate.category == category)

    if visibility:
        query = query.filter(PromptTemplate.visibility == visibility)

    if is_active is not None:
        query = query.filter(PromptTemplate.is_active == is_active)

    if search:
        query = query.filter(
            or_(
                PromptTemplate.name.ilike(f"%{search}%"),
                PromptTemplate.description.ilike(f"%{search}%")
            )
        )

    # Get total count
    total = await db.scalar(select(func.count()).select_from(query.subquery()))

    # Get templates with pagination
    result = await db.execute(
        query
        .order_by(PromptTemplate.created_at.desc())
        .offset(pagination.skip)
        .limit(pagination.limit)
    )
    templates = result.scalars().all()

    logger.debug(f"Retrieved {len(templates)} prompt templates")
    return ListResponse(
        items=[PromptTemplateResponse.model_validate(t) for t in templates],
        total=total,
        skip=pagination.skip,
        limit=pagination.limit
    )


@router.get("/categories", response_model=List[str])
async def get_categories(db: AsyncSession = Depends(get_async_db)):
    """Get list of all categories"""
    result = await db.execute(
        select(PromptTemplate.category)
        .filter(PromptTemplate.is_active == True)
        .distinct()
    )
    categories = result.all()

    return [cat[0] for cat in categories]


@router.get("/{template_id}", response_model=PromptTemplateResponse)
async def get_prompt_template(
    template_id: UUID,
    db: AsyncSession = Depends(get_async_db)
):
    """Get a specific prompt template"""
    result = await db.execute(
        select(PromptTemplate).filter(PromptTemplate.id == template_id)
    )
    template = result.scalars().first()

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
    db: AsyncSession = Depends(get_async_db)
):
    """Update a prompt template"""
    result = await db.execute(
        select(PromptTemplate).filter(PromptTemplate.id == template_id)
    )
    template = result.scalars().first()

    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prompt template not found"
        )

    # Check name uniqueness if changing
    if data.name and data.name != template.name:
        result = await db.execute(
            select(PromptTemplate).filter(
                PromptTemplate.name == data.name,
                PromptTemplate.id != template_id
            )
        )
        existing = result.scalars().first()

        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Prompt template with name '{data.name}' already exists"
            )

    # Update fields
    update_data = data.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        if field == "variables" and value is not None:
            # value is already a list of dicts after model_dump()
            setattr(template, field, value)
        elif field == "settings" and value is not None:
            # value is already a dict after model_dump()
            setattr(template, field, value)
        else:
            setattr(template, field, value)

    # Increment version
    template.version += 1

    await db.commit()
    await db.refresh(template)

    logger.info(f"Prompt template updated: {template_id}, new version: {template.version}")
    return template


@router.delete("/{template_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_prompt_template(
    template_id: UUID,
    hard_delete: bool = Query(False, description="Permanently delete (vs soft delete)"),
    db: AsyncSession = Depends(get_async_db)
):
    """Delete or deactivate a prompt template"""
    result = await db.execute(
        select(PromptTemplate).filter(PromptTemplate.id == template_id)
    )
    template = result.scalars().first()

    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prompt template not found"
        )

    if hard_delete:
        # Check if template is in use
        if template.conversations:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete template that is in use by conversations"
            )
        await db.delete(template)
    else:
        # Soft delete
        template.is_active = False

    await db.commit()
    logger.info(f"Prompt template {'deleted' if hard_delete else 'deactivated'}: {template_id}")
    return None


# =============================================================================
# Special Operations
# =============================================================================

@router.post("/{template_id}/duplicate", response_model=PromptTemplateResponse)
async def duplicate_prompt_template(
    template_id: UUID,
    new_name: Optional[str] = Query(None, description="Name for duplicated template"),
    db: AsyncSession = Depends(get_async_db)
):
    """Duplicate an existing prompt template"""
    result = await db.execute(
        select(PromptTemplate).filter(PromptTemplate.id == template_id)
    )
    original = result.scalars().first()

    if not original:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prompt template not found"
        )

    # Generate new name
    if not new_name:
        new_name = f"{original.name} (Copy)"

    # Check uniqueness
    result = await db.execute(
        select(PromptTemplate).filter(PromptTemplate.name == new_name)
    )
    existing = result.scalars().first()

    if existing:
        # Add number suffix
        counter = 1
        while existing:
            new_name = f"{original.name} (Copy {counter})"
            result = await db.execute(
                select(PromptTemplate).filter(PromptTemplate.name == new_name)
            )
            existing = result.scalars().first()
            counter += 1

    # Create duplicate
    duplicate = PromptTemplate(
        name=new_name,
        description=original.description,
        category=original.category,
        visibility=VisibilityType.PRIVATE,  # Duplicates are private by default
        system_prompt=original.system_prompt,
        user_prompt_template=original.user_prompt_template,
        variables=original.variables.copy(),
        settings=original.settings.copy(),
        version=1
    )

    db.add(duplicate)
    await db.commit()
    await db.refresh(duplicate)

    return duplicate


@router.get("/{template_id}/export")
async def export_prompt_template(
    template_id: UUID,
    db: AsyncSession = Depends(get_async_db)
):
    """Export prompt template as JSON"""
    result = await db.execute(
        select(PromptTemplate).filter(PromptTemplate.id == template_id)
    )
    template = result.scalars().first()

    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prompt template not found"
        )

    # Convert to exportable format
    export_data = {
        "name": template.name,
        "description": template.description,
        "category": template.category,
        "system_prompt": template.system_prompt,
        "user_prompt_template": template.user_prompt_template,
        "variables": template.variables,
        "settings": template.settings,
        "version": template.version
    }

    return export_data


@router.post("/import", response_model=PromptTemplateResponse)
async def import_prompt_template(
    data: dict,
    db: AsyncSession = Depends(get_async_db)
):
    """Import a prompt template from JSON"""
    # Validate required fields
    required_fields = ["name", "category", "system_prompt"]
    missing = [f for f in required_fields if f not in data]

    if missing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required fields: {', '.join(missing)}"
        )

    # Check if name exists
    result = await db.execute(
        select(PromptTemplate).filter(PromptTemplate.name == data["name"])
    )
    existing = result.scalars().first()

    if existing:
        # Add suffix
        data["name"] = f"{data['name']} (Imported)"

    # Create template
    template = PromptTemplate(
        name=data["name"],
        description=data.get("description"),
        category=data["category"],
        visibility=VisibilityType.PRIVATE,
        system_prompt=data["system_prompt"],
        user_prompt_template=data.get("user_prompt_template"),
        variables=data.get("variables", []),
        settings=data.get("settings", {}),
        version=1
    )

    db.add(template)
    await db.commit()
    await db.refresh(template)

    return template
