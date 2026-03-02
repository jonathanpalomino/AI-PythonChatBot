# =============================================================================
# src/api/v1/projects.py
# Projects Controller - HTTP Layer
# =============================================================================
"""
REST API endpoints for Project management.

Following Spring Boot @RestController pattern:
- Thin controller (no business logic)
- HTTP concerns only (request/response)
- Delegates to service layer
- Input validation via Pydantic
"""

from uuid import UUID

from fastapi import APIRouter, Depends, File, UploadFile, status

from src.dependencies import get_project_service, get_redis_client
from src.schemas.schemas import (
    ConversationResponse,
    FileResponse,
    ListResponse,
    PaginationParams,
    ProjectCreate,
    ProjectResponse,
    ProjectUpdate,
)
from src.services.project_service import ProjectService
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


# =============================================================================
# CRUD Operations
# =============================================================================


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    data: ProjectCreate,
    service: ProjectService = Depends(get_project_service),
):
    """
    Create a new project.
    
    Args:
        data: Project creation data (name, description)
        service: Injected project service
        
    Returns:
        Created project
    """
    project = await service.create_project(data)
    return project


@router.get("", response_model=ListResponse)
async def list_projects(
    pagination: PaginationParams = Depends(),
    service: ProjectService = Depends(get_project_service),
):
    """
    List all projects with pagination.
    
    Args:
        pagination: Pagination parameters (skip, limit)
        service: Injected project service
        
    Returns:
        Paginated list of projects
    """
    projects, total = await service.list_projects(pagination)
    
    return ListResponse(
        items=[ProjectResponse.model_validate(p) for p in projects],
        total=total,
        skip=pagination.skip,
        limit=pagination.limit,
    )


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: UUID,
    service: ProjectService = Depends(get_project_service),
):
    """
    Get a specific project by ID.
    
    Args:
        project_id: Project UUID
        service: Injected project service
        
    Returns:
        Project details
        
    Raises:
        HTTPException: 404 if project not found
    """
    project = await service.get_project(project_id)
    if not project:
        from fastapi import HTTPException
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Project not found"
        )
    
    return project


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: UUID,
    data: ProjectUpdate,
    service: ProjectService = Depends(get_project_service),
):
    """
    Update a project.
    
    Args:
        project_id: Project UUID
        data: Update data (name, description, is_active)
        service: Injected project service
        
    Returns:
        Updated project
        
    Raises:
        HTTPException: 404 if project not found
    """
    project = await service.update_project(project_id, data)
    return project


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: UUID,
    service: ProjectService = Depends(get_project_service),
):
    """
    Delete a project (cascades to files and conversations).
    
    Args:
        project_id: Project UUID
        service: Injected project service
        
    Returns:
        None (204 No Content)
        
    Raises:
        HTTPException: 404 if project not found
    """
    await service.delete_project_cascade(project_id)
    return None


# =============================================================================
# Project Resources (Nested Endpoints)
# =============================================================================


@router.get("/{project_id}/conversations", response_model=ListResponse)
async def list_project_conversations(
    project_id: UUID,
    pagination: PaginationParams = Depends(),
    service: ProjectService = Depends(get_project_service),
):
    """
    List conversations in a project.
    
    Args:
        project_id: Project UUID
        pagination: Pagination parameters
        service: Injected project service
        
    Returns:
        Paginated list of conversations
        
    Raises:
        HTTPException: 404 if project not found
    """
    conversations, total = await service.list_project_conversations(
        project_id, pagination
    )
    
    return ListResponse(
        items=[ConversationResponse.model_validate(c) for c in conversations],
        total=total,
        skip=pagination.skip,
        limit=pagination.limit,
    )


@router.post("/{project_id}/files", response_model=FileResponse)
async def upload_project_file(
    project_id: UUID,
    file: UploadFile = File(...),
    service: ProjectService = Depends(get_project_service),
    redis_client=Depends(get_redis_client),
):
    """
    Upload a file to a project.
    
    Args:
        project_id: Project UUID
        file: File to upload
        service: Injected project service
        redis_client: Redis client for progress tracking
        
    Returns:
        Uploaded file record
        
    Raises:
        HTTPException: 404 if project not found, 400 if validation fails
    """
    file_record = await service.upload_file_to_project(
        project_id, file, redis_client
    )
    return file_record


@router.get("/{project_id}/files", response_model=ListResponse)
async def list_project_files(
    project_id: UUID,
    pagination: PaginationParams = Depends(),
    service: ProjectService = Depends(get_project_service),
):
    """
    List files in a project.
    
    Args:
        project_id: Project UUID
        pagination: Pagination parameters
        service: Injected project service
        
    Returns:
        Paginated list of files
        
    Raises:
        HTTPException: 404 if project not found
    """
    files, total = await service.list_project_files(project_id, pagination)
    
    return ListResponse(
        items=[FileResponse.model_validate(f) for f in files],
        total=total,
        skip=pagination.skip,
        limit=pagination.limit,
    )
