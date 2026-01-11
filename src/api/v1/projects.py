# =============================================================================
# api/v1/projects.py
# Projects API endpoints
# =============================================================================
"""
API endpoints for managing Projects
"""
from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query
from sqlalchemy import select, func, desc, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.database.connection import get_async_db
from src.models.models import Project, Conversation, File as FileModel, ProcessingStatus
from src.schemas.schemas import (
    ProjectCreate, ProjectUpdate, ProjectResponse,
    ConversationResponse, FileResponse as FileResponseSchema,
    PaginationParams, ListResponse
)
from src.utils.logger import get_logger

# Import upload logic reuse (optional, or implement directly)
from src.api.v1.files import upload_file

logger = get_logger(__name__)
router = APIRouter()


# =============================================================================
# CRUD Operations
# =============================================================================

@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    data: ProjectCreate,
    db: AsyncSession = Depends(get_async_db)
):
    """Create a new project"""
    logger.info("Creating new project", extra={"project_name": data.name})

    project = Project(
        name=data.name,
        description=data.description
    )

    db.add(project)
    await db.commit()
    await db.refresh(project)

    logger.info("Project created successfully", extra={"project_id": str(project.id)})
    return project


@router.get("", response_model=ListResponse)
async def list_projects(
    pagination: PaginationParams = Depends(),
    db: AsyncSession = Depends(get_async_db)
):
    """List all projects"""
    # Get total
    total = await db.scalar(select(func.count(Project.id)))

    # Get items
    query = (
        select(Project)
        .order_by(desc(Project.updated_at))
        .offset(pagination.skip)
        .limit(pagination.limit)
    )
    
    result = await db.execute(query)
    projects = result.scalars().all()

    return ListResponse(
        items=[ProjectResponse.model_validate(p) for p in projects],
        total=total,
        skip=pagination.skip,
        limit=pagination.limit
    )


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: UUID,
    db: AsyncSession = Depends(get_async_db)
):
    """Get a specific project"""
    result = await db.execute(select(Project).filter(Project.id == project_id))
    project = result.scalars().first()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )

    return project


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: UUID,
    data: ProjectUpdate,
    db: AsyncSession = Depends(get_async_db)
):
    """Update a project"""
    result = await db.execute(select(Project).filter(Project.id == project_id))
    project = result.scalars().first()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )

    if data.name is not None:
        project.name = data.name
    if data.description is not None:
        project.description = data.description
    if data.is_active is not None:
        project.is_active = data.is_active

    await db.commit()
    await db.refresh(project)

    return project


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: UUID,
    db: AsyncSession = Depends(get_async_db)
):
    """Delete a project (cascades to files/conversations)"""
    result = await db.execute(select(Project).filter(Project.id == project_id))
    project = result.scalars().first()

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )

    # Cascade delete: Delete chunks for all files in the project
    try:
        # Get all files for this project
        from src.services.file_processor import FileProcessor
        
        files_result = await db.execute(select(FileModel).filter(FileModel.project_id == project_id))
        files = files_result.scalars().all()
        
        processor = FileProcessor(db)
        
        # Delete chunks for each file
        for file in files:
            try:
                await processor.delete_file_chunks(file.id)
            except Exception as e:
                logger.error(f"Failed to delete chunks for file {file.id} in project {project_id}: {e}")
                
    except Exception as e:
        logger.error(f"Error during project deletion cascade: {e}")

    await db.delete(project)
    await db.commit()
    return None


# =============================================================================
# Project Resources
# =============================================================================

@router.get("/{project_id}/conversations", response_model=ListResponse)
async def list_project_conversations(
    project_id: UUID,
    pagination: PaginationParams = Depends(),
    db: AsyncSession = Depends(get_async_db)
):
    """List conversations in a project"""
    # Check project exists
    result = await db.execute(select(Project).filter(Project.id == project_id))
    if not result.scalars().first():
        raise HTTPException(status_code=404, detail="Project not found")

    # Get total
    total = await db.scalar(
        select(func.count(Conversation.id))
        .filter(Conversation.project_id == project_id)
    )

    # Get items
    query = (
        select(Conversation)
        .filter(Conversation.project_id == project_id)
        .order_by(desc(Conversation.updated_at))
        .offset(pagination.skip)
        .limit(pagination.limit)
    )
    
    result = await db.execute(query)
    conversations = result.scalars().all()
    
    # We use a Simplified Conversation Response manually or reuse valid schema
    # For now reusing ConversationResponse but note it sends full details
    # Ideally use ConversationListItem logic
    items = []
    for conv in conversations:
        items.append(ConversationResponse.model_validate(conv))

    return ListResponse(
        items=items,
        total=total,
        skip=pagination.skip,
        limit=pagination.limit
    )


@router.post("/{project_id}/files", response_model=FileResponseSchema)
async def upload_project_file(
    project_id: UUID,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Upload a file to a project.
    
    Reuses the core upload logic from files.py but sets project_id explicitly.
    Note: file.py:upload_file endpoint checks for conversation_id. 
    Here we need to adapt it. 
    Instead of calling the endpoint function directly (which expects Query params),
    we'll reimplement the core logic or call a service if we had one extracted.
    
    Since upload_file in files.py is an endpoint function, calling it directly is tricky
    due to Dependency Injection. We'll duplicate the upload logic slightly for now 
    to ensure clean project_id handling, or better, we modify files.py to be reusable.
    
    For now, let's call the `upload_file` endpoint logic by reusing the implementation
    but passing project_id as meta context.
    
    Actually, to keep it clean, let's just make a service call or direct implementation.
    The implementation in files.py is coupled with HTTPExceptions.
    """
    # Check project
    result = await db.execute(select(Project).filter(Project.id == project_id))
    if not result.scalars().first():
        raise HTTPException(status_code=404, detail="Project not found")
        
    # We will use the files.py router dependency - but we can't easily.
    # We will invoke the upload_file function from files.py?
    # No, FastAPI doesn't support calling other view functions easily.
    
    # We'll implement the upload logic here to be safe and explicit.
    # This avoids import cycles and dependency injection mess.
    
    from src.api.v1.files import validate_file, get_file_storage_path, calculate_file_hash, redis_client
    from src.config.settings import settings
    from src.services.file_processor import process_file_async, FileProcessingError
    from pathlib import Path
    
    logger.info("Uploading file to project", extra={"project_id": str(project_id), "file_name": file.filename})

    validate_file(file)
    
    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
        
    file_hash = calculate_file_hash(content)
    
    # Check duplicate in project
    query = select(FileModel).filter(
        func.jsonb_extract_path_text(FileModel.extra_metadata, 'hash') == file_hash,
        FileModel.project_id == project_id
    )
    result = await db.execute(query)
    existing = result.scalars().first()
    
    if existing:
        return existing
        
    # Create file record
    file_record = FileModel(
        project_id=project_id,  # LINK TO PROJECT
        file_name=file.filename,
        file_type=Path(file.filename).suffix.lower(),
        file_size=len(content),
        storage_path="",
        mime_type=file.content_type,
        processed=False,
        processing_status=ProcessingStatus.PENDING,
        extra_metadata={"hash": file_hash, "original_name": file.filename}
    )
    
    db.add(file_record)
    await db.flush()
    
    # Save to disk
    storage_path = get_file_storage_path(file_record.id, file.filename)
    
    total_size = len(content)
    chunk_size = 1024 * 1024
    uploaded_size = 0
    
    with open(storage_path, "wb") as f:
        for chunk in [content[i:i + chunk_size] for i in range(0, total_size, chunk_size)]:
            f.write(chunk)
            uploaded_size += len(chunk)
            progress = int((uploaded_size / total_size) * 100)
            redis_client.set(str(file_record.id), progress)
            
    file_record.storage_path = str(storage_path)
    await db.commit()
    await db.refresh(file_record)
    
    # Auto-process
    if settings.AUTO_PROCESS_FILES:
        try:
            await process_file_async(file_record.id, db)
        except Exception as e:
            logger.error(f"Auto-process failed: {e}")
            # Don't fail the upload request
            
    return file_record


@router.get("/{project_id}/files", response_model=ListResponse)
async def list_project_files(
    project_id: UUID,
    pagination: PaginationParams = Depends(),
    db: AsyncSession = Depends(get_async_db)
):
    """List files in a project"""
    # Check project exists
    result = await db.execute(select(Project).filter(Project.id == project_id))
    if not result.scalars().first():
        raise HTTPException(status_code=404, detail="Project not found")

    # Get total
    total = await db.scalar(
        select(func.count(FileModel.id))
        .filter(FileModel.project_id == project_id)
    )

    # Get items
    query = (
        select(FileModel)
        .filter(FileModel.project_id == project_id)
        .order_by(desc(FileModel.uploaded_at))
        .offset(pagination.skip)
        .limit(pagination.limit)
    )
    
    result = await db.execute(query)
    files = result.scalars().all()

    return ListResponse(
        items=[FileResponseSchema.model_validate(f) for f in files],
        total=total,
        skip=pagination.skip,
        limit=pagination.limit
    )
