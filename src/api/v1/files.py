# =============================================================================
# api/v1/files.py
# File Management API endpoints
# =============================================================================
"""
API endpoints para gestión de archivos
"""
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, status
from fastapi.responses import FileResponse

from src.dependencies import get_file_service
from src.schemas.schemas import FileResponse as FileResponseSchema, PaginationParams, ListResponse
from src.services.file_service import FileService
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


# =============================================================================
# Configuration Endpoints
# =============================================================================

@router.get("/supported-extensions")
async def get_supported_extensions():
    """
    Get list of supported file extensions and their corresponding loaders
    """
    from src.document_loaders import DocumentLoaderFactory
    return DocumentLoaderFactory.get_loader_info()


# =============================================================================
# Upload & Download
# =============================================================================

@router.post("/upload", response_model=FileResponseSchema, status_code=status.HTTP_201_CREATED)
async def upload_file(
    file: UploadFile = File(...),
    conversation_id: Optional[UUID] = Query(None, description="Associate with conversation"),
    async_processing: bool = Query(True, description="Process asynchronously (default: true)"),
    service: FileService = Depends(get_file_service)
):
    """Upload a file with async processing"""
    try:
        file_record = await service.upload_file(file, conversation_id, async_processing)
        return file_record
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to upload file: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}"
        )


@router.get("/{file_id}/upload-progress")
async def get_upload_progress(
    file_id: UUID,
    service: FileService = Depends(get_file_service)
):
    """Get upload progress for a file"""
    return await service.get_upload_progress(file_id)


@router.get("/{file_id}/status")
async def get_file_processing_status(
    file_id: UUID,
    service: FileService = Depends(get_file_service)
):
    """Get detailed processing status for a file"""

    file_record = await service.get_file(file_id)
    if not file_record:
        raise HTTPException(status_code=404, detail="File not found")

    response = {
        "file_id": str(file_id),
        "status": file_record.processing_status.value,
        "processed": file_record.processed,
        "file_name": file_record.file_name,
    }

    if file_record.extra_metadata.get("celery_task_id"):
        response["celery_task_id"] = file_record.extra_metadata["celery_task_id"]

    if file_record.extra_metadata.get("processing_error"):
        response["error"] = file_record.extra_metadata["processing_error"]

    if file_record.extra_metadata.get("chunks"):
        response["chunks"] = file_record.extra_metadata["chunks"]

    return response


@router.get("/{file_id}/download")
async def download_file(
    file_id: UUID,
    service: FileService = Depends(get_file_service)
):
    """Download a file"""
    try:
        storage_path, file_record = await service.download_file(file_id)
        return FileResponse(
            path=storage_path,
            filename=file_record.file_name,
            media_type=file_record.mime_type
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


# =============================================================================
# CRUD Operations
# =============================================================================

@router.get("", response_model=ListResponse)
async def list_files(
    conversation_id: Optional[UUID] = Query(None, description="Filter by conversation"),
    file_type: Optional[str] = Query(None, description="Filter by file type"),
    processed: Optional[bool] = Query(None, description="Filter by processed status"),
    pagination: PaginationParams = Depends(),
    service: FileService = Depends(get_file_service)
):
    """List files"""
    return await service.list_files(
        conversation_id=conversation_id,
        file_type=file_type,
        processed=processed,
        skip=pagination.skip,
        limit=pagination.limit
    )


@router.get("/{file_id}", response_model=FileResponseSchema)
async def get_file(
    file_id: UUID,
    service: FileService = Depends(get_file_service)
):
    """Get file information"""
    file_record = await service.get_file(file_id)
    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found"
        )
    return file_record


@router.delete("/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_file(
    file_id: UUID,
    delete_from_disk: bool = Query(True, description="Also delete from disk"),
    service: FileService = Depends(get_file_service)
):
    """Delete a file"""
    deleted = await service.delete_file(file_id, delete_from_disk)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found"
        )
    return None


# =============================================================================
# Processing
# =============================================================================

@router.post("/{file_id}/process", response_model=FileResponseSchema)
async def trigger_file_processing(
    file_id: UUID,
    service: FileService = Depends(get_file_service)
):
    """Trigger file processing (extraction, embedding, etc.)"""
    try:
        file_record = await service.process_file(file_id)
        return file_record
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST if "already processed" in str(e) else status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/{file_id}/content")
async def get_file_content(
    file_id: UUID,
    service: FileService = Depends(get_file_service)
):
    """Get extracted content from a processed file"""
    try:
        return await service.get_file_content(file_id)
    except ValueError as e:
        status_code = status.HTTP_404_NOT_FOUND
        if "not yet processed" in str(e):
            status_code = status.HTTP_400_BAD_REQUEST
        raise HTTPException(status_code=status_code, detail=str(e))


# =============================================================================
# Batch Operations
# =============================================================================

@router.post("/batch/upload", response_model=List[FileResponseSchema])
async def batch_upload_files(
    files: List[UploadFile] = File(...),
    conversation_id: Optional[UUID] = Query(None),
    service: FileService = Depends(get_file_service)
):
    """Upload multiple files at once"""
    uploaded_files = await service.batch_upload_files(files, conversation_id)
    return uploaded_files


@router.delete("/batch/delete", status_code=status.HTTP_204_NO_CONTENT)
async def batch_delete_files(
    file_ids: List[UUID],
    delete_from_disk: bool = Query(True),
    service: FileService = Depends(get_file_service)
):
    """Delete multiple files"""
    await service.batch_delete_files(file_ids, delete_from_disk)
    return None
