# =============================================================================
# api/v1/files.py
# File Management API endpoints
# =============================================================================
"""
API endpoints para gestión de archivos
"""
import hashlib
import asyncio
from pathlib import Path
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, status
from fastapi.responses import FileResponse
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import settings
from src.database.connection import get_async_db
from src.models.models import File as FileModel, ProcessingStatus, Conversation
from src.schemas.schemas import FileResponse as FileResponseSchema, PaginationParams, ListResponse
from src.services.file_processor import process_file_async, FileProcessingError, FileProcessor
from src.utils.logger import get_logger, set_conversation_context

logger = get_logger(__name__)
router = APIRouter()

# Configurar Redis para almacenar el progreso de la carga
import redis

# Initialize Redis client using REDIS_URL from settings
redis_client = redis.Redis.from_url(settings.REDIS_URL)


# =============================================================================
# Configuration Endpoints
# =============================================================================

@router.get("/supported-extensions")
async def get_supported_extensions():
    """
    Get list of supported file extensions and their corresponding loaders

    Returns:
        dict: {
            'loaders': [...],
            'all_extensions': [...]
        }
    """
    from src.document_loaders import DocumentLoaderFactory

    return DocumentLoaderFactory.get_loader_info()


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_file_hash(content: bytes) -> str:
    """Calculate SHA256 hash of file content"""
    return hashlib.sha256(content).hexdigest()


def get_file_storage_path(file_id: UUID, filename: str) -> Path:
    """Generate storage path for file"""
    # Organize by first 2 chars of UUID for distribution
    subdir = str(file_id)[:2]
    storage_dir = settings.UPLOAD_DIR / subdir
    storage_dir.mkdir(parents=True, exist_ok=True)

    # Use UUID as filename to avoid collisions
    extension = Path(filename).suffix
    return storage_dir / f"{file_id}{extension}"


def validate_file(file: UploadFile) -> None:
    """Validate uploaded file"""
    # Import here to avoid circular dependency
    from src.document_loaders import DocumentLoaderFactory

    # Check extension
    if file.filename:
        extension = Path(file.filename).suffix.lower()
        allowed_extensions = DocumentLoaderFactory.get_supported_extensions()

        if extension not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type {extension} not allowed. Allowed: {', '.join(sorted(allowed_extensions))}"
            )

    # Size check will be done during upload


# =============================================================================
# Upload & Download
# =============================================================================

@router.post("/upload", response_model=FileResponseSchema, status_code=status.HTTP_201_CREATED)
async def upload_file(
    file: UploadFile = File(...),
    conversation_id: Optional[UUID] = Query(None, description="Associate with conversation"),
    db: AsyncSession = Depends(get_async_db)
):
    """Upload a file"""
    logger.info(
        "Uploading file",
        extra={"file_name": file.filename,
               "conversation_id": str(conversation_id) if conversation_id else None}
    )
    if conversation_id:
        set_conversation_context(str(conversation_id))

    # Validate file
    validate_file(file)

    # Validate conversation if provided
    if conversation_id:
        result = await db.execute(
            select(Conversation).filter(Conversation.id == conversation_id)
        )
        conversation = result.scalars().first()
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )

    try:
        # Read file content
        content = await file.read()

        # Check size
        if len(content) > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Max size: {settings.MAX_UPLOAD_SIZE / 1024 / 1024:.1f}MB"
            )

        # Calculate hash
        file_hash = calculate_file_hash(content)

        # Check for duplicate in the SAME conversation
        # Allow same file in different conversations (for temporary collections)
        # Check for duplicate in the SAME conversation
        # Allow same file in different conversations (for temporary collections)
        query = select(FileModel).filter(
            func.jsonb_extract_path_text(FileModel.extra_metadata, 'hash') == file_hash
        )

        if conversation_id:
            # If uploading to a conversation, check duplicate in that conversation only
            query = query.filter(
                FileModel.conversation_id == conversation_id
            )

        result = await db.execute(query)
        existing = result.scalars().first()

        if existing:
            # File already exists in this conversation
            logger.info(f"File already exists, returning existing: {existing.id}")
            return existing

        # Create file record
        file_record = FileModel(
            conversation_id=conversation_id,
            file_name=file.filename,
            file_type=Path(file.filename).suffix.lower(),
            file_size=len(content),
            storage_path="",  # Will be set after saving
            mime_type=file.content_type,
            processed=False,
            processing_status=ProcessingStatus.PENDING,
            extra_metadata={"hash": file_hash, "original_name": file.filename}
        )

        db.add(file_record)
        await db.flush()  # Get ID

        # Save file to disk
        storage_path = get_file_storage_path(file_record.id, file.filename)
        
        # Simular progreso de carga
        total_size = len(content)
        chunk_size = 1024 * 1024  # 1MB
        uploaded_size = 0
        
        with open(storage_path, "wb") as f:
            for chunk in [content[i:i + chunk_size] for i in range(0, total_size, chunk_size)]:
                f.write(chunk)
                uploaded_size += len(chunk)
                progress = int((uploaded_size / total_size) * 100)
                redis_client.set(str(file_record.id), progress)

        # Update storage path
        file_record.storage_path = str(storage_path)

        await db.commit()
        await db.refresh(file_record)

        # Auto-process file if enabled
        if settings.AUTO_PROCESS_FILES:
            try:
                print(f"🔄 Auto-processing file: {file.filename}")
                process_result = await process_file_async(file_record.id, db)
                logger.info(f"File auto-processed: {process_result['chunks']} chunks indexed")
                print(f"✅ File processed: {process_result['chunks']} chunks indexed")
            except FileProcessingError as e:
                logger.error(f"File processing failed: {e}", exc_info=True)
                print(f"⚠️  File uploaded but processing failed: {e}")
                # Processing failed - rollback file record and disk file
                print(f"🔄 Rolling back failed upload...")
                try:
                    # Delete from disk
                    if storage_path.exists():
                        storage_path.unlink()
                    # Delete from database
                    await db.delete(file_record)
                    await db.commit()
                    print(f"✅ Rollback completed")
                except Exception as cleanup_error:
                    print(f"❌ Rollback failed: {cleanup_error}")

                # Re-raise the original error so user knows upload failed
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"File processing failed: {str(e)}"
                )

        logger.info(
            "File uploaded successfully",
            extra={"file_id": str(file_record.id), "size_bytes": file_record.file_size}
        )
        return file_record

    except HTTPException:
        # Re-raise HTTP exceptions (like processing failure above)
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to upload file: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}"
        )

@router.get("/{file_id}/upload-progress")
async def get_upload_progress(
    file_id: UUID
):
    """Get upload progress for a file"""
    progress = redis_client.get(str(file_id))
    progress = int(progress) if progress else 0
    
    # Check processing progress
    processing_progress = redis_client.get(f"processing:{file_id}")
    processing_progress = int(processing_progress) if processing_progress else 0
    
    # Determine overall status and combined progress
    status_category = "uploading"
    if progress >= 100:
        if processing_progress >= 100:
            status_category = "completed"
        elif processing_progress > 0:
            status_category = "processing"
        else:
             # Upload done, processing maybe pending or about to start
             status_category = "pending_processing"

    return {
        "file_id": str(file_id), 
        "upload_progress": progress,
        "processing_progress": processing_progress,
        "status": status_category
    }


@router.get("/{file_id}/download")
async def download_file(
    file_id: UUID,
    db: AsyncSession = Depends(get_async_db)
):
    """Download a file"""
    result = await db.execute(
        select(FileModel).filter(FileModel.id == file_id)
    )
    file_record = result.scalars().first()

    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found"
        )

    storage_path = Path(file_record.storage_path)

    if not storage_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found on disk"
        )

    return FileResponse(
        path=storage_path,
        filename=file_record.file_name,
        media_type=file_record.mime_type
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
    db: AsyncSession = Depends(get_async_db)
):
    """List files"""
    query = select(FileModel)

    # Apply filters
    if conversation_id:
        query = query.filter(FileModel.conversation_id == conversation_id)

    if file_type:
        query = query.filter(FileModel.file_type == file_type)

    if processed is not None:
        query = query.filter(FileModel.processed == processed)

    # Get total
    total = await db.scalar(select(func.count()).select_from(query.subquery()))

    # Get files
    result = await db.execute(
        query
        .order_by(FileModel.uploaded_at.desc())
        .offset(pagination.skip)
        .limit(pagination.limit)
    )
    files = result.scalars().all()

    return ListResponse(
        items=[FileResponseSchema.model_validate(f) for f in files],
        total=total,
        skip=pagination.skip,
        limit=pagination.limit
    )


@router.get("/{file_id}", response_model=FileResponseSchema)
async def get_file(
    file_id: UUID,
    db: AsyncSession = Depends(get_async_db)
):
    """Get file information"""
    result = await db.execute(
        select(FileModel).filter(FileModel.id == file_id)
    )
    file_record = result.scalars().first()

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
    db: AsyncSession = Depends(get_async_db)
):
    """Delete a file"""
    result = await db.execute(
        select(FileModel).filter(FileModel.id == file_id)
    )
    file_record = result.scalars().first()

    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found"
        )

    # Delete chunks from Qdrant
    try:
        processor = FileProcessor(db)
        await processor.delete_file_chunks(file_id)
    except Exception as e:
        logger.error(f"Failed to delete chunks for file {file_id}: {e}")
        # Continue with file deletion even if chunk deletion fails

    # Delete from disk
    if delete_from_disk:
        storage_path = Path(file_record.storage_path)
        if storage_path.exists():
            storage_path.unlink()

    # Delete from database
    await db.delete(file_record)
    await db.commit()

    return None


# =============================================================================
# Processing
# =============================================================================

@router.post("/{file_id}/process", response_model=FileResponseSchema)
async def trigger_file_processing(
    file_id: UUID,
    db: AsyncSession = Depends(get_async_db)
):
    """
    Trigger file processing (extraction, embedding, etc.)
    """
    result = await db.execute(
        select(FileModel).filter(FileModel.id == file_id)
    )
    file_record = result.scalars().first()

    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found"
        )

    if file_record.processed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File already processed"
        )

    # Trigger file processing
    try:
        logger.info(f"Triggering file processing: {file_id}")
        result = await process_file_async(file_id, db)
        await db.refresh(file_record)
        logger.info(f"File processing completed: {result['chunks']} chunks")
        return file_record
    except FileProcessingError as e:
        logger.error(f"File processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/{file_id}/content")
async def get_file_content(
    file_id: UUID,
    db: AsyncSession = Depends(get_async_db)
):
    """
    Get extracted content from a processed file
    """
    result = await db.execute(
        select(FileModel).filter(FileModel.id == file_id)
    )
    file_record = result.scalars().first()

    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found"
        )

    if not file_record.processed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File not yet processed"
        )

    extracted_text = file_record.extra_metadata.get("extracted_text")

    if not extracted_text:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No extracted content available"
        )

    return {
        "file_id": file_record.id,
        "file_name": file_record.file_name,
        "content": extracted_text,
        "language": file_record.extra_metadata.get("language"),
        "analysis": file_record.extra_metadata.get("analysis_result"),
        "code_analysis": file_record.extra_metadata.get("code_analysis")
        # Include code analysis if available
    }


# =============================================================================
# Batch Operations
# =============================================================================

@router.post("/batch/upload", response_model=List[FileResponseSchema])
async def batch_upload_files(
    files: List[UploadFile] = File(...),
    conversation_id: Optional[UUID] = Query(None),
    db: AsyncSession = Depends(get_async_db)
):
    """Upload multiple files at once"""
    uploaded_files = []
    tasks = []

    # Create tasks for parallel execution
    for file in files:
        # We wrap the call to capture exceptions per file
        async def upload_wrapper(f):
            try:
                return await upload_file(f, conversation_id, db)
            except Exception as e:
                print(f"Failed to upload {f.filename}: {e}")
                return None

        tasks.append(upload_wrapper(file))

    # Run in parallel
    results = await asyncio.gather(*tasks)
    
    # Filter successful uploads
    uploaded_files = [r for r in results if r is not None]

    return uploaded_files


@router.delete("/batch/delete", status_code=status.HTTP_204_NO_CONTENT)
async def batch_delete_files(
    file_ids: List[UUID],
    delete_from_disk: bool = Query(True),
    db: AsyncSession = Depends(get_async_db)
):
    """Delete multiple files"""
    tasks = []
    
    for file_id in file_ids:
        async def delete_wrapper(fid):
            try:
                await delete_file(fid, delete_from_disk, db)
            except:
                pass
        
        tasks.append(delete_wrapper(file_id))
        
    await asyncio.gather(*tasks)

    return None
