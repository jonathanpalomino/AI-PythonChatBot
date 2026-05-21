# =============================================================================
# src/api/v1/collections_ingest.py
# Files Collections API - Administrative Ingestion
# =============================================================================
"""
API endpoints for administrative file ingestion into collections.
"""
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form

from src.dependencies import get_collection_ingest_service
from src.schemas.schemas import FolderIngestRequest, IngestionStats
from src.services.collection_ingest_service import CollectionIngestService
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

@router.post("/{collection_id}/ingest/folder", response_model=IngestionStats)
async def ingest_folder(
    collection_id: str,
    request: FolderIngestRequest,
    service: CollectionIngestService = Depends(get_collection_ingest_service)
):
    """
    Ingest a local folder into a Qdrant collection.
    Supports recursive scanning and incremental updates.
    """
    try:
        stats = await service.ingest_local_folder(
            collection_id=collection_id,
            folder_path=request.folder_path,
            recursive=request.recursive,
            embedding_model=request.embedding_model
        )
        return stats
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error during folder ingestion: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during ingestion."
        )

@router.post("/{collection_id}/ingest/upload", response_model=IngestionStats)
async def upload_files(
    collection_id: str,
    files: List[UploadFile] = File(...),
    embedding_model: Optional[str] = Form(None),
    service: CollectionIngestService = Depends(get_collection_ingest_service)
):
    """
    Ingest multiple uploaded files into a Qdrant collection.
    Ideal for distributed environments where client has the files.
    """
    try:
        stats = await service.ingest_uploaded_files(
            collection_id=collection_id,
            files=files,
            embedding_model=embedding_model
        )
        return stats
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error during file upload ingestion: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during upload."
        )


@router.get("/{collection_id}/ingest/status/{job_id}")
async def get_ingest_status(
    collection_id: str,
    job_id: str,
    # In a real app, we'd fetch this from Redis via service
):
    """
    Get the status of a long-running ingestion job.
    (Currently placeholder for future expansion)
    """
    return {"status": "Not implemented - use synchronous call for now"}
