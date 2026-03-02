# =============================================================================
# api/v1/collections.py
# Qdrant Collections API endpoints
# =============================================================================
"""
API endpoints para gestión de colecciones Qdrant
"""
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, status
from src.dependencies import get_collection_service
from src.models.models import VisibilityType
from src.schemas.schemas import (
    QdrantCollectionCreate,
    QdrantCollectionUpdate,
    QdrantCollectionResponse,
    PaginationParams,
    ListResponse
)
from src.services.collection_service import CollectionService
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


# =============================================================================
# CRUD Endpoints
# =============================================================================

@router.post("", response_model=QdrantCollectionResponse, status_code=status.HTTP_201_CREATED)
async def create_collection(
    data: QdrantCollectionCreate,
    create_in_qdrant: bool = Query(True, description="Create collection in Qdrant"),
    vector_size: Optional[int] = Query(None, description="Vector dimension (auto-detected if not provided)"),
    service: CollectionService = Depends(get_collection_service)
):
    """Create a new Qdrant collection registry"""
    try:
        collection = await service.create_collection(data, create_in_qdrant, vector_size)
        return collection
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("", response_model=ListResponse)
async def list_collections(
    category: Optional[str] = Query(None, description="Filter by category"),
    visibility: Optional[VisibilityType] = Query(None, description="Filter by visibility"),
    is_active: bool = Query(True, description="Filter by active status"),
    search: Optional[str] = Query(None, description="Search in name/description"),
    pagination: PaginationParams = Depends(),
    service: CollectionService = Depends(get_collection_service)
):
    """
    List ALL Qdrant collections (registered and unregistered).
    Returns collections from Qdrant with metadata from DB if registered.
    """
    return await service.list_collections(
        category=category,
        visibility=visibility,
        is_active=is_active,
        search=search,
        skip=pagination.skip,
        limit=pagination.limit
    )


@router.get("/categories", response_model=List[str])
async def get_categories(
    service: CollectionService = Depends(get_collection_service)
):
    """Get list of categories"""
    return await service.get_categories()


@router.get("/{collection_id}", response_model=QdrantCollectionResponse)
async def get_collection(
    collection_id: UUID,
    service: CollectionService = Depends(get_collection_service)
):
    """Get a specific collection"""
    collection = await service.get_collection(collection_id)
    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found"
        )
    return collection


@router.patch("/{collection_id}", response_model=QdrantCollectionResponse)
async def update_collection(
    collection_id: UUID,
    data: QdrantCollectionUpdate,
    service: CollectionService = Depends(get_collection_service)
):
    """Update a collection"""
    try:
        collection = await service.update_collection(collection_id, data)
        if not collection:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Collection not found"
            )
        return collection
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_collection(
    collection_id: UUID,
    delete_from_qdrant: bool = Query(False, description="Also delete from Qdrant"),
    service: CollectionService = Depends(get_collection_service)
):
    """Delete a collection registry"""
    try:
        deleted = await service.delete_collection(collection_id, delete_from_qdrant)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Collection not found"
            )
        return None
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# =============================================================================
# Special Operations
# =============================================================================

@router.post("/{collection_id}/sync", response_model=QdrantCollectionResponse)
async def sync_collection(
    collection_id: UUID,
    service: CollectionService = Depends(get_collection_service)
):
    """Sync collection with Qdrant to update vector count"""
    try:
        collection = await service.sync_collection(collection_id)
        if not collection:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Collection not found"
            )
        return collection
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/{collection_id}/stats")
async def get_collection_stats(
    collection_id: UUID,
    service: CollectionService = Depends(get_collection_service)
):
    """Get detailed collection statistics from Qdrant"""
    try:
        stats = await service.get_collection_stats(collection_id)
        if not stats:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Collection not found"
            )
        return stats
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/{collection_id}/search")
async def search_in_collection(
    collection_id: UUID,
    query: str = Query(..., description="Search query"),
    limit: int = Query(5, ge=1, le=20),
    score_threshold: float = Query(0.5, ge=0.0, le=1.0),
    service: CollectionService = Depends(get_collection_service)
):
    """
    Search in a specific collection (for testing)
    This is a simplified version - production use should go through RAG tool
    """
    try:
        result = await service.search_in_collection(
            collection_id, query, limit, score_threshold
        )
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Collection not found"
            )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
