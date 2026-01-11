# =============================================================================
# api/v1/collections.py
# Qdrant Collections API endpoints
# =============================================================================
"""
API endpoints para gestión de colecciones Qdrant
"""
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import get_qdrant_config
from src.database.connection import get_async_db
from src.models.models import QdrantCollection, VisibilityType
from src.schemas.schemas import (
    QdrantCollectionCreate,
    QdrantCollectionUpdate,
    QdrantCollectionResponse,
    PaginationParams,
    ListResponse
)

router = APIRouter()


# =============================================================================
# Helper Functions
# =============================================================================

def get_qdrant_client():
    """Get Qdrant client instance"""
    config = get_qdrant_config()
    return QdrantClient(**config)


# =============================================================================
# CRUD Endpoints
# =============================================================================

@router.post("", response_model=QdrantCollectionResponse, status_code=status.HTTP_201_CREATED)
async def create_collection(
    data: QdrantCollectionCreate,
    create_in_qdrant: bool = Query(True, description="Create collection in Qdrant"),
    vector_size: Optional[int] = Query(None, description="Vector dimension (auto-detected if not provided)"),
    db: AsyncSession = Depends(get_async_db)
):
    """Create a new Qdrant collection registry"""
    # Check if name already exists
    result = await db.execute(
        select(QdrantCollection).filter(QdrantCollection.name == data.name)
    )
    existing = result.scalars().first()

    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Collection '{data.name}' already exists"
        )

    # Determine vector size if not provided
    if create_in_qdrant and vector_size is None:
        try:
            from src.services.embedding_service import EmbeddingService
            embedding_service = EmbeddingService(db)
            vector_size = await embedding_service.get_embedding_dimension(db=db)
        except Exception as e:
            # Fallback to default in settings
            config = get_qdrant_config() # This is just for logging/context
            from src.config.settings import settings
            vector_size = settings.VECTOR_SIZE

    # Create in Qdrant if requested
    if create_in_qdrant:
        try:
            qdrant = get_qdrant_client()

            # Check if collection exists in Qdrant
            try:
                qdrant.get_collection(data.name)
                # Collection exists, just register it
            except:
                # Collection doesn't exist, create it
                qdrant.create_collection(
                    collection_name=data.name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create collection in Qdrant: {str(e)}"
            )

    # Create registry entry
    collection = QdrantCollection(
        name=data.name,
        display_name=data.display_name,
        description=data.description,
        category=data.category,
        visibility=data.visibility,
        extra_metadata=data.metadata
    )

    db.add(collection)
    await db.commit()
    await db.refresh(collection)

    return collection


@router.get("", response_model=ListResponse)
async def list_collections(
    category: Optional[str] = Query(None, description="Filter by category"),
    visibility: Optional[VisibilityType] = Query(None, description="Filter by visibility"),
    is_active: bool = Query(True, description="Filter by active status"),
    search: Optional[str] = Query(None, description="Search in name/description"),
    pagination: PaginationParams = Depends(),
    db: AsyncSession = Depends(get_async_db)
):
    """List Qdrant collections (only those that exist physically in Qdrant)"""
    query = select(QdrantCollection)

    # Apply filters
    if category:
        query = query.filter(QdrantCollection.category == category)

    if visibility:
        query = query.filter(QdrantCollection.visibility == visibility)

    if is_active is not None:
        query = query.filter(QdrantCollection.is_active == is_active)

    if search:
        query = query.filter(
            or_(
                QdrantCollection.name.ilike(f"%{search}%"),
                QdrantCollection.display_name.ilike(f"%{search}%"),
                QdrantCollection.description.ilike(f"%{search}%")
            )
        )

    # Get collections from DB
    result = await db.execute(
        query
        .order_by(QdrantCollection.created_at.desc())
        .offset(pagination.skip)
        .limit(pagination.limit)
    )
    db_collections = result.scalars().all()
    qdrant = get_qdrant_client()
    qdrant_collections = {c.name for c in qdrant.get_collections().collections}

    # Filter only those that exist physically in Qdrant
    existing_collections = [c for c in db_collections if c.name in qdrant_collections]

    return ListResponse(
        items=[QdrantCollectionResponse.model_validate(c) for c in existing_collections],
        total=len(existing_collections),
        skip=pagination.skip,
        limit=pagination.limit
    )


@router.get("/categories", response_model=List[str])
async def get_categories(db: AsyncSession = Depends(get_async_db)):
    """Get list of categories"""
    result = await db.execute(
        select(QdrantCollection.category)
        .filter(QdrantCollection.is_active == True)
        .distinct()
    )
    categories = result.all()

    return [cat[0] for cat in categories if cat[0]]


@router.get("/{collection_id}", response_model=QdrantCollectionResponse)
async def get_collection(
    collection_id: UUID,
    db: AsyncSession = Depends(get_async_db)
):
    """Get a specific collection"""
    result = await db.execute(
        select(QdrantCollection).filter(QdrantCollection.id == collection_id)
    )
    collection = result.scalars().first()

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
    db: AsyncSession = Depends(get_async_db)
):
    """Update a collection"""
    result = await db.execute(
        select(QdrantCollection).filter(QdrantCollection.id == collection_id)
    )
    collection = result.scalars().first()

    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found"
        )

    # Update fields
    update_data = data.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        if field == "metadata" and value is not None:
            # Merge metadata
            current_meta = collection.extra_metadata or {}
            current_meta.update(value)
            setattr(collection, "extra_metadata", current_meta)
        else:
            # Map to correct attribute name
            if field == "metadata":
                field = "extra_metadata"
            setattr(collection, field, value)

    await db.commit()
    await db.refresh(collection)

    return collection


@router.delete("/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_collection(
    collection_id: UUID,
    delete_from_qdrant: bool = Query(False, description="Also delete from Qdrant"),
    db: AsyncSession = Depends(get_async_db)
):
    """Delete a collection registry"""
    result = await db.execute(
        select(QdrantCollection).filter(QdrantCollection.id == collection_id)
    )
    collection = result.scalars().first()

    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found"
        )

    # Delete from Qdrant if requested
    if delete_from_qdrant:
        try:
            qdrant = get_qdrant_client()
            qdrant.delete_collection(collection.name)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete from Qdrant: {str(e)}"
            )

    # Delete registry
    await db.delete(collection)
    await db.commit()

    return None


# =============================================================================
# Special Operations
# =============================================================================

@router.post("/{collection_id}/sync", response_model=QdrantCollectionResponse)
async def sync_collection(
    collection_id: UUID,
    db: AsyncSession = Depends(get_async_db)
):
    """
    Sync collection with Qdrant to update vector count
    """
    result = await db.execute(
        select(QdrantCollection).filter(QdrantCollection.id == collection_id)
    )
    collection = result.scalars().first()

    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found"
        )

    try:
        qdrant = get_qdrant_client()
        collection_info = qdrant.get_collection(collection.name)

        # Update stats
        collection.vector_count = collection_info.points_count
        collection.last_synced = datetime.now()

        await db.commit()
        await db.refresh(collection)

        return collection

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to sync with Qdrant: {str(e)}"
        )


@router.get("/{collection_id}/stats")
async def get_collection_stats(
    collection_id: UUID,
    db: AsyncSession = Depends(get_async_db)
):
    """Get detailed collection statistics from Qdrant"""
    result = await db.execute(
        select(QdrantCollection).filter(QdrantCollection.id == collection_id)
    )
    collection = result.scalars().first()

    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found"
        )

    try:
        qdrant = get_qdrant_client()
        collection_info = qdrant.get_collection(collection.name)

        return {
            "name": collection.name,
            "display_name": collection.display_name,
            "points_count": collection_info.points_count,
            "segments_count": collection_info.segments_count,
            "status": collection_info.status,
            "optimizer_status": collection_info.optimizer_status,
            "vectors_config": {
                "size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance.value
            },
            "last_synced": collection.last_synced
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats from Qdrant: {str(e)}"
        )


@router.post("/{collection_id}/search")
async def search_in_collection(
    collection_id: UUID,
    query: str = Query(..., description="Search query"),
    limit: int = Query(5, ge=1, le=20),
    score_threshold: float = Query(0.5, ge=0.0, le=1.0),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Search in a specific collection (for testing)
    This is a simplified version - production use should go through RAG tool
    """
    result = await db.execute(
        select(QdrantCollection).filter(QdrantCollection.id == collection_id)
    )
    collection = result.scalars().first()

    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection not found"
        )

    try:
        # Import here to avoid circular dependency
        from src.tools.rag_tool import RAGTool

        rag_tool = RAGTool()
        result = await rag_tool.execute(
            query=query,
            collections=[collection.name],
            k=limit,
            score_threshold=score_threshold
        )

        if not result.success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.error
            )

        return result.data

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )
