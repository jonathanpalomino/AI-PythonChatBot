# =============================================================================
# src/services/collection_service.py
# Collection Service - Business Logic
# =============================================================================
"""
Business logic for Qdrant collection operations

REFACTORED: Service now receives Repositories directly, not UnitOfWork.
This follows the Repository pattern correctly:
    Service → Repository → Session
"""
from typing import List, Optional
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.config.settings import get_qdrant_config
from src.models.models import QdrantCollection, VisibilityType
from src.schemas.schemas import (
    QdrantCollectionCreate,
    QdrantCollectionUpdate,
    QdrantCollectionResponse,
    ListResponse
)
from src.utils.date_utils import get_current_utc
from src.utils.logger import get_logger
from src.utils.transactional import transactional

logger = get_logger(__name__)


class CollectionService:
    """
    Service for Qdrant collection business logic.

    REFACTORED: Receives Repositories directly, not UnitOfWork.
    This follows the Repository pattern correctly.
    """

    def __init__(self, qdrant_collection_repo):
        """
        Initialize CollectionService with repository.

        Args:
            qdrant_collection_repo: QdrantCollectionRepository instance
        """
        self.qdrant_collection_repo = qdrant_collection_repo
        self._qdrant_client = None

    @property
    def qdrant(self) -> QdrantClient:
        """Lazy initialization of Qdrant client"""
        if self._qdrant_client is None:
            config = get_qdrant_config()
            self._qdrant_client = QdrantClient(**config)
        return self._qdrant_client

    @transactional
    async def create_collection(
        self,
        data: QdrantCollectionCreate,
        create_in_qdrant: bool = True,
        vector_size: Optional[int] = None
    ) -> QdrantCollection:
        """
        Create new collection

        Args:
            data: Collection creation data
            create_in_qdrant: Whether to create in Qdrant
            vector_size: Vector dimension

        Returns:
            Created collection

        Raises:
            ValueError: If name already exists
        """
        logger.info(f"Creating collection: {data.name}")

        # Check if name already exists
        existing = await self.qdrant_collection_repo.get_by_name(data.name)
        if existing:
            raise ValueError(f"Collection '{data.name}' already exists")

        # Determine vector size if not provided
        if create_in_qdrant and vector_size is None:
            try:
                from src.services.embedding.embedding_service import get_embedding_service
                embedding_service = await get_embedding_service()
                vector_size = await embedding_service.get_embedding_dimension()
            except Exception as e:
                from src.config.settings import settings
                vector_size = settings.VECTOR_SIZE
                logger.warning(f"Using default vector size: {vector_size}")

        # Create in Qdrant if requested
        if create_in_qdrant:
            try:
                # Check if collection exists in Qdrant
                try:
                    self.qdrant.get_collection(data.name)
                    logger.info(f"Collection exists in Qdrant, registering it")
                except:
                    # Collection doesn't exist, create it
                    self.qdrant.create_collection(
                        collection_name=data.name,
                        vectors_config=VectorParams(
                            size=vector_size,
                            distance=Distance.COSINE
                        )
                    )
                    logger.info(f"Collection created in Qdrant: {data.name}")
            except Exception as e:
                raise ValueError(f"Failed to create collection in Qdrant: {str(e)}")

        # Create registry entry - repository handles flush/refresh internally
        collection = await self.qdrant_collection_repo.create(
            name=data.name,
            display_name=data.display_name,
            description=data.description,
            category=data.category,
            visibility=data.visibility,
            extra_metadata=data.metadata
        )

        logger.info(f"Collection created: {collection.id}")
        return collection

    async def list_collections(
        self,
        category: Optional[str] = None,
        visibility: Optional[VisibilityType] = None,
        is_active: bool = True,
        search: Optional[str] = None,
        skip: int = 0,
        limit: int = 20
    ) -> ListResponse:
        """
        List all collections (registered and unregistered)
        Combines data from Qdrant and database
        """
        # Get all collections from Qdrant
        qdrant_collections_info = self.qdrant.get_collections().collections

        # Get registered collections from DB with filters
        db_collections_list, _ = await self.qdrant_collection_repo.list_with_filters(
            category=category,
            visibility=visibility,
            is_active=is_active,
            search=search,
            skip=0,
            limit=10000  # Get all to match with Qdrant
        )

        # Create dict for fast lookup
        db_collections = {c.name: c for c in db_collections_list}

        # Build response combining both sources
        all_collections = []
        for qdrant_col in qdrant_collections_info:
            col_name = qdrant_col.name

            if col_name in db_collections:
                # Registered collection
                collection_dict = QdrantCollectionResponse.model_validate(
                    db_collections[col_name]
                ).model_dump()
                collection_dict["is_registered_bd"] = True
                all_collections.append(collection_dict)
            else:
                # Unregistered collection
                if search and search.lower() not in col_name.lower():
                    continue

                # Get detailed info from Qdrant
                try:
                    col_info = self.qdrant.get_collection(col_name)
                    vector_size = col_info.config.params.vectors.size if col_info.config.params.vectors else 0
                    points_count = col_info.points_count
                except:
                    vector_size = 0
                    points_count = 0

                all_collections.append({
                    "id": None,
                    "name": col_name,
                    "display_name": col_name,
                    "description": "Unregistered collection",
                    "category": "unregistered",
                    "visibility": VisibilityType.PRIVATE.value,
                    "vector_count": points_count,
                    "is_active": True,
                    "created_at": None,
                    "updated_at": None,
                    "last_synced": None,
                    "extra_metadata": {"vector_size": vector_size},
                    "is_registered_bd": False
                })

        # Apply pagination
        total = len(all_collections)
        paginated = all_collections[skip:skip + limit]

        return ListResponse(
            items=paginated,
            total=total,
            skip=skip,
            limit=limit
        )

    async def get_categories(self) -> List[str]:
        """Get list of categories"""
        return await self.qdrant_collection_repo.get_categories()

    async def get_collection(self, collection_id: UUID) -> Optional[QdrantCollection]:
        """Get collection by ID"""
        return await self.qdrant_collection_repo.get_by_id(collection_id)

    @transactional
    async def update_collection(
        self,
        collection_id: UUID,
        data: QdrantCollectionUpdate
    ) -> Optional[QdrantCollection]:
        """Update collection"""
        logger.info(f"Updating collection: {collection_id}")
        collection = await self.qdrant_collection_repo.get_by_id(collection_id)
        if not collection:
            return None

        # Update fields
        update_data = data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            if field == "metadata" and value is not None:
                current_meta = collection.extra_metadata or {}
                current_meta.update(value)
                setattr(collection, "extra_metadata", current_meta)
            else:
                if field == "metadata":
                    field = "extra_metadata"
                setattr(collection, field, value)

        # Repository handles flush/refresh internally
        collection = await self.qdrant_collection_repo.save(collection)
        logger.info(f"Collection updated: {collection_id}")
        return collection

    @transactional
    async def delete_collection(
        self,
        collection_id: UUID,
        delete_from_qdrant: bool = False
    ) -> bool:
        """Delete collection"""
        logger.info(f"Deleting collection: {collection_id}")
        collection = await self.qdrant_collection_repo.get_by_id(collection_id)
        if not collection:
            return False

        # Delete from Qdrant if requested
        if delete_from_qdrant:
            try:
                self.qdrant.delete_collection(collection.name)
                logger.info(f"Deleted from Qdrant: {collection.name}")
            except Exception as e:
                raise ValueError(f"Failed to delete from Qdrant: {str(e)}")

        await self.qdrant_collection_repo.delete(collection_id)
        logger.info(f"Collection deleted: {collection_id}")
        return True

    @transactional
    async def sync_collection(self, collection_id: UUID) -> Optional[QdrantCollection]:
        """Sync collection with Qdrant to update vector count"""
        logger.info(f"Syncing collection: {collection_id}")
        collection = await self.qdrant_collection_repo.get_by_id(collection_id)
        if not collection:
            return None

        try:
            collection_info = self.qdrant.get_collection(collection.name)
            collection.vector_count = collection_info.points_count
            collection.last_synced = get_current_utc()

            # Repository handles flush/refresh internally
            collection = await self.qdrant_collection_repo.save(collection)
            logger.info(f"Collection synced: {collection_id}")
            return collection
        except Exception as e:
            raise ValueError(f"Failed to sync with Qdrant: {str(e)}")

    async def get_collection_stats(self, collection_id: UUID) -> dict:
        """Get detailed collection statistics from Qdrant"""
        collection = await self.qdrant_collection_repo.get_by_id(collection_id)
        if not collection:
            return None

        try:
            collection_info = self.qdrant.get_collection(collection.name)
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
            raise ValueError(f"Failed to get stats from Qdrant: {str(e)}")

    async def search_in_collection(
        self,
        collection_id: UUID,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.5
    ) -> dict:
        """Search in a specific collection (for testing)"""
        collection = await self.qdrant_collection_repo.get_by_id(collection_id)
        if not collection:
            return None

        try:
            from src.tools.rag_tool import RAGTool
            rag_tool = RAGTool()
            result = await rag_tool.execute(
                query=query,
                collections=[collection.name],
                k=limit,
                score_threshold=score_threshold
            )

            if not result.success:
                raise ValueError(result.error)

            return result.data
        except Exception as e:
            raise ValueError(f"Search failed: {str(e)}")
