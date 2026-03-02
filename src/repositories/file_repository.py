# =============================================================================
# src/repositories/file_repository.py
# File Repository
# =============================================================================
"""
Repository for File entity operations.
"""
from typing import List, Optional
from uuid import UUID
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from src.models.models import File as FileModel, ProcessingStatus
from src.repositories.base_repository import BaseRepository


class FileRepository(BaseRepository[FileModel]):
    """Repository for managing files."""
    
    def __init__(self, db: AsyncSession):
        super().__init__(FileModel, db)
    
    async def get_by_conversation(
        self,
        conversation_id: UUID,
        skip: int = 0,
        limit: int = 100
    ) -> List[FileModel]:
        """Get all files for a conversation."""
        try:
            stmt = (
                select(FileModel)
                .where(FileModel.conversation_id == conversation_id)
                .order_by(FileModel.uploaded_at.desc())
                .offset(skip)
                .limit(limit)
            )
            result = await self.db.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            self.logger.error(f"Error getting files for conversation: {e}")
            raise

    async def get_by_ids(self, file_ids: List[UUID]) -> List[FileModel]:
        """Get files by a list of IDs."""
        try:
            if not file_ids:
                return []
            stmt = (
                select(FileModel)
                .where(FileModel.id.in_(file_ids))
            )
            result = await self.db.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            self.logger.error(f"Error getting files by IDs: {e}")
            raise

    async def get_by_id_with_conversation(self, file_id: UUID) -> Optional[FileModel]:
        """Get file by id with conversation relationship eagerly loaded."""
        try:
            stmt = (
                select(FileModel)
                .options(selectinload(FileModel.conversation))
                .where(FileModel.id == file_id)
            )
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            self.logger.error(f"Error getting File by id with conversation: {e}")
            raise
    
    async def get_conversation_files(
        self,
        conversation_id: UUID,
        skip: int = 0,
        limit: int = 100
    ) -> List[FileModel]:
        """Alias for get_by_conversation to maintain compatibility with ChatOrchestrator."""
        return await self.get_by_conversation(conversation_id, skip, limit)
    
    async def count_filtered(
        self,
        conversation_id: Optional[UUID] = None,
        file_type: Optional[str] = None,
        processed: Optional[bool] = None
    ) -> int:
        """Count files with filters."""
        try:
            query = select(func.count(FileModel.id))
            
            if conversation_id:
                query = query.filter(FileModel.conversation_id == conversation_id)
            if file_type:
                query = query.filter(FileModel.file_type == file_type)
            if processed is not None:
                query = query.filter(FileModel.processed == processed)
            
            result = await self.db.execute(query)
            return result.scalar() or 0
        except Exception as e:
            self.logger.error(f"Error counting files: {e}")
            raise
    
    async def list_filtered(
        self,
        conversation_id: Optional[UUID] = None,
        file_type: Optional[str] = None,
        processed: Optional[bool] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[FileModel]:
        """List files with filters."""
        try:
            query = select(FileModel)
            
            if conversation_id:
                query = query.filter(FileModel.conversation_id == conversation_id)
            if file_type:
                query = query.filter(FileModel.file_type == file_type)
            if processed is not None:
                query = query.filter(FileModel.processed == processed)
            
            query = query.order_by(FileModel.uploaded_at.desc()).offset(skip).limit(limit)
            result = await self.db.execute(query)
            return list(result.scalars().all())
        except Exception as e:
            self.logger.error(f"Error listing files: {e}")
            raise
    
    async def get_by_hash(
        self,
        file_hash: str,
        conversation_id: Optional[UUID] = None
    ) -> Optional[FileModel]:
        """Get file by hash, optionally filtered by conversation."""
        try:
            query = select(FileModel).filter(
                func.jsonb_extract_path_text(FileModel.extra_metadata, 'hash') == file_hash
            )
            
            if conversation_id:
                query = query.filter(FileModel.conversation_id == conversation_id)
            
            result = await self.db.execute(query)
            return result.scalars().first()
        except Exception as e:
            self.logger.error(f"Error getting file by hash: {e}")
            raise

    # =========================================================================
    # REPOSITORY PATTERN: Methods to encapsulate DB operations
    # Services should NEVER access self.db directly
    # =========================================================================

    async def update_storage_path(
        self,
        file_id: UUID,
        storage_path: str
    ) -> Optional[FileModel]:
        """
        Update the storage path for a file.
        
        Args:
            file_id: File UUID
            storage_path: New storage path
            
        Returns:
            Updated file model or None if not found
        """
        return await self.update(file_id, storage_path=storage_path)

    async def update_processing_status(
        self,
        file_id: UUID,
        status: ProcessingStatus,
        processed: Optional[bool] = None,
        extra_metadata: Optional[dict] = None
    ) -> Optional[FileModel]:
        """
        Update file processing status and optionally metadata.
        
        Args:
            file_id: File UUID
            status: New processing status
            processed: Whether file is processed
            extra_metadata: Optional metadata to merge
            
        Returns:
            Updated file model or None if not found
        """
        update_data = {"processing_status": status}
        if processed is not None:
            update_data["processed"] = processed
        
        if extra_metadata:
            # Get current file to merge metadata
            current = await self.get_by_id(file_id)
            if current:
                current_metadata = current.extra_metadata or {}
                current_metadata.update(extra_metadata)
                update_data["extra_metadata"] = current_metadata
        
        return await self.update(file_id, **update_data)

    async def update_metadata(
        self,
        file_id: UUID,
        metadata: dict,
        merge: bool = True
    ) -> Optional[FileModel]:
        """
        Update file metadata.
        
        Args:
            file_id: File UUID
            metadata: Metadata to set/merge
            merge: If True, merge with existing; if False, replace
            
        Returns:
            Updated file model or None if not found
        """
        if merge:
            current = await self.get_by_id(file_id)
            if current:
                current_metadata = current.extra_metadata or {}
                current_metadata.update(metadata)
                return await self.update(file_id, extra_metadata=current_metadata)
            return None
        else:
            return await self.update(file_id, extra_metadata=metadata)
