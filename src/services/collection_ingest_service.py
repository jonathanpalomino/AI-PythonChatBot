# =============================================================================
# src/services/collection_ingest_service.py
# Administrative File Ingestion Service
# =============================================================================
"""
Service for administrative file ingestion into RAG collections.
Supports recursive folder scanning and incremental updates using Postgres.
"""
import hashlib
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from uuid import UUID
import os
import aiofiles
from fastapi import UploadFile

from src.models.models import File as FileModel, ProcessingStatus, QdrantCollection, VisibilityType
from src.repositories.file_repository import FileRepository
from src.repositories.qdrant_collection_repository import QdrantCollectionRepository
from src.services.processing.file_processor import FileProcessor
from src.document_loaders import DocumentLoaderFactory
from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)

class CollectionIngestService:
    """Service for managing administrative file ingestion into collections."""

    def __init__(
        self,
        file_repo: FileRepository,
        collection_repo: QdrantCollectionRepository,
        file_processor: FileProcessor,
        redis_client = None
    ):
        self.file_repo = file_repo
        self.collection_repo = collection_repo
        self.file_processor = file_processor
        self.redis_client = redis_client

    async def ingest_local_folder(
        self,
        collection_id: str,
        folder_path: str,
        recursive: bool = True,
        embedding_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Scan a local folder and ingest new/modified files into a collection.
        
        Args:
            collection_id: Collection UUID or Name
            folder_path: Local path on the server
            recursive: Whether to scan subfolders
            embedding_model: Optional override for embedding model
            
        Returns:
            Stats of the ingestion process
        """
        path = Path(folder_path)
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Invalid folder path: {folder_path}")

        # 1. Resolve collection
        collection = await self._resolve_collection(collection_id)
        if not collection:
            raise ValueError(f"Collection not found: {collection_id}")

        # 2. Get supported extensions
        supported_exts = DocumentLoaderFactory.get_supported_extensions()
        
        # 3. Find files
        files_to_process = []
        pattern = "**/*" if recursive else "*"
        
        for p in path.glob(pattern):
            if p.is_file() and p.suffix.lower() in supported_exts:
                # Avoid hidden folders (like .obsidian, .git)
                if not any(part.startswith('.') for part in p.parts):
                    files_to_process.append(p)

        logger.info(f"Found {len(files_to_process)} candidate files in {folder_path}")

        stats = {
            "total_found": len(files_to_process),
            "processed": 0,
            "added": 0,
            "updated": 0,
            "skipped": 0,
            "errors": 0,
            "error_details": []
        }

        # 4. Process each file
        for i, file_path in enumerate(files_to_process):
            try:
                # Update progress in Redis if available
                if self.redis_client:
                    job_id = f"ingest_{collection_id}"
                    progress = int((i / len(files_to_process)) * 100)
                    # We might use a more complex status object in Redis
                    await self.redis_client.set(f"status:{job_id}", progress, ex=3600)

                # Compute hash
                content_hash = self._compute_hash(file_path)
                
                # Check if file exists in this collection with same path
                existing_file = await self._find_existing_file(str(file_path), collection.id)
                
                if existing_file:
                    last_hash = existing_file.extra_metadata.get("hash")
                    if last_hash == content_hash and existing_file.processed:
                        logger.debug(f"Skipping unchanged file: {file_path}")
                        stats["skipped"] += 1
                        continue
                    else:
                        logger.info(f"Updating changed file: {file_path}")
                        stats["updated"] += 1
                        # Update metadata and reset status
                        await self.file_repo.update_processing_status(
                            existing_file.id,
                            ProcessingStatus.PENDING,
                            processed=False,
                            extra_metadata={"hash": content_hash}
                        )
                        file_record = existing_file
                else:
                    logger.info(f"Adding new file: {file_path}")
                    stats["added"] += 1
                    # Create new file record
                    file_record = await self.file_repo.create(
                        file_name=file_path.name,
                        file_type=file_path.suffix.lower()[1:],
                        file_size=file_path.stat().st_size,
                        storage_path=str(file_path),
                        mime_type=None,
                        extra_metadata={
                            "hash": content_hash,
                            "original_name": file_path.name,
                            "collection_id": str(collection.id),
                            "collection_name": collection.name,
                            "is_admin_ingest": True,
                            "embedding_model_override": embedding_model
                        }
                    )

                # TRIGGER PROCESSING
                # We call the file processor directly. 
                # Note: FileProcessor usually gets model/collection from conversation.
                # We need to make sure FileProcessor can handle these custom fields.
                await self.file_processor.process_file(file_record.id)
                stats["processed"] += 1

            except Exception as e:
                logger.error(f"Error ingesting file {file_path}: {e}", exc_info=True)
                stats["errors"] += 1
                stats["error_details"].append({"file": str(file_path), "error": str(e)})

        return stats

    async def ingest_uploaded_files(
        self,
        collection_id: str,
        files: List[UploadFile],
        embedding_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Handle multipart file uploads and ingest them into a collection.
        
        Args:
            collection_id: Collection UUID or Name
            files: List of FastAPI UploadFile objects
            embedding_model: Optional override for embedding model
            
        Returns:
            Stats of the ingestion process
        """
        # 1. Resolve collection
        collection = await self._resolve_collection(collection_id)
        if not collection:
            raise ValueError(f"Collection not found: {collection_id}")

        stats = {
            "total_found": len(files),
            "processed": 0,
            "added": 0,
            "updated": 0,
            "skipped": 0,
            "errors": 0,
            "error_details": []
        }

        # Ensure upload dir exists
        upload_dir = settings.UPLOAD_DIR / "admin_ingest" / str(collection_id)
        upload_dir.mkdir(parents=True, exist_ok=True)

        for i, file in enumerate(files):
            try:
                # Read content to calculate hash
                content = await file.read()
                content_hash = hashlib.sha256(content).hexdigest()
                
                # Check for existing file by hash IN THIS COLLECTION
                existing_file = await self._find_existing_file_by_hash(content_hash, collection.id)
                
                if existing_file and existing_file.processed:
                    logger.debug(f"Skipping identical file upload: {file.filename}")
                    stats["skipped"] += 1
                    continue

                if not existing_file:
                    # Check by filename if hash is different
                    existing_file = await self._find_existing_file_by_name(file.filename, collection.id)

                # Save to disk
                file_path = upload_dir / f"{content_hash[:10]}_{file.filename}"
                async with aiofiles.open(file_path, "wb") as f:
                    await f.write(content)

                if existing_file:
                    logger.info(f"Updating file from upload: {file.filename}")
                    stats["updated"] += 1
                    await self.file_repo.update_processing_status(
                        existing_file.id,
                        ProcessingStatus.PENDING,
                        processed=False,
                        extra_metadata={"hash": content_hash}
                    )
                    file_record = existing_file
                else:
                    logger.info(f"Adding new file from upload: {file.filename}")
                    stats["added"] += 1
                    file_record = await self.file_repo.create(
                        file_name=file.filename,
                        file_type=Path(file.filename).suffix.lower()[1:],
                        file_size=len(content),
                        storage_path=str(file_path),
                        mime_type=file.content_type,
                        extra_metadata={
                            "hash": content_hash,
                            "original_name": file.filename,
                            "collection_id": str(collection.id),
                            "collection_name": collection.name,
                            "is_admin_ingest": True,
                            "embedding_model_override": embedding_model
                        }
                    )

                # Trigger processing
                await self.file_processor.process_file(file_record.id)
                stats["processed"] += 1

            except Exception as e:
                logger.error(f"Error ingesting uploaded file {file.filename}: {e}", exc_info=True)
                stats["errors"] += 1
                stats["error_details"].append({"file": file.filename, "error": str(e)})
            finally:
                await file.close()

        return stats

    def _compute_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file content."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    async def _find_existing_file(self, storage_path: str, collection_id: UUID) -> Optional[FileModel]:
        """Find a file record by storage path and collection association."""
        # Custom query needed here since base repo doesn't support complex JSONB filtering
        from sqlalchemy import select
        stmt = select(FileModel).where(
            FileModel.storage_path == storage_path,
            FileModel.extra_metadata["collection_id"].astext == str(collection_id)
        )
        result = await self.file_repo.db.execute(stmt)
        return result.scalars().first()

    async def _resolve_collection(self, collection_id: str) -> Optional[QdrantCollection]:
        """Resolve a collection ID (UUID) or Name to a QdrantCollection object."""
        collection = None
        # Try as UUID first
        try:
            uuid_val = UUID(collection_id)
            collection = await self.collection_repo.get_by_id(uuid_val)
        except ValueError:
            # If not a valid UUID, try as name
            collection = await self.collection_repo.get_by_name(collection_id)

        if collection:
            return collection

        # If not found in DB, check Qdrant directly
        try:
            from qdrant_client import QdrantClient
            from src.config.settings import get_qdrant_config
            qdrant = QdrantClient(**get_qdrant_config())
            
            # This will raise an exception if collection doesn't exist in Qdrant
            qdrant.get_collection(collection_id)
            
            # Found in Qdrant! Auto-register in DB to allow tracking
            logger.info(f"Auto-registering existing Qdrant collection in DB: {collection_id}")
            collection = await self.collection_repo.create_collection(
                name=collection_id,
                display_name=collection_id,
                description="Auto-registered during admin ingestion",
                category="admin",
                visibility=VisibilityType.PRIVATE,
                extra_metadata={"auto_registered": True}
            )
            return collection
        except Exception as e:
            logger.warning(f"Collection {collection_id} not found in Qdrant either: {e}")
            return None

    async def _find_existing_file_by_hash(self, content_hash: str, collection_id: UUID) -> Optional[FileModel]:
        """Find a file record by hash and collection association."""
        from sqlalchemy import select
        stmt = select(FileModel).where(
            FileModel.extra_metadata["hash"].astext == content_hash,
            FileModel.extra_metadata["collection_id"].astext == str(collection_id)
        )
        result = await self.file_repo.db.execute(stmt)
        return result.scalars().first()

    async def _find_existing_file_by_name(self, filename: str, collection_id: UUID) -> Optional[FileModel]:
        """Find a file record by original name and collection association."""
        from sqlalchemy import select
        stmt = select(FileModel).where(
            FileModel.extra_metadata["original_name"].astext == filename,
            FileModel.extra_metadata["collection_id"].astext == str(collection_id)
        )
        result = await self.file_repo.db.execute(stmt)
        return result.scalars().first()
