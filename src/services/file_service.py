# =============================================================================
# src/services/file_service.py
# File Service - Business Logic
# =============================================================================
"""
Business logic for file operations

REFACTORED: Service now receives Repositories directly, not UnitOfWork.
This follows the Repository pattern correctly:
    Service → Repository → Session

The Service does NOT know about UnitOfWork - it only uses Repositories.
"""
import hashlib
from pathlib import Path
from typing import List, Optional
from uuid import UUID

from fastapi import UploadFile

from src.config.settings import settings
from src.models.models import File as FileModel, ProcessingStatus
from src.repositories.conversation_repository import ConversationRepository
from src.repositories.file_repository import FileRepository
from src.repositories.qdrant_collection_repository import QdrantCollectionRepository
from src.schemas.schemas import FileResponse, ListResponse
from src.services.processing.file_processor import FileProcessor, FileProcessingError
from src.utils.logger import get_logger, set_conversation_context

logger = get_logger(__name__)


class FileService:
    """
    Service for file business logic.

    REFACTORED: Receives Repositories directly, not UnitOfWork.
    This follows the Repository pattern correctly:
        Service → Repository → Session
    """

    def __init__(
        self,
        file_repo: FileRepository,
        conversation_repo: ConversationRepository,
        qdrant_collection_repo: QdrantCollectionRepository,
    ):
        self.file_repo = file_repo
        self.conversation_repo = conversation_repo
        self.qdrant_collection_repo = qdrant_collection_repo
        self._processor = None

    @property
    def processor(self) -> FileProcessor:
        """Lazy initialization of FileProcessor"""
        if self._processor is None:
            self._processor = FileProcessor(
                self.file_repo,
                self.conversation_repo,
                self.qdrant_collection_repo
            )
        return self._processor

    @staticmethod
    def calculate_file_hash(content: bytes) -> str:
        """Calculate SHA256 hash of file content"""
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def get_file_storage_path(file_id: UUID, filename: str) -> Path:
        """Generate storage path for file"""
        subdir = str(file_id)[:2]
        storage_dir = settings.UPLOAD_DIR / subdir
        storage_dir.mkdir(parents=True, exist_ok=True)
        extension = Path(filename).suffix
        return storage_dir / f"{file_id}{extension}"

    @staticmethod
    def validate_file(file: UploadFile) -> None:
        """Validate uploaded file"""
        from src.utils.file_utils import validate_file_extension

        if file.filename:
            validate_file_extension(file.filename)

    async def upload_file(
        self,
        file: UploadFile,
        conversation_id: Optional[UUID] = None,
        async_processing: bool = True
    ) -> FileModel:
        """
        Upload a file with async processing.

        Args:
            file: Uploaded file
            conversation_id: Optional conversation to associate with
            async_processing: If True (default), process asynchronously.

        Returns:
            Created file record with PENDING status

        Raises:
            ValueError: If file validation fails or size exceeds limit
        """
        logger.info(
            "Uploading file",
            extra={"file_name": file.filename, "conversation_id": str(conversation_id) if conversation_id else None}
        )

        if conversation_id:
            set_conversation_context(str(conversation_id))

        # Validate file
        self.validate_file(file)

        # Validate conversation if provided
        if conversation_id:
            conversation = await self.conversation_repo.get_by_id(conversation_id)
            if not conversation:
                raise ValueError("Conversation not found")

        # Read file content
        content = await file.read()

        # Check size
        if len(content) > settings.MAX_UPLOAD_SIZE:
            raise ValueError(
                f"File too large. Max size: {settings.MAX_UPLOAD_SIZE / 1024 / 1024:.1f}MB"
            )

        # Calculate hash
        file_hash = self.calculate_file_hash(content)

        # Check for duplicate in the same conversation
        existing = await self.file_repo.get_by_hash(file_hash, conversation_id)
        if existing:
            logger.info(f"File already exists, returning existing: {existing.id}")
            return existing

        # Create file record
        file_record = await self.file_repo.create(
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

        # Save file to disk
        storage_path = self.get_file_storage_path(file_record.id, file.filename)

        # Simulate upload progress (write in chunks)
        total_size = len(content)
        chunk_size = 1024 * 1024  # 1MB
        uploaded_size = 0

        with open(storage_path, "wb") as f:
            for chunk in [content[i:i + chunk_size] for i in range(0, total_size, chunk_size)]:
                f.write(chunk)
                uploaded_size += len(chunk)
                progress = int((uploaded_size / total_size) * 100)
                # Store progress in Redis if available
                try:
                    import redis
                    redis_client = redis.Redis.from_url(settings.REDIS_URL)
                    redis_client.set(str(file_record.id), progress)
                except:
                    pass

        # Update storage path via repository
        file_record = await self.file_repo.update_storage_path(file_record.id, str(storage_path))

        # Commit the file record to ensure it's persisted before returning
        await self.file_repo.commit()
        logger.info(f"File record committed to database: {file_record.id}")

        # Process file based on async_processing flag
        if async_processing:
            # Try to enqueue Celery task (uses Redis DB 1)
            # Fall back to sync processing if Celery is unavailable
            try:
                from src.tasks.file_tasks import process_file_task
                task = process_file_task.delay(str(file_record.id))

                # Guardar task_id para tracking via repository
                await self.file_repo.update_metadata(
                    file_record.id,
                    {"celery_task_id": task.id, "async_processing": True}
                )

                logger.info(f"File queued for async processing: {file_record.id}, task_id: {task.id}")
            except Exception as celery_error:
                # Celery/Redis unavailable - fall back to sync processing
                logger.warning(
                    f"Celery unavailable (Redis connection failed), falling back to sync processing: {celery_error}"
                )
                logger.info(f"Processing file synchronously (fallback): {file_record.id}")

                try:
                    process_result = await self.processor.process_file(file_record.id)
                    await self.file_repo.update_processing_status(
                        file_record.id,
                        ProcessingStatus.COMPLETED,
                        processed=True,
                        extra_metadata={
                            "chunks": process_result.get("chunks", 0),
                            "async_processing": False,
                            "celery_fallback": True
                        }
                    )
                    logger.info(f"File processed synchronously (fallback): {process_result['chunks']} chunks indexed")
                except FileProcessingError as e:
                    logger.error(f"Sync processing (fallback) failed: {e}", exc_info=True)
                    # Rollback file record and disk file
                    logger.info("Rolling back failed upload...")
                    try:
                        if storage_path.exists():
                            storage_path.unlink()
                        await self.file_repo.delete(file_record.id)
                    except Exception as cleanup_error:
                        logger.error(f"Rollback failed: {cleanup_error}")
                    raise ValueError(f"File processing failed: {str(e)}")
        elif settings.AUTO_PROCESS_FILES:
            # Procesamiento síncrono (backwards compatibility)
            logger.info(f"Processing file synchronously (backwards compat): {file_record.id}")
            try:
                process_result = await self.processor.process_file(file_record.id)
                await self.file_repo.update_processing_status(
                    file_record.id,
                    ProcessingStatus.COMPLETED,
                    processed=True,
                    extra_metadata={
                        "chunks": process_result.get("chunks", 0),
                        "async_processing": False
                    }
                )
                logger.info(f"File auto-processed: {process_result['chunks']} chunks indexed")
            except FileProcessingError as e:
                logger.error(f"Sync processing failed: {e}", exc_info=True)
                # Rollback file record and disk file
                logger.info("Rolling back failed upload...")
                try:
                    if storage_path.exists():
                        storage_path.unlink()
                    await self.file_repo.delete(file_record.id)
                except Exception as cleanup_error:
                    logger.error(f"Rollback failed: {cleanup_error}")
                raise ValueError(f"File processing failed: {str(e)}")
        else:
            # No processing - file stays in PENDING status
            await self.file_repo.update_metadata(
                file_record.id,
                {"async_processing": False}
            )
            logger.info(f"File uploaded without processing: {file_record.id}")

        logger.info(
            "File uploaded successfully",
            extra={"file_id": str(file_record.id), "size_bytes": file_record.file_size}
        )

        return file_record

    async def get_upload_progress(self, file_id: UUID) -> dict:
        """Get upload and processing progress for a file"""
        try:
            import redis
            redis_client = redis.Redis.from_url(settings.REDIS_URL)

            progress = redis_client.get(str(file_id))
            progress = int(progress) if progress else 0

            processing_progress = redis_client.get(f"processing:{file_id}")
            processing_progress = int(processing_progress) if processing_progress else 0

            # Determine status
            status_category = "uploading"
            if progress >= 100:
                if processing_progress >= 100:
                    status_category = "completed"
                elif processing_progress > 0:
                    status_category = "processing"
                else:
                    status_category = "pending_processing"

            return {
                "file_id": str(file_id),
                "upload_progress": progress,
                "processing_progress": processing_progress,
                "status": status_category
            }
        except:
            return {
                "file_id": str(file_id),
                "upload_progress": 0,
                "processing_progress": 0,
                "status": "unknown"
            }

    async def download_file(self, file_id: UUID) -> tuple[Path, FileModel]:
        """Get file path and record for download"""
        file_record = await self.file_repo.get_by_id(file_id)
        if not file_record:
            raise ValueError("File not found")

        storage_path = Path(file_record.storage_path)
        if not storage_path.exists():
            raise ValueError("File not found on disk")

        return storage_path, file_record

    async def list_files(
        self,
        conversation_id: Optional[UUID] = None,
        file_type: Optional[str] = None,
        processed: Optional[bool] = None,
        skip: int = 0,
        limit: int = 20
    ) -> ListResponse:
        """List files with filters"""
        total = await self.file_repo.count_filtered(
            conversation_id=conversation_id,
            file_type=file_type,
            processed=processed
        )

        files = await self.file_repo.list_filtered(
            conversation_id=conversation_id,
            file_type=file_type,
            processed=processed,
            skip=skip,
            limit=limit
        )

        return ListResponse(
            items=[FileResponse.model_validate(f) for f in files],
            total=total,
            skip=skip,
            limit=limit
        )

    async def get_file(self, file_id: UUID) -> Optional[FileModel]:
        """Get file by ID"""
        return await self.file_repo.get_by_id(file_id)

    async def delete_file(
        self,
        file_id: UUID,
        delete_from_disk: bool = True
    ) -> bool:
        """Delete a file"""
        logger.info(f"Deleting file: {file_id}")
        file_record = await self.file_repo.get_by_id(file_id)
        if not file_record:
            return False

        # Delete chunks from Qdrant
        try:
            await self.processor.delete_file_chunks(file_id)
        except Exception as e:
            logger.error(f"Failed to delete chunks for file {file_id}: {e}")

        # Delete from disk
        if delete_from_disk:
            storage_path = Path(file_record.storage_path)
            if storage_path.exists():
                storage_path.unlink()

        # Delete from database
        await self.file_repo.delete(file_id)
        logger.info(f"File deleted: {file_id}")
        return True

    async def process_file(self, file_id: UUID) -> FileModel:
        """Trigger file processing"""
        logger.info(f"Triggering file processing: {file_id}")
        file_record = await self.file_repo.get_by_id(file_id)
        if not file_record:
            raise ValueError("File not found")

        if file_record.processed:
            raise ValueError("File already processed")

        # Trigger processing
        try:
            result = await self.processor.process_file(file_id)
            # Fetch updated record via repository
            file_record = await self.file_repo.get_by_id(file_id)
            logger.info(f"File processing completed: {result['chunks']} chunks")
            return file_record
        except FileProcessingError as e:
            logger.error(f"File processing failed: {e}", exc_info=True)
            raise ValueError(str(e))

    async def get_file_content(self, file_id: UUID) -> dict:
        """Get extracted content from a processed file"""
        file_record = await self.file_repo.get_by_id(file_id)
        if not file_record:
            raise ValueError("File not found")

        if not file_record.processed:
            raise ValueError("File not yet processed")

        extracted_text = file_record.extra_metadata.get("extracted_text")
        if not extracted_text:
            raise ValueError("No extracted content available")

        return {
            "file_id": file_record.id,
            "file_name": file_record.file_name,
            "content": extracted_text,
            "language": file_record.extra_metadata.get("language"),
            "analysis": file_record.extra_metadata.get("analysis_result"),
            "code_analysis": file_record.extra_metadata.get("code_analysis")
        }

    async def batch_upload_files(
        self,
        files: List[UploadFile],
        conversation_id: Optional[UUID] = None
    ) -> List[FileModel]:
        """Upload multiple files at once"""
        import asyncio

        uploaded_files = []

        async def upload_wrapper(f: UploadFile):
            try:
                return await self.upload_file(f, conversation_id)
            except Exception as e:
                logger.error(f"Failed to upload {f.filename}: {e}")
                return None

        tasks = [upload_wrapper(file) for file in files]
        results = await asyncio.gather(*tasks)

        uploaded_files = [r for r in results if r is not None]
        return uploaded_files

    async def batch_delete_files(
        self,
        file_ids: List[UUID],
        delete_from_disk: bool = True
    ) -> None:
        """Delete multiple files"""
        import asyncio

        async def delete_wrapper(fid: UUID):
            try:
                await self.delete_file(fid, delete_from_disk)
            except:
                pass

        tasks = [delete_wrapper(file_id) for file_id in file_ids]
        await asyncio.gather(*tasks)
