# =============================================================================
# src/services/project_service.py
# Project Service Layer - Business Logic
# =============================================================================
"""
Service layer for Project business logic.
Orchestrates operations between multiple repositories and services.

Following Spring Boot @Service pattern:
- Business logic encapsulation
- Transaction management
- Multi-repository orchestration
- Reusable from controllers, CLI, background tasks
"""

from pathlib import Path
from uuid import UUID

from fastapi import HTTPException, UploadFile, status

from src.config.settings import settings
from src.models.models import ProcessingStatus
from src.repositories.conversation_repository import ConversationRepository
from src.repositories.file_repository import FileRepository
from src.repositories.project_repository import ProjectRepository
from src.schemas.schemas import (
    PaginationParams,
    ProjectCreate,
    ProjectUpdate,
)
from src.services.processing.file_processor import FileProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ProjectService:
    """
    Service layer for Project entity.

    Responsibilities:
    - Business logic for projects
    - Cascade operations (delete project + files + chunks)
    - File upload orchestration
    - Transaction coordination

    Equivalent to Spring Boot @Service
    """

    def __init__(
        self,
        project_repo: ProjectRepository,
        file_repo: FileRepository,
        conversation_repo: ConversationRepository,
        file_processor: FileProcessor,
    ):
        """
        Initialize service with injected dependencies.

        Args:
            project_repo: Project repository
            file_repo: File repository
            conversation_repo: Conversation repository
            file_processor: File processing service
        """
        self.project_repo = project_repo
        self.file_repo = file_repo
        self.conversation_repo = conversation_repo
        self.file_processor = file_processor
        self.logger = logger

    async def create_project(self, data: ProjectCreate):
        """
        Create a new project.

        Args:
            data: Project creation data

        Returns:
            Created project entity

        Raises:
            Exception: If creation fails
        """
        self.logger.info("Creating new project", extra={"project_name": data.name})

        try:
            project = await self.project_repo.create(
                name=data.name, description=data.description
            )
            await self.project_repo.commit()

            self.logger.info(
                "Project created successfully",
                extra={"project_id": str(project.id), "project_name": project.name},
            )
            return project

        except Exception as e:
            self.logger.error(
                f"Failed to create project: {e}",
                exc_info=True,
                extra={"project_name": data.name},
            )
            await self.project_repo.rollback()
            raise

    async def get_project(self, project_id: UUID):
        """
        Get project by ID.

        Args:
            project_id: Project UUID

        Returns:
            Project entity or None
        """
        self.logger.debug("Fetching project", extra={"project_id": str(project_id)})

        project = await self.project_repo.get_by_id(project_id)

        if not project:
            self.logger.warning(
                "Project not found", extra={"project_id": str(project_id)}
            )

        return project

    async def list_projects(self, pagination: PaginationParams):
        """
        List all projects with pagination.

        Args:
            pagination: Pagination parameters

        Returns:
            Tuple of (projects list, total count)
        """
        self.logger.debug(
            "Listing projects",
            extra={"skip": pagination.skip, "limit": pagination.limit},
        )

        total = await self.project_repo.count()
        projects = await self.project_repo.get_all(
            skip=pagination.skip,
            limit=pagination.limit,
            order_by="updated_at",
            descending=True,
        )

        self.logger.debug(
            "Projects retrieved",
            extra={"total": total, "returned": len(projects)},
        )

        return projects, total

    async def update_project(self, project_id: UUID, data: ProjectUpdate):
        """
        Update project.

        Args:
            project_id: Project UUID
            data: Update data

        Returns:
            Updated project entity

        Raises:
            HTTPException: If project not found
        """
        self.logger.info("Updating project", extra={"project_id": str(project_id)})

        # Verify existence
        project = await self.project_repo.get_by_id(project_id)
        if not project:
            self.logger.warning(
                "Cannot update: project not found",
                extra={"project_id": str(project_id)},
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Project not found"
            )

        try:
            # Build update dict
            update_data = {}
            if data.name is not None:
                update_data["name"] = data.name
            if data.description is not None:
                update_data["description"] = data.description
            if data.is_active is not None:
                update_data["is_active"] = data.is_active

            # Perform update
            updated_project = await self.project_repo.update(project_id, **update_data)
            await self.project_repo.commit()

            self.logger.info(
                "Project updated successfully",
                extra={"project_id": str(project_id), "fields": list(update_data.keys())},
            )

            return updated_project

        except Exception as e:
            self.logger.error(
                f"Failed to update project: {e}",
                exc_info=True,
                extra={"project_id": str(project_id)},
            )
            await self.project_repo.rollback()
            raise

    async def delete_project_cascade(self, project_id: UUID) -> None:
        """
        Delete project and cascade to all related entities.

        Business logic:
        1. Verify project exists
        2. Get all associated files
        3. Delete vector chunks for each file
        4. Delete project (DB cascade handles files/conversations)
        5. Commit transaction

        Args:
            project_id: Project UUID

        Raises:
            HTTPException: If project not found
        """
        self.logger.info(
            "Starting cascade deletion for project",
            extra={"project_id": str(project_id)},
        )

        # Verify existence
        project = await self.project_repo.get_by_id(project_id)
        if not project:
            self.logger.warning(
                "Cannot delete: project not found",
                extra={"project_id": str(project_id)},
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Project not found"
            )

        try:
            # Get all files in the project
            files = await self.file_repo.get_all(project_id=project_id)
            file_count = len(files)

            self.logger.info(
                f"Found {file_count} files to clean up",
                extra={"project_id": str(project_id), "file_count": file_count},
            )

            # Delete vector chunks for each file
            deleted_chunks_count = 0
            failed_chunks = []

            for file in files:
                try:
                    success = await self.file_processor.delete_file_chunks(file.id)
                    if success:
                        deleted_chunks_count += 1
                    else:
                        failed_chunks.append(str(file.id))

                except Exception as e:
                    self.logger.error(
                        f"Failed to delete chunks for file {file.id}: {e}",
                        exc_info=True,
                        extra={"file_id": str(file.id), "project_id": str(project_id)},
                    )
                    failed_chunks.append(str(file.id))

            if failed_chunks:
                self.logger.warning(
                    f"Some file chunks failed to delete: {len(failed_chunks)} files",
                    extra={
                        "project_id": str(project_id),
                        "failed_files": failed_chunks,
                    },
                )

            # Delete project (DB cascade will handle files/conversations)
            await self.project_repo.delete(project_id)
            await self.project_repo.commit()

            self.logger.info(
                "Project deleted successfully",
                extra={
                    "project_id": str(project_id),
                    "files_cleaned": deleted_chunks_count,
                    "total_files": file_count,
                },
            )

        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise

        except Exception as e:
            self.logger.error(
                f"Failed to delete project: {e}",
                exc_info=True,
                extra={"project_id": str(project_id)},
            )
            await self.project_repo.rollback()
            raise

    async def upload_file_to_project(
        self,
        project_id: UUID,
        file: UploadFile,
        redis_client=None,
    ):
        """
        Upload and process a file for a project.

        Business logic:
        1. Validate project exists
        2. Validate file (type, size)
        3. Check for duplicates (hash)
        4. Create file record
        5. Save to disk
        6. Auto-process if enabled

        Args:
            project_id: Project UUID
            file: Uploaded file
            redis_client: Redis client for progress tracking

        Returns:
            File entity

        Raises:
            HTTPException: If validation fails
        """
        self.logger.info(
            "Uploading file to project",
            extra={
                "project_id": str(project_id),
                "file_name": file.filename,
                "content_type": file.content_type,
            },
        )

        # Validate project exists
        if not await self.project_repo.exists(project_id):
            self.logger.warning(
                "Cannot upload: project not found",
                extra={"project_id": str(project_id)},
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Project not found"
            )

        # Validate file
        self._validate_file(file)

        # Read content
        content = await file.read()
        file_size = len(content)

        # Validate size
        if file_size > settings.MAX_UPLOAD_SIZE:
            self.logger.warning(
                "File too large",
                extra={
                    "file_name": file.filename,
                    "size": file_size,
                    "max_size": settings.MAX_UPLOAD_SIZE,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Max size: {settings.MAX_UPLOAD_SIZE} bytes",
            )

        # Calculate hash for duplicate detection
        file_hash = self._calculate_file_hash(content)

        # Check for duplicates in project
        existing_files = await self.file_repo.get_all(project_id=project_id)
        for existing in existing_files:
            if (
                existing.extra_metadata
                and existing.extra_metadata.get("hash") == file_hash
            ):
                self.logger.info(
                    "Duplicate file detected, returning existing",
                    extra={
                        "file_id": str(existing.id),
                        "file_name": file.filename,
                        "hash": file_hash,
                    },
                )
                return existing

        try:
            # Create file record
            file_record = await self.file_repo.create(
                project_id=project_id,
                file_name=file.filename,
                file_type=Path(file.filename).suffix.lower(),
                file_size=file_size,
                storage_path="",  # Will be set after saving
                mime_type=file.content_type,
                processed=False,
                processing_status=ProcessingStatus.PENDING,
                extra_metadata={
                    "hash": file_hash,
                    "original_name": file.filename,
                },
            )
            await self.file_repo.flush()

            # Save to disk
            storage_path = self._get_file_storage_path(file_record.id, file.filename)
            self._save_file_to_disk(
                content, storage_path, file_record.id, redis_client
            )

            # Update storage path
            file_record.storage_path = str(storage_path)
            await self.file_repo.commit()
            await self.file_repo.refresh(file_record)

            self.logger.info(
                "File uploaded successfully",
                extra={
                    "file_id": str(file_record.id),
                    "file_name": file.filename,
                    "size": file_size,
                    "storage_path": str(storage_path),
                },
            )

            # Auto-process if enabled
            if settings.AUTO_PROCESS_FILES:
                try:
                    await self.process_file_async(file_record.id)
                except Exception as e:
                    self.logger.error(
                        f"Auto-processing failed: {e}",
                        exc_info=True,
                        extra={"file_id": str(file_record.id)},
                    )
                    # Don't fail the upload if processing fails

            return file_record

        except Exception as e:
            self.logger.error(
                f"Failed to upload file: {e}",
                exc_info=True,
                extra={"project_id": str(project_id), "file_name": file.filename},
            )
            await self.file_repo.rollback()
            raise

    async def process_file_async(self, file_id: UUID) -> None:
        """
        Process a file asynchronously.

        Business logic:
        1. Update status to PROCESSING
        2. Extract text, chunk, embed
        3. Store in vector database
        4. Update status to COMPLETED

        Args:
            file_id: File UUID

        Raises:
            Exception: If processing fails
        """
        self.logger.info(
            "Starting async file processing", extra={"file_id": str(file_id)}
        )

        file_record = await self.file_repo.get_by_id(file_id)
        if not file_record:
            self.logger.error(
                "File not found for processing", extra={"file_id": str(file_id)}
            )
            return

        try:
            # Update status to processing
            file_record.processing_status = ProcessingStatus.PROCESSING
            await self.file_repo.commit()

            self.logger.info(
                "Processing file",
                extra={"file_id": str(file_id), "file_name": file_record.file_name},
            )

            # Process file (extract, chunk, embed, store)
            await self.file_processor.process_file(file_record.id)

            # Update status to completed
            file_record.processed = True
            file_record.processing_status = ProcessingStatus.COMPLETED
            await self.file_repo.commit()
            await self.file_repo.refresh(file_record)

            self.logger.info(
                "File processing completed successfully",
                extra={"file_id": str(file_id), "file_name": file_record.file_name},
            )

        except Exception as e:
            self.logger.error(
                f"File processing failed: {e}",
                exc_info=True,
                extra={"file_id": str(file_id)},
            )

            # Update status to error
            try:
                file_record.processing_status = ProcessingStatus.ERROR
                file_record.extra_metadata = file_record.extra_metadata or {}
                file_record.extra_metadata["error"] = str(e)
                await self.file_repo.commit()
            except Exception as update_error:
                self.logger.error(
                    f"Failed to update error status: {update_error}",
                    extra={"file_id": str(file_id)},
                )

            raise

    async def list_project_conversations(
        self, project_id: UUID, pagination: PaginationParams
    ):
        """
        List conversations in a project.

        Args:
            project_id: Project UUID
            pagination: Pagination parameters

        Returns:
            Tuple of (conversations list, total count)

        Raises:
            HTTPException: If project not found
        """
        self.logger.debug(
            "Listing project conversations", extra={"project_id": str(project_id)}
        )

        # Verify project exists
        if not await self.project_repo.exists(project_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Project not found"
            )

        total = await self.conversation_repo.count(project_id=project_id)
        conversations = await self.conversation_repo.get_all(
            skip=pagination.skip,
            limit=pagination.limit,
            order_by="updated_at",
            descending=True,
            project_id=project_id,
        )

        return conversations, total

    async def list_project_files(
        self, project_id: UUID, pagination: PaginationParams
    ):
        """
        List files in a project.

        Args:
            project_id: Project UUID
            pagination: Pagination parameters

        Returns:
            Tuple of (files list, total count)

        Raises:
            HTTPException: If project not found
        """
        self.logger.debug(
            "Listing project files", extra={"project_id": str(project_id)}
        )

        # Verify project exists
        if not await self.project_repo.exists(project_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Project not found"
            )

        total = await self.file_repo.count(project_id=project_id)
        files = await self.file_repo.get_all(
            skip=pagination.skip,
            limit=pagination.limit,
            order_by="uploaded_at",
            descending=True,
            project_id=project_id,
        )

        return files, total

    # =========================================================================
    # Private helper methods
    # =========================================================================

    def _validate_file(self, file: UploadFile) -> None:
        """Validate uploaded file type and name."""
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Filename is required",
            )

        # Validate extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type not allowed. Allowed: {settings.ALLOWED_EXTENSIONS}",
            )

    def _calculate_file_hash(self, content: bytes) -> str:
        """Calculate SHA-256 hash of file content."""
        import hashlib

        return hashlib.sha256(content).hexdigest()

    def _get_file_storage_path(self, file_id: UUID, filename: str) -> Path:
        """Generate storage path for file."""
        uploads_dir = Path(settings.UPLOAD_DIR)
        uploads_dir.mkdir(parents=True, exist_ok=True)

        # Use file_id + original extension
        file_ext = Path(filename).suffix
        return uploads_dir / f"{file_id}{file_ext}"

    def _save_file_to_disk(
        self,
        content: bytes,
        storage_path: Path,
        file_id: UUID,
        redis_client=None,
    ) -> None:
        """Save file content to disk with progress tracking."""
        total_size = len(content)
        chunk_size = 1024 * 1024  # 1MB chunks
        uploaded_size = 0

        with open(storage_path, "wb") as f:
            for i in range(0, total_size, chunk_size):
                chunk = content[i : i + chunk_size]
                f.write(chunk)
                uploaded_size += len(chunk)

                # Update progress in Redis
                if redis_client:
                    try:
                        progress = int((uploaded_size / total_size) * 100)
                        redis_client.setex(
                            f"upload:{file_id}", 3600, progress
                        )  # 1 hour TTL
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to update upload progress: {e}",
                            extra={"file_id": str(file_id)},
                        )

        self.logger.debug(
            "File saved to disk",
            extra={
                "file_id": str(file_id),
                "storage_path": str(storage_path),
                "size": total_size,
            },
        )
