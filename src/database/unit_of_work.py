# =============================================================================
# src/database/unit_of_work.py
# Unit of Work Pattern - Manejo de transacciones
# =============================================================================
"""
Unit of Work pattern para manejar transacciones
Similar a como Spring Data maneja EntityManager internamente
"""
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from src.repositories.message_repository import MessageRepository
from src.repositories.prompt_template_repository import PromptTemplateRepository
from src.repositories.file_repository import FileRepository
from src.repositories.conversation_repository import ConversationRepository
from src.repositories.custom_tool_repository import CustomToolRepository
from src.repositories.qdrant_collection_repository import QdrantCollectionRepository
from src.repositories.tool_configuration_repository import ToolConfigurationRepository
from src.utils.logger import get_logger


logger = get_logger(__name__)


class UnitOfWork:
    """
    Unit of Work pattern - Maneja transacciones y repositorios
    El Service solo interactúa con UnitOfWork, NO con AsyncSession
    """
    
    def __init__(self, session: AsyncSession):
        self._session = session  # ✅ Variable consistente
        
        # Lazy initialization de repositories
        self._message_repository: Optional[MessageRepository] = None
        self._prompt_template_repository: Optional[PromptTemplateRepository] = None
        self._file_repository: Optional[FileRepository] = None
        self._conversation_repository: Optional[ConversationRepository] = None
        self._custom_tool_repository: Optional[CustomToolRepository] = None
        self._qdrant_collection_repository: Optional[QdrantCollectionRepository] = None
        self._tool_configuration_repository: Optional[ToolConfigurationRepository] = None
    
    @property
    def messages(self) -> MessageRepository:
        if self._message_repository is None:
            self._message_repository = MessageRepository(self._session)  # ✅ Cambio: _db → _session
        return self._message_repository
    
    @property
    def prompt_templates(self) -> PromptTemplateRepository:
        if self._prompt_template_repository is None:
            self._prompt_template_repository = PromptTemplateRepository(self._session)  # ✅ Cambio
        return self._prompt_template_repository
    
    @property
    def files(self) -> FileRepository:
        if self._file_repository is None:
            self._file_repository = FileRepository(self._session)  # ✅ Cambio
        return self._file_repository
    
    @property
    def conversations(self) -> ConversationRepository:
        if self._conversation_repository is None:
            self._conversation_repository = ConversationRepository(self._session)  # ✅ Cambio
        return self._conversation_repository
    
    @property
    def custom_tools(self) -> CustomToolRepository:
        if self._custom_tool_repository is None:
            self._custom_tool_repository = CustomToolRepository(self._session)  # ✅ Cambio
        return self._custom_tool_repository
    
    @property
    def qdrant_collections(self) -> QdrantCollectionRepository:
        if self._qdrant_collection_repository is None:
            self._qdrant_collection_repository = QdrantCollectionRepository(self._session)  # ✅ Cambio
        return self._qdrant_collection_repository
    
    @property
    def tool_configurations(self) -> ToolConfigurationRepository:
        if self._tool_configuration_repository is None:
            self._tool_configuration_repository = ToolConfigurationRepository(self._session)  # ✅ Cambio
        return self._tool_configuration_repository
    
    def get_session(self) -> AsyncSession:
        """
        Obtiene la sesión de SQLAlchemy para operaciones directas.
        
        Usar solo cuando sea necesario acceso directo a la sesión.
        La mayoría de operaciones deberían usar repositories.
        
        Returns:
            AsyncSession: La sesión de SQLAlchemy activa
        """
        return self._session
    
    async def commit(self):
        """Commit la transacción actual"""
        await self._session.commit()  # ✅ Cambio: _db → _session
        logger.debug("Transaction committed")
    
    async def rollback(self):
        """Rollback la transacción actual"""
        await self._session.rollback()  # ✅ Cambio: _db → _session
        logger.debug("Transaction rolled back")
    
    async def flush(self):
        """
        Sincroniza cambios pendientes con la BD sin hacer commit.
        Útil para obtener IDs generados antes del commit.
        """
        await self._session.flush()  # ✅ Nuevo método
        logger.debug("Session flushed")
    
    async def refresh(self, instance):
        """Refresh una instancia desde la BD"""
        await self._session.refresh(instance)  # ✅ Cambio: _db → _session
        logger.debug(f"Instance refreshed: {type(instance).__name__}")
