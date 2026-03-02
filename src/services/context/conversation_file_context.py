"""
Contexto persistente de archivos por conversación.
Trackea archivos adjuntos + mencionados durante toda la conversación.

CHANGES:
- ✅ Convertido a lazy singleton (era eager)
- ✅ Factory con asyncio.Lock para thread-safety
- ✅ Agregado registro de archivos procesados con metadata
- ✅ Agregado método get_files_metadata para TargetFileDetector
"""

import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Any
from uuid import UUID

from src.repositories.file_repository import FileRepository
from src.repositories.message_repository import MessageRepository
from src.utils.cache import cache_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConversationFileContext:
    """
    Mantiene contexto de archivos por conversación:
    - Archivos adjuntos en mensajes previos
    - Archivos mencionados en contexto RAG
    - TTL de 1 hora para limpieza automática

    NUEVO: Soporta conversation_id por defecto para simplificar llamadas.
    """

    CACHE_NAMESPACE = 'conversation_files'
    DEFAULT_MAX_FILES = 10
    DEFAULT_TTL = 3600  # 1 hora

    def __init__(
        self,
        max_files_per_conv: int = DEFAULT_MAX_FILES,
        ttl_seconds: int = DEFAULT_TTL,
        default_conversation_id: Optional[str] = None
    ):
        self.max_files = max_files_per_conv
        self.ttl = ttl_seconds
        self.logger = logger
        self._default_conversation_id = default_conversation_id

        # Crear caché centralizado con TTL
        cache_manager.create_cache(
            namespace=self.CACHE_NAMESPACE,
            max_size=1000,  # 1000 conversaciones en caché
            ttl=ttl_seconds
        )

    def set_conversation(self, conversation_id: str):
        """Establece el conversation_id por defecto."""
        self._default_conversation_id = conversation_id

    def add_files(self, conversation_id: str, file_ids: List[UUID]):
        """Agrega archivos al contexto conversacional."""
        if not file_ids:
            return

        # Obtener contexto actual (o dict vacío)
        cache_key = self._get_cache_key(conversation_id)
        current_context = cache_manager.get(self.CACHE_NAMESPACE, cache_key) or {}

        # Agregar nuevos archivos (manteniendo orden de inserción)
        for file_id in file_ids:
            file_id_str = str(file_id)
            # Si existe, lo mueve al final (más reciente)
            if file_id_str in current_context:
                del current_context[file_id_str]
            current_context[file_id_str] = True  # Valor dummy, solo nos importa el orden

        # Limitar a max_files (eliminar más antiguos)
        if len(current_context) > self.max_files:
            # Obtener keys más antiguas y eliminarlas
            keys_to_remove = list(current_context.keys())[:-self.max_files]
            for key in keys_to_remove:
                del current_context[key]

        # Guardar en caché
        cache_manager.set(self.CACHE_NAMESPACE, cache_key, current_context)
        self.logger.debug(
            f"Conversation {conversation_id[:8]} - Files context: {len(current_context)} files",
            extra={"files": list(current_context.keys())[:5]}
        )

    async def get_recent_files(
        self,
        conversation_id: str,
        file_repo: FileRepository,
        message_repo: Optional[MessageRepository] = None,
        max_files: int = 5
    ) -> List[UUID]:
        """
        Obtiene archivos recientes vinculados a la conversación.

        Pasos:
        1. Cache en memoria: Rápido y mantiene orden.
        2. Validación via Repository: Asegura que los archivos aún existen.
        3. Fallback via History (si hay message_repo): Busca archivos en mensajes pasados.
        """
        # 1. Intentar obtener desde caché
        cache_key = self._get_cache_key(conversation_id)
        cached_context = cache_manager.get(self.CACHE_NAMESPACE, cache_key)

        if cached_context:
            cached_ids = [UUID(fid) for fid in cached_context.keys()]
            try:
                # Validar existencia via FileRepository
                valid_files = await file_repo.get_by_ids(cached_ids)
                valid_ids = [f.id for f in valid_files]

                if valid_ids:
                    self.logger.debug(
                        f"Found {len(valid_ids)} valid files in context for {conversation_id[:8]}"
                    )
                    return valid_ids[-max_files:]
            except Exception as e:
                self.logger.warning(f"File context validation failed: {e}")

        # 2. Fallback: Buscar en el historial de mensajes si se provee el repositorio
        if message_repo:
            try:
                # Usar el repositorio para obtener mensajes recientes
                messages = await message_repo.get_last_n_messages(
                    conversation_id=UUID(conversation_id),
                    n=20
                )

                all_file_ids = []
                for msg in messages:
                    if msg.attachments:
                        for att in msg.attachments:
                            if isinstance(att, dict) and "file_id" in att:
                                try:
                                    all_file_ids.append(UUID(att["file_id"]))
                                except (ValueError, TypeError):
                                    continue

                # Deduplicar manteniendo orden (los más recientes al final)
                seen = set()
                unique_ids = []
                for fid in all_file_ids:
                    if fid not in seen:
                        seen.add(fid)
                        unique_ids.append(fid)

                if unique_ids:
                    # Actualizar caché para futuras llamadas rápidas
                    self.add_files(conversation_id, unique_ids)
                    self.logger.info(
                        f"Loaded {len(unique_ids)} files from history via Repository",
                        extra={"conversation_id": conversation_id[:8]}
                    )
                    return unique_ids[-max_files:]
            except Exception as e:
                self.logger.error(f"Failed to load files from history repository: {e}")

        return []


    def remove_file(self, conversation_id: str, file_id: UUID):
        """Remueve un archivo específico del contexto."""
        cache_key = self._get_cache_key(conversation_id)
        context = cache_manager.get(self.CACHE_NAMESPACE, cache_key)

        if not context:
            return

        file_id_str = str(file_id)
        if file_id_str in context:
            del context[file_id_str]
            cache_manager.set(self.CACHE_NAMESPACE, cache_key, context)
            self.logger.debug(
                f"Removed file {file_id_str[:8]} from conversation {conversation_id[:8]}"
            )

    def clear_conversation(self, conversation_id: str):
        """Limpia todo el contexto de una conversación."""
        cache_key = self._get_cache_key(conversation_id)
        cache_manager.delete(self.CACHE_NAMESPACE, cache_key)
        self.logger.info(f"Cleared file context for conversation {conversation_id[:8]}")

    def get_stats(self) -> Dict:
        """Retorna estadísticas del caché de contextos."""
        cache_stats = cache_manager.get_stats(self.CACHE_NAMESPACE)
        return {
            'namespace': self.CACHE_NAMESPACE,
            'conversations_cached': cache_stats.get(self.CACHE_NAMESPACE, {}).get('size', 0),
            'max_files_per_conv': self.max_files,
            'ttl_seconds': self.ttl,
            **cache_stats.get(self.CACHE_NAMESPACE, {})
        }

    def _get_cache_key(self, conversation_id: str) -> str:
        """Genera key de caché para una conversación."""
        return f"conv_files:{conversation_id}"

    # =============================================================================
    # NUEVOS MÉTODOS PARA SISTEMA DE CONTEXTO INTELIGENTE
    # =============================================================================

    def register_file_processed(
        self,
        file_id: UUID,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None,
        conversation_id: Optional[str] = None
    ):
        """
        Registra un archivo cuando es procesado por una herramienta.

        Esto permite trackear qué archivos han sido procesados y cuándo,
        para que el TargetFileDetector pueda usar esta información.

        Args:
            file_id: ID del archivo procesado
            filename: Nombre del archivo
            metadata: Metadata adicional (opcional)
            conversation_id: ID de la conversación (opcional, usa default si no se provee)
        """
        conv_id = conversation_id or self._default_conversation_id
        if not conv_id:
            self.logger.warning("Cannot register file: no conversation_id set")
            return

        cache_key = self._get_processed_files_key(conv_id)
        processed = cache_manager.get(self.CACHE_NAMESPACE, cache_key) or {}

        file_id_str = str(file_id)
        processed[file_id_str] = {
            'id': file_id,
            'filename': filename,
            'processed_at': datetime.utcnow().isoformat(),
            'metadata': metadata or {}
        }

        # Limitar a max_files
        if len(processed) > self.max_files:
            # Eliminar los más antiguos
            sorted_items = sorted(
                processed.items(),
                key=lambda x: x[1].get('processed_at', '')
            )
            for key, _ in sorted_items[:-self.max_files]:
                del processed[key]

        cache_manager.set(self.CACHE_NAMESPACE, cache_key, processed)
        self.logger.debug(
            f"Registered processed file: {filename}",
            extra={"conversation_id": conv_id[:8], "file_id": str(file_id)[:8]}
        )

    def get_files_metadata(
        self,
        conversation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtiene la lista de archivos procesados con metadata.

        Usado por TargetFileDetector para detectar el archivo objetivo.

        Args:
            conversation_id: ID de la conversación (opcional, usa default si no se provee)

        Returns:
            Lista de dicts con: id, filename, uploaded_at, metadata
        """
        conv_id = conversation_id or self._default_conversation_id
        if not conv_id:
            return []

        cache_key = self._get_processed_files_key(conv_id)
        processed = cache_manager.get(self.CACHE_NAMESPACE, cache_key) or {}

        result = []
        for file_id_str, info in processed.items():
            result.append({
                'id': UUID(file_id_str),
                'filename': info.get('filename', 'unknown'),
                'uploaded_at': datetime.fromisoformat(info['processed_at']) if info.get('processed_at') else datetime.min,
                'metadata': info.get('metadata', {})
            })

        # Ordenar por fecha de procesamiento
        result.sort(key=lambda x: x.get('uploaded_at', datetime.min))

        return result

    def get_last_processed_file(
        self,
        conversation_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Obtiene el último archivo procesado en la conversación.

        Args:
            conversation_id: ID de la conversación (opcional, usa default si no se provee)

        Returns:
            Dict con info del archivo o None si no hay archivos
        """
        files = self.get_files_metadata(conversation_id)
        if files:
            return files[-1]
        return None

    def _get_processed_files_key(self, conversation_id: str) -> str:
        """Genera key de caché para archivos procesados."""
        return f"conv_processed:{conversation_id}"


# =============================================================================
# ✅ SINGLETON FACTORY (Lazy Loading)
# =============================================================================
_conversation_file_ctx: Optional[ConversationFileContext] = None
_ctx_lock = asyncio.Lock()


async def get_conversation_file_context(
    max_files_per_conv: int = ConversationFileContext.DEFAULT_MAX_FILES,
    ttl_seconds: int = ConversationFileContext.DEFAULT_TTL
) -> ConversationFileContext:
    """
    Lazy singleton factory para ConversationFileContext.
    Thread-safe con asyncio.Lock + double-checked locking.

    Args:
        max_files_per_conv: Máximo de archivos por conversación (solo primera llamada)
        ttl_seconds: TTL del cache en segundos (solo primera llamada)

    Returns:
        La única instancia de ConversationFileContext.
    """
    global _conversation_file_ctx

    # Fast path: already initialized
    if _conversation_file_ctx is not None:
        return _conversation_file_ctx

    # Slow path: need to initialize with lock
    async with _ctx_lock:
        # Double-check after acquiring lock
        if _conversation_file_ctx is None:
            _conversation_file_ctx = ConversationFileContext(
                max_files_per_conv=max_files_per_conv,
                ttl_seconds=ttl_seconds
            )
            logger.info("ConversationFileContext singleton initialized")

        return _conversation_file_ctx


# =============================================================================
# AGREGAR AL FINAL DE ConversationFileContext (antes del singleton)
# =============================================================================

def get_previous_file(
    self,
    exclude_file_id: Optional[UUID] = None,
    conversation_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Obtiene el archivo anterior al especificado.

    Args:
        exclude_file_id: Archivo actual a excluir (opcional)
        conversation_id: ID de la conversación (opcional)

    Returns:
        Dict con info del archivo previo o None

    Examples:
         # Usuario procesó: A, B, C (C es el último)
         get_previous_file(exclude_file_id=C)
        {'id': B_id, 'filename': 'B.py', ...}
    """
    files = self.get_files_metadata(conversation_id)

    if len(files) < 2:
        return None

    if exclude_file_id:
        # Buscar el archivo a excluir
        try:
            idx = next(i for i, f in enumerate(files) if f['id'] == exclude_file_id)
            # Retornar el anterior
            if idx > 0:
                self.logger.debug(
                    f"Previous file to {files[idx]['filename']}: "
                    f"{files[idx - 1]['filename']}"
                )
                return files[idx - 1]
        except StopIteration:
            pass

    # Sin exclude, retornar penúltimo
    return files[-2]

# =============================================================================
# ⚠️ BACKWARD COMPATIBILITY (deprecar en futuro)
# =============================================================================
# Si ya tienes código que usa conversation_file_ctx directamente,
# puedes mantener esta línea temporalmente:
# conversation_file_ctx = ConversationFileContext()  # ❌ DEPRECADO
