"""
TargetFileDetector: Detecta el archivo objetivo de una query.

Lógica de detección:
1. Si hay file_ids adjuntos en la consulta actual:
   - Si es un solo archivo → ese es el objetivo
   - Si son múltiples → buscar mención explícita en la query
   - Si no hay mención → usar el último archivo adjunto

2. Si no hay file_ids adjuntos:
   - Buscar mención explícita en la query (nombre de archivo)
   - Si no hay mención → usar el último archivo del historial conversacional

RESPONSABILIDAD: Detección de archivo objetivo (SRP)
NO accede directamente a la BD - usa Repository pattern
"""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from uuid import UUID

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.repositories import FileRepository
    from src.services.context.conversation_file_context import ConversationFileContext

logger = get_logger(__name__)


@dataclass
class TargetInfo:
    """Información del archivo objetivo detectado."""
    file_id: UUID
    filename: str
    source: str  # 'attached', 'mentioned', 'history', 'fallback'
    confidence: float


class TargetFileDetector:
    """
    Detecta el archivo objetivo basándose en:
    1. file_ids adjuntos en la consulta actual
    2. Mención explícita en la query
    3. Historial conversacional

    RESPONSABILIDAD ÚNICA: Detección de archivo objetivo
    NO accede directamente a la BD - usa datos proporcionados
    """

    # Patrones para detectar mención de archivo
    FILE_PATTERNS = [
        r'archivo\s+["\']?(\S+?\.\w+)["\']?',  # archivo "nombre.py"
        r'fichero\s+["\']?(\S+?\.\w+)["\']?',  # fichero nombre.py
        r'file\s+["\']?(\S+?\.\w+)["\']?',     # file nombre.py
        r'(\S+?\.py)',                         # nombre.py (archivos Python)
        r'(\S+?\.js)',                         # nombre.js (archivos JS)
        r'(\S+?\.java)',                       # nombre.java (archivos Java)
        r'(\S+?\.ts)',                         # nombre.ts (archivos TS)
        r'(\S+?\.sql)',                        # nombre.sql (archivos SQL)
    ]

    # Referencias textuales comunes
    THIS_FILE_PATTERNS = [
        r'este\s+archivo',
        r'este\s+fichero',
        r'this\s+file',
        r'el\s+archivo',
    ]

    OTHER_FILE_PATTERNS = [
        r'el\s+otro\s+archivo',
        r'el\s+otro\s+fichero',
        r'the\s+other\s+file',
        r'otro\s+archivo',
    ]

    PREVIOUS_FILE_PATTERNS = [
        r'el\s+anterior',
        r'el\s+archivo\s+anterior',
        r'archivo\s+previo',
        r'the\s+previous',
        r'the\s+previous\s+file',
        r'previous\s+file',
    ]

    FIRST_FILE_PATTERNS = [
        r'el\s+primer\s+archivo',
        r'el\s+primero',
        r'the\s+first\s+file',
    ]

    LAST_FILE_PATTERNS = [
        r'el\s+[úu]ltimo\s+archivo',
        r'el\s+[úu]ltimo',
        r'the\s+last\s+file',
    ]

    def __init__(self, file_repo: Optional['FileRepository'] = None):
        self.logger = logger
        self.file_repo = file_repo

    async def detect_target(
        self,
        conversation_id: UUID,
        attached_file_ids: List[UUID],
        user_message: str,
        conversation_file_context: Optional['ConversationFileContext'] = None
    ) -> Optional[TargetInfo]:
        """
        Método async para detectar el archivo objetivo.

        Este método integra:
        1. Los file_ids adjuntos en la consulta actual
        2. El contexto de archivos de la conversación
        3. El historial de archivos procesados

        Args:
            conversation_id: ID de la conversación
            attached_file_ids: file_ids adjuntos en la consulta actual
            user_message: Mensaje del usuario
            conversation_file_context: Contexto de archivos de la conversación (opcional)

        Returns:
            TargetInfo con información del archivo objetivo, o None si no se detecta
        """
        self.logger.debug(
            f"Detecting target file (async)",
            extra={
                "conversation_id": str(conversation_id),
                "attached_file_ids_count": len(attached_file_ids) if attached_file_ids else 0,
                "user_message_preview": user_message[:50]
            }
        )

        # Obtener archivos de la conversación
        conversation_files = []

        # 1. Desde el contexto de archivos de la conversación
        if conversation_file_context:
            conversation_files = conversation_file_context.get_files_metadata()

        # 2. Si hay file_repo, obtener archivos adicionales de la BD
        if self.file_repo and not conversation_files:
            try:
                files = await self.file_repo.get_by_conversation(conversation_id)
                conversation_files = [
                    {
                        "id": f.id,
                        "filename": f.file_name,  # CORREGIDO: usar file_name, no filename
                        "uploaded_at": f.uploaded_at  # CORREGIDO: usar uploaded_at, no created_at
                    }
                    for f in files
                ]
            except Exception as e:
                self.logger.warning(f"Failed to get conversation files: {e}")

        # Usar el método síncrono para la detección
        return self.detect_target_sync(
            query=user_message,
            current_file_ids=attached_file_ids,
            conversation_files=conversation_files
        )

    def detect_target_sync(
        self,
        query: str,
        current_file_ids: Optional[List[UUID]],
        conversation_files: List[Dict[str, Any]],
        message_history: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[TargetInfo]:
        """
        Detecta el archivo objetivo para la query actual (método síncrono).

        Args:
            query: Mensaje del usuario
            current_file_ids: file_ids adjuntos en la consulta actual
            conversation_files: Lista de archivos en la conversación con metadata:
                [{"id": UUID, "filename": str, "uploaded_at": datetime}, ...]
            message_history: Historial de mensajes (opcional) para detectar contexto

        Returns:
            TargetInfo con información del archivo objetivo, o None si no se detecta
        """
        self.logger.debug(
            f"Detecting target file (sync)",
            extra={
                "query_preview": query[:50],
                "current_file_ids_count": len(current_file_ids) if current_file_ids else 0,
                "conversation_files_count": len(conversation_files)
            }
        )

        # Paso 1: Si hay file_ids adjuntos, priorizar
        if current_file_ids and len(current_file_ids) > 0:
            result = self._detect_from_attached(query, current_file_ids, conversation_files)
            if result:
                return result

        # Paso 2: Buscar mención explícita en la query
        result = self._detect_from_mention(query, conversation_files)
        if result:
            return result

        # Paso 3: Detectar referencias textuales ("este archivo", "el otro", etc.)
        result = self._detect_from_reference(query, conversation_files)
        if result:
            return result

        # Paso 4: Usar el último archivo del historial
        result = self._detect_from_history(conversation_files)
        if result:
            return result

        self.logger.debug("No target file detected")
        return None

    def _detect_from_attached(
        self,
        query: str,
        current_file_ids: List[UUID],
        conversation_files: List[Dict[str, Any]]
    ) -> Optional[TargetInfo]:
        """
        Detecta archivo objetivo desde file_ids adjuntos.

        Lógica:
        - Un solo archivo → ese es el objetivo
        - Múltiples archivos → buscar mención explícita
        - Sin mención → usar el último (más reciente)
        """
        if len(current_file_ids) == 1:
            # Un solo archivo adjunto → ese es el objetivo
            file_info = self._find_file_info(current_file_ids[0], conversation_files)
            if file_info:
                self.logger.info(f"🎯 Single attached file detected: {file_info['filename']}")
                return TargetInfo(
                    file_id=current_file_ids[0],
                    filename=file_info['filename'],
                    source='attached',
                    confidence=1.0
                )

        # Múltiples archivos → buscar mención explícita
        mentioned = self._find_mentioned_file(query, current_file_ids, conversation_files)
        if mentioned:
            self.logger.info(f"🎯 Mentioned file detected from attached: {mentioned.filename}")
            return mentioned

        # Sin mención → usar el último adjunto (más reciente)
        if conversation_files:
            # Filtrar solo los archivos adjuntos actuales
            attached_files = [
                f for f in conversation_files
                if f['id'] in current_file_ids
            ]
            if attached_files:
                # Ordenar por fecha de subida (más reciente último)
                attached_files.sort(key=lambda x: x.get('uploaded_at', datetime.min))
                last_file = attached_files[-1]
                self.logger.info(f"🎯 Last attached file as fallback: {last_file['filename']}")
                return TargetInfo(
                    file_id=last_file['id'],
                    filename=last_file['filename'],
                    source='attached',
                    confidence=0.8
                )

        return None

    def _detect_from_mention(
        self,
        query: str,
        conversation_files: List[Dict[str, Any]]
    ) -> Optional[TargetInfo]:
        """
        Busca si la query menciona explícitamente un archivo por nombre.
        """
        query_lower = query.lower()

        for pattern in self.FILE_PATTERNS:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            for match in matches:
                # Buscar archivo que coincida
                for file_info in conversation_files:
                    filename = file_info['filename'].lower()
                    if match.lower() in filename or filename in match.lower():
                        self.logger.info(f"🎯 File mentioned in query: {file_info['filename']}")
                        return TargetInfo(
                            file_id=file_info['id'],
                            filename=file_info['filename'],
                            source='mentioned',
                            confidence=0.9
                        )

        return None

    def _detect_from_reference(
        self,
        query: str,
        conversation_files: List[Dict[str, Any]]
    ) -> Optional[TargetInfo]:
        """
        Detecta referencias textuales como "este archivo", "el otro", etc.
        """
        if not conversation_files:
            return None

        query_lower = query.lower()

        # "este archivo" → último archivo
        for pattern in self.THIS_FILE_PATTERNS:
            if re.search(pattern, query_lower):
                last_file = conversation_files[-1]
                self.logger.info(f"🎯 'This file' reference detected: {last_file['filename']}")
                return TargetInfo(
                    file_id=last_file['id'],
                    filename=last_file['filename'],
                    source='mentioned',
                    confidence=0.85
                )

        # "el otro archivo" → penúltimo archivo
        for pattern in self.OTHER_FILE_PATTERNS:
            if re.search(pattern, query_lower):
                if len(conversation_files) >= 2:
                    other_file = conversation_files[-2]
                    self.logger.info(f"🎯 'Other file' reference detected: {other_file['filename']}")
                    return TargetInfo(
                        file_id=other_file['id'],
                        filename=other_file['filename'],
                        source='mentioned',
                        confidence=0.8
                    )

        # ✅ NUEVO: "el anterior", "previo"
        for pattern in self.PREVIOUS_FILE_PATTERNS:
            if re.search(pattern, query_lower):
                if len(conversation_files) >= 2:
                    previous_file = conversation_files[-2]
                    self.logger.info(
                        f"🎯 'Previous file' reference detected: {previous_file['filename']}")
                    return TargetInfo(
                        file_id=previous_file['id'],
                        filename=previous_file['filename'],
                        source='mentioned',
                        confidence=0.82
                    )

        # "el primer archivo" → primer archivo
        for pattern in self.FIRST_FILE_PATTERNS:
            if re.search(pattern, query_lower):
                first_file = conversation_files[0]
                self.logger.info(f"🎯 'First file' reference detected: {first_file['filename']}")
                return TargetInfo(
                    file_id=first_file['id'],
                    filename=first_file['filename'],
                    source='mentioned',
                    confidence=0.85
                )

        # "el último archivo" → último archivo
        for pattern in self.LAST_FILE_PATTERNS:
            if re.search(pattern, query_lower):
                last_file = conversation_files[-1]
                self.logger.info(f"🎯 'Last file' reference detected: {last_file['filename']}")
                return TargetInfo(
                    file_id=last_file['id'],
                    filename=last_file['filename'],
                    source='mentioned',
                    confidence=0.85
                )

        return None

    def _detect_from_history(
        self,
        conversation_files: List[Dict[str, Any]]
    ) -> Optional[TargetInfo]:
        """
        Usa el último archivo del historial como fallback.
        """
        if not conversation_files:
            return None

        # Ordenar por fecha y tomar el más reciente
        sorted_files = sorted(
            conversation_files,
            key=lambda x: x.get('uploaded_at', datetime.min),
            reverse=True
        )

        if sorted_files:
            last_file = sorted_files[0]
            self.logger.info(f"🎯 Last file from history as fallback: {last_file['filename']}")
            return TargetInfo(
                file_id=last_file['id'],
                filename=last_file['filename'],
                source='history',
                confidence=0.6
            )

        return None

    def _find_file_info(
        self,
        file_id: UUID,
        conversation_files: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Busca información de un archivo por ID."""
        for file_info in conversation_files:
            if file_info['id'] == file_id:
                return file_info
        return None

    def _find_mentioned_file(
        self,
        query: str,
        file_ids: List[UUID],
        conversation_files: List[Dict[str, Any]]
    ) -> Optional[TargetInfo]:
        """Busca archivo mencionado explícitamente en la query."""
        query_lower = query.lower()

        for pattern in self.FILE_PATTERNS:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            for match in matches:
                for file_info in conversation_files:
                    if file_info['id'] not in file_ids:
                        continue
                    filename = file_info['filename'].lower()
                    if match.lower() in filename or filename in match.lower():
                        return TargetInfo(
                            file_id=file_info['id'],
                            filename=file_info['filename'],
                            source='mentioned',
                            confidence=0.9
                        )

        return None


# =============================================================================
# SINGLETON FACTORY (Lazy Loading)
# =============================================================================
_target_detector: Optional[TargetFileDetector] = None


def get_target_file_detector() -> TargetFileDetector:
    """
    Singleton factory para TargetFileDetector.

    Returns:
        La única instancia de TargetFileDetector.
    """
    global _target_detector

    if _target_detector is None:
        _target_detector = TargetFileDetector()
        logger.info("TargetFileDetector singleton initialized")

    return _target_detector
