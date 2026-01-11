# =============================================================================
# src/document_loaders/base_loader.py
# =============================================================================
"""
Clase base para todos los loaders de documentos
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any


@dataclass
class DocumentSection:
    """Representa una secciÃ³n de un documento"""
    title: str
    content: str
    level: int
    metadata: Dict[str, Any]


@dataclass
class ProcessedDocument:
    """Documento procesado listo para indexar"""
    file_path: str
    file_name: str
    content: str
    sections: List[DocumentSection]
    metadata: Dict[str, Any]
    original_filename: str = None


class BaseDocumentLoader(ABC):
    """Clase base para cargar y procesar documentos"""

    def __init__(self):
        self.supported_extensions = set()

    def can_load(self, file_path: Path) -> bool:
        """Verifica si este loader puede procesar el archivo"""
        return file_path.suffix.lower() in self.supported_extensions

    @abstractmethod
    def load(self, file_path: Path, original_filename: str = None) -> ProcessedDocument:
        """Carga y procesa un documento
        
        Args:
            file_path: Ruta al archivo
            original_filename: Nombre original del archivo (opcional, para conservar nombre original)
        """
        pass

    @abstractmethod
    def extract_sections(self, content: str) -> List[DocumentSection]:
        """Extrae secciones del contenido"""
        pass
