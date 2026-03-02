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
    recommended_chunk_size: int = None  # Loader-defined optimal chunk size


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

    # =============================================================================
    # Métodos Estáticos Compartidos (para reducir duplicación)
    # =============================================================================

    @staticmethod
    def get_relative_path(file_path: Path) -> Path:
        """
        Convert file path to relative path from cwd.
        
        Args:
            file_path: Path to convert
            
        Returns:
            Relative path if possible, otherwise absolute path
        """
        abs_path = file_path if file_path.is_absolute() else file_path.resolve()
        try:
            return abs_path.relative_to(Path.cwd())
        except ValueError:
            return abs_path

    @staticmethod
    def read_file_with_encodings(file_path: Path, encodings: List[str] = None) -> tuple:
        """
        Read file trying multiple encodings.
        
        Args:
            file_path: Path to the file
            encodings: List of encodings to try (default: utf-8, latin-1, cp1252)
            
        Returns:
            Tuple of (content, encoding_used)
            
        Raises:
            ValueError: If file cannot be read with any encoding
        """
        if encodings is None:
            encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read(), encoding
            except UnicodeDecodeError:
                continue
        raise ValueError(f"No se pudo leer {file_path} con ninguna codificación")

    def _generate_full_content(self, sections: List[DocumentSection]) -> str:
        """
        Generate full content from sections (default implementation).
        Can be overridden by subclasses for custom formatting.
        
        Args:
            sections: List of document sections
            
        Returns:
            Formatted full content string
        """
        full_content = []
        for section in sections:
            header_prefix = '#' * (section.level if section.level > 0 else 1)
            full_content.append(f"{header_prefix} {section.title}")
            full_content.append(section.content)
            full_content.append("")  # Línea en blanco entre secciones
        return '\n'.join(full_content).strip()
