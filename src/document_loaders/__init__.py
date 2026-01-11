# =============================================================================
# src/document_loaders/__init__.py
# =============================================================================
"""
Factory para obtener el loader apropiado segÃºn el tipo de archivo
"""
from pathlib import Path
from typing import Optional

from .code_loader import CodeLoader
from .sql_plsql_loader import SqlPlsqlLoader
from .base_loader import BaseDocumentLoader
from .csv_loader import CSVLoader
from .excel_loader import ExcelLoader
from .html_loader import HTMLLoader
from .json_loader import JSONLoader
from .markdown_loader import MarkdownLoader
from .pdf_loader import PDFLoader
from .pptx_loader import PPTXLoader
from .text_loader import TextLoader
from .word_loader import WordLoaderUniversal


class DocumentLoaderFactory:
    """Factory para obtener el loader apropiado"""

    _loaders = [
        MarkdownLoader(),
        WordLoaderUniversal(),
        TextLoader(),
        PDFLoader(),
        PPTXLoader(),
        JSONLoader(),
        CSVLoader(),
        HTMLLoader(),
        ExcelLoader(),
        SqlPlsqlLoader(),
        CodeLoader(),  # Descomentar si se desea incluir el CodeLoader
        # Agregar mÃ¡s loaders aquÃ­ segÃºn se implementen
    ]

    @classmethod
    def get_loader(cls, file_path: Path) -> Optional[BaseDocumentLoader]:
        """Obtiene el loader apropiado para un archivo"""
        for loader in cls._loaders:
            if loader.can_load(file_path):
                return loader
        return None

    @classmethod
    def get_supported_extensions(cls) -> set:
        """Retorna todas las extensiones soportadas"""
        extensions = set()
        for loader in cls._loaders:
            extensions.update(loader.supported_extensions)
        return extensions

    @classmethod
    def get_loader_info(cls) -> dict:
        """
        Retorna información detallada sobre cada loader y sus extensiones

        Returns:
            dict: {
                'loaders': [
                    {
                        'name': 'PDFLoader',
                        'extensions': ['.pdf'],
                        'description': 'Loader for PDF documents'
                    },
                    ...
                ],
                'all_extensions': ['.pdf', '.docx', ...]
            }
        """
        loader_details = []
        all_extensions = set()

        for loader in cls._loaders:
            loader_name = loader.__class__.__name__
            extensions = list(loader.supported_extensions)
            all_extensions.update(extensions)

            loader_details.append({
                'name': loader_name,
                'extensions': sorted(extensions),
                'description': f'Loader for {", ".join(extensions)} files'
            })

        return {
            'loaders': loader_details,
            'all_extensions': sorted(list(all_extensions))
        }
