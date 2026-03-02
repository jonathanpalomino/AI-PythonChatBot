# =============================================================================
# src/document_loaders/text_loader.py
# =============================================================================
"""
Loader para archivos de texto plano (.txt)
"""
from pathlib import Path
from typing import List, Dict

from .base_loader import BaseDocumentLoader, DocumentSection, ProcessedDocument


class TextLoader(BaseDocumentLoader):
    """Carga y procesa archivos de texto plano"""

    def __init__(self):
        super().__init__()
        self.supported_extensions = {'.txt'}

    def load(self, file_path: Path, original_filename: str = None) -> ProcessedDocument:
        """Carga un archivo de texto plano"""
        # Usar método compartido para leer con múltiples codificaciones
        content, encoding_used = self.read_file_with_encodings(
            file_path, 
            encodings=['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        )

        # Extract basic metadata
        metadata = self._extract_metadata(content, encoding_used)
        sections = self.extract_sections(content)

        # Generate full content from sections (usar método de clase base)
        full_content = self._generate_full_content(sections)

        # Usar método compartido para obtener ruta relativa
        relative_path = self.get_relative_path(file_path)

        return ProcessedDocument(
            file_path=str(relative_path),
            file_name=file_path.name,
            original_filename=original_filename or file_path.name,
            content=full_content,
            sections=sections,
            metadata=metadata
        )

    def _extract_metadata(self, content: str, encoding: str) -> Dict:
        """Extract basic metadata from text content"""
        lines = content.split('\n')

        return {
            'line_count': len(lines),
            'char_count': len(content),
            'non_empty_lines': len([l for l in lines if l.strip()]),
            'encoding': encoding
        }

    def extract_sections(self, content: str) -> List[DocumentSection]:
        """
        Extract sections from text file
        For plain text, we split by double newlines (paragraphs)
        """
        sections = []

        # Split by double newlines (paragraphs)
        paragraphs = content.split('\n\n')

        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # Use first line as title if short, otherwise use generic title
            lines = paragraph.split('\n')
            first_line = lines[0].strip()

            # If first line is short (likely a title), use it
            if len(first_line) < 100 and len(lines) > 1:
                title = first_line
                section_content = '\n'.join(lines[1:]).strip()
            else:
                title = f"Paragraph {i + 1}"
                section_content = paragraph

            sections.append(DocumentSection(
                title=title,
                content=section_content,
                level=1,
                metadata={'paragraph_number': i + 1}
            ))

        # If no paragraphs found, treat entire content as one section
        if not sections:
            first_50 = content[:50].replace('\n', ' ').strip()
            title = first_50 + "..." if len(content) > 50 else "Text Content"

            sections.append(DocumentSection(
                title=title,
                content=content,
                level=1,
                metadata={'paragraph_number': 1}
            ))

        return sections
