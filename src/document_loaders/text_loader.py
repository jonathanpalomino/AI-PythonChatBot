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
        # Try multiple encodings
        content = None
        encoding_used = 'utf-8'

        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                encoding_used = encoding
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            raise ValueError(f"No se pudo decodificar el archivo {file_path}")

        # Extract basic metadata
        metadata = self._extract_metadata(content, encoding_used)
        sections = self.extract_sections(content)

        # Generate full content from sections
        full_content = self._generate_full_content(sections)

        # Convertir a ruta relativa
        abs_path = file_path if file_path.is_absolute() else file_path.resolve()
        try:
            relative_path = abs_path.relative_to(Path.cwd())
        except ValueError:
            relative_path = abs_path

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

    def _generate_full_content(self, sections: List[DocumentSection]) -> str:
        """Genera el contenido completo del documento desde las secciones"""
        full_content = []

        for section in sections:
            full_content.append(f"# {section.title}")
            full_content.append(section.content)
            full_content.append("")  # Línea en blanco entre secciones

        return '\n'.join(full_content).strip()
