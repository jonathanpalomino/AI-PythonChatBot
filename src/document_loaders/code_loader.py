# =============================================================================
# src/document_loaders/code_loader.py
# =============================================================================
from pathlib import Path
from typing import List, Dict, Any
import re

from .base_loader import BaseDocumentLoader, DocumentSection, ProcessedDocument

LANGUAGE_BY_EXTENSION = {
    '.py': 'python',
    '.java': 'java',
    '.cs': 'csharp',
    '.js': 'javascript',
    '.ts': 'typescript',
    '.go': 'go'
}

CLASS_REGEX = {
    'python': re.compile(r'^class\s+(\w+)', re.MULTILINE),
    'java': re.compile(r'\bclass\s+(\w+)'),
    'csharp': re.compile(r'\bclass\s+(\w+)'),
}

FUNCTION_REGEX = {
    'python': re.compile(r'^def\s+(\w+)', re.MULTILINE),
    'java': re.compile(r'\b(?:public|private|protected)?\s+\w+\s+(\w+)\s*\(', re.MULTILINE),
    'csharp': re.compile(r'\b(?:public|private|protected)?\s+\w+\s+(\w+)\s*\(', re.MULTILINE),
}

class CodeLoader(BaseDocumentLoader):
    """Loader genérico para código fuente"""

    def __init__(self):
        super().__init__()
        self.supported_extensions = set(LANGUAGE_BY_EXTENSION.keys())

    def load(self, file_path: Path, original_filename: str = None) -> ProcessedDocument:
        content, encoding = self._read_file(file_path)

        language = LANGUAGE_BY_EXTENSION.get(file_path.suffix.lower(), 'unknown')

        metadata = {
            'type': 'code',
            'language': language,
            'line_count': content.count('\n'),
            'char_count': len(content),
            'encoding': encoding
        }

        sections = self.extract_sections(content, language)

        return ProcessedDocument(
            file_path=str(file_path),
            file_name=file_path.name,
            original_filename=original_filename or file_path.name,
            content=content,
            sections=sections,
            metadata=metadata
        )

    def extract_sections(self, content: str, language: str) -> List[DocumentSection]:
        sections = []
        section_number = 1

        # 1️⃣ Clases
        for match in CLASS_REGEX.get(language, []).finditer(content):
            sections.append(DocumentSection(
                title=f"Class: {match.group(1)}",
                content=match.group(0),
                level=1,
                metadata={
                    'symbol_type': 'class',
                    'name': match.group(1),
                    'language': language,
                    'section_number': section_number
                }
            ))
            section_number += 1

        # 2️⃣ Funciones / métodos
        for match in FUNCTION_REGEX.get(language, []).finditer(content):
            sections.append(DocumentSection(
                title=f"Function: {match.group(1)}",
                content=match.group(0),
                level=2,
                metadata={
                    'symbol_type': 'function',
                    'name': match.group(1),
                    'language': language,
                    'section_number': section_number
                }
            ))
            section_number += 1

        # 3️⃣ Fallback: archivo completo
        if not sections:
            sections.append(DocumentSection(
                title="Full File",
                content=content,
                level=0,
                metadata={'language': language}
            ))

        return sections

    def _read_file(self, file_path: Path):
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read(), encoding
            except UnicodeDecodeError:
                continue
        raise ValueError(f"No se pudo leer {file_path}")
