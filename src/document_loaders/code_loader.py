import logging
from pathlib import Path
from typing import List

from .base_loader import BaseDocumentLoader, DocumentSection, ProcessedDocument
from src.config.constants import CODE_CHUNK_SIZE
from ..services.analysis.codebase_analyzer import CodebaseAnalyzer, LANGUAGE_BY_EXTENSION, \
    CodeSymbol

logger = logging.getLogger(__name__)

class CodeLoader(BaseDocumentLoader):
    """
    Advanced Code Loader.
    Delegates all analysis to CodebaseAnalyzer (Multi-language).
    """

    def __init__(self):
        super().__init__()
        self.supported_extensions = set(LANGUAGE_BY_EXTENSION.keys())
        self.analyzer = CodebaseAnalyzer()

    def extract_sections(self, content: str) -> List[DocumentSection]:
        """
        Implementation to satisfy abstract method.
        For code, we typically use load() which has file context.
        """
        return [self._create_full_file_section(content, "unknown")]

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

        sections = []
        try:
            # Universal analysis request
            analysis = self.analyzer.analyze_file(content, file_path.name)

            if "error" in analysis:
                logger.warning(f"Analysis error for {file_path}: {analysis['error']}")
                sections = [self._create_full_file_section(content, language)]
            else:
                symbols: List[CodeSymbol] = analysis.get("symbols", [])
                metadata['complexity'] = analysis.get("complexity", 1)
                metadata['imports'] = analysis.get("imports", [])

                # Convert symbols to sections
                symbols.sort(key=lambda x: x.start_line)
                for i, sym in enumerate(symbols):
                    title_prefix = sym.type.capitalize()
                    title = f"{title_prefix}: {sym.name}"
                    if sym.parent:
                        title = f"{title_prefix}: {sym.parent}.{sym.name}"

                    sections.append(DocumentSection(
                        title=title,
                        content=sym.content,
                        level=1 if sym.type == 'class' else 2,
                        metadata={
                            'symbol_type': sym.type,
                            'name': sym.name,
                            'parent': sym.parent,
                            'start_line': sym.start_line,
                            'end_line': sym.end_line,
                            'dependencies': list(sym.dependencies),
                            'section_number': i + 1,
                            'language': language
                        }
                    ))

                # Fallback if no symbols found
                if not sections:
                     sections = [self._create_full_file_section(content, language)]

        except Exception as e:
            logger.error(f"Analysis failed for {file_path}: {e}")
            sections = [self._create_full_file_section(content, language)]

        return ProcessedDocument(
            file_path=str(file_path),
            file_name=file_path.name,
            original_filename=original_filename or file_path.name,
            content=content,
            sections=sections,
            metadata=metadata,
            recommended_chunk_size=CODE_CHUNK_SIZE  # Large chunks to preserve code units
        )

    def _create_full_file_section(self, content: str, language: str) -> DocumentSection:
        return DocumentSection(
            title="Full File",
            content=content,
            level=0,
            metadata={'language': language, 'symbol_type': 'file'}
        )

    def _read_file(self, file_path: Path):
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read(), encoding
            except UnicodeDecodeError:
                continue
        raise ValueError(f"No se pudo leer {file_path}")
