# =============================================================================
# src/document_loaders/csv_loader.py
# =============================================================================
"""
Loader para archivos CSV
"""
import csv
from pathlib import Path
from typing import List, Dict, Any

from .base_loader import BaseDocumentLoader, DocumentSection, ProcessedDocument


class CSVLoader(BaseDocumentLoader):
    """Carga y procesa archivos CSV"""

    def __init__(self, max_rows_per_section: int = 100):
        super().__init__()
        self.supported_extensions = {'.csv'}
        self.max_rows_per_section = max_rows_per_section

    def load(self, file_path: Path, original_filename: str = None) -> ProcessedDocument:
        """Carga un archivo CSV"""
        # Detectar delimitador y encoding
        delimiter, encoding = self._detect_csv_format(file_path)

        with open(file_path, 'r', encoding=encoding, newline='') as f:
            reader = csv.DictReader(f, delimiter=delimiter)

            # Convertir a lista para poder procesar
            rows = list(reader)
            headers = reader.fieldnames if reader.fieldnames else []

        # Extract metadata
        metadata = self._extract_metadata(rows, headers, delimiter, encoding)

        # Extract sections (dividir en chunks si es muy grande)
        sections = self.extract_sections(rows, headers)

        # Generate full content
        full_content = self._generate_full_content(sections, headers)

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

    def _detect_csv_format(self, file_path: Path) -> tuple:
        """Detecta el delimitador y encoding del CSV"""
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    sample = f.read(4096)
                    sniffer = csv.Sniffer()
                    delimiter = sniffer.sniff(sample).delimiter
                    return delimiter, encoding
            except (UnicodeDecodeError, csv.Error):
                continue

        # Default fallback
        return ',', 'utf-8'

    def _extract_metadata(self, rows: List[Dict], headers: List[str],
                          delimiter: str, encoding: str) -> Dict[str, Any]:
        """Extract metadata from CSV"""
        return {
            'row_count': len(rows),
            'column_count': len(headers),
            'columns': headers,
            'delimiter': delimiter,
            'encoding': encoding,
            'has_headers': bool(headers)
        }

    def extract_sections(self, rows: List[Dict], headers: List[str]) -> List[DocumentSection]:
        """
        Extract sections from CSV
        Each section contains a chunk of rows
        """
        sections = []

        # Si el CSV es pequeño, una sola sección
        if len(rows) <= self.max_rows_per_section:
            content = self._format_rows_as_text(rows, headers)
            sections.append(DocumentSection(
                title=f"CSV Data (1-{len(rows)} rows)",
                content=content,
                level=1,
                metadata={
                    'row_start': 1,
                    'row_end': len(rows),
                    'row_count': len(rows)
                }
            ))
        else:
            # Dividir en chunks
            for i in range(0, len(rows), self.max_rows_per_section):
                chunk = rows[i:i + self.max_rows_per_section]
                row_start = i + 1
                row_end = i + len(chunk)

                content = self._format_rows_as_text(chunk, headers)
                sections.append(DocumentSection(
                    title=f"CSV Data (rows {row_start}-{row_end})",
                    content=content,
                    level=1,
                    metadata={
                        'row_start': row_start,
                        'row_end': row_end,
                        'row_count': len(chunk)
                    }
                ))

        return sections

    def _format_rows_as_text(self, rows: List[Dict], headers: List[str]) -> str:
        """Format CSV rows as readable text"""
        lines = []

        # Add header
        lines.append(' | '.join(headers))
        lines.append('-' * (sum(len(h) for h in headers) + 3 * len(headers)))

        # Add rows
        for row in rows:
            row_values = [str(row.get(h, '')) for h in headers]
            lines.append(' | '.join(row_values))

        return '\n'.join(lines)

    def _generate_full_content(self, sections: List[DocumentSection],
                               headers: List[str]) -> str:
        """Generate full content from sections"""
        full_content = []

        full_content.append(f"# CSV File")
        full_content.append(f"Columns: {', '.join(headers)}")
        full_content.append("")

        for section in sections:
            full_content.append(f"## {section.title}")
            full_content.append(section.content)
            full_content.append("")

        return '\n'.join(full_content).strip()
