# =============================================================================
# src/document_loaders/json_loader.py
# =============================================================================
"""
Loader para archivos JSON
"""
import json
from pathlib import Path
from typing import List, Dict, Any

from .base_loader import BaseDocumentLoader, DocumentSection, ProcessedDocument


class JSONLoader(BaseDocumentLoader):
    """Carga y procesa archivos JSON"""

    def __init__(self, max_depth: int = 5):
        super().__init__()
        self.supported_extensions = {'.json', '.jsonl'}
        self.max_depth = max_depth

    def load(self, file_path: Path, original_filename: str = None) -> ProcessedDocument:
        """Carga un archivo JSON"""
        # Detectar si es JSON o JSONL
        is_jsonl = file_path.suffix == '.jsonl'

        if is_jsonl:
            data = self._load_jsonl(file_path)
        else:
            data = self._load_json(file_path)

        # Extract metadata
        metadata = self._extract_metadata(data, is_jsonl)

        # Extract sections
        sections = self.extract_sections(data, is_jsonl)

        # Generate full content
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

    def _load_json(self, file_path: Path) -> Any:
        """Load regular JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_jsonl(self, file_path: Path) -> List[Dict]:
        """Load JSONL (JSON Lines) file"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def _extract_metadata(self, data: Any, is_jsonl: bool) -> Dict[str, Any]:
        """Extract metadata from JSON"""
        metadata = {
            'format': 'jsonl' if is_jsonl else 'json',
            'type': type(data).__name__
        }

        if isinstance(data, list):
            metadata['item_count'] = len(data)
            if data and isinstance(data[0], dict):
                metadata['keys'] = list(data[0].keys())
        elif isinstance(data, dict):
            metadata['keys'] = list(data.keys())

        return metadata

    def extract_sections(self, data: Any, is_jsonl: bool) -> List[DocumentSection]:
        """Extract sections from JSON"""
        sections = []

        if is_jsonl or isinstance(data, list):
            # JSONL or array: each item is a section
            items = data if isinstance(data, list) else [data]

            for i, item in enumerate(items, 1):
                title = self._generate_item_title(item, i)
                content = self._format_json_content(item, depth=0)

                sections.append(DocumentSection(
                    title=title,
                    content=content,
                    level=1,
                    metadata={'item_index': i}
                ))

        elif isinstance(data, dict):
            # Object: each top-level key is a section
            for key, value in data.items():
                content = self._format_json_content(value, depth=0)

                sections.append(DocumentSection(
                    title=str(key),
                    content=content,
                    level=1,
                    metadata={'key': key}
                ))

        else:
            # Primitive value
            sections.append(DocumentSection(
                title="JSON Content",
                content=str(data),
                level=1,
                metadata={}
            ))

        return sections

    def _generate_item_title(self, item: Any, index: int) -> str:
        """Generate a title for a JSON item"""
        if isinstance(item, dict):
            # Try common title fields
            for key in ['title', 'name', 'id', 'key']:
                if key in item:
                    return f"{key}: {item[key]}"

            # Use first key-value pair
            if item:
                first_key = list(item.keys())[0]
                first_value = item[first_key]
                return f"{first_key}: {first_value}"

        return f"Item {index}"

    def _format_json_content(self, obj: Any, depth: int = 0,
                             indent: str = "  ") -> str:
        """Format JSON content as readable text"""
        if depth > self.max_depth:
            return f"{indent * depth}[Max depth reached]"

        if isinstance(obj, dict):
            lines = []
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{indent * depth}{key}:")
                    lines.append(self._format_json_content(value, depth + 1, indent))
                else:
                    lines.append(f"{indent * depth}{key}: {value}")
            return '\n'.join(lines)

        elif isinstance(obj, list):
            lines = []
            for i, item in enumerate(obj):
                if isinstance(item, (dict, list)):
                    lines.append(f"{indent * depth}[{i}]:")
                    lines.append(self._format_json_content(item, depth + 1, indent))
                else:
                    lines.append(f"{indent * depth}[{i}]: {item}")
            return '\n'.join(lines)

        else:
            return f"{indent * depth}{obj}"

    def _generate_full_content(self, sections: List[DocumentSection]) -> str:
        """Generate full content from sections"""
        full_content = []

        full_content.append("# JSON Content")
        full_content.append("")

        for section in sections:
            full_content.append(f"## {section.title}")
            full_content.append(section.content)
            full_content.append("")

        return '\n'.join(full_content).strip()
