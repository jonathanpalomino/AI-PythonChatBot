import logging
import re
from pathlib import Path
from typing import List

from src.config.constants import CODE_CHUNK_SIZE
from .base_loader import BaseDocumentLoader, DocumentSection, ProcessedDocument

logger = logging.getLogger(__name__)

class JavaLoader(BaseDocumentLoader):
    """
    Specialized Java Loader.
    Uses regex + brace counting to segment Java files into logical sections.
    """

    # Regex to find class/interface/enum declarations
    CLASS_PATTERN = re.compile(
        r'(?:public|protected|private|static|final|abstract|sealed|non-sealed)?\s*'
        r'(?:class|interface|enum|record)\s+(\w+)',
        re.MULTILINE
    )

    # Regex to find method declarations
    # Matches patterns like: public void methodName(Args...) {
    # It focuses on the signature and the opening brace.
    METHOD_PATTERN = re.compile(
        r'(?:public|protected|private|static|final|abstract|synchronized|native|default)?\s*'
        r'(?:<[\w\s,<>?]+>\s*)?' # Generics
        r'(?:[\w.\[\]<>?]+\s+)+' # Return type and name parts
        r'(?!(?:if|for|while|switch|catch|return)\b)(\w+)\s*\([^)]*\)\s*'  # Method name and params
        r'(?:throws\s+[\w\s,.]+)?\s*' # Throws
        r'\{', # Opening brace
        re.MULTILINE
    )

    def __init__(self):
        super().__init__()
        self.supported_extensions = {'.java'}

    def load(self, file_path: Path, original_filename: str = None) -> ProcessedDocument:
        content = None
        encoding_used = 'utf-8'

        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                encoding_used = encoding
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            raise ValueError(f"No se pudo leer {file_path}")

        sections = self.extract_sections(content)

        return ProcessedDocument(
            file_path=str(file_path),
            file_name=file_path.name,
            original_filename=original_filename or file_path.name,
            content=content,
            sections=sections,
            metadata={
                'language': 'java',
                'line_count': content.count('\n') + 1,
                'char_count': len(content),
                'encoding': encoding_used
            },
            recommended_chunk_size=CODE_CHUNK_SIZE
        )

    def extract_sections(self, content: str) -> List[DocumentSection]:
        sections = []
        lines = content.splitlines()

        # We'll use a scanning approach to find class/method starts while ignoring comments/strings
        i = 0
        last_processed_pos = 0

        # 1. Capture Header (Package & Imports)
        class_match = self.CLASS_PATTERN.search(content)
        if class_match:
            # Check if this match is inside a comment
            # (Simple check for now: find occurrences of class declarations and verify with brace logic)
            pass

        # Let's use the scan + pattern match approach
        while i < len(content):
            char = content[i]

            # Skip strings
            if char == '"':
                i += 1
                while i < len(content) and content[i] != '"':
                    if content[i] == '\\': i += 1
                    i += 1
            # Skip chars
            elif char == "'":
                i += 1
                while i < len(content) and content[i] != "'":
                    if content[i] == '\\': i += 1
                    i += 1
            # Skip comments
            elif char == '/' and i + 1 < len(content):
                if content[i+1] == '/':
                    i = content.find('\n', i)
                    if i == -1: i = len(content)
                elif content[i+1] == '*':
                    i = content.find('*/', i)
                    if i == -1: i = len(content)
                    else: i += 1
            # Potential Class or Method
            else:
                # Check for class declaration at this position
                # We check a small window
                window = content[i:i+200]
                class_m = self.CLASS_PATTERN.match(window)
                if class_m:
                    # Capture preceding content as header if it's the first class
                    if last_processed_pos == 0 and i > 0:
                        header = content[last_processed_pos:i].strip()
                        if header:
                            start_line = 1
                            end_line = content[:i].count('\n')
                            sections.append(DocumentSection(
                                title="Package & Imports",
                                content=header,
                                level=1,
                                metadata={'type': 'header', 'start_line': start_line, 'end_line': end_line}
                            ))

                    class_name = class_m.group(1)
                    # Find opening brace
                    brace_pos = content.find('{', i + class_m.end())
                    if brace_pos != -1:
                        end_pos = self._find_closing_brace(content, brace_pos)
                        class_content = content[i:end_pos]

                        # Process class internal content
                        class_sections = self._segment_class(class_content, class_name, i, content)
                        sections.extend(class_sections)

                        i = end_pos - 1 # Jump to end
                        last_processed_pos = end_pos

            i += 1

        # 3. Capture remaining content
        if last_processed_pos < len(content):
            footer_content = content[last_processed_pos:].strip()
            if footer_content:
                start_line = content[:last_processed_pos].count('\n') + 1
                sections.append(DocumentSection(
                    title="Footer / Trailing Code",
                    content=footer_content,
                    level=1,
                    metadata={'type': 'footer', 'start_line': start_line, 'end_line': len(lines)}
                ))

        return [s for s in sections if s.content.strip()]

    def _find_closing_brace(self, content: str, open_brace_pos: int) -> int:
        """Finds matching closing brace while skipping strings and comments."""
        count = 0
        i = open_brace_pos
        while i < len(content):
            char = content[i]

            # Skip strings
            if char == '"':
                i += 1
                while i < len(content) and content[i] != '"':
                    if content[i] == '\\': i += 1 # Skip escaped chars
                    i += 1
            # Skip char literals
            elif char == "'":
                i += 1
                while i < len(content) and content[i] != "'":
                    if content[i] == '\\': i += 1
                    i += 1
            # Skip comments
            elif char == '/' and i + 1 < len(content):
                if content[i+1] == '/': # Single line
                    i = content.find('\n', i)
                    if i == -1: i = len(content)
                elif content[i+1] == '*': # Multi line
                    i = content.find('*/', i)
                    if i == -1: i = len(content)
                    else: i += 1
            # Brace counting
            elif char == '{':
                count += 1
            elif char == '}':
                count -= 1
                if count == 0:
                    return i + 1
            i += 1
        return len(content)

    def _segment_class(self, class_content: str, class_name: str, offset_in_full: int, full_content: str) -> List[DocumentSection]:
        """Segments a class into header and methods."""
        sections = []

        # Find methods within class content
        method_matches = list(self.METHOD_PATTERN.finditer(class_content))

        if not method_matches:
            # Whole class as one section if no methods found
            start_line = full_content[:offset_in_full].count('\n') + 1
            end_line = start_line + class_content.count('\n')
            return [DocumentSection(
                title=f"Class: {class_name}",
                content=class_content,
                level=1,
                metadata={'type': 'class', 'name': class_name, 'start_line': start_line, 'end_line': end_line}
            )]

        # Class Header (from start of class to first method)
        first_method = method_matches[0]
        header_content = class_content[:first_method.start()].strip()
        if header_content:
            start_line = full_content[:offset_in_full].count('\n') + 1
            end_line = start_line + header_content.count('\n')
            sections.append(DocumentSection(
                title=f"Class Header: {class_name}",
                content=header_content,
                level=1,
                metadata={'type': 'class_header', 'name': class_name, 'start_line': start_line, 'end_line': end_line}
            ))

        # Individual Methods
        for i, match in enumerate(method_matches):
            method_name = match.group(1)
            # Find method end
            # The signature ends at { which is where the match ends
            opening_brace_pos = match.end() - 1

            # Find relative end pos in class_content
            rel_end_pos = self._find_closing_brace(class_content, opening_brace_pos)

            # Extract method content
            method_content = class_content[match.start():rel_end_pos]

            start_line = full_content[:offset_in_full + match.start()].count('\n') + 1
            end_line = start_line + method_content.count('\n')

            sections.append(DocumentSection(
                title=f"Method: {class_name}.{method_name}",
                content=method_content,
                level=2,
                metadata={'type': 'method', 'name': method_name, 'parent': class_name, 'start_line': start_line, 'end_line': end_line}
            ))

            # Capture gaps between methods (comments or vars)
            if i + 1 < len(method_matches):
                next_match = method_matches[i+1]
                gap_content = class_content[rel_end_pos:next_match.start()].strip()
                if gap_content:
                    g_start = full_content[:offset_in_full + rel_end_pos].count('\n') + 1
                    g_end = g_start + gap_content.count('\n')
                    sections.append(DocumentSection(
                        title=f"Constants / Vars ({class_name})",
                        content=gap_content,
                        level=2,
                        metadata={'type': 'class_vars', 'parent': class_name, 'start_line': g_start, 'end_line': g_end}
                    ))
            else:
                # Capture trailing content in class
                trailing = class_content[rel_end_pos:].strip()
                # Remove class closing brace if it's there (usually it's the very last char)
                if trailing == '}': trailing = ""

                if trailing:
                    t_start = full_content[:offset_in_full + rel_end_pos].count('\n') + 1
                    t_end = t_start + trailing.count('\n')
                    sections.append(DocumentSection(
                        title=f"Trailing class code ({class_name})",
                        content=trailing,
                        level=2,
                        metadata={'type': 'class_footer', 'parent': class_name, 'start_line': t_start, 'end_line': t_end}
                    ))

        return sections
