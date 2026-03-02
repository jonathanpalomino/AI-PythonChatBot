# =============================================================================
# src/document_loaders/python_loader.py
# =============================================================================
"""
Professional Python Document Loader for RAG Systems.

This loader uses AST (Abstract Syntax Tree) to intelligently segment Python
code into semantic chunks based on language constructs, making it ideal for
RAG (Retrieval-Augmented Generation) systems.

Features:
- Semantic chunking by functions, methods, and classes
- Rich metadata extraction (signatures, docstrings, type hints, decorators)
- Nested function support
- Context-aware chunk generation for better RAG retrieval
- Import analysis and dependency tracking
"""

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from src.config.constants import CODE_CHUNK_SIZE
from .base_loader import BaseDocumentLoader, DocumentSection, ProcessedDocument

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes for Rich Metadata
# =============================================================================

@dataclass
class FunctionMetadata:
    """Metadata extracted from a function or method."""
    name: str
    full_name: str  # Includes class name if method
    signature: str  # Full function signature
    docstring: Optional[str]
    parameters: List[Dict[str, str]]  # [{name, annotation, default}]
    return_type: Optional[str]
    decorators: List[str]
    is_async: bool
    is_method: bool
    parent_class: Optional[str]
    start_line: int
    end_line: int
    line_count: int
    complexity_hints: List[str] = field(default_factory=list)


@dataclass
class ClassMetadata:
    """Metadata extracted from a class definition."""
    name: str
    docstring: Optional[str]
    decorators: List[str]
    base_classes: List[str]
    methods: List[str]
    attributes: List[str]
    start_line: int
    end_line: int
    line_count: int


@dataclass
class ModuleMetadata:
    """Metadata extracted from a Python module."""
    docstring: Optional[str]
    imports: List[Dict[str, str]]
    global_variables: List[str]
    classes: List[str]
    functions: List[str]
    total_lines: int


# =============================================================================
# Python Loader Implementation
# =============================================================================

class PythonLoader(BaseDocumentLoader):
    """
    Professional Python Loader for RAG Systems.

    Uses AST to segment Python files into semantic chunks based on:
    - Module-level docstrings and imports
    - Class definitions with their methods
    - Function definitions (including nested functions)
    - Global variables and constants

    Each chunk includes rich metadata for improved RAG retrieval.
    """

    def __init__(self):
        super().__init__()
        self.supported_extensions = {'.py'}

    def load(self, file_path: Path, original_filename: str = None) -> ProcessedDocument:
        """
        Load and process a Python file into semantic sections.

        Args:
            file_path: Path to the Python file
            original_filename: Original filename if different from path

        Returns:
            ProcessedDocument with semantic sections optimized for RAG
        """
        content, encoding_used = self.read_file_with_encodings(file_path)

        # Parse AST and extract sections
        sections = self.extract_sections(content)

        # Calculate module-level metadata
        module_meta = self._extract_module_metadata(content)

        return ProcessedDocument(
            file_path=str(file_path),
            file_name=file_path.name,
            original_filename=original_filename or file_path.name,
            content=content,
            sections=sections,
            metadata={
                'language': 'python',
                'line_count': content.count('\n') + 1,
                'char_count': len(content),
                'encoding': encoding_used,
                'module_docstring': module_meta.docstring,
                'imports': module_meta.imports,
                'classes': module_meta.classes,
                'functions': module_meta.functions,
                'global_variables': module_meta.global_variables,
            },
            recommended_chunk_size=CODE_CHUNK_SIZE
        )

    def extract_sections(self, content: str) -> List[DocumentSection]:
        """
        Extract semantic sections from Python code.

        Creates chunks based on logical code units:
        1. Module header (docstring + imports)
        2. Global variables and constants
        3. Each class with context
        4. Each function with full signature and docstring

        Args:
            content: Python source code

        Returns:
            List of DocumentSection objects with rich metadata
        """
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.warning(f"Syntax error parsing Python file: {e}")
            return [DocumentSection(
                title="Full File (Syntax Error)",
                content=content,
                level=0,
                metadata={'type': 'error', 'error': str(e)}
            )]

        lines = content.splitlines()
        sections = []

        # Track processed lines to avoid duplicates
        processed_ranges: List[Tuple[int, int]] = []

        # 1. Extract module header (docstring + imports)
        header_section = self._extract_module_header(tree, lines)
        if header_section:
            sections.append(header_section)
            processed_ranges.append((
                header_section.metadata['start_line'],
                header_section.metadata['end_line']
            ))

        # 2. Extract global variables/constants
        global_sections = self._extract_global_variables(tree, lines, processed_ranges)
        sections.extend(global_sections)

        # 3. Process top-level nodes (classes and functions)
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                class_sections = self._process_class(node, lines)
                sections.extend(class_sections)
                processed_ranges.append((self._get_start_line(node), node.end_lineno))

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_section = self._process_function(node, lines)
                sections.append(func_section)
                processed_ranges.append((self._get_start_line(node), node.end_lineno))

        # 4. Capture any remaining code (footer, main block, etc.)
        footer_section = self._extract_footer(tree, lines, processed_ranges)
        if footer_section:
            sections.append(footer_section)

        return [s for s in sections if s.content.strip()]

    # =========================================================================
    # AST Helper Methods
    # =========================================================================

    def _get_start_line(self, node: ast.AST) -> int:
        """Get the true start line including decorators."""
        if hasattr(node, 'decorator_list') and node.decorator_list:
            return node.decorator_list[0].lineno
        return node.lineno

    def _get_node_content(self, lines: List[str], start: int, end: int) -> str:
        """Extract content from lines (1-indexed to 0-indexed conversion)."""
        return "\n".join(lines[start - 1:end])

    def _extract_docstring(self, node: ast.AST) -> Optional[str]:
        """Extract docstring from a node."""
        return ast.get_docstring(node)

    def _get_decorators(self, node: ast.AST) -> List[str]:
        """Extract decorator names from a node."""
        if not hasattr(node, 'decorator_list'):
            return []

        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(f"@{dec.id}")
            elif isinstance(dec, ast.Attribute):
                decorators.append(f"@{self._get_attribute_name(dec)}")
            elif isinstance(dec, ast.Call):
                decorators.append(f"@{self._get_call_name(dec)}")
        return decorators

    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """Get full attribute name (e.g., 'module.function')."""
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return '.'.join(reversed(parts))

    def _get_call_name(self, node: ast.Call) -> str:
        """Get function name from a call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return self._get_attribute_name(node.func)
        return "unknown"

    def _get_base_classes(self, node: ast.ClassDef) -> List[str]:
        """Extract base class names from a class definition."""
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(self._get_attribute_name(base))
            elif isinstance(base, ast.Subscript):
                # Handle generic types like Generic[T]
                if isinstance(base.value, ast.Name):
                    bases.append(base.value.id)
                elif isinstance(base.value, ast.Attribute):
                    bases.append(self._get_attribute_name(base.value))
        return bases

    def _get_parameters(self, node: ast.FunctionDef) -> List[Dict[str, str]]:
        """Extract function parameters with type hints and defaults."""
        params = []
        args = node.args

        # Regular arguments
        for i, arg in enumerate(args.args):
            param = {
                'name': arg.arg,
                'annotation': self._get_annotation(arg.annotation) if arg.annotation else None,
                'default': None
            }
            # Calculate default value index
            default_offset = len(args.args) - len(args.defaults)
            if i >= default_offset and args.defaults:
                default_idx = i - default_offset
                param['default'] = self._get_default_value(args.defaults[default_idx])
            params.append(param)

        # *args
        if args.vararg:
            params.append({
                'name': f"*{args.vararg.arg}",
                'annotation': self._get_annotation(args.vararg.annotation) if args.vararg.annotation else None,
                'default': None
            })

        # **kwargs
        if args.kwarg:
            params.append({
                'name': f"**{args.kwarg.arg}",
                'annotation': self._get_annotation(args.kwarg.annotation) if args.kwarg.annotation else None,
                'default': None
            })

        return params

    def _get_annotation(self, node: ast.AST) -> str:
        """Convert annotation AST node to string."""
        if node is None:
            return None
        try:
            return ast.unparse(node)
        except AttributeError:
            # Fallback for older Python versions
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                return self._get_attribute_name(node)
            elif isinstance(node, ast.Subscript):
                return f"{self._get_annotation(node.value)}[...]"
            return str(type(node).__name__)

    def _get_default_value(self, node: ast.AST) -> str:
        """Convert default value AST node to string representation."""
        try:
            return ast.unparse(node)
        except AttributeError:
            if isinstance(node, ast.Constant):
                return repr(node.value)
            elif isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.List):
                return "[]"
            elif isinstance(node, ast.Dict):
                return "{}"
            elif isinstance(node, ast.Tuple):
                return "()"
            return "..."

    def _get_return_type(self, node: ast.FunctionDef) -> Optional[str]:
        """Extract return type annotation."""
        if node.returns:
            return self._get_annotation(node.returns)
        return None

    def _build_signature(self, node: ast.FunctionDef) -> str:
        """Build full function signature string."""
        params = []

        # Regular parameters
        for param in self._get_parameters(node):
            param_str = param['name']
            if param['annotation']:
                param_str += f": {param['annotation']}"
            if param['default']:
                param_str += f" = {param['default']}"
            params.append(param_str)

        # Build signature
        async_prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
        signature = f"{async_prefix}def {node.name}({', '.join(params)})"

        return_type = self._get_return_type(node)
        if return_type:
            signature += f" -> {return_type}"

        return signature

    def _detect_complexity_hints(self, node: ast.FunctionDef) -> List[str]:
        """Detect complexity indicators in a function."""
        hints = []

        for child in ast.walk(node):
            if isinstance(child, ast.For):
                hints.append("loop")
            elif isinstance(child, ast.While):
                hints.append("while_loop")
            elif isinstance(child, ast.If):
                hints.append("conditional")
            elif isinstance(child, ast.Try):
                hints.append("exception_handling")
            elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                hints.append("comprehension")
            elif isinstance(child, ast.Await):
                hints.append("async_await")
            elif isinstance(child, ast.Yield) or isinstance(child, ast.YieldFrom):
                hints.append("generator")

        # Remove duplicates while preserving order
        seen = set()
        return [h for h in hints if not (h in seen or seen.add(h))]

    # =========================================================================
    # Section Extraction Methods
    # =========================================================================

    def _extract_module_metadata(self, content: str) -> ModuleMetadata:
        """Extract module-level metadata."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return ModuleMetadata(
                docstring=None,
                imports=[],
                global_variables=[],
                classes=[],
                functions=[],
                total_lines=content.count('\n') + 1
            )

        docstring = ast.get_docstring(tree)
        imports = []
        global_vars = []
        classes = []
        functions = []

        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'type': 'import',
                        'module': alias.name,
                        'alias': alias.asname
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append({
                        'type': 'from_import',
                        'module': module,
                        'name': alias.name,
                        'alias': alias.asname
                    })
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        global_vars.append(target.id)
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    global_vars.append(node.target.id)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node.name)

        return ModuleMetadata(
            docstring=docstring,
            imports=imports,
            global_variables=global_vars,
            classes=classes,
            functions=functions,
            total_lines=content.count('\n') + 1
        )

    def _extract_module_header(self, tree: ast.Module, lines: List[str]) -> Optional[DocumentSection]:
        """Extract module header (docstring + imports)."""
        header_lines = []
        end_line = 0

        for node in tree.body:
            # Stop at first class or function
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                break

            node_start = self._get_start_line(node)
            node_end = node.end_lineno

            # Include imports and module-level assignments
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Expr, ast.Assign, ast.AnnAssign)):
                header_lines.append((node_start, node_end))
                end_line = max(end_line, node_end)

        if not header_lines:
            return None

        # Combine all header content
        content_parts = []
        for start, end in header_lines:
            content_parts.append(self._get_node_content(lines, start, end))

        return DocumentSection(
            title="Module Header",
            content="\n\n".join(content_parts),
            level=0,
            metadata={
                'type': 'module_header',
                'start_line': header_lines[0][0],
                'end_line': end_line
            }
        )

    def _extract_global_variables(
        self,
        tree: ast.Module,
        lines: List[str],
        processed_ranges: List[Tuple[int, int]]
    ) -> List[DocumentSection]:
        """Extract global variables and constants."""
        sections = []

        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                start = self._get_start_line(node)
                end = node.end_lineno

                # Skip if already processed
                if any(start >= p[0] and end <= p[1] for p in processed_ranges):
                    continue

                # Skip if it's inside a class or function (shouldn't happen at module level)
                if isinstance(node, ast.Assign):
                    names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                else:
                    names = [node.target.id] if isinstance(node.target, ast.Name) else []

                # Check if it's a constant (uppercase) or configuration
                is_constant = all(n.isupper() or n.startswith('_') for n in names)

                content = self._get_node_content(lines, start, end)

                sections.append(DocumentSection(
                    title=f"Global: {', '.join(names)}",
                    content=content,
                    level=1,
                    metadata={
                        'type': 'constant' if is_constant else 'global_variable',
                        'names': names,
                        'start_line': start,
                        'end_line': end
                    }
                ))

        return sections

    def _process_class(self, node: ast.ClassDef, lines: List[str]) -> List[DocumentSection]:
        """Process a class definition into semantic sections."""
        sections = []
        class_start = self._get_start_line(node)
        class_end = node.end_lineno

        # Extract class metadata
        class_meta = ClassMetadata(
            name=node.name,
            docstring=self._extract_docstring(node),
            decorators=self._get_decorators(node),
            base_classes=self._get_base_classes(node),
            methods=[n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))],
            attributes=self._extract_class_attributes(node),
            start_line=class_start,
            end_line=class_end,
            line_count=class_end - class_start + 1
        )

        # Find methods
        methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

        if not methods:
            # Simple class without methods - return as single section
            content = self._get_node_content(lines, class_start, class_end)
            return [DocumentSection(
                title=f"Class: {node.name}",
                content=content,
                level=1,
                metadata={
                    'type': 'class',
                    'name': node.name,
                    'docstring': class_meta.docstring,
                    'decorators': class_meta.decorators,
                    'base_classes': class_meta.base_classes,
                    'attributes': class_meta.attributes,
                    'start_line': class_start,
                    'end_line': class_end,
                    'line_count': class_meta.line_count
                }
            )]

        # Create class header section (decorators, class definition, docstring, class attributes)
        first_method_start = self._get_start_line(methods[0])
        header_content = self._get_node_content(lines, class_start, first_method_start - 1)

        if header_content.strip():
            sections.append(DocumentSection(
                title=f"Class: {node.name}",
                content=header_content,
                level=1,
                metadata={
                    'type': 'class_header',
                    'name': node.name,
                    'docstring': class_meta.docstring,
                    'decorators': class_meta.decorators,
                    'base_classes': class_meta.base_classes,
                    'attributes': class_meta.attributes,
                    'methods': class_meta.methods,
                    'start_line': class_start,
                    'end_line': first_method_start - 1
                }
            ))

        # Process each method
        for i, method in enumerate(methods):
            m_start = self._get_start_line(method)
            m_end = method.end_lineno

            # Adjust end to include any trailing comments/blank lines before next method
            if i + 1 < len(methods):
                next_start = self._get_start_line(methods[i + 1])
                # Include lines up to (but not including) the next method
                m_end = next_start - 1

            method_section = self._process_method(method, lines, node.name, m_start, m_end)
            sections.append(method_section)

        return sections

    def _extract_class_attributes(self, node: ast.ClassDef) -> List[str]:
        """Extract class-level attributes (not instance attributes)."""
        attributes = []
        for child in node.body:
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        attributes.append(target.id)
            elif isinstance(child, ast.AnnAssign):
                if isinstance(child.target, ast.Name):
                    attributes.append(child.target.id)
        return attributes

    def _process_method(
        self,
        node: ast.FunctionDef,
        lines: List[str],
        class_name: str,
        start_line: int,
        end_line: int
    ) -> DocumentSection:
        """Process a method into a semantic section."""
        content = self._get_node_content(lines, start_line, end_line)

        # Build rich metadata
        func_meta = FunctionMetadata(
            name=node.name,
            full_name=f"{class_name}.{node.name}",
            signature=self._build_signature(node),
            docstring=self._extract_docstring(node),
            parameters=self._get_parameters(node),
            return_type=self._get_return_type(node),
            decorators=self._get_decorators(node),
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_method=True,
            parent_class=class_name,
            start_line=start_line,
            end_line=end_line,
            line_count=end_line - start_line + 1,
            complexity_hints=self._detect_complexity_hints(node)
        )

        # Build searchable content with context
        searchable_content = self._build_searchable_content(content, func_meta)

        return DocumentSection(
            title=f"Method: {class_name}.{node.name}",
            content=searchable_content,
            level=2,
            metadata={
                'type': 'method',
                'name': node.name,
                'full_name': func_meta.full_name,
                'signature': func_meta.signature,
                'docstring': func_meta.docstring,
                'parameters': func_meta.parameters,
                'return_type': func_meta.return_type,
                'decorators': func_meta.decorators,
                'is_async': func_meta.is_async,
                'parent_class': class_name,
                'start_line': start_line,
                'end_line': end_line,
                'line_count': func_meta.line_count,
                'complexity_hints': func_meta.complexity_hints
            }
        )

    def _process_function(
        self,
        node: ast.FunctionDef,
        lines: List[str]
    ) -> DocumentSection:
        """Process a standalone function into a semantic section."""
        start = self._get_start_line(node)
        end = node.end_lineno
        content = self._get_node_content(lines, start, end)

        # Build rich metadata
        func_meta = FunctionMetadata(
            name=node.name,
            full_name=node.name,
            signature=self._build_signature(node),
            docstring=self._extract_docstring(node),
            parameters=self._get_parameters(node),
            return_type=self._get_return_type(node),
            decorators=self._get_decorators(node),
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_method=False,
            parent_class=None,
            start_line=start,
            end_line=end,
            line_count=end - start + 1,
            complexity_hints=self._detect_complexity_hints(node)
        )

        # Build searchable content with context
        searchable_content = self._build_searchable_content(content, func_meta)

        title_prefix = "Async Function" if func_meta.is_async else "Function"

        return DocumentSection(
            title=f"{title_prefix}: {node.name}",
            content=searchable_content,
            level=1,
            metadata={
                'type': 'function',
                'name': node.name,
                'full_name': func_meta.full_name,
                'signature': func_meta.signature,
                'docstring': func_meta.docstring,
                'parameters': func_meta.parameters,
                'return_type': func_meta.return_type,
                'decorators': func_meta.decorators,
                'is_async': func_meta.is_async,
                'start_line': start,
                'end_line': end,
                'line_count': func_meta.line_count,
                'complexity_hints': func_meta.complexity_hints
            }
        )

    def _build_searchable_content(self, content: str, meta: FunctionMetadata) -> str:
        """
        Build content optimized for RAG retrieval.

        Includes structured metadata at the top for better semantic search.
        """
        parts = []

        # Add signature as header (important for search)
        parts.append(f"# {meta.signature}")

        # Add docstring if present (highly valuable for RAG)
        if meta.docstring:
            parts.append(f'"""')
            parts.append(meta.docstring)
            parts.append(f'"""')

        # Add the actual code
        parts.append("")
        parts.append(content)

        return "\n".join(parts)

    def _extract_footer(
        self,
        tree: ast.Module,
        lines: List[str],
        processed_ranges: List[Tuple[int, int]]
    ) -> Optional[DocumentSection]:
        """Extract footer content (main block, remaining code)."""
        footer_nodes = []

        for node in tree.body:
            start = self._get_start_line(node)
            end = node.end_lineno

            # Check if this node was already processed
            is_processed = any(
                start >= p[0] and end <= p[1]
                for p in processed_ranges
            )

            if not is_processed:
                # Check if it's a main block or other important footer content
                if isinstance(node, ast.If):
                    # Check for if __name__ == "__main__"
                    if self._is_main_block(node):
                        footer_nodes.append((start, end, 'main_block'))
                elif isinstance(node, (ast.Expr, ast.Pass, ast.Break, ast.Continue)):
                    footer_nodes.append((start, end, 'statement'))

        if not footer_nodes:
            return None

        content_parts = []
        min_start = float('inf')
        max_end = 0

        for start, end, node_type in footer_nodes:
            content_parts.append(self._get_node_content(lines, start, end))
            min_start = min(min_start, start)
            max_end = max(max_end, end)

        return DocumentSection(
            title="Module Footer",
            content="\n\n".join(content_parts),
            level=1,
            metadata={
                'type': 'module_footer',
                'start_line': min_start,
                'end_line': max_end
            }
        )

    def _is_main_block(self, node: ast.If) -> bool:
        """Check if an if statement is the main block."""
        test = node.test
        # Check for: if __name__ == "__main__"
        if isinstance(test, ast.Compare):
            if isinstance(test.left, ast.Name) and test.left.id == '__name__':
                for op, comp in zip(test.ops, test.comparators):
                    if isinstance(op, ast.Eq) and isinstance(comp, ast.Constant):
                        if comp.value == '__main__':
                            return True
        return False
