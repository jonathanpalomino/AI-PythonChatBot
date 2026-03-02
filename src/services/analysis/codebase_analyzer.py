import ast
import bisect
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

# =============================================================================
# MULTI-LANGUAGE PATTERNS (Moved from CodeLoader)
# =============================================================================
LANGUAGE_BY_EXTENSION = {
    '.py': 'python',
    '.java': 'java',
    '.cs': 'csharp',
    '.js': 'javascript',
    '.ts': 'typescript',
    '.jsx': 'javascript',
    '.tsx': 'typescript',
    '.go': 'go',
    '.sql': 'sql',
    '.pls': 'plsql',
    '.plsql': 'plsql',
    '.pck': 'plsql',
    '.pkb': 'plsql',
    '.pks': 'plsql',
    '.spc': 'plsql',
    '.bdy': 'plsql',
    '.fnc': 'plsql',
    '.prc': 'plsql',
    '.typ': 'plsql',
    '.tps': 'plsql',
    '.tpb': 'plsql',
    '.c': 'c',
    '.cpp': 'cpp',
    '.h': 'cpp',
    '.hpp': 'cpp',
    '.rs': 'rust',
    '.php': 'php',
    '.rb': 'ruby',
}

CLASS_REGEX = {
    'java': re.compile(r'\bclass\s+(\w+)'),
    'csharp': re.compile(r'\bclass\s+(\w+)'),
    'javascript': re.compile(r'\bclass\s+(\w+)'),
    'typescript': re.compile(r'\b(?:export\s+)?class\s+(\w+)'),
}

FUNCTION_REGEX = {
    'java': re.compile(r'\b(?:public|private|protected)?\s+\w+\s+(\w+)\s*\(', re.MULTILINE),
    'csharp': re.compile(r'\b(?:public|private|protected)?\s+\w+\s+(\w+)\s*\(', re.MULTILINE),
    'javascript': re.compile(r'\bfunction\s+(\w+)\s*\('),
    'typescript': re.compile(r'\b(?:public|private|protected)?\s*(?:async\s+)?(\w+)\s*\('),
    'go': re.compile(r'func\s+(\w+)\s*\('),
}

@dataclass
class CodeSymbol:
    name: str
    type: str  # 'class', 'function', 'method', 'module'
    start_line: int
    end_line: int
    content: str
    docstring: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)  # Modules or symbols this node depends on
    children: List['CodeSymbol'] = field(default_factory=list)
    parent: Optional[str] = None  # Name of parent class/module

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert CodeSymbol to a JSON-serializable dictionary.
        
        Handles:
        - Set[str] -> List[str] conversion
        - Recursive children conversion
        - Optional fields
        """
        return {
            'name': self.name,
            'type': self.type,
            'start_line': self.start_line,
            'end_line': self.end_line,
            'content': self.content,
            'docstring': self.docstring,
            'decorators': self.decorators,
            'dependencies': list(self.dependencies),  # Set -> List for JSON
            'children': [child.to_dict() for child in self.children],
            'parent': self.parent
        }

class CodebaseAnalyzer:
    """
    Universal Code Analyzer.
    Uses AST for Python and Regex for other languages (Java, TS, C#, etc).
    """

    def analyze_file(self, content: str, file_name: str) -> Dict[str, Any]:
        """
        Analyzes a single file content and returns a structure of symbols and dependencies.
        Auto-detects language from definition.
        """
        ext = Path(file_name).suffix.lower()
        language = LANGUAGE_BY_EXTENSION.get(ext, 'unknown')

        if language == 'python':
            return self._analyze_python(content, file_name)
        else:
            return self._analyze_regex(content, language, file_name)

    def _analyze_regex(self, content: str, language: str, file_path: str) -> Dict[str, Any]:
        """
        Fallback analysis using Regex for non-Python languages.
        Optimized with bisect for O(n + m log n) line number calculation.
        """
        symbols = []
        imports = []  # Regex for imports could be added later

        # OPTIMIZATION: Precalculate line offsets O(n) once
        line_offsets = [0]
        for i, char in enumerate(content):
            if char == '\n':
                line_offsets.append(i + 1)

        def get_line_number(offset: int) -> int:
            """Binary search for line number - O(log n)."""
            return bisect.bisect_right(line_offsets, offset)

        # Extract Classes
        if language in CLASS_REGEX:
            for match in CLASS_REGEX[language].finditer(content):
                line_no = get_line_number(match.start())
                symbols.append(CodeSymbol(
                    name=match.group(1),
                    type='class',
                    start_line=line_no,
                    end_line=line_no,  # Placeholder
                    content=match.group(0)  # Just the declaration line usually
                ))

        # Extract Functions
        if language in FUNCTION_REGEX:
            for match in FUNCTION_REGEX[language].finditer(content):
                line_no = get_line_number(match.start())
                symbols.append(CodeSymbol(
                    name=match.group(1),
                    type='function',
                    start_line=line_no,
                    end_line=line_no,
                    content=match.group(0)
                ))

        # Basic Complexity Estimate (keyword counting)
        complexity = 1
        complexity_indicators = [r"\bif\b", r"\bfor\b", r"\bwhile\b", r"\bswitch\b",
                                 r"\bcatch\b"]
        for pattern in complexity_indicators:
            complexity += len(re.findall(pattern, content))

        return {
            "file_path": file_path,
            "language": language,
            "imports": imports,
            "symbols": symbols,
            "complexity": complexity,
            "metrics": {
                "classes": len([s for s in symbols if s.type == 'class']),
                "functions": len([s for s in symbols if s.type == 'function'])
            }
        }
    def _analyze_python(self, content: str, file_path: str) -> Dict[str, Any]:
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return {"error": f"SyntaxError in {file_path}: {e}", "symbols": []}

        try:
            visitor = SymbolVisitor(content.splitlines())
            visitor.visit(tree)

            complexity_visitor = ComplexityVisitor()
            complexity_visitor.visit(tree)

            return {
                "file_path": file_path,
                "language": "python",
                "imports": visitor.imports,
                "symbols": visitor.symbols,
                "complexity": complexity_visitor.complexity,
                "metrics": {
                    "classes": len([s for s in visitor.symbols if s.type == 'class']),
                    "functions": len(
                        [s for s in visitor.symbols if s.type in ('function', 'method')])
                }
            }
        except RecursionError:
            return {"error": "RecursionError: too deeply nested", "symbols": []}
        except Exception as e:
            return {"error": f"Analysis error: {e}", "symbols": []}




class SymbolVisitor(ast.NodeVisitor):
    def __init__(self, source_lines: List[str]):
        self.source_lines = source_lines
        self.symbols: List[CodeSymbol] = []
        self.imports: List[str] = []
        self.current_scope: List[str] = [] # Stack of parent names

    def _get_source_segment(self, node: ast.AST) -> str:
        """Extracts the source code for a given node."""
        # AST line numbers are 1-based
        start = node.lineno - 1
        end = node.end_lineno if node.end_lineno else len(self.source_lines)
        return "\n".join(self.source_lines[start:end])

    def _get_docstring(self, node: ast.AST) -> Optional[str]:
        return ast.get_docstring(node)

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            self.imports.append(node.module)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        symbol = CodeSymbol(
            name=node.name,
            type='class',
            start_line=node.lineno,
            end_line=node.end_lineno,
            content=self._get_source_segment(node),
            docstring=self._get_docstring(node),
            decorators=[d.id for d in node.decorator_list if isinstance(d, ast.Name)],
            parent=".".join(self.current_scope) if self.current_scope else None
        )

        # Track dependencies (bases)
        for base in node.bases:
            if isinstance(base, ast.Name):
                symbol.dependencies.add(base.id)
            elif isinstance(base, ast.Attribute):
                 # Try to capture full name like module.Class
                 parts = []
                 curr = base
                 while isinstance(curr, ast.Attribute):
                     parts.append(curr.attr)
                     curr = curr.value
                 if isinstance(curr, ast.Name):
                     parts.append(curr.id)
                 symbol.dependencies.add(".".join(reversed(parts)))

        self.symbols.append(symbol)

        self.current_scope.append(node.name)
        # Visit children (methods)
        # We manually visit body to keep track of methods belonging to this class
        # But for flattened list, we just continue visiting.
        # Ideally we might want a hierarchical structure, but flat is easier for RAG chunking often.
        # Let's keep it flat for now, but 'parent' field helps reconstruct hierarchy.
        self.generic_visit(node)
        self.current_scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        is_method = bool(self.current_scope) # Approximate check
        symbol = CodeSymbol(
            name=node.name,
            type='method' if is_method else 'function',
            start_line=node.lineno,
            end_line=node.end_lineno,
            content=self._get_source_segment(node),
            docstring=self._get_docstring(node),
            decorators=[d.id for d in node.decorator_list if isinstance(d, ast.Name)],
            parent=".".join(self.current_scope) if self.current_scope else None
        )

        # Naive dependency extraction from body calls
        # This is expensive/complex to do perfectly, but we can catch simple calls
        call_visitor = CallVisitor()
        call_visitor.visit(node)
        symbol.dependencies.update(call_visitor.calls)

        self.symbols.append(symbol)

        self.current_scope.append(node.name) # inner functions
        self.generic_visit(node)
        self.current_scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        # Treat same as sync function for now
        self.visit_FunctionDef(node)

class CallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.calls = set()

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            self.calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            self.calls.add(node.func.attr)
        self.generic_visit(node)

class ComplexityVisitor(ast.NodeVisitor):
    """Calculates Cyclomatic Complexity"""
    def __init__(self):
        self.complexity = 1 # Base complexity

    def visit_If(self, node: ast.If):
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        self.complexity += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor):
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While):
        self.complexity += 1
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try):
        self.complexity += 1
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # We don't increment for function def itself when counting *file* complexity,
        # but if we visitor a function node directly we might.
        # Here we visit children.
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp):
        self.complexity += (len(node.values) - 1)
        self.generic_visit(node)
