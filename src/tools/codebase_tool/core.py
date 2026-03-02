# =============================================================================
# src/tools/codebase_tool/core.py
# Professional Codebase Analysis Tool - Core Implementation
# =============================================================================
"""
Professional Codebase Analysis Tool - Core Implementation.

This is the main entry point for the professional codebase analysis tool.
It integrates all components and provides a comprehensive API for code analysis.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from uuid import UUID

from .code_smells import CodeSmellDetector
from .llm_formatter import LLMFormatter
from .metrics import MetricsCalculator
from .models import (
    AnalysisResult,
    FileAnalysisResult,
    SymbolInfo,
    CodeLocation
)
from .refactoring import RefactoringSuggester
from .sarif import SarifGenerator
from .security import SecurityAnalyzer
from ..base_tool import BaseTool, ToolCategory, ToolParameter, ToolResult
from ...config.constants import IGNORED_DIRS, IGNORED_EXTENSIONS
from ...config.intent_patterns import CodebaseAction
from ...services.analysis.codebase_analyzer import (
    CodebaseAnalyzer,
    LANGUAGE_BY_EXTENSION
)

if TYPE_CHECKING:
    from ...repositories.file_repository import FileRepository

logger = logging.getLogger(__name__)


class CodebaseTool(BaseTool):
    """
    Professional Codebase Analysis Tool.

    A comprehensive static code analysis tool that provides:
    - Advanced code quality metrics
    - Code smell detection
    - Security vulnerability scanning
    - Refactoring suggestions
    - Multi-language support
    - SARIF report generation
    - LLM-optimized output

    This is a professional-grade tool suitable for production use.
    """

    def __init__(self, uow: Optional['UnitOfWork'] = None):
        """
        Initialize CodebaseTool.

        Args:
            uow: UnitOfWork instance for database operations (deprecated)
        """
        super().__init__()

        # Core components
        self.analyzer = CodebaseAnalyzer()
        self.metrics_calculator = MetricsCalculator()
        self.code_smell_detector = CodeSmellDetector()
        self.security_analyzer = SecurityAnalyzer()
        self.refactoring_suggester = RefactoringSuggester()
        self.sarif_generator = SarifGenerator()
        self.llm_formatter = LLMFormatter()

        # Configuration
        self.root_dir = Path(os.getcwd())
        self.ignored_dirs = IGNORED_DIRS
        self.ignored_extensions = IGNORED_EXTENSIONS

        # Dependencies
        self._uow = uow  # Kept for backward compatibility
        self.file_repo: Optional['FileRepository'] = None  # Injected by chat_orchestrator

        # Lazy load RAG tool
        self._rag_tool = None

        logger.info(f"CodebaseTool v2.0.0 initialized with root: {self.root_dir}")

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def rag_tool(self):
        """Lazy load RAG tool."""
        if self._rag_tool is None:
            from ..rag_tool import RAGTool
            self._rag_tool = RAGTool()
        return self._rag_tool

    @property
    def uow(self) -> Optional['UnitOfWork']:
        """Get UnitOfWork instance."""
        return self._uow

    @uow.setter
    def uow(self, value: 'UnitOfWork'):
        """Set UnitOfWork instance."""
        self._uow = value
        logger.debug("UnitOfWork set for CodebaseTool")

    @property
    def name(self) -> str:
        return "codebase_tool"

    @property
    def description(self) -> str:
        return (
            "Professional static code analysis tool. Provides comprehensive "
            "code quality metrics, code smell detection, security vulnerability "
            "scanning, and refactoring suggestions for uploaded files."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description="Action to perform",
                required=True,
                enum=CodebaseAction.ALL_ACTIONS
            ),
            ToolParameter(
                name="target",
                type="string",
                description="Symbol name or file path (optional)",
                required=False
            ),
            ToolParameter(
                name="context_files",
                type="array",
                description="Files to limit analysis from RAG context (optional)",
                required=False
            ),
            ToolParameter(
                name="file_ids",
                type="array",
                description="UUIDs of uploaded files to analyze",
                required=False
            ),
            ToolParameter(
                name="format",
                type="string",
                description="Output format for formatforllm action",
                required=False,
                enum=["detailed", "summary", "compact","markdown"]
            )
        ]

    # =========================================================================
    # Main Execution Method
    # =========================================================================

    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute codebase analysis action.

        Args:
            **kwargs: Action parameters

        Returns:
            ToolResult with analysis results
        """
        try:
            logger.info(f"CodebaseTool execution started with kwargs: {kwargs}")
            await self.validate_input(**kwargs)

            action = kwargs["action"]
            file_ids = kwargs.get("file_ids", [])
            context_files = kwargs.get("context_files", [])
            target = kwargs.get("target")
            format_type = kwargs.get("format", "detailed")
            sub_action = kwargs.get("sub_action")  # Hint granular del intent router
            logger.info(f"Executing action '{action}' with {len(file_ids)} file_ids and {len(context_files)} context_files")

            # Tratar "markdown" como "detailed" (ya es markdown por defecto)
            if format_type == "markdown":
                format_type = "detailed"
            # Analyze context files from RAG (lower priority than file_ids)
            if context_files and not file_ids:
                analysis_result = await self.analyze_context_files(
                    context_files,
                    action,
                    target
                )

                # Format output based on action
                if action == "generatesarif":
                    sarif_json = self.sarif_generator.generate_sarif_json(analysis_result)
                    return ToolResult(success=True, data=sarif_json)
                elif action == "formatforllm":
                    formatted = self.llm_formatter.format_analysis_result(
                        analysis_result,
                        format_type=format_type
                    )
                    return ToolResult(success=True, data=formatted)
                elif action == CodebaseAction.BASIC_ANALYZE_FILE:
                    # Specialized structural formatting
                    formatted = self.llm_formatter.format_analysis_result(
                        analysis_result,
                        format_type="basic", sub_action=sub_action
                    )
                    return ToolResult(success=True, data=formatted)
                elif action == CodebaseAction.ANALYZE_FILE:
                    # Full detailed analysis
                    formatted = self.llm_formatter.format_analysis_result(
                        analysis_result,
                        format_type="detailed"
                    )
                    return ToolResult(success=True, data=formatted)
                else:
                    # Return formatted output instead of raw dict for other actions
                    formatted_output = self._format_analysis_result_dict(analysis_result.to_dict())
                    return ToolResult(
                        success=True,
                        data=formatted_output
                    )

            # Analyze uploaded files (higher priority)
            if file_ids:
                analysis_result = await self.analyze_uploaded_files(
                    file_ids,
                    action,
                    target
                )

                # Format output based on action
                if action == "generatesarif":
                    sarif_json = self.sarif_generator.generate_sarif_json(analysis_result)
                    return ToolResult(success=True, data=sarif_json)
                elif action == "formatforllm":
                    formatted = self.llm_formatter.format_analysis_result(
                        analysis_result,
                        format_type=format_type
                    )
                    return ToolResult(success=True, data=formatted)
                elif action == CodebaseAction.GET_METHOD_CONTENT:
                    # For content retrieval, return the result content directly
                    content = ""
                    if analysis_result.results:
                        content = analysis_result.results[0].content
                    return ToolResult(
                        success=True,
                        data=content if content else f"Symbol '{target}' not found",
                        metadata={"target": target}
                    )
                elif action == CodebaseAction.BASIC_ANALYZE_FILE:
                    # Specialized structural formatting
                    formatted = self.llm_formatter.format_analysis_result(
                        analysis_result,
                        format_type="basic",sub_action=sub_action
                    )
                    return ToolResult(success=True, data=formatted)
                elif action == CodebaseAction.ANALYZE_FILE:
                    # Full detailed analysis
                    formatted = self.llm_formatter.format_analysis_result(
                        analysis_result,
                        format_type="detailed"
                    )
                    return ToolResult(success=True, data=formatted)
                else:
                    # Return formatted output instead of raw dict
                    formatted_output = self._format_analysis_result_dict(analysis_result.to_dict())
                    return ToolResult(
                        success=True,
                        data=formatted_output
                    )

            # No files provided
            return ToolResult(
                success=False,
                data=None,
                error="No files provided for analysis. Please provide file_ids, context_files, or target path."
            )

        except Exception as e:
            logger.error(f"Error executing CodebaseTool: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))

    # =========================================================================
    # File Analysis Methods
    # =========================================================================

    async def analyze_uploaded_files(
        self,
        file_ids: List[str],
        action: str,
        target: Optional[str] = None
    ) -> AnalysisResult:
        """
        Analyze uploaded files by UUID from database.

        Args:
            file_ids: List of file UUIDs
            action: Action to perform
            target: Optional symbol name for specific analysis

        Returns:
            AnalysisResult with complete analysis
        """
        try:
            if not self.file_repo:
                logger.error("file_repo not available for CodebaseTool")
                raise ValueError("file_repo not configured")

            file_results = []

            for file_id_str in file_ids:
                logger.info(f"Analyzing uploaded file: {file_id_str}")
                file_id = UUID(file_id_str)

                try:
                    # Get file from database
                    file_record = await self.file_repo.get_by_id(file_id)
                    if not file_record:
                        logger.warning(f"File not found: {file_id}")
                        continue

                    # Read file content
                    storage_path = Path(file_record.storage_path)
                    content = None

                    if storage_path.exists():
                        try:
                            content = storage_path.read_text(encoding='utf-8', errors='ignore')
                        except Exception as e:
                            logger.warning(f"Failed to read local file: {e}")

                    # Fallback to RAG
                    if content is None:
                        collection_name = getattr(file_record, 'collection_name', 'documentation')
                        content = await self.rag_tool.get_full_document_content(
                            str(file_id),
                            collection_name
                        )

                    if not content:
                        logger.warning(f"Failed to get content for file: {file_id}")
                        continue
                      # Choose analysis type based on action
                    if action == CodebaseAction.BASIC_ANALYZE_FILE:
                        file_result = await self._analyze_file_structural(
                            content,
                            file_record.file_name,
                            str(file_id)
                        )
                    elif action == CodebaseAction.GET_METHOD_CONTENT:
                         # Analyze structural first
                        temp_result = await self._analyze_file_structural(
                            content,
                            file_record.file_name,
                            str(file_id)
                        )
                        # Filter for specific method
                        file_result = self._find_symbol_content(temp_result, target)
                    else:
                        file_result = await self._analyze_file_comprehensive(
                            content,
                            file_record.file_name,
                            str(file_id)
                        )

                    file_results.append(file_result)

                except Exception as e:
                    logger.error(f"Error analyzing file {file_id}: {e}", exc_info=True)

            # Create analysis result
            analysis_result = AnalysisResult(
                action=action,
                target=target,
                files_analyzed=len(file_results),
                results=file_results
            )

            # Calculate summary
            analysis_result.calculate_summary()

            logger.info(f"Analysis completed: {len(file_results)} files analyzed")
            return analysis_result

        except Exception as e:
            logger.error(f"Error in analyze_uploaded_files: {e}", exc_info=True)
            raise

    async def analyze_context_files(
        self,
        context_files: List[str],
        action: str,
        target: Optional[str] = None
    ) -> AnalysisResult:
        """
        Analyze files from RAG context by filename.

        This method reads files from the filesystem using the filenames
        provided in context_files (from RAG search results).

        Args:
            context_files: List of filenames to analyze
            action: Action to perform
            target: Optional symbol name for specific analysis

        Returns:
            AnalysisResult with complete analysis
        """
        try:
            file_results = []

            for filename in context_files:
                logger.info(f"Analyzing context file: {filename}")

                # Try to find the file in the current directory
                file_path = self.root_dir / filename

                if not file_path.exists():
                    # Try to find the file recursively
                    found = False
                    for root, dirs, files in os.walk(self.root_dir):
                        if filename in files:
                            file_path = Path(root) / filename
                            found = True
                            break

                    if not found:
                        logger.warning(f"File not found: {filename}")
                        continue

                # Read file content
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                except Exception as e:
                    logger.warning(f"Failed to read file {filename}: {e}")
                    continue

                # Choose analysis type based on action
                if action == CodebaseAction.BASIC_ANALYZE_FILE:
                    file_result = await self._analyze_file_structural(
                        content,
                        filename,
                        "" # No file_id for context files usually
                    )
                elif action == CodebaseAction.GET_METHOD_CONTENT:
                     # Analyze structural first
                    temp_result = await self._analyze_file_structural(
                        content,
                        filename,
                        ""
                    )
                    # Filter for specific method
                    file_result = self._find_symbol_content(temp_result, target)
                else:
                    file_result = await self._analyze_file_comprehensive(
                        content,
                        filename,
                        ""
                    )
                file_results.append(file_result)

            # Create analysis result
            analysis_result = AnalysisResult(
                action=action,
                target=target,
                files_analyzed=len(file_results),
                results=file_results
            )

            # Calculate summary
            analysis_result.calculate_summary()

            logger.info(f"Context files analysis completed: {len(file_results)} files analyzed")
            return analysis_result

        except Exception as e:
            logger.error(f"Error in analyze_context_files: {e}", exc_info=True)
            raise

    async def _analyze_file_structural(
        self,
        content: str,
        filename: str,
        file_id: str
    ) -> FileAnalysisResult:
        """
        Perform structural analysis of a single file (no quality metrics).

        Args:
            content: File content
            filename: File name
            file_id: File UUID

        Returns:
            FileAnalysisResult with only structural information
        """
        # Detect language
        ext = Path(filename).suffix.lower()
        language = LANGUAGE_BY_EXTENSION.get(ext, 'unknown')

        # Analyze structure
        structure_analysis = self.analyzer.analyze_file(content, filename)

        # Extract symbols
        symbols = self._extract_symbols(structure_analysis, content)

        # Extract structure info
        classes = [s.name for s in structure_analysis.get("symbols", []) if s.type == "class"]
        functions = [s.name for s in structure_analysis.get("symbols", []) if s.type in ("function", "method")]
        imports = structure_analysis.get("imports", [])

        return FileAnalysisResult(
            file_id=file_id,
            file_path=filename,
            language=language,
            symbols=symbols,
            classes=classes,
            functions=functions,
            imports=imports,
            content=content
        )

    async def _analyze_file_comprehensive(
        self,
        content: str,
        filename: str,
        file_id: str
    ) -> FileAnalysisResult:
        """
        Perform comprehensive analysis of a single file.

        Args:
            content: File content
            filename: File name
            file_id: File UUID

        Returns:
            FileAnalysisResult with complete analysis
        """
        # Detect language
        ext = Path(filename).suffix.lower()
        language = LANGUAGE_BY_EXTENSION.get(ext, 'unknown')

        # Analyze structure
        structure_analysis = self.analyzer.analyze_file(content, filename)

        # Extract symbols
        symbols = self._extract_symbols(structure_analysis, content)

        # Calculate metrics
        metrics = self.metrics_calculator.calculate_all_metrics(
            content,
            language
        )

        # Detect code smells
        code_smells = self.code_smell_detector.detect_all_smells(
            content,
            filename,
            language,
            symbols
        )

        # Detect security issues
        security_issues = self.security_analyzer.detect_all_vulnerabilities(
            content,
            filename,
            language
        )

        # Generate refactoring suggestions
        refactoring_suggestions = self.refactoring_suggester.generate_all_suggestions(
            content,
            filename,
            language,
            symbols,
            code_smells,
            security_issues
        )

        # Extract structure info
        classes = [s.name for s in structure_analysis.get("symbols", []) if s.type == "class"]
        functions = [s.name for s in structure_analysis.get("symbols", []) if s.type in ("function", "method")]
        imports = structure_analysis.get("imports", [])

        return FileAnalysisResult(
            file_id=file_id,
            file_path=filename,
            language=language,
            symbols=symbols,
            classes=classes,
            functions=functions,
            imports=imports,
            metrics=metrics,
            code_smells=code_smells,
            security_issues=security_issues,
            solid_violations=[],  # TODO: Implement SOLID violation detection
            refactoring_suggestions=refactoring_suggestions,
            content=content
        )

    def _extract_symbols(
        self,
        structure_analysis: Dict[str, Any],
        content: str
    ) -> List[SymbolInfo]:
        """
        Extract symbol information from structure analysis.

        Args:
            structure_analysis: Structure analysis from CodebaseAnalyzer
            content: File content

        Returns:
            List of SymbolInfo objects
        """
        symbols = []

        for sym in structure_analysis.get("symbols", []):
            location = CodeLocation(
                file_path=structure_analysis.get("file_path", ""),
                start_line=sym.start_line,
                end_line=sym.end_line
            )

            symbol_info = SymbolInfo(
                name=sym.name,
                symbol_type=sym.type,
                location=location,
                docstring=sym.docstring,
                decorators=sym.decorators,
                dependencies=sym.dependencies,
                parent=sym.parent,
                content=sym.content
            )

            symbols.append(symbol_info)

        return symbols

    def _find_symbol_content(
        self,
        analysis_result: FileAnalysisResult,
        target_name: Optional[str]
    ) -> FileAnalysisResult:
        """
        Filter analysis result to keep only the target symbol content.

        Args:
            analysis_result: Full file analysis result
            target_name: Name of symbol to find

        Returns:
            Filtered FileAnalysisResult
        """
        if not target_name:
            return analysis_result

        target_lower = target_name.lower()
        found_symbol = None

        # Search in symbols
        for symbol in analysis_result.symbols:
            if symbol.name.lower() == target_lower:
                found_symbol = symbol
                break

        if found_symbol:
            # Create a new result with only this symbol
            return FileAnalysisResult(
                file_id=analysis_result.file_id,
                file_path=analysis_result.file_path,
                language=analysis_result.language,
                symbols=[found_symbol],
                classes=[], # Irrelevant for content view
                functions=[],
                imports=[],
                content=found_symbol.content # Replace full content with symbol content
            )

        # If not found, return empty or error-like result
        return FileAnalysisResult(
            file_id=analysis_result.file_id,
            file_path=analysis_result.file_path,
            language=analysis_result.language,
            symbols=[],
            classes=[],
            functions=[],
            imports=[],
            content=f"Method/Function '{target_name}' not found in {analysis_result.file_path}"
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def is_ignored(self, path: Path) -> bool:
        """
        Check if a path should be ignored.

        Args:
            path: Path to check

        Returns:
            True if path should be ignored
        """
        # Check extension
        if path.suffix.lower() in self.ignored_extensions:
            return True

        # Check directory components
        for part in path.parts:
            if part in self.ignored_dirs:
                return True

        return False

    def format_output(self, result: 'ToolResult') -> str:
        """
        Format tool output for LLM consumption.

        Args:
            result: Tool execution result

        Returns:
            Formatted string for LLM
        """
        if not result.success:
            return f"Error in {self.name}: {result.error}"

        data = result.data

        # Handle SARIF output
        if isinstance(data, str) and '"version": "2.1.0"' in data:
            return f"""# SARIF Report Generated

SARIF 2.1.0 report has been generated successfully.

```json
{data[:1000]}...
```

*Full report available in data field*
"""

        # Handle formatted LLM output
        if isinstance(data, str) and data.startswith('#'):
            return data

        # Handle analysis result dict
        if isinstance(data, dict):
            if "files_analyzed" in data:
                return self._format_analysis_result_dict(data)

        # Default formatting
        return str(data)

    def _format_analysis_result_dict(self, data: Dict[str, Any]) -> str:
        """Format analysis result dictionary."""
        lines = []
        lines.append(f"# Codebase Analysis Results")
        lines.append(f"**Files Analyzed:** {data.get('files_analyzed', 0)}")
        lines.append("")

        summary = data.get('summary', {})
        if summary:
            lines.append("## Summary")
            lines.append(f"- Code Smells: {summary.get('total_code_smells', 0)}")
            lines.append(f"- Security Issues: {summary.get('total_security_issues', 0)}")
            lines.append(f"- Refactoring Suggestions: {summary.get('total_refactoring_suggestions', 0)}")
            lines.append("")

        results = data.get('results', [])
        for result in results[:3]:  # Limit to 3 files
            # Support both filename and file_path
            filename = result.get('filename') or result.get('file_path', 'Unknown')
            lines.append(f"## 📄 {filename}")
            lines.append(f"**Language:** {result.get('language', 'unknown')}")

            # Show classes and functions
            classes = result.get('classes', [])
            functions = result.get('functions', [])

            if classes:
                classes_str = ', '.join(classes)
                lines.append(f"**Classes ({len(classes)}):** {classes_str}")

            if functions:
                functions_str = ', '.join(functions)
                lines.append(f"**Functions ({len(functions)}):** {functions_str}")

            metrics = result.get('metrics', {})
            if metrics:
                maintainability = metrics.get('maintainability', {})
                lines.append(f"**Maintainability Index:** {maintainability.get('maintainability_index', 0):.1f}")

            code_smells = result.get('code_smells', [])
            security_issues = result.get('security_issues', [])

            lines.append(f"**Issues:** {len(code_smells)} code smells, {len(security_issues)} security issues")
            lines.append("")

        if len(results) > 3:
            lines.append(f"... and {len(results) - 3} more files")

        return "\n".join(lines)
