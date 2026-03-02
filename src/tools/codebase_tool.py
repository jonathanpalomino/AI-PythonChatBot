# =============================================================================
# src/tools/codebase_tool.py
# Tool for deep static analysis of the codebase
# =============================================================================
"""
CodebaseTool - Análisis estático profundo del código fuente.
Permite analizar archivos, buscar definiciones, navegar dependencias,
y realizar análisis de calidad del código.

NOTE: This file now uses the professional codebase_tool module.
For the latest professional implementation, see src/tools/codebase_tool/
"""
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from uuid import UUID

from src.config.constants import IGNORED_DIRS, IGNORED_EXTENSIONS
from src.config.intent_patterns import CodebaseAction
from src.document_loaders import DocumentLoaderFactory
from src.repositories.file_repository import FileRepository
from src.services.analysis.codebase_analyzer import (
    CodebaseAnalyzer,
    CodeSymbol,
    LANGUAGE_BY_EXTENSION
)
from src.services.code_graph_builder import CodeGraphBuilder
from src.tools.codebase_tool.llm_formatter import LLMFormatter
from .base_tool import BaseTool, ToolCategory, ToolParameter, ToolResult
from .codebase_tool import CodebaseTool as ProfessionalCodebaseTool

if TYPE_CHECKING:
    from src.database.unit_of_work import UnitOfWork

logger = logging.getLogger(__name__)


class CodebaseTool(BaseTool):
    """
    Tool for deep static analysis of the codebase.

    Can analyze specific files, navigate dependency graphs,
    find definitions, usages, and analyze code quality.
    """

    _INTENT_TO_CODEBASE = {
        "count_methods": CodebaseAction.BASIC_ANALYZE_FILE,
        "count_classes": CodebaseAction.BASIC_ANALYZE_FILE,
        "list_methods": CodebaseAction.BASIC_ANALYZE_FILE,
        "list_classes": CodebaseAction.BASIC_ANALYZE_FILE,
        "file_summary": CodebaseAction.BASIC_ANALYZE_FILE,
        "get_method": CodebaseAction.GET_METHOD_CONTENT,
        "get_class": CodebaseAction.ANALYZE_FILE,
        "search_symbol": CodebaseAction.FIND_DEFINITION,
    }

    def __init__(self, uow: Optional['UnitOfWork'] = None):
        """
        Initialize CodebaseTool.

        Args:
            uow: UnitOfWork instance for database operations (deprecated, use file_repo instead)
        """
        super().__init__()
        self.analyzer = CodebaseAnalyzer()
        self.root_dir = Path(os.getcwd())
        self.ignored_dirs = IGNORED_DIRS
        self.ignored_extensions = IGNORED_EXTENSIONS
        self.graph_builder = CodeGraphBuilder(str(self.root_dir))
        self.llm_formatter = LLMFormatter()
        self._uow = uow  # Kept for backward compatibility
        self.file_repo: Optional[FileRepository] = None  # Injected by chat_orchestrator

        # Lazy load RAG tool to avoid circular imports
        self._rag_tool = None

        # Temporary cache for analyzed files (file_id -> {content, analysis})
        self._analysis_cache: Dict[str, Dict[str, Any]] = {}

        logger.info(f"CodebaseTool initialized with root: {self.root_dir}")

    @property
    def rag_tool(self):
        """Lazy load RAG tool."""
        if self._rag_tool is None:
            from src.tools.rag_tool import RAGTool
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
            "Perform deep static analysis of source code. Use this to find "
            "class/function definitions, analyze file structure, navigate "
            "dependency graphs (callers/callees), and understand code logic "
            "better than generic search."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description="Action to perform. Use 'analyze_file' to count methods/classes or get file structure.",
                required=True,
                enum=CodebaseAction.ALL_ACTIONS
            ),
            ToolParameter(
                name="target",
                type="string",
                description="File path, symbol name, or method name (optional)",
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
                description="UUIDs of uploaded files to analyze directly (takes precedence over context_files)",
                required=False
            ),
            ToolParameter(
                name="new_content",
                type="string",
                description="New content for the method (required for modifymethod action)",
                required=False
            )
        ]

    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute codebase analysis action.

        Priority order:
        1. file_ids (uploaded files by UUID) - Uses professional tool
        2. analyzequality with context_files (from RAG)
        3. Standard filesystem actions
        """
        try:
            logger.info(f"CodebaseTool execution started with kwargs: {kwargs}")
            await self.validate_input(**kwargs)

            action = kwargs["action"]
            target = kwargs.get("target")
            context_files = kwargs.get("context_files", [])
            file_ids = kwargs.get("file_ids", [])

            logger.info(f"Executing action '{action}' with target '{target}'")

            # Safety net: normaliza intent action_names → CodebaseAction values.
            # Cubre el caso donde intent_action llega directamente sin pasar por action_mapping.
            if action in _INTENT_TO_CODEBASE:
                kwargs["sub_action"] = kwargs.get("sub_action", action)
                action = _INTENT_TO_CODEBASE[action]
                kwargs["action"] = action  # ProfessionalCodebaseTool recibe el action correcto
                logger.info(f"Normalized intent action '{kwargs['sub_action']}' → '{action}'")

            # ============================================================
            # PRIORITY 1: Uploaded files by UUID - Use Professional Tool
            # ============================================================
            if file_ids:
                logger.info(f"Analyzing uploaded files with professional tool: {file_ids}")

                # Special handling for getmethodcontent action
                if action == CodebaseAction.GET_METHOD_CONTENT:
                    return await self.get_method_content(file_ids, target)

                # Special handling for modifymethod action
                if action == CodebaseAction.MODIFY_METHOD:
                    new_content = kwargs.get("new_content")
                    if not new_content:
                        return ToolResult(
                            success=False,
                            data=None,
                            error="new_content parameter is required for modifymethod action"
                        )
                    return await self.modify_method(file_ids, target, new_content)

                # Use professional codebase tool for uploaded files
                professional_tool = ProfessionalCodebaseTool(uow=self._uow)
                professional_tool.file_repo = self.file_repo

                return await professional_tool.execute(**kwargs)

            # ============================================================
            # PRIORITY 2: Code quality with context files from RAG
            # ============================================================
            if action == CodebaseAction.ANALYZE_QUALITY:
                logger.info(
                    f"Analyzing code quality for '{target}' with context files: {context_files}")
                return await self.analyze_code_quality(target, context_files)

            # ============================================================
            # PRIORITY 2.5: Analyze file with context files from RAG
            # ============================================================
            if action == CodebaseAction.ANALYZE_FILE and context_files:
                logger.info(
                    f"Analyzing files from RAG context: {context_files}")
                return await self.analyze_context_files(context_files, target)

            # ============================================================
            # PRIORITY 3: Graph refresh (no target needed)
            # ============================================================
            if action == CodebaseAction.REFRESH_GRAPH:
                logger.info("Refreshing global dependency graph")
                self.graph_builder.build_graph()
                logger.info("Global dependency graph rebuilt successfully")
                return ToolResult(
                    success=True,
                    data="Global dependency graph rebuilt successfully."
                )

            # ============================================================
            # PRIORITY 4: Filesystem actions (require target OR file_ids)
            # ============================================================
            if not target and not file_ids:
                logger.warning(f"Target parameter is required for action '{action}' when no file_ids provided")
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Target parameter is required for action '{action}'"
                )

            if action == CodebaseAction.ANALYZE_FILE:
                logger.info(f"Analyzing file: {target}")
                return self.analyze_file(target)

            elif action == CodebaseAction.BASIC_ANALYZE_FILE:
                logger.info(f"Basic analysis of file: {target}")
                result = self.analyze_file(target)
                if result.success and isinstance(result.data, dict):
                    # Convert dict to FileAnalysisResult object for LLMFormatter
                    from src.tools.codebase_tool.models import FileAnalysisResult

                    # Ensure symbols are formatted as expected by FileAnalysisResult/LLMFormatter
                    # The legacy analyze_file returns 'classes' and 'functions' as lists of names
                    # We might need to map it or LLMFormatter might fail if it expects objects

                    # Actually, LLMFormatter expects FileAnalysisResult objects in some methods,
                    # but _format_file_structural_only iterates strings if that's what's there?
                    # Let's check _format_file_structural_only implementation.
                    # It iterates file_result.classes and .functions.

                    # We need to construct a minimal compatible object or dict
                    file_result = FileAnalysisResult(
                        file_path=str(result.data.get("file", target)),
                        language=result.data.get("type", "unknown"),
                        classes=result.data.get("classes", []),
                        functions=result.data.get("functions", []),
                        imports=result.data.get("imports", []),
                        content=result.data.get("content", "")
                    )

                    formatted_output = self.llm_formatter.format_file_result(
                        file_result,
                        action="basic_analyze_file"
                    )
                    return ToolResult(success=True, data=formatted_output)

                return result

            elif action == CodebaseAction.FIND_DEFINITION:
                logger.info(f"Finding definition for: {target}")
                return self.find_definition(target)

            elif action == CodebaseAction.FIND_REFERENCES:
                logger.info(f"Finding references for: {target}")
                return self.find_references(target)

            elif action == CodebaseAction.GET_CALLERS:
                logger.info(f"Getting callers for: {target}")
                callers = self.graph_builder.get_callers(target)
                logger.info(f"Found {len(callers)} callers for '{target}'")
                return ToolResult(
                    success=True,
                    data={
                        "symbol": target,
                        "callers": callers[:50],  # Limit to 50
                        "count": len(callers)
                    }
                )

            elif action == CodebaseAction.GET_DEPENDENCIES:
                logger.info(f"Getting dependencies for: {target}")
                deps = self.graph_builder.get_dependencies(target)
                logger.info(f"Found {len(deps)} dependencies for '{target}'")
                return ToolResult(
                    success=True,
                    data={
                        "symbol": target,
                        "dependencies": deps[:50],  # Limit to 50
                        "count": len(deps)
                    }
                )

            elif action == CodebaseAction.EXPLAIN:
                logger.info(f"Explaining file: {target}")
                return self.analyze_file(target)

            elif action == CodebaseAction.GET_METHOD_CONTENT:
                logger.info(f"Getting method content for: {target}")
                return self.get_method_content_filesystem(target)

            elif action == CodebaseAction.MODIFY_METHOD:
                logger.info(f"Modifying method for: {target}")
                new_content = kwargs.get("new_content")
                if not new_content:
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"new_content parameter is required for {CodebaseAction.MODIFY_METHOD} action"
                    )
                return self.modify_method_filesystem(target, new_content)

            else:
                logger.error(f"Unknown action: {action}")
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Unknown action: {action}"
                )

        except Exception as e:
            logger.error(f"Error executing CodebaseTool: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))

    # =========================================================================
    # UPLOADED FILES ANALYSIS (Priority 1)
    # =========================================================================

    async def analyze_uploaded_files(
        self,
        file_ids: List[str],
        action: str,
        target: Optional[str] = None
    ) -> ToolResult:
        """
        Analyze uploaded files by UUID from database.
        Highest priority - takes precedence over context_files.

        Args:
            file_ids: List of file UUIDs
            action: Action to perform (analyze_file, analyze_quality, etc.)
            target: Optional symbol/method name for specific analysis

        Returns:
            ToolResult with analysis results
        """
        try:
            if not self.file_repo:
                logger.error("file_repo not available for CodebaseTool")
                return ToolResult(
                    success=False,
                    data=None,
                    error="file_repo not configured. Cannot analyze uploaded files."
                )

            analysis_results = []

            for file_id_str in file_ids:
                logger.info(f"Analyzing uploaded file: {file_id_str}")
                file_id = UUID(file_id_str)
                logger.info(f"File ID: {file_id}")
                try:
                    # Get file from database using injected file_repo
                    file_record = await self.file_repo.get_by_id(file_id)
                    if not file_record:
                        logger.warning(f"File not found in database: {file_id}")
                        analysis_results.append({
                            "file_id": str(file_id),
                            "error": "File not found in database"
                        })
                        continue

                    # FIX: Initialize storage_path from file_record
                    storage_path = Path(file_record.storage_path)
                    logger.info(f"Storage path: {storage_path}")
                    content = None
                    # Try reading from disk first
                    if storage_path.exists():
                        try:
                            content = storage_path.read_text(encoding='utf-8', errors='ignore')
                            #logger.info(f"Content readed from disk: {content}")
                        except Exception as e:
                            logger.warning(f"Failed to read local file {file_id}: {e}")

                    # Fallback to RAG if local read failed
                    if content is None:
                        logger.info(f"File {file_id} not found locally, attempting RAG reconstruction")
                        # Try to get collection from file record or default
                        collection_name = getattr(file_record, 'collection_name', 'documentation')
                        content = await self.rag_tool.get_full_document_content(str(file_id), collection_name)

                        if content:
                             logger.info(f"Successfully reconstructed file {file_id} from RAG")
                        else:
                             logger.warning(f"Failed to reconstruct file {file_id} from RAG")
                             analysis_results.append({
                                "file_id": str(file_id),
                                "error": "File content not found locally or in RAG"
                            })
                             continue

                    filename = file_record.file_name

                    # Perform analysis based on action
                    if action in (CodebaseAction.ANALYZE_FILE, CodebaseAction.EXPLAIN):
                        analysis = self.analyzer.analyze_file(content, filename)

                        # Cache the analysis for future getmethodcontent calls
                        self._analysis_cache[str(file_id)] = {
                            "content": content,
                            "analysis": analysis,
                            "filename": filename
                        }

                        # Include full symbol details for AST analysis
                        symbols_detail = []
                        methods_per_class = {}  # Group methods by class
                        methods_count = 0

                        for sym in analysis.get("symbols", []):
                            # Track methods per class
                            if sym.type == "method":
                                methods_count += 1
                                parent = sym.parent or "_global"
                                if parent not in methods_per_class:
                                    methods_per_class[parent] = []
                                methods_per_class[parent].append({
                                    "name": sym.name,
                                    "start_line": sym.start_line,
                                    "end_line": sym.end_line,
                                    "content": sym.content[:200] + "..." if len(sym.content) > 200 else sym.content
                                })

                            symbols_detail.append({
                                "name": sym.name,
                                "type": sym.type,
                                "start_line": sym.start_line,
                                "end_line": sym.end_line,
                                "content": sym.content,
                                "docstring": sym.docstring,
                                "decorators": sym.decorators,
                                "dependencies": list(sym.dependencies) if sym.dependencies else [],
                                "parent": sym.parent
                            })

                        # Get all classes
                        all_classes = [s.name for s in analysis.get("symbols", []) if s.type == "class"]

                        # Get all functions (methods + standalone functions)
                        all_functions = [s.name for s in analysis.get("symbols", []) if s.type in ("function", "method")]

                        analysis_results.append({
                            "file_id": str(file_id),
                            "filename": filename,
                            "language": analysis.get("language", "unknown"),
                            "symbols": symbols_detail,  # Full AST symbol details
                            "classes": all_classes,
                            "classes_count": len(all_classes),
                            "functions": all_functions,
                            "functions_count": len(all_functions),
                            "methods_count": methods_count,  # Total methods count
                            "methods_per_class": methods_per_class,  # Methods grouped by class
                            "imports": analysis.get("imports", []),
                            "imports_count": len(analysis.get("imports", [])),
                            "complexity": analysis.get("complexity", 0),
                            "content": content  # Return full content for reference
                        })

                    elif action == CodebaseAction.ANALYZE_QUALITY and target:
                        symbol_analysis = self.find_symbol_and_analyze_quality(
                            content,
                            target
                        )
                        analysis_results.append({
                            "file_id": str(file_id),
                            "filename": filename,
                            "target": target,
                            "symbol_analysis": symbol_analysis
                        })

                    else:
                        analysis_results.append({
                            "file_id": str(file_id),
                            "filename": filename,
                            "error": f"Action '{action}' not supported for uploaded files"
                        })

                except Exception as e:
                    logger.error(f"Error analyzing uploaded file {file_id}: {e}", exc_info=True)
                    analysis_results.append({
                        "file_id": str(file_id),
                        "error": str(e)
                    })

            if analysis_results:
                return ToolResult(
                    success=True,
                    data={
                        "action": action,
                        "target": target,
                        "files_analyzed": len(analysis_results),
                        "results": analysis_results
                    }
                )
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error="No valid content found in uploaded files"
                )

        except Exception as e:
            logger.error(f"Error in analyze_uploaded_files: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))

    # =========================================================================
    # CONTEXT FILES ANALYSIS (from RAG)
    # =========================================================================

    async def analyze_context_files(
        self,
        context_files: List[str],
        target: Optional[str] = None
    ) -> ToolResult:
        """
        Analyze files from RAG context by filename.

        This method reads files from the filesystem using the filenames
        provided in context_files (from RAG search results).

        Args:
            context_files: List of filenames to analyze
            target: Optional symbol/method name for specific analysis

        Returns:
            ToolResult with analysis results
        """
        try:
            analysis_results = []

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
                        analysis_results.append({
                            "filename": filename,
                            "error": "File not found in workspace"
                        })
                        continue

                # Read file content
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                except Exception as e:
                    logger.warning(f"Failed to read file {filename}: {e}")
                    analysis_results.append({
                        "filename": filename,
                        "error": f"Failed to read file: {str(e)}"
                    })
                    continue

                # Perform analysis
                analysis = self.analyzer.analyze_file(content, filename)

                # Extract symbol details
                symbols_detail = []
                methods_per_class = {}
                methods_count = 0

                for sym in analysis.get("symbols", []):
                    # Track methods per class
                    if sym.type == "method":
                        methods_count += 1
                        parent = sym.parent or "_global"
                        if parent not in methods_per_class:
                            methods_per_class[parent] = []
                        methods_per_class[parent].append({
                            "name": sym.name,
                            "start_line": sym.start_line,
                            "end_line": sym.end_line,
                            "content": sym.content[:200] + "..." if len(sym.content) > 200 else sym.content
                        })

                    symbols_detail.append({
                        "name": sym.name,
                        "type": sym.type,
                        "start_line": sym.start_line,
                        "end_line": sym.end_line,
                        "content": sym.content,
                        "docstring": sym.docstring,
                        "decorators": sym.decorators,
                        "dependencies": list(sym.dependencies) if sym.dependencies else [],
                        "parent": sym.parent
                    })

                # Get all classes and functions
                all_classes = [s.name for s in analysis.get("symbols", []) if s.type == "class"]
                all_functions = [s.name for s in analysis.get("symbols", []) if s.type in ("function", "method")]

                analysis_results.append({
                    "filename": filename,
                    "language": analysis.get("language", "unknown"),
                    "symbols": symbols_detail,
                    "classes": all_classes,
                    "classes_count": len(all_classes),
                    "functions": all_functions,
                    "functions_count": len(all_functions),
                    "methods_count": methods_count,
                    "methods_per_class": methods_per_class,
                    "imports": analysis.get("imports", []),
                    "imports_count": len(analysis.get("imports", [])),
                    "complexity": analysis.get("complexity", 0),
                    "content": content
                })

            if analysis_results:
                return ToolResult(
                    success=True,
                    data={
                        "action": "analyze_file",
                        "target": target,
                        "files_analyzed": len(analysis_results),
                        "results": analysis_results
                    }
                )
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error="No valid content found in context files"
                )

        except Exception as e:
            logger.error(f"Error in analyze_context_files: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))

    # =========================================================================
    # METHOD CONTENT RETRIEVAL (New Feature)
    # =========================================================================

    async def get_method_content(
        self,
        file_ids: List[str],
        method_name: Optional[str] = None
    ) -> ToolResult:
        """
        Get the complete content of a specific method from uploaded files.

        This method uses the cache to avoid re-reading files from disk/RAG.

        Args:
            file_ids: List of file UUIDs to search
            method_name: Name of the method to retrieve (optional, if None returns all methods)

        Returns:
            ToolResult with method content
        """
        try:
            if not self.file_repo:
                logger.error("file_repo not available for CodebaseTool")
                return ToolResult(
                    success=False,
                    data=None,
                    error="file_repo not configured. Cannot retrieve method content."
                )

            if not method_name:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"method_name parameter is required for {CodebaseAction.GET_METHOD_CONTENT} action"
                )

            logger.info(f"Getting method content for '{method_name}' from {len(file_ids)} files")

            # Search for the method in all provided files
            for file_id_str in file_ids:
                file_id = file_id_str

                # Check cache first
                if file_id in self._analysis_cache:
                    logger.info(f"Using cached analysis for file: {file_id}")
                    cached_data = self._analysis_cache[file_id]
                    analysis = cached_data.get("analysis")
                    filename = cached_data.get("filename")
                else:
                    # Load file and analyze
                    logger.info(f"Loading and analyzing file: {file_id}")
                    file_record = await self.file_repo.get_by_id(UUID(file_id))
                    if not file_record:
                        logger.warning(f"File not found in database: {file_id}")
                        continue

                    # Read file content
                    storage_path = Path(file_record.storage_path)
                    content = None

                    if storage_path.exists():
                        try:
                            content = storage_path.read_text(encoding='utf-8', errors='ignore')
                        except Exception as e:
                            logger.warning(f"Failed to read local file {file_id}: {e}")

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

                    # Analyze file
                    analysis = self.analyzer.analyze_file(content, file_record.file_name)
                    filename = file_record.file_name

                    # Cache the analysis
                    self._analysis_cache[file_id] = {
                        "content": content,
                        "analysis": analysis,
                        "filename": filename
                    }

                # Search for the method in the analysis
                for sym in analysis.get("symbols", []):
                    if sym.name == method_name:
                        logger.info(f"Found method '{method_name}' in file '{filename}'")

                        # Format the response with complete method content
                        return ToolResult(
                            success=True,
                            data={
                                "file_id": file_id,
                                "filename": filename,
                                "method_name": method_name,
                                "method_type": sym.type,
                                "start_line": sym.start_line,
                                "end_line": sym.end_line,
                                "content": sym.content,
                                "docstring": sym.docstring,
                                "decorators": sym.decorators,
                                "dependencies": list(sym.dependencies) if sym.dependencies else [],
                                "parent": sym.parent
                            }
                        )

            # Method not found in any file
            logger.warning(f"Method '{method_name}' not found in any of the provided files")
            return ToolResult(
                success=False,
                data=None,
                error=f"Method '{method_name}' not found in any of the provided files"
            )

        except Exception as e:
            logger.error(f"Error in get_method_content: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))

    def get_method_content_filesystem(self, target: str) -> ToolResult:
        """
        Get the complete content of a specific method from a filesystem file.

        The target parameter should be in the format: "filepath:methodname"
        Example: "src/services/chat_orchestrator.py:process_message"

        Args:
            target: File path and method name separated by colon

        Returns:
            ToolResult with method content
        """
        try:
            # Parse target (format: "filepath:methodname")
            if ":" not in target:
                return ToolResult(
                    success=False,
                    data=None,
                    error="Target must be in format 'filepath:methodname'. Example: 'src/services/chat_orchestrator.py:process_message'"
                )

            filepath_str, method_name = target.split(":", 1)
            filepath = self.root_dir / filepath_str

            logger.info(f"Getting method content for '{method_name}' from file: {filepath}")

            # Try to find file if not found
            if not filepath.exists():
                matches = list(self.root_dir.rglob(filepath_str))
                if matches:
                    filepath = matches[0]
                else:
                    logger.error(f"File not found: {filepath_str}")
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"File not found: {filepath_str}"
                    )

            # Read file content
            try:
                content = filepath.read_text(encoding='utf-8', errors='ignore')
            except Exception as e:
                logger.error(f"Failed to read file {filepath}: {e}")
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Failed to read file: {e}"
                )

            # Analyze file
            analysis = self.analyzer.analyze_file(content, filepath.name)

            # Search for the method
            for sym in analysis.get("symbols", []):
                if sym.name == method_name:
                    logger.info(f"Found method '{method_name}' in file '{filepath}'")

                    return ToolResult(
                        success=True,
                        data={
                            "filepath": str(filepath),
                            "method_name": method_name,
                            "method_type": sym.type,
                            "start_line": sym.start_line,
                            "end_line": sym.end_line,
                            "content": sym.content,
                            "docstring": sym.docstring,
                            "decorators": sym.decorators,
                            "dependencies": list(sym.dependencies) if sym.dependencies else [],
                            "parent": sym.parent
                        }
                    )

            # Method not found
            logger.warning(f"Method '{method_name}' not found in file '{filepath}'")
            return ToolResult(
                success=False,
                data=None,
                error=f"Method '{method_name}' not found in file '{filepath}'"
            )

        except Exception as e:
            logger.error(f"Error in get_method_content_filesystem: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))

    async def modify_method(
        self,
        file_ids: List[str],
        method_name: Optional[str] = None,
        new_content: Optional[str] = None
    ) -> ToolResult:
        """
        Modify a specific method in uploaded files and return the complete modified file content.

        This method replaces the content of a method with new content and returns
        the complete file content for easy copying.

        Args:
            file_ids: List of file UUIDs to search
            method_name: Name of the method to modify
            new_content: New content for the method

        Returns:
            ToolResult with complete modified file content
        """
        try:
            if not self.file_repo:
                logger.error("file_repo not available for CodebaseTool")
                return ToolResult(
                    success=False,
                    data=None,
                    error="file_repo not configured. Cannot modify method."
                )

            if not method_name or not new_content:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"method_name and new_content parameters are required for {CodebaseAction.MODIFY_METHOD} action"
                )

            logger.info(f"Modifying method '{method_name}' in {len(file_ids)} files")

            # Search for the method in all provided files
            for file_id_str in file_ids:
                file_id = file_id_str

                # Check cache first
                if file_id in self._analysis_cache:
                    logger.info(f"Using cached analysis for file: {file_id}")
                    cached_data = self._analysis_cache[file_id]
                    content = cached_data.get("content")
                    analysis = cached_data.get("analysis")
                    filename = cached_data.get("filename")
                else:
                    # Load file and analyze
                    logger.info(f"Loading and analyzing file: {file_id}")
                    file_record = await self.file_repo.get_by_id(UUID(file_id))
                    if not file_record:
                        logger.warning(f"File not found in database: {file_id}")
                        continue

                    # Read file content
                    storage_path = Path(file_record.storage_path)
                    content = None

                    if storage_path.exists():
                        try:
                            content = storage_path.read_text(encoding='utf-8', errors='ignore')
                        except Exception as e:
                            logger.warning(f"Failed to read local file {file_id}: {e}")

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

                    # Analyze file
                    analysis = self.analyzer.analyze_file(content, file_record.file_name)
                    filename = file_record.file_name

                    # Cache the analysis
                    self._analysis_cache[file_id] = {
                        "content": content,
                        "analysis": analysis,
                        "filename": filename
                    }

                # Search for the method in the analysis
                for sym in analysis.get("symbols", []):
                    if sym.name == method_name:
                        logger.info(f"Found method '{method_name}' in file '{filename}'")

                        # Split content into lines
                        lines = content.split('\n')

                        # Replace the method content (adjust for 0-based indexing)
                        start_idx = sym.start_line - 1
                        end_idx = sym.end_line

                        # Create new content with the modified method
                        new_lines = lines[:start_idx] + [new_content] + lines[end_idx:]
                        new_file_content = '\n'.join(new_lines)

                        # Update cache
                        self._analysis_cache[file_id]["content"] = new_file_content

                        # Return the complete modified file content
                        return ToolResult(
                            success=True,
                            data={
                                "file_id": file_id,
                                "filename": filename,
                                "method_name": method_name,
                                "method_type": sym.type,
                                "start_line": sym.start_line,
                                "end_line": sym.end_line,
                                "old_content": sym.content,
                                "new_content": new_content,
                                "complete_file_content": new_file_content,
                                "message": f"Method '{method_name}' has been modified. Below is the complete file content for copying."
                            }
                        )

            # Method not found in any file
            logger.warning(f"Method '{method_name}' not found in any of the provided files")
            return ToolResult(
                success=False,
                data=None,
                error=f"Method '{method_name}' not found in any of the provided files"
            )

        except Exception as e:
            logger.error(f"Error in modify_method: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))

    def modify_method_filesystem(self, target: str, new_content: str) -> ToolResult:
        """
        Modify a specific method in a filesystem file and return the complete modified file content.

        The target parameter should be in the format: "filepath:methodname"
        Example: "src/services/chat_orchestrator.py:process_message"

        Args:
            target: File path and method name separated by colon
            new_content: New content for the method

        Returns:
            ToolResult with complete modified file content
        """
        try:
            # Parse target (format: "filepath:methodname")
            if ":" not in target:
                return ToolResult(
                    success=False,
                    data=None,
                    error="Target must be in format 'filepath:methodname'. Example: 'src/services/chat_orchestrator.py:process_message'"
                )

            filepath_str, method_name = target.split(":", 1)
            filepath = self.root_dir / filepath_str

            logger.info(f"Modifying method '{method_name}' in file: {filepath}")

            # Try to find file if not found
            if not filepath.exists():
                matches = list(self.root_dir.rglob(filepath_str))
                if matches:
                    filepath = matches[0]
                else:
                    logger.error(f"File not found: {filepath_str}")
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"File not found: {filepath_str}"
                    )

            # Read file content
            try:
                content = filepath.read_text(encoding='utf-8', errors='ignore')
            except Exception as e:
                logger.error(f"Failed to read file {filepath}: {e}")
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Failed to read file: {e}"
                )

            # Analyze file
            analysis = self.analyzer.analyze_file(content, filepath.name)

            # Search for the method
            for sym in analysis.get("symbols", []):
                if sym.name == method_name:
                    logger.info(f"Found method '{method_name}' in file '{filepath}'")

                    # Split content into lines
                    lines = content.split('\n')

                    # Replace the method content (adjust for 0-based indexing)
                    start_idx = sym.start_line - 1
                    end_idx = sym.end_line

                    # Create new content with the modified method
                    new_lines = lines[:start_idx] + [new_content] + lines[end_idx:]
                    new_file_content = '\n'.join(new_lines)

                    # Return the complete modified file content
                    return ToolResult(
                        success=True,
                        data={
                            "filepath": str(filepath),
                            "method_name": method_name,
                            "method_type": sym.type,
                            "start_line": sym.start_line,
                            "end_line": sym.end_line,
                            "old_content": sym.content,
                            "new_content": new_content,
                            "complete_file_content": new_file_content,
                            "message": f"Method '{method_name}' has been modified. Below is the complete file content for copying."
                        }
                    )

            # Method not found
            logger.warning(f"Method '{method_name}' not found in file '{filepath}'")
            return ToolResult(
                success=False,
                data=None,
                error=f"Method '{method_name}' not found in file '{filepath}'"
            )

        except Exception as e:
            logger.error(f"Error in modify_method_filesystem: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))

    # =========================================================================
    # CODE QUALITY ANALYSIS (Priority 2)
    # =========================================================================

    async def analyze_code_quality(
        self,
        symbol: str,
        context_files: List[str]
    ) -> ToolResult:
        """
        Analyze code quality for a specific method/function.

        Args:
            symbol: Method/function name to analyze
            context_files: Files from RAG context to search

        Returns:
            ToolResult with quality report
        """
        logger.info(f"Analyzing code quality for symbol '{symbol}'")

        # Determine files to search
        files_to_search = context_files if context_files else self.get_all_code_files()
        logger.info(f"Searching for symbol in {len(files_to_search)} files")

        target_filepath = None
        target_content = None

        # Search for symbol definition
        for filepath in files_to_search:
            filepath = self.root_dir / filepath
            if not filepath.exists():
                continue

            try:
                content = filepath.read_text(encoding='utf-8', errors='ignore')
            except Exception as e:
                logger.warning(f"Failed to read {filepath}: {e}")
                continue

            # Look for function/method definition
            if f"def {symbol}" in content or f"async def {symbol}" in content:
                target_filepath = filepath
                target_content = content
                logger.info(f"Found definition for '{symbol}' in {filepath}")
                break

        if not target_filepath or not target_content:
            logger.warning(f"Method '{symbol}' not found in provided context")
            return ToolResult(
                success=False,
                data=None,
                error=f"Method '{symbol}' not found in {len(files_to_search)} files"
            )

        # Analyze file
        logger.info(f"Analyzing file: {target_filepath}")
        analysis = self.analyzer.analyze_file(target_content, target_filepath.name)

        # Find target symbol
        target_symbol = None
        for sym in analysis.get("symbols", []):
            if sym.name == symbol:
                target_symbol = sym
                break

        if not target_symbol:
            logger.error(f"Symbol '{symbol}' found in file but analysis failed")
            return ToolResult(
                success=False,
                data=None,
                error=f"Symbol '{symbol}' analysis failed despite being found"
            )

        # Generate quality report
        quality_report = {
            "symbol": symbol,
            "file": str(target_filepath),
            "type": target_symbol.type,
            "line_range": f"{target_symbol.start_line}-{target_symbol.end_line}",
            "code": target_symbol.content,
            "docstring": target_symbol.docstring,
            "complexity_score": self.calculate_complexity(target_symbol),
            "dependencies": list(target_symbol.dependencies),
            "issues": self.detect_code_smells(target_symbol),
            "suggestions": self.generate_suggestions(target_symbol)
        }

        logger.info(f"Code quality analysis completed for '{symbol}'")
        return ToolResult(success=True, data=quality_report)

    def find_symbol_and_analyze_quality(
        self,
        content: str,
        target: str
    ) -> Dict[str, Any]:
        """
        Find specific symbol in content and analyze its quality.

        Args:
            content: File content
            target: Symbol name to find

        Returns:
            Quality analysis dict or error
        """
        analysis = self.analyzer.analyze_file(content, "temp_file")

        # Find target symbol
        target_symbol = None
        for symbol in analysis.get("symbols", []):
            if symbol.name == target:
                target_symbol = symbol
                break

        if not target_symbol:
            return {"error": f"Symbol '{target}' not found in file"}

        return {
            "symbol": target_symbol.name,
            "type": target_symbol.type,
            "complexity": self.calculate_complexity(target_symbol),
            "issues": self.detect_code_smells(target_symbol),
            "suggestions": self.generate_suggestions(target_symbol)
        }

    # =========================================================================
    # FILESYSTEM ANALYSIS (Priority 3)
    # =========================================================================

    def analyze_file(self, filepath_str: str) -> ToolResult:
        """
        Analyze a specific file using AST (Python) or Regex (others).

        Args:
            filepath_str: Relative or absolute file path

        Returns:
            ToolResult with file analysis
        """
        filepath = self.root_dir / filepath_str
        logger.info(f"Analyzing file: {filepath}")

        # Try to find file if not found
        if not filepath.exists():
            matches = list(self.root_dir.rglob(filepath_str))
            if matches:
                filepath = matches[0]
            else:
                logger.error(f"File not found: {filepath_str}")
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"File not found: {filepath_str}"
                )

        try:
            content = filepath.read_text(encoding='utf-8', errors='ignore')
            logger.info(f"File content read successfully for {filepath}")

            if filepath.suffix == ".py":
                # Python AST analysis
                analysis = self.analyzer.analyze_file(content, filepath.name)
                summary = {
                    "file": str(filepath),
                    "type": "python_ast",
                    "classes": [
                        s.name for s in analysis.get("symbols", [])
                        if s.type == "class"
                    ],
                    "functions": [
                        s.name for s in analysis.get("symbols", [])
                        if s.type == "function"
                    ],
                    "imports": analysis.get("imports", []),
                    "dependencies_detected": [],
                    "content": content  # Return full content for analysis
                }

                # Aggregate dependencies from symbols
                all_deps = set()
                for sym in analysis.get("symbols", []):
                    all_deps.update(sym.dependencies)
                summary["dependencies_detected"] = list(all_deps)

                logger.info(f"File analysis completed for {filepath}")
                return ToolResult(success=True, data=summary)
            else:
                # Generic analysis
                analysis = self.analyzer.analyze_file(content, filepath.name)
                summary = {
                    "file": str(filepath),
                    "type": f"{filepath.suffix.lower()}_analysis",
                    "classes": [
                        s.name for s in analysis.get("symbols", [])
                        if s.type == "class"
                    ],
                    "functions": [
                        s.name for s in analysis.get("symbols", [])
                        if s.type == "function"
                    ],
                    "note": f"Structural analysis via CodebaseAnalyzer (Language: {filepath.suffix})",
                    "content": content  # Return full content for analysis
                }

                logger.info(f"File analysis completed for {filepath} \n\n {summary}")
                return ToolResult(success=True, data=summary)

        except Exception as e:
            logger.error(f"Error analyzing file: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Error analyzing file: {e}"
            )

    def find_definition(self, symbol: str) -> ToolResult:
        """
        Scan files to find where a class or function is defined.

        Args:
            symbol: Symbol name to find

        Returns:
            ToolResult with file paths containing definition
        """
        logger.info(f"Finding definition for symbol '{symbol}'")
        matches = []

        # Get code file extensions
        extensions = [f"*{ext}" for ext in DocumentLoaderFactory.get_code_extensions()]

        for glob_pattern in extensions:
            for file_p in self.root_dir.rglob(glob_pattern):
                if self.is_ignored(file_p):
                    continue

                try:
                    content = file_p.read_text(encoding='utf-8', errors='ignore')
                    lang = LANGUAGE_BY_EXTENSION.get(file_p.suffix.lower())
                    is_match = False

                    # Language-specific checks
                    if file_p.suffix == ".py":
                        if f"class {symbol}" in content or f"def {symbol}" in content:
                            is_match = True
                    elif lang in ("typescript", "javascript", "java", "csharp"):
                        if (f"class {symbol}" in content or
                            f"function {symbol}" in content or
                            f"interface {symbol}" in content):
                            is_match = True
                    else:
                        # Generic check
                        if f"class {symbol}" in content or f"function {symbol}" in content:
                            is_match = True

                    if is_match:
                        matches.append(str(file_p))
                except:
                    continue

        if matches:
            logger.info(f"Found {len(matches)} definitions for '{symbol}'")
            return ToolResult(success=True, data=matches)
        else:
            logger.info(f"No definition found for '{symbol}'")
            return ToolResult(
                success=True,
                data="No definition found (checked typical source files)."
            )

    def find_references(self, symbol: str) -> ToolResult:
        """
        Find text occurrences of the symbol (references).

        Args:
            symbol: Symbol name to find

        Returns:
            ToolResult with files containing references
        """
        logger.info(f"Finding references for symbol '{symbol}'")
        references = []

        # Get code file extensions + HTML
        extensions = [f"*{ext}" for ext in DocumentLoaderFactory.get_code_extensions()]
        if ".html" not in extensions:
            extensions.append("*.html")

        for glob_pattern in extensions:
            for file_p in self.root_dir.rglob(glob_pattern):
                if self.is_ignored(file_p):
                    continue

                try:
                    content = file_p.read_text(encoding='utf-8', errors='ignore')
                    if symbol in content:
                        references.append(str(file_p))
                except:
                    continue

        logger.info(f"Found {len(references)} references for '{symbol}'")
        return ToolResult(
            success=True,
            data={
                "symbol": symbol,
                "reference_count": len(references),
                "files": references[:20]  # Limit to 20 files
            }
        )

    # =========================================================================
    # CODE QUALITY HELPERS
    # =========================================================================

    def calculate_complexity(self, symbol: CodeSymbol) -> int:
        """
        Calculate approximate cyclomatic complexity.

        Args:
            symbol: CodeSymbol to analyze

        Returns:
            Complexity score
        """
        code = symbol.content
        complexity = 1  # Base complexity

        # Count control structures
        complexity += code.count("if ")
        complexity += code.count("elif ")
        complexity += code.count("for ")
        complexity += code.count("while ")
        complexity += code.count(" and ")
        complexity += code.count(" or ")
        complexity += code.count("except ")

        return complexity

    def detect_code_smells(self, symbol: CodeSymbol) -> List[str]:
        """
        Detect problematic patterns in code.

        Args:
            symbol: CodeSymbol to analyze

        Returns:
            List of detected issues
        """
        issues = []
        code = symbol.content

        # Check line count
        line_count = code.count("\n")
        if line_count > 50:
            issues.append(f"Method too long ({line_count} lines, recommended < 50)")

        # Check parameter count
        param_match = re.search(r"def .*?\((.*?)\)", code)
        if param_match:
            params = [p.strip() for p in param_match.group(1).split(",") if p.strip()]
            if len(params) > 5:
                issues.append(f"Too many parameters ({len(params)}, recommended < 5)")

        # Check docstring
        if not symbol.docstring:
            issues.append("Missing docstring")

        # Check coupling
        if len(symbol.dependencies) > 8:
            issues.append(f"High coupling ({len(symbol.dependencies)} dependencies)")

        # Check exception handling
        if "except:" in code or "except Exception" in code:
            issues.append("Bare except clause detected (catches all exceptions)")

        return issues

    def generate_suggestions(self, symbol: CodeSymbol) -> List[str]:
        """
        Generate improvement suggestions.

        Args:
            symbol: CodeSymbol to analyze

        Returns:
            List of suggestions
        """
        suggestions = []
        code = symbol.content

        if not symbol.docstring:
            suggestions.append("Add docstring with parameters, returns, and raises")

        if self.calculate_complexity(symbol) > 10:
            suggestions.append("Consider refactoring into smaller functions (complexity > 10)")

        if "TODO" in code or "FIXME" in code:
            suggestions.append("Address TODO/FIXME comments")

        if symbol.type == "method" and "self" in code:
            if not re.search(r"def __init__", code):
                suggestions.append("Consider if method should be static or class method")

        return suggestions

    # =========================================================================
    # UTILITY METHODS
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
        Provides readable, structured output for code analysis results.

        Args:
            result: Tool execution result

        Returns:
            Formatted string for LLM
        """
        if not result.success:
            return f"Error in {self.name}: {result.error}"

        data = result.data

        # If data is already a formatted string (from professional tool), return it directly
        if isinstance(data, str):
            return data

        if isinstance(data, dict):
            # Handle different result types
            if "files_analyzed" in data:  # analyze_uploaded_files results
                files_info = []
                for f in data.get("results", []):
                    if "error" in f:
                        files_info.append(f"  - {f.get('filename', f.get('file_id', f.get('file_path')))}: ERROR: {f['error']}")
                    else:
                        # Support both filename and file_path
                        filename = f.get('filename') or f.get('file_path', 'unknown')
                        lang = f.get('language', 'unknown')
                        classes = f.get("classes", [])
                        functions = f.get("functions", [])
                        methods_count = f.get("methods_count", 0)
                        methods_per_class = f.get("methods_per_class", {})
                        complexity = f.get("complexity", 0)
                        imports_count = f.get("imports_count", 0)

                        # Format methods per class
                        methods_str = []
                        for class_name, methods in methods_per_class.items():
                            method_names = [m.get("name", "?") for m in methods]
                            methods_str.append(f"{class_name}: {', '.join(method_names)}")

                        file_info = [
                            f"  - {filename} ({lang})",
                            f"    Classes ({len(classes)}): {', '.join(classes) if classes else 'None'}",
                            f"    Methods ({methods_count}): {', '.join(methods_str) if methods_str else 'None'}",
                            f"    Functions (standalone): {', '.join([f for f in functions if f not in [m.get('name') for methods in methods_per_class.values() for m in methods]])}",
                            f"    Imports: {imports_count}",
                            f"    Complexity: {complexity}"
                        ]
                        files_info.append("\n".join(file_info))

                files_info_text = "\n".join(files_info)

                # Add instructions for getting method content
                instructions = (
                    f"\n\n💡 To get the complete code of a specific method, use:\n"
                    f"   - For uploaded files: action='{CodebaseAction.GET_METHOD_CONTENT}', file_ids=['<file_id>'], target='<method_name>'\n"
                    f"   - For filesystem files: action='{CodebaseAction.GET_METHOD_CONTENT}', target='<filepath>:<method_name>'\n"
                    f"   Example: action='{CodebaseAction.GET_METHOD_CONTENT}', file_ids=['abc123'], target='process_message'"
                )

                return (
                    f"Codebase Analysis Results\n"
                    f"   Files analyzed: {data.get('files_analyzed', 0)}\n"
                    f"\n{files_info_text}"
                    f"{instructions}"
                )

            elif "symbol" in data and "callers" in data:  # CodebaseAction.GET_CALLERS results
                callers = data.get("callers", [])
                callers_str = "\n".join(f"  - {c}" for c in callers[:10])
                if len(callers) > 10:
                    callers_str += f"\n  ... and {len(callers) - 10} more"
                return (
                    f"Callers of `{data['symbol']}` ({data.get('count', len(callers))} total):\n"
                    f"{callers_str}"
                )

            elif "symbol" in data and "dependencies" in data:  # CodebaseAction.GET_DEPENDENCIES results
                deps = data.get("dependencies", [])
                deps_str = "\n".join(f"  - {d}" for d in deps[:10])
                if len(deps) > 10:
                    deps_str += f"\n  ... and {len(deps) - 10} more"
                return (
                    f"Dependencies of `{data['symbol']}` ({data.get('count', len(deps))} total):\n"
                    f"{deps_str}"
                )

            elif "action" in data:  # General action results
                return f"{data.get('action', 'Analysis')} completed: {data.get('files_analyzed', 0)} files processed"

            elif "file" in data:  # analyze_file results
                classes = data.get("classes", [])
                functions = data.get("functions", [])
                imports = data.get("imports", [])
                return (
                    f"File: {data.get('file', 'Unknown')}\n"
                    f"   Classes: {', '.join(classes) if classes else 'None'}\n"
                    f"   Functions: {', '.join(functions) if functions else 'None'}\n"
                    f"   Imports: {len(imports)} import{'s' if len(imports) != 1 else ''}"
                )

            elif "method_name" in data and "content" in data:  # CodebaseAction.GET_METHOD_CONTENT results
                method_name = data.get("method_name")
                method_type = data.get("method_type", "unknown")
                start_line = data.get("start_line")
                end_line = data.get("end_line")
                content = data.get("content")
                docstring = data.get("docstring")
                decorators = data.get("decorators", [])
                dependencies = data.get("dependencies", [])
                parent = data.get("parent")

                # Build the response
                response_parts = [
                    f"Method: {method_name} ({method_type})",
                    f"Location: Lines {start_line}-{end_line}"
                ]

                if parent:
                    response_parts.append(f"Parent: {parent}")

                if decorators:
                    response_parts.append(f"Decorators: {', '.join(decorators)}")

                if docstring:
                    response_parts.append(f"\nDocstring:\n{docstring}")

                if dependencies:
                    response_parts.append(f"\nDependencies: {', '.join(dependencies)}")

                response_parts.append(f"\nCode:\n```python\n{content}\n```")

                return "\n".join(response_parts)

            elif "complete_file_content" in data:  # CodebaseAction.MODIFY_METHOD results
                method_name = data.get("method_name")
                filename = data.get("filename") or data.get("filepath", "unknown")
                message = data.get("message", "")
                complete_file_content = data.get("complete_file_content")

                # Build response
                response_parts = [
                    f"✅ {message}",
                    f"File: {filename}",
                    f"Method: {method_name}",
                    f"\n📋 Complete file content (ready to copy):",
                    f"```python\n{complete_file_content}\n```"
                ]

                return "\n".join(response_parts)

            elif "error" in data:  # Error in data
                return f"Error: {data['error']}"

            # Default dict formatting
            return json.dumps(data, indent=2, default=str)

        elif isinstance(data, list):
            if len(data) == 0:
                return "No results found"
            elif len(data) == 1:
                return f"Found: {data[0]}"
            else:
                return f"Found {len(data)} results:\n" + "\n".join(f"  - {item}" for item in data[:10])

        else:
            return str(data)

    def get_all_code_files(self) -> List[str]:
        """
        Get all code files in the project.

        Returns:
            List of relative file paths (limited to 100)
        """
        try:
            code_files = []
            extensions = DocumentLoaderFactory.get_code_extensions()

            for ext in extensions:
                for filepath in self.root_dir.rglob(f"*{ext}"):
                    if self.is_ignored(filepath):
                        continue

                    relative_path = filepath.relative_to(self.root_dir)
                    code_files.append(str(relative_path))

            logger.info(f"Found {len(code_files)} code files in project")
            return code_files[:100]  # Limit to 100 files
        except Exception as e:
            logger.error(f"Error getting code files: {e}")
            return []

