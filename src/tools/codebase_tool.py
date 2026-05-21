# =============================================================================
# src/tools/codebase_tool.py
# Professional Codebase Analysis Tool - Core Implementation
# =============================================================================
"""
Herramienta de análisis de código base.

Sirve como punto de entrada principal para el análisis estático.
Integra componentes para proveer una API completa de análisis de código.
"""

import ast
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from uuid import UUID

from src.tools.codebase_tool_deps.code_smells import CodeSmellDetector
from src.tools.codebase_tool_deps.llm_formatter import LLMFormatter
from src.tools.codebase_tool_deps.metrics import MetricsCalculator
from src.tools.codebase_tool_deps.models import (
    AnalysisResult,
    FileAnalysisResult,
    SymbolInfo,
    CodeLocation
)
from src.tools.codebase_tool_deps.refactoring import RefactoringSuggester
from src.tools.codebase_tool_deps.sarif import SarifGenerator
from src.tools.codebase_tool_deps.security import SecurityAnalyzer
from src.tools.base_tool import BaseTool, ToolCategory, ToolParameter, ToolResult, ExecutionContext
from src.config.constants import IGNORED_DIRS, IGNORED_EXTENSIONS
from src.config.intent_patterns import CodebaseAction
from src.document_loaders import DocumentLoaderFactory
from src.services.code_graph_builder import CodeGraphBuilder
from src.services.analysis.codebase_analyzer import (
    CodebaseAnalyzer,
    CodeSymbol,
    LANGUAGE_BY_EXTENSION
)

if TYPE_CHECKING:
    from src.repositories.file_repository import FileRepository

logger = logging.getLogger(__name__)


class CodebaseTool(BaseTool):
    """
    Herramienta profesional de análisis de código base.

    Ofrece:
    - Métricas de calidad de código avanzadas
    - Detección de "code smells"
    - Escaneo de vulnerabilidades de seguridad
    - Sugerencias de refactorización
    - Soporte multilingüe
    - Generación de reportes SARIF
    - Salidas optimizadas para LLM
    """

    def __init__(self):
        """
        Initialize CodebaseTool.
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

        # Dependencies (injected by chat_orchestrator)
        self.file_repo: Optional['FileRepository'] = None
        self.file_service: Optional[Any] = None  # FileService — for safe file creation

        # Lazy load RAG tool
        self._rag_tool = None
        self._graph_builder = None
        self._analysis_cache: Dict[str, Any] = {}

        logger.info(f"CodebaseTool v2.2.0 initialized with root: {self.root_dir}")

    # Atributos

    @property
    def graph_builder(self) -> CodeGraphBuilder:
        if not hasattr(self, '_graph_builder') or self._graph_builder is None:
            self._graph_builder = CodeGraphBuilder(str(self.root_dir))
        return self._graph_builder

    @property
    def project_graph(self) -> Any:
        if not hasattr(self, '_project_graph') or self._project_graph is None:
            logger.info("Initializing project dependency graph...")
            self._project_graph = self.graph_builder.build_graph()
        return self._project_graph

    @property
    def rag_tool(self):
        """Lazy load RAG tool."""
        if self._rag_tool is None:
            from .rag_tool import RAGTool
            self._rag_tool = RAGTool()
        return self._rag_tool

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

    @property
    def file_dependent_actions(self) -> set:
        """Acciones que requieren archivos para ejecutarse."""
        return {
            "analyze_file", "basic_analyze_file", "analyze_quality",
            "explain", "get_method_content", "get_class_content",
            "modify_method", "export_refactored",
            "find_symbol", "find_class", "find_method"
        }

    @property
    def required_dependencies(self) -> list:
        """Dependencias de infraestructura que esta tool necesita."""
        return ["file_repo"]

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
                enum=["detailed", "summary", "compact", "markdown"]
            ),
            ToolParameter(
                name="sub_action",
                type="string",
                description="Granular hint intended for BASIC_ANALYZE_FILE (e.g. count_methods, list_methods)",
                required=False
            ),
            ToolParameter(
                name="method_name",
                type="string",
                description="Name of the method to replace or export",
                required=False
            ),
            ToolParameter(
                name="new_content",
                type="string",
                description="New content to replace the method/class with",
                required=False
            ),
            ToolParameter(
                name="conversation_id",
                type="string",
                description="Conversation ID for linking exported files",
                required=False
            ),
            ToolParameter(
                name="caller_updates",
                type="object",
                description="Mapping of caller method signatures to their rewrites from the LLM",
                required=False
            )
        ]

    # Método Principal de Ejecución

    async def execute(self, **kwargs) -> ToolResult:
        """
        Ejecuta la acción de análisis de código.
        Actúa como envoltura para registrar el resultado final de la ejecución interna.
        """
        try:
            # Invocar la lógica interna
            result = await self._execute_internal(**kwargs)

            action = kwargs.get("action", "unknown_action")
            target = kwargs.get("target", "None")

            # Formatear la salida para el log
            if result.success:
                data_str = str(result.data)

                logger.info(
                    f"\n{'='*60}\n"
                    f" CODEBASE TOOL RESULT (SUCCESS) \n"
                    f" Action: {action} | Target: {target}\n"
                    f"{'-'*60}\n"
                    f"{data_str}\n"
                    f"{'='*60}"
                )
            else:
                logger.error(
                    f"\n{'='*60}\n"
                    f" CODEBASE TOOL RESULT (ERROR) \n"
                    f" Action: {action} | Target: {target}\n"
                    f"{'-'*60}\n"
                    f"{result.error}\n"
                    f"{'='*60}"
                )

            return result

        except Exception as e:
            logger.error(f"Unhandled error executing CodebaseTool wrapper: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))

    async def _execute_internal(self, **kwargs) -> ToolResult:
        """
        Lógica de ejecución interna para la herramienta de código.

        Args:
            **kwargs: Parámetros de la acción a ejecutar.

        Returns:
            ToolResult con los resultados del análisis.
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

            logger.info(
                f"Executing action '{action}' with {len(file_ids)} file_ids "
                f"and {len(context_files)} context_files"
            )

            # Tratar "markdown" como "detailed" (ya es markdown por defecto)
            if format_type == "markdown":
                format_type = "detailed"

            # Acciones globales (independientes de archivos)
            if action == CodebaseAction.REFRESH_GRAPH:
                logger.info("Force refreshing project dependency graph...")
                self._project_graph = self.graph_builder.build_graph()
                return ToolResult(
                    success=True,
                    data="Project dependency graph has been successfully rebuilt and refreshed."
                )

            # Prioridad secundaria: archivos de contexto de RAG
            if context_files and not file_ids:
                analysis_result = await self.analyze_context_files(
                    context_files, action, target
                )
                # Note: rewrite not supported for context_files yet (would need file resolution)
                return self._format_action_result(
                    analysis_result, action, target, format_type, sub_action
                )

            # Prioridad principal: archivos subidos por el usuario
            if file_ids:
                if action == CodebaseAction.MODIFY_METHOD:
                    return await self.modify_method(
                        file_ids=file_ids,
                        method_name=target,
                        new_content=kwargs.get('new_content'),
                    )

                if action == CodebaseAction.EXPORT_REFACTORED:
                    # Prefer explicit method_name kwarg (enriched from history),
                    # fall back to target (extracted by intent router).
                    resolved_method = kwargs.get('method_name') or target
                    return await self.export_refactored_file(
                        file_ids=file_ids,
                        method_name=resolved_method,
                        new_content=kwargs.get('new_content'),
                        conversation_id=kwargs.get('conversation_id'),
                    )

                analysis_result = await self.analyze_uploaded_files(
                    file_ids, action, target
                )
                return self._format_action_result(
                    analysis_result, action, target, format_type, sub_action
                )

            # Análisis alternativo sobre el sistema de archivos local
            if not file_ids and not context_files:
                logger.info(
                    f"No file_ids provided. Falling back to local filesystem "
                    f"analysis for action '{action}' with target '{target}'"
                )

                # Verificar target requerido
                needs_target = [
                    CodebaseAction.FIND_DEFINITION, CodebaseAction.FIND_REFERENCES,
                    CodebaseAction.GET_CALLERS, CodebaseAction.GET_DEPENDENCIES,
                    CodebaseAction.ANALYZE_QUALITY, CodebaseAction.GET_METHOD_CONTENT,
                    CodebaseAction.MODIFY_METHOD, CodebaseAction.ANALYZE_FILE
                ]

                if action in needs_target and not target:
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"Parameter 'target' is required for local action '{action}'"
                    )

                if action == CodebaseAction.ANALYZE_FILE:
                    return self.analyze_file(target)
                elif action == CodebaseAction.FIND_DEFINITION:
                    return self.find_definition(target)
                elif action == CodebaseAction.FIND_REFERENCES:
                    return self.find_references(target)
                elif action == CodebaseAction.GET_CALLERS:
                    return self.get_callers(target)
                elif action == CodebaseAction.GET_DEPENDENCIES:
                    return self.get_dependencies(target)
                elif action == CodebaseAction.EXPLAIN:
                    content_res = self.get_method_content_filesystem(target)
                    if not content_res.success:
                        return content_res
                    return ToolResult(
                        success=True,
                        data=f"Please explain this code:\n\n{content_res.data}"
                    )
                elif action == CodebaseAction.GET_METHOD_CONTENT:
                    return self.get_method_content_filesystem(target)
                elif action == CodebaseAction.MODIFY_METHOD:
                    new_content = kwargs.get('new_content')
                    if not new_content:
                        return ToolResult(success=False, data=None, error="new_content required for modifying")
                    return self.modify_method_filesystem(target, new_content)
                elif action == CodebaseAction.ANALYZE_QUALITY:
                    return self.find_symbol_and_analyze_quality(target)
                elif action == CodebaseAction.MODIFY_METHOD_ALL_CALLERS:
                    new_content = kwargs.get('new_content')
                    if not new_content or not target:
                        return ToolResult(
                            success=False,
                            data=None,
                            error="new_content and target are required to propagate changes"
                        )
                    return await self.propagate_method_change(
                        target_method=target, 
                        new_content=new_content,
                        caller_updates=kwargs.get('caller_updates'),
                        conversation_id=kwargs.get('conversation_id')
                    )
                # Acción no soportada en filesystem
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Action '{action}' is not supported for local filesystem fallback."
                )
            # ------------- END FALLBACK -------------

        except Exception as e:
            logger.error(f"Error executing CodebaseTool logic: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))

    def _format_action_result(
        self,
        analysis_result: AnalysisResult,
        action: str,
        target: Optional[str],
        format_type: str,
        sub_action: Optional[str]
    ) -> ToolResult:
        """
        Centraliza el formateo del resultado según la acción solicitada.

        Extrae la lógica de formateo duplicada que existía en los paths
        context_files y file_ids de execute(). Único lugar donde se decide
        cómo presentar cada tipo de resultado al LLM.

        Args:
            analysis_result: Resultado del análisis (structural o comprehensive)
            action:          Acción ejecutada (CodebaseAction.*)
            target:          Símbolo objetivo (nombre del método/clase)
            format_type:     Formato de salida ("detailed", "summary", "compact")
            sub_action:      Hint granular del IntentRouter (count_methods, etc.)

        Returns:
            ToolResult con data formateada para el LLM
        """
        # SARIF export — retorna JSON crudo sin formatear
        if action == "generatesarif":
            sarif_json = self.sarif_generator.generate_sarif_json(analysis_result)
            return ToolResult(success=True, data=sarif_json)

        # Format for LLM — delega en LLMFormatter con formato configurable
        if action == "formatforllm":
            formatted = self.llm_formatter.format_analysis_result(
                analysis_result, format_type=format_type
            )
            return ToolResult(success=True, data=formatted)

        # Extraer contenido de método desde el AST.
        if action == CodebaseAction.GET_METHOD_CONTENT:
            if not analysis_result.results:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"No files analyzed for target '{target}'"
                )

            file_result = analysis_result.results[0]

            # Buscar el símbolo específico por nombre en la lista de symbols (AST)
            symbol = next(
                (s for s in (file_result.symbols or []) if s.name == target),
                None
            )

            if symbol and symbol.content:
                lang = file_result.language or "python"
                return ToolResult(
                    success=True,
                    data=f"```{lang}\n{symbol.content}\n```",
                    metadata={
                        "target": target,
                        "type": symbol.symbol_type,
                        "start_line": symbol.location.start_line if symbol.location else None,
                        "end_line": symbol.location.end_line if symbol.location else None,
                        "file": file_result.file_path
                    }
                )
            else:
                # Símbolo no encontrado — listar disponibles para debug
                available = [s.name for s in (file_result.symbols or [])]
                return ToolResult(
                    success=False,
                    data=None,
                    error=(
                        f"Symbol '{target}' not found in '{file_result.file_path}'. "
                        f"Available symbols: {available}"
                    )
                )

        # Extraer clase desde el AST.
        if action == CodebaseAction.GET_CLASS_CONTENT:
            if not analysis_result.results:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"No files analyzed for target '{target}'"
                )

            file_result = analysis_result.results[0]

            # Buscar la clase por nombre — type puede ser "class"
            symbol = next(
                (s for s in (file_result.symbols or [])
                 if s.name == target and s.symbol_type == "class"),
                None
            )

            if symbol and symbol.content:
                lang = file_result.language or "python"
                return ToolResult(
                    success=True,
                    data=f"```{lang}\n{symbol.content}\n```",
                    metadata={
                        "target": target,
                        "type": "class",
                        "start_line": symbol.location.start_line if symbol.location else None,
                        "end_line": symbol.location.end_line if symbol.location else None,
                        "file": file_result.file_path
                    }
                )
            else:
                available_classes = [
                    s.name for s in (file_result.symbols or [])
                    if s.symbol_type == "class"
                ]
                return ToolResult(
                    success=False,
                    data=None,
                    error=(
                        f"Class '{target}' not found in '{file_result.file_path}'. "
                        f"Available classes: {available_classes}"
                    )
                )

        # EXPLAIN: Generar prompt detallado basado en el contenido del símbolo de AST.
        if action == CodebaseAction.EXPLAIN:
            if not analysis_result.results:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"No files analyzed for target '{target}'"
                )

            file_result = analysis_result.results[0]

            if target:
                # Explicar símbolo específico
                symbol = next(
                    (s for s in (file_result.symbols or []) if s.name == target),
                    None
                )
                code_to_explain = symbol.content if symbol else file_result.content
            else:
                # Explicar el archivo completo
                code_to_explain = file_result.content

            lang = file_result.language or "python"
            return ToolResult(
                success=True,
                data=(
                    f"Por favor explica este código de `{file_result.file_path}`:\n\n"
                    f"```{lang}\n{code_to_explain}\n```"
                )
            )

        # Análisis estructural básico (count_methods, list_methods, etc.)
        if action == CodebaseAction.BASIC_ANALYZE_FILE:
            formatted = self.llm_formatter.format_analysis_result(
                analysis_result, format_type="basic", sub_action=sub_action
            )
            return ToolResult(success=True, data=formatted)

        # Análisis completo detallado
        if action == CodebaseAction.ANALYZE_FILE:
            formatted = self.llm_formatter.format_analysis_result(
                analysis_result, format_type="detailed"
            )
            return ToolResult(success=True, data=formatted)

        # Default: formatear el dict del resultado para acciones no mapeadas explícitamente
        formatted_output = self._format_analysis_result_dict(analysis_result.to_dict())
        return ToolResult(success=True, data=formatted_output)

    # Métodos de Análisis de Archivos

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

                    # Read file content from local storage
                    storage_path = Path(file_record.storage_path)
                    content = None

                    if storage_path.exists():
                        try:
                            content = storage_path.read_text(encoding='utf-8', errors='ignore')
                        except Exception as e:
                            logger.warning(f"Failed to read local file: {e}")

                    # Fallback a RAG si el archivo local no está disponible
                    if content is None:
                        collection_name = getattr(file_record, 'collection_name', 'documentation')
                        content = await self.rag_tool.get_full_document_content(
                            str(file_id),
                            collection_name
                        )

                    if not content:
                        logger.warning(f"Failed to get content for file: {file_id}")
                        continue

                    # Elegir tipo de análisis según la acción solicitada
                    if action == CodebaseAction.BASIC_ANALYZE_FILE:
                        file_result = await self._analyze_file_structural(
                            content, file_record.file_name, str(file_id)
                        )
                    elif action in (
                        CodebaseAction.GET_METHOD_CONTENT,
                        CodebaseAction.GET_CLASS_CONTENT,
                        CodebaseAction.EXPLAIN  # FIX 5: EXPLAIN también necesita structural primero
                    ):
                        # Análisis estructural para extracción de símbolos via AST
                        temp_result = await self._analyze_file_structural(
                            content, file_record.file_name, str(file_id)
                        )
                        # Filtrar al símbolo específico si se proporcionó target
                        file_result = self._find_symbol_content(temp_result, target)
                    else:
                        # Análisis comprensivo completo (métricas, smells, seguridad)
                        file_result = await self._analyze_file_comprehensive(
                            content, file_record.file_name, str(file_id)
                        )

                    file_results.append(file_result)

                except Exception as e:
                    logger.error(f"Error analyzing file {file_id}: {e}", exc_info=True)

            # Construir resultado agregado
            analysis_result = AnalysisResult(
                action=action,
                target=target,
                files_analyzed=len(file_results),
                results=file_results
            )
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

                # Buscar el archivo primero en root_dir, luego recursivamente
                file_path = self.root_dir / filename

                if not file_path.exists():
                    found = False
                    for root, dirs, files in os.walk(self.root_dir):
                        if filename in files:
                            file_path = Path(root) / filename
                            found = True
                            break

                    if not found:
                        logger.warning(f"File not found: {filename}")
                        continue

                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                except Exception as e:
                    logger.warning(f"Failed to read file {filename}: {e}")
                    continue

                # FIX 3: Misma lógica de selección de análisis que analyze_uploaded_files
                # Antes solo manejaba BASIC_ANALYZE_FILE y GET_METHOD_CONTENT, sin GET_CLASS_CONTENT ni EXPLAIN
                if action == CodebaseAction.BASIC_ANALYZE_FILE:
                    file_result = await self._analyze_file_structural(
                        content, filename, ""  # No file_id para context files
                    )
                elif action in (
                    CodebaseAction.GET_METHOD_CONTENT,
                    CodebaseAction.GET_CLASS_CONTENT,
                    CodebaseAction.EXPLAIN  # FIX 5: EXPLAIN necesita structural
                ):
                    temp_result = await self._analyze_file_structural(
                        content, filename, ""
                    )
                    file_result = self._find_symbol_content(temp_result, target)
                else:
                    file_result = await self._analyze_file_comprehensive(
                        content, filename, ""
                    )

                file_results.append(file_result)

            # Construir resultado agregado
            analysis_result = AnalysisResult(
                action=action,
                target=target,
                files_analyzed=len(file_results),
                results=file_results
            )
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

        Suitable for: BASIC_ANALYZE_FILE, GET_METHOD_CONTENT, GET_CLASS_CONTENT.
        No ejecuta metrics_calculator, code_smell_detector ni security_analyzer
        para mantener la latencia baja en queries de estructura.

        Args:
            content: File content
            filename: File name
            file_id: File UUID

        Returns:
            FileAnalysisResult with only structural information
        """
        # Detectar lenguaje por extensión
        ext = Path(filename).suffix.lower()
        language = LANGUAGE_BY_EXTENSION.get(ext, 'unknown')

        # Análisis estructural via AST (Python) o regex (otros lenguajes)
        structure_analysis = self.analyzer.analyze_file(content, filename)

        # Extraer SymbolInfo con content incluido (del AST)
        symbols = self._extract_symbols(structure_analysis, content)

        # Listas planas de nombres para el formatter
        classes = [s.name for s in structure_analysis.get("symbols", []) if s.type == "class"]
        functions = [
            s.name for s in structure_analysis.get("symbols", [])
            if s.type in ("function", "method")
        ]
        imports = structure_analysis.get("imports", [])

        return FileAnalysisResult(
            file_id=file_id,
            file_path=filename,
            language=language,
            symbols=symbols,
            classes=classes,
            functions=functions,
            imports=imports,
            content=content  # Necesario como fallback en EXPLAIN sin target
        )

    async def _analyze_file_comprehensive(
        self,
        content: str,
        filename: str,
        file_id: str
    ) -> FileAnalysisResult:
        """
        Perform comprehensive analysis of a single file.

        Ejecuta el pipeline completo: estructura + métricas + code smells +
        seguridad + sugerencias de refactoring.

        Args:
            content: File content
            filename: File name
            file_id: File UUID

        Returns:
            FileAnalysisResult with complete analysis
        """
        # Detectar lenguaje por extensión
        ext = Path(filename).suffix.lower()
        language = LANGUAGE_BY_EXTENSION.get(ext, 'unknown')

        # Análisis estructural
        structure_analysis = self.analyzer.analyze_file(content, filename)
        symbols = self._extract_symbols(structure_analysis, content)

        # Métricas de calidad
        metrics = self.metrics_calculator.calculate_all_metrics(content, language)

        # Detección de code smells
        code_smells = self.code_smell_detector.detect_all_smells(
            content, filename, language, symbols
        )

        # Análisis de seguridad
        security_issues = self.security_analyzer.detect_all_vulnerabilities(
            content, filename, language
        )

        # Sugerencias de refactoring
        refactoring_suggestions = self.refactoring_suggester.generate_all_suggestions(
            content, filename, language, symbols, code_smells, security_issues
        )

        # Listas planas de nombres para el formatter
        classes = [s.name for s in structure_analysis.get("symbols", []) if s.type == "class"]
        functions = [
            s.name for s in structure_analysis.get("symbols", [])
            if s.type in ("function", "method")
        ]
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
            List of SymbolInfo objects con content del AST incluido
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
                content=sym.content  # Contenido exacto del símbolo (AST get_source_segment)
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

        Retorna un FileAnalysisResult reducido donde content es el código del
        símbolo específico, no el archivo completo. Esto evita truncación en
        el LLMFormatter cuando se pide un método largo en un archivo grande.

        Args:
            analysis_result: Full file analysis result
            target_name: Name of symbol to find (case-insensitive)

        Returns:
            FileAnalysisResult con solo el símbolo objetivo, o el resultado
            original si target_name es None.
        """
        if not target_name:
            return analysis_result

        target_lower = target_name.lower()

        # Búsqueda case-insensitive en la lista de símbolos
        found_symbol = next(
            (s for s in analysis_result.symbols if s.name.lower() == target_lower),
            None
        )

        if found_symbol:
            # Resultado reducido: content = solo el código del símbolo
            return FileAnalysisResult(
                file_id=analysis_result.file_id,
                file_path=analysis_result.file_path,
                language=analysis_result.language,
                symbols=[found_symbol],
                classes=[],   # Irrelevante para vista de contenido
                functions=[],
                imports=[],
                content=found_symbol.content  # ← AST content, no archivo completo
            )

        # Símbolo no encontrado — result vacío con mensaje descriptivo
        logger.warning(
            f"Symbol '{target_name}' not found in '{analysis_result.file_path}'. "
            f"Available: {[s.name for s in analysis_result.symbols]}"
        )
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

        # Handle formatted LLM output (markdown)
        if isinstance(data, str) and data.startswith('#'):
            return data

        # Handle analysis result dict
        if isinstance(data, dict):
            if "files_analyzed" in data:
                return self._format_analysis_result_dict(data)

        # Default formatting
        return str(data)

    def _format_analysis_result_dict(self, data: Dict[str, Any]) -> str:
        """Format analysis result dictionary for LLM consumption."""
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
                lines.append(
                    f"**Maintainability Index:** "
                    f"{maintainability.get('maintainability_index', 0):.1f}"
                )

            code_smells = result.get('code_smells', [])
            security_issues = result.get('security_issues', [])
            lines.append(
                f"**Issues:** {len(code_smells)} code smells, "
                f"{len(security_issues)} security issues"
            )
            lines.append("")

        if len(results) > 3:
            lines.append(f"... and {len(results) - 3} more files")

        return "\n".join(lines)

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
                    error=(
                        "Target must be in format 'filepath:methodname'. "
                        "Example: 'src/services/chat_orchestrator.py:process_message'"
                    )
                )

            filepath_str, method_name = target.split(":", 1)
            filepath = self.root_dir / filepath_str

            logger.info(f"Getting method content for '{method_name}' from file: {filepath}")

            # Búsqueda recursiva si el path exacto no existe
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
            except Exception as e:
                logger.error(f"Failed to read file {filepath}: {e}")
                return ToolResult(success=False, data=None, error=f"Failed to read file: {e}")

            # Análisis AST para extraer el símbolo
            analysis = self.analyzer.analyze_file(content, filepath.name)

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

            # Método no encontrado
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
        new_content: Optional[str] = None,
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
                    error=(
                        f"method_name and new_content parameters are required "
                        f"for {CodebaseAction.MODIFY_METHOD} action"
                    )
                )

            logger.info(f"Modifying method '{method_name}' in {len(file_ids)} files")

            # Buscar el método en todos los archivos proporcionados
            for file_id_str in file_ids:
                file_id = file_id_str

                # Verificar cache primero para evitar relectura
                if file_id in self._analysis_cache:
                    logger.info(f"Using cached analysis for file: {file_id}")
                    cached_data = self._analysis_cache[file_id]
                    content = cached_data.get("content")
                    analysis = cached_data.get("analysis")
                    filename = cached_data.get("filename")
                else:
                    logger.info(f"Loading and analyzing file: {file_id}")
                    file_record = await self.file_repo.get_by_id(UUID(file_id))
                    if not file_record:
                        logger.warning(f"File not found in database: {file_id}")
                        continue

                    storage_path = Path(file_record.storage_path)
                    content = None

                    if storage_path.exists():
                        try:
                            content = storage_path.read_text(encoding='utf-8', errors='ignore')
                        except Exception as e:
                            logger.warning(f"Failed to read local file {file_id}: {e}")

                    # Fallback a RAG
                    if content is None:
                        collection_name = getattr(file_record, 'collection_name', 'documentation')
                        content = await self.rag_tool.get_full_document_content(
                            str(file_id), collection_name
                        )

                    if not content:
                        logger.warning(f"Failed to get content for file: {file_id}")
                        continue

                    analysis = self.analyzer.analyze_file(content, file_record.file_name)
                    filename = file_record.file_name

                    # Cachear para reutilización en la misma sesión
                    self._analysis_cache[file_id] = {
                        "content": content,
                        "analysis": analysis,
                        "filename": filename
                    }

                # Buscar el método y reemplazar su contenido
                for sym in analysis.get("symbols", []):
                    if sym.name == method_name:
                        logger.info(f"Found method '{method_name}' in file '{filename}'")

                        lines = content.split('\n')

                        # Reemplazar contenido (ajuste a índice 0-based)
                        start_idx = sym.start_line - 1
                        end_idx = sym.end_line

                        new_lines = lines[:start_idx] + [new_content] + lines[end_idx:]
                        new_file_content = '\n'.join(new_lines)

                        # Actualizar cache con el nuevo contenido
                        self._analysis_cache[file_id]["content"] = new_file_content

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
                                "message": (
                                    f"Method '{method_name}' has been modified in memory. "
                                    f"Use 'export_refactored' action to generate a downloadable versioned file."
                                )
                            }
                        )

            # Método no encontrado en ningún archivo
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
                    error=(
                        "Target must be in format 'filepath:methodname'. "
                        "Example: 'src/services/chat_orchestrator.py:process_message'"
                    )
                )

            filepath_str, method_name = target.split(":", 1)
            filepath = self.root_dir / filepath_str

            logger.info(f"Modifying method '{method_name}' in file: {filepath}")

            # Búsqueda recursiva si el path exacto no existe
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
            except Exception as e:
                logger.error(f"Failed to read file {filepath}: {e}")
                return ToolResult(success=False, data=None, error=f"Failed to read file: {e}")

            # Análisis AST
            analysis = self.analyzer.analyze_file(content, filepath.name)

            for sym in analysis.get("symbols", []):
                if sym.name == method_name:
                    logger.info(f"Found method '{method_name}' in file '{filepath}'")

                    lines = content.split('\n')

                    # Reemplazar contenido (ajuste a índice 0-based)
                    start_idx = sym.start_line - 1
                    end_idx = sym.end_line

                    new_lines = lines[:start_idx] + [new_content] + lines[end_idx:]
                    new_file_content = '\n'.join(new_lines)

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
                            "message": (
                                f"Method '{method_name}' has been modified in memory. "
                                f"Use 'export_refactored' action to generate a downloadable versioned file."
                            )
                        }
                    )

            # Método no encontrado
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

        files_to_search = context_files if context_files else self.get_all_code_files()
        logger.info(f"Searching for symbol in {len(files_to_search)} files")

        target_filepath = None
        target_content = None

        for filepath in files_to_search:
            filepath = self.root_dir / filepath
            if not filepath.exists():
                continue

            try:
                content = filepath.read_text(encoding='utf-8', errors='ignore')
            except Exception as e:
                logger.warning(f"Failed to read {filepath}: {e}")
                continue

            # FIX 6: Añadir "async def {symbol}" — antes solo buscaba "def {symbol}"
            # Los métodos async no se detectaban, devolviendo falso negativo.
            if (
                f"def {symbol}" in content
                or f"async def {symbol}" in content
            ):
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

        logger.info(f"Analyzing file: {target_filepath}")
        analysis = self.analyzer.analyze_file(target_content, target_filepath.name)

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
        target: str
    ) -> ToolResult:
        """
        Find specific symbol in filesystem and analyze its quality.

        FIX 4: Firma original era (content, target) — pero el caller en execute()
        lo invocaba con un solo argumento (target). Corregido para aceptar solo
        target y resolver content internamente buscando en el filesystem.

        Args:
            target: Symbol name to find (busca en todos los archivos del proyecto)

        Returns:
            ToolResult con quality analysis o error
        """
        logger.info(f"find_symbol_and_analyze_quality: searching for '{target}'")

        # Buscar el símbolo en los archivos del proyecto
        code_files = self.get_all_code_files()

        for filepath_str in code_files:
            filepath = self.root_dir / filepath_str
            if not filepath.exists():
                continue

            try:
                content = filepath.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                continue

            # Verificación rápida antes de parsear el AST completo
            if f"def {target}" not in content and f"async def {target}" not in content:
                continue

            analysis = self.analyzer.analyze_file(content, filepath.name)

            target_symbol = next(
                (sym for sym in analysis.get("symbols", []) if sym.name == target),
                None
            )

            if not target_symbol:
                continue

            return ToolResult(
                success=True,
                data={
                    "symbol": target_symbol.name,
                    "file": filepath_str,
                    "type": target_symbol.type,
                    "complexity": self.calculate_complexity(target_symbol),
                    "issues": self.detect_code_smells(target_symbol),
                    "suggestions": self.generate_suggestions(target_symbol)
                }
            )

        return ToolResult(
            success=False,
            data=None,
            error=f"Symbol '{target}' not found in {len(code_files)} project files"
        )

    # =========================================================================
    # FILESYSTEM ANALYSIS (Priority 3)
    # =========================================================================

    def analyze_file(self, filepath_str: str) -> ToolResult:
        """
        Analiza un archivo específico usando AST (Python) o Regex (otros).

        Args:
            filepath_str: Ruta relativa o absoluta del archivo.

        Returns:
            ToolResult con el análisis del archivo.
        """
        filepath = self.root_dir / filepath_str
        logger.info(f"Analyzing file: {filepath}")

        # Búsqueda recursiva si el path exacto no existe
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

                # Agregar dependencias únicas de todos los símbolos
                all_deps = set()
                for sym in analysis.get("symbols", []):
                    all_deps.update(sym.dependencies)
                summary["dependencies_detected"] = list(all_deps)

                logger.info(f"File analysis completed for {filepath}")
                return ToolResult(success=True, data=summary)
            else:
                # Generic analysis para otros lenguajes
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
                    "note": (
                        f"Structural analysis via CodebaseAnalyzer "
                        f"(Language: {filepath.suffix})"
                    ),
                    "content": content  # Return full content for analysis
                }

                logger.info(f"File analysis completed for {filepath}\n\n{summary}")
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
        Escanea archivos para encontrar dónde se define una clase o función.

        Args:
            symbol: Nombre del símbolo a buscar.

        Returns:
            ToolResult con las rutas de archivos que contienen la definición.
        """
        logger.info(f"Finding definition for symbol '{symbol}'")
        matches = []

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
                        if (
                            f"class {symbol}" in content
                            or f"function {symbol}" in content
                            or f"interface {symbol}" in content
                        ):
                            is_match = True
                    else:
                        # Generic check
                        if f"class {symbol}" in content or f"function {symbol}" in content:
                            is_match = True

                    if is_match:
                        matches.append(str(file_p))
                except Exception:
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
        Encuentra ocurrencias de texto de un símbolo (referencias).

        Args:
            symbol: Nombre del símbolo a buscar.

        Returns:
            ToolResult con los archivos que contienen referencias.
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
                except Exception:
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

    # Utilidades de Calidad de Código

    def calculate_complexity(self, symbol: CodeSymbol) -> int:
        """
        Calcula la complejidad ciclomática aproximada.

        Args:
            symbol: CodeSymbol a analizar.

        Returns:
            Puntaje de complejidad.
        """
        code = symbol.content
        complexity = 1  # Base complexity

        # Contar estructuras de control
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
        Detecta patrones problemáticos en el código.

        Args:
            symbol: CodeSymbol a analizar.

        Returns:
            Lista de problemas detectados.
        """
        issues = []
        code = symbol.content

        # Método demasiado largo
        line_count = code.count("\n")
        if line_count > 50:
            issues.append(f"Método demasiado largo ({line_count} líneas, recomendado < 50)")

        # Demasiados parámetros
        param_match = re.search(r"def .*?\((.*?)\)", code)
        if param_match:
            params = [p.strip() for p in param_match.group(1).split(",") if p.strip()]
            if len(params) > 5:
                issues.append(f"Demasiados parámetros ({len(params)}, recomendado < 5)")

        # Sin docstring
        if not symbol.docstring:
            issues.append("Falta docstring")

        # Alto acoplamiento
        if len(symbol.dependencies) > 8:
            issues.append(f"Alto acoplamiento ({len(symbol.dependencies)} dependencias)")

        # Except demasiado amplio
        if "except:" in code or "except Exception" in code:
            issues.append("Cláusula except vacía detectada (captura todas las excepciones)")

        return issues

    def generate_suggestions(self, symbol: CodeSymbol) -> List[str]:
        """
        Genera sugerencias de mejora.

        Args:
            symbol: CodeSymbol a analizar.

        Returns:
            Lista de sugerencias.
        """
        suggestions = []
        code = symbol.content

        if not symbol.docstring:
            suggestions.append("Añadir docstring con parámetros, retornos y excepciones.")

        if self.calculate_complexity(symbol) > 10:
            suggestions.append(
                "Considere refactorizar en funciones más pequeñas (complejidad > 10)."
            )

        if "TODO" in code or "FIXME" in code:
            suggestions.append("Atender comentarios TODO/FIXME.")

        if symbol.type == "method" and "self" in code:
            if not re.search(r"def __init__", code):
                suggestions.append(
                    "Considere si el método debería ser estático o de clase."
                )

        return suggestions

    # Métodos de Utilidad

    def get_callers(self, target: str) -> ToolResult:
        """
        Obtiene todos los llamadores de un símbolo vía el grafo de dependencias.

        Args:
            target: Nombre del símbolo para buscar llamadores.

        Returns:
            ToolResult con la lista de llamadores (limitada a 50).
        """
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

    def get_dependencies(self, target: str) -> ToolResult:
        """
        Obtiene todas las dependencias de un símbolo vía el grafo de dependencias.

        Args:
            target: Nombre del símbolo para buscar dependencias.

        Returns:
            ToolResult con la lista de dependencias (limitada a 50).
        """
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

    def get_all_code_files(self) -> List[str]:
        """
        Obtiene todos los archivos de código del proyecto.

        Returns:
            Lista de rutas relativas de archivos (limitada a 100).
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

    async def propagate_method_change(
        self, 
        target_method: str, 
        new_content: str, 
        caller_updates: Optional[Dict[str, str]] = None,
        conversation_id: Optional[str] = None
    ) -> ToolResult:
        """
        Propaga cambios a un método y a todos sus llamadores.
        
        Fase 1 (Sin caller_updates): Valida la sintaxis del método objetivo, consulta el grafo
        para encontrar llamadores, obtiene su código fuente y lo retorna para su edición.
        
        Fase 2 (Con caller_updates): Usa el mapeo proporcionado para reemplazar los bloques AST
        de los llamadores y guardarlos de forma segura como borradores versionados.
        """
        try:
            # Validate target method syntax
            target_syntax_valid = self.validate_syntax(new_content, "target.py")
            if not target_syntax_valid.success:
                return ToolResult(success=False, data=None, error=f"Syntax Error in new target method: {target_syntax_valid.error}")

            # Get Callers from graph directly
            callers = self.graph_builder.get_callers(target_method)
            
            # Phase 1: Provide dry-run context for LLM
            if not caller_updates:
                caller_context = {}
                for caller_path, caller_method in callers[:10]: # Limit for context window
                    caller_file = caller_path.split(":")[0]
                    content_res = self.get_method_content_filesystem(f"{caller_file}:{caller_method}")
                    if content_res.success:
                        caller_context[f"{caller_file}:{caller_method}"] = content_res.data

                return ToolResult(
                    success=True,
                    data={
                        "message": "Target method is valid. Here are the callers' current source code. Please provide 'caller_updates' mapping each 'file:method' to its new rewritten python code.",
                        "target_method": target_method,
                        "callers_to_update": caller_context,
                        "total_callers": len(callers)
                    }
                )

            # Phase 2: Execute safe AST replacement
            modified_files = []
            failed_updates = []
            
            # Apply update to the target method itself (using modify_method_filesystem logic)
            # Find all parts for target_method (We don't know the exact filepath here, but we can query it)
            logger.info(f"Applying updates for {target_method} and its callers")
            
            # Since CodebaseTool does not have an easy unified way to export drafts by `target` (it uses file_ids),
            # we will iterate through caller_updates map.
            # caller_updates is dict like: {"src/file.py:caller1": "def caller1()..."}
            
            for identifier, rewritten_code in caller_updates.items():
                if ":" not in identifier:
                    failed_updates.append({identifier: "Invalid format. Expected 'filepath:method'"})
                    continue
                
                filepath_str, meth_name = identifier.split(":", 1)
                
                # Validate syntax of updated caller
                caller_syntax_valid = self.validate_syntax(rewritten_code, filepath_str)
                if not caller_syntax_valid.success:
                    failed_updates.append({identifier: f"Syntax error: {caller_syntax_valid.error}"})
                    continue
                
                # Use modify_method_filesystem to get the integrated content
                update_result = self.modify_method_filesystem(identifier, rewritten_code)
                if not update_result.success:
                    failed_updates.append({identifier: f"Modification failed: {update_result.error}"})
                    continue
                
                # If we have file_repo, we can save it explicitly as a safe draft
                if self.file_repo:
                    # Find original file in DB by path to generate draft
                    file_records = await self.file_repo.get_files(storage_path=update_result.data["filepath"])
                    if file_records:
                        import uuid
                        import datetime
                        from src.services.file.file_repository import FileRecord
                        original_record = file_records[0]
                        orig_path = Path(original_record.storage_path)
                        draft_id = uuid.uuid4()
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        draft_name = f"{orig_path.stem}_refactored_{timestamp}{orig_path.suffix}"
                        draft_path = orig_path.parent / draft_name
                        
                        # Save physically
                        draft_path.write_text(update_result.data["complete_file_content"], encoding='utf-8')
                        
                        # Register in DB
                        new_record = FileRecord(
                            id=draft_id,
                            user_id=original_record.user_id,
                            # Use conversation id from kwarg, OR from the original file.  Need fallback logic.
                            conversation_id=uuid.UUID(conversation_id) if conversation_id else original_record.conversation_id,
                            file_name=draft_name,
                            file_hash=f"{original_record.file_hash}_refactored",
                            storage_path=str(draft_path),
                            mime_type=original_record.mime_type,
                            file_size=len(update_result.data["complete_file_content"].encode('utf-8')),
                            processed=False,
                            is_draft=True
                        )
                        await self.file_repo.create(new_record)
                        modified_files.append({"id": str(draft_id), "name": draft_name, "identifier": identifier})
                    else:
                        # Could not find in DB, just track success (in-memory mode for tests)
                        modified_files.append({"identifier": identifier, "status": "In-memory success (no DB record found)"})
                else:
                    modified_files.append({"identifier": identifier, "status": "In-memory success (no file_repo)"})

            return ToolResult(
                success=True,
                data={
                    "message": (
                        "Propagación completada. Archivos de borrador generados.\n"
                        "INSTRUCCIÓN PARA EL LLM: Los archivos han sido entregados al usuario. "
                        "No repita el código generado. Brinde un resumen breve de los llamadores actualizados."
                    ),
                    "modified_drafts": modified_files,
                    "failed_updates": failed_updates
                }
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))

    # Persistencia y Validación

    def validate_syntax(self, content: str, filename: str) -> ToolResult:
        """
        Valida la sintaxis del código fuente antes de guardar.
        
        Args:
            content: Contenido del código a validar.
            filename: Nombre del archivo para determinar el lenguaje.
            
        Returns:
            ToolResult indicando éxito o error detallado de sintaxis.
        """
        ext = Path(filename).suffix.lower()
        if ext == ".py":
            try:
                ast.parse(content)
                logger.info(f"Syntax validation SUCCESS for {filename}")
                return ToolResult(success=True, data="Syntax is valid.")
            except SyntaxError as e:
                logger.error(f"Syntax validation FAILED for {filename}: {e}")
                return ToolResult(
                    success=False,
                    error=f"SyntaxError detected: {e.msg} at line {e.lineno}",
                    data={"line": e.lineno, "offset": e.offset, "text": e.text}
                )
            except Exception as e:
                return ToolResult(success=False, data=None, error=f"Validation error: {e}")
        
        # Default for other languages (no deep validation yet)
        return ToolResult(success=True, data="Language not supported for deep validation; skipping.")

    async def _write_back_to_storage(self, file_id: str, content: str) -> ToolResult:
        """
        [DEPRECADO] Método mantenido por compatibilidad.
        Use export_refactored_file() para crear copias seguras versionadas.
        """
        logger.warning("_write_back_to_storage is deprecated. Use export_refactored_file() instead.")
        return ToolResult(success=False, data=None, error="Use export_refactored action instead.")

    async def export_refactored_file(
        self,
        file_ids: List[str],
        method_name: Optional[str],
        new_content: Optional[str],
        conversation_id: Optional[str] = None
    ) -> ToolResult:
        """
        Genera un archivo refactorizado seguro y versionado a partir de un archivo subido.

        Decisiones de diseño:
        - NUNCA muta el archivo original (seguridad).
        - Guarda un NUEVO archivo en disco con un nombre con marca de tiempo.
        - Lo registra en la BD con processed=False e is_draft=True.
        - NO dispara la indexación de Qdrant (los borradores no son la fuente de verdad).
        - Retorna un download_url que apunta al endpoint de descarga existente.

        Args:
            file_ids: UUIDs de archivos subidos para buscar el método objetivo.
            method_name: Nombre del método a reemplazar.
            new_content: Código de reemplazo (versión corregida).
            conversation_id: Conversación opcional para vincular el nuevo archivo.

        Returns:
            ToolResult con file_id, file_name y download_url.
        """
        import datetime

        if not self.file_repo:
            return ToolResult(success=False, data=None, error="file_repo not available for CodebaseTool")

        if not method_name or not new_content:
            return ToolResult(
                success=False,
                data=None,
                error="method_name and new_content are required for export_refactored action"
            )

        logger.info(f"export_refactored_file: searching for '{method_name}' in {len(file_ids)} files")

        for file_id_str in file_ids:
            try:
                file_id = UUID(file_id_str)
                file_record = await self.file_repo.get_by_id(file_id)
                if not file_record:
                    logger.warning(f"File not found: {file_id_str}")
                    continue

                # --- READ ORIGINAL CONTENT ---
                storage_path = Path(file_record.storage_path)
                original_content = None

                if storage_path.exists():
                    try:
                        original_content = storage_path.read_text(encoding='utf-8', errors='ignore')
                    except Exception as e:
                        logger.warning(f"Failed to read local file: {e}")

                if original_content is None:
                    # Fallback to RAG
                    collection_name = getattr(file_record, 'collection_name', 'documentation')
                    original_content = await self.rag_tool.get_full_document_content(
                        str(file_id), collection_name
                    )

                if not original_content:
                    logger.warning(f"Could not get content for {file_id_str}")
                    continue

                # --- FIND AND REPLACE METHOD VIA AST ---
                analysis = self.analyzer.analyze_file(original_content, file_record.file_name)
                
                # method_name might be a comma-separated list of potential names 
                # (e.g. "ExecutionStats,__init__,to_dict") extracted by _enrich_export_refactored_params
                potential_names = [n.strip() for n in method_name.split(",")]
                target_sym = None
                matched_name = None
                
                for name_to_try in potential_names:
                    target_sym = next(
                        (s for s in analysis.get("symbols", []) if s.name == name_to_try),
                        None
                    )
                    if target_sym:
                        matched_name = name_to_try
                        break

                if not target_sym:
                    logger.info(f"Method(s) '{method_name}' not found in {file_record.file_name}, skipping")
                    continue

                # Surgical replacement (0-indexed)
                lines = original_content.split('\n')
                start_idx = target_sym.start_line - 1
                end_idx = target_sym.end_line
                new_lines = lines[:start_idx] + [new_content] + lines[end_idx:]
                refactored_content = '\n'.join(new_lines)

                # --- SYNTAX VALIDATION ---
                validation = self.validate_syntax(refactored_content, file_record.file_name)
                if not validation.success:
                    return validation

                # --- SAVE AS NEW FILE ON DISK ---
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                original_stem = Path(file_record.file_name).stem
                original_ext = Path(file_record.file_name).suffix
                new_file_name = f"{original_stem}_refactored_{timestamp}{original_ext}"

                # Use same storage structure as FileService.get_file_storage_path
                from src.config.settings import settings
                import uuid as uuid_module
                new_file_id = uuid_module.uuid4()
                subdir = str(new_file_id)[:2]
                new_storage_dir = settings.UPLOAD_DIR / subdir
                new_storage_dir.mkdir(parents=True, exist_ok=True)
                new_storage_path = new_storage_dir / f"{new_file_id}{original_ext}"

                try:
                    new_storage_path.write_text(refactored_content, encoding='utf-8')
                    logger.info(f"Refactored file saved to disk: {new_storage_path}")
                except Exception as e:
                    logger.error(f"Failed to save refactored file: {e}")
                    return ToolResult(success=False, data=None, error=f"Failed to save file: {e}")

                # --- REGISTER IN DB (NO Qdrant indexing) ---
                # processed=False prevents Qdrant indexing.
                # is_draft=True marks it as a non-authoritative version.
                conv_id = UUID(conversation_id) if conversation_id else file_record.conversation_id
                from src.models.models import ProcessingStatus

                new_file_record = await self.file_repo.create(
                    conversation_id=conv_id,
                    file_name=new_file_name,
                    file_type=original_ext.lower(),
                    file_size=len(refactored_content.encode('utf-8')),
                    storage_path=str(new_storage_path),
                    mime_type=file_record.mime_type,
                    processed=False,
                    processing_status=ProcessingStatus.PENDING,
                    extra_metadata={
                        "is_draft": True,
                        "source": "refactored",
                        "original_file_id": str(file_id),
                        "original_file_name": file_record.file_name,
                        "modified_method": method_name,
                        "refactored_at": timestamp,
                        # Note: processed=False ensures no Qdrant indexing is triggered
                        # Drafts are NOT indexed to avoid polluting semantic search
                        "indexing_skipped_reason": "draft_file_not_source_of_truth"
                    }
                )
                await self.file_repo.commit()

                new_id = str(new_file_record.id)
                download_url = f"/api/v1/files/{new_id}/download"

                logger.info(
                    f"Refactored file created: {new_file_name} "
                    f"(id={new_id}, NOT indexed in Qdrant)"
                )

                return ToolResult(
                    success=True,
                    data={
                        "file_id": new_id,
                        "file_name": new_file_name,
                        "download_url": download_url,
                        "original_file": file_record.file_name,
                        "modified_method": method_name,
                        "message": (
                            f"Archivo refactorizado generado: **{new_file_name}**\n"
                            f"Descarga: `{download_url}`\n\n"
                            f"El archivo original `{file_record.file_name}` no ha cambiado. "
                            f"Este borrador no está indexado en la base de conocimientos.\n"
                            f"INSTRUCCIÓN PARA EL LLM: El archivo ha sido entregado. "
                            f"NO repita el código generado. Brinde solo el enlace de descarga."
                        )
                    }
                )

            except Exception as e:
                logger.error(f"Error processing file {file_id_str}: {e}", exc_info=True)
                continue

        return ToolResult(
            success=False,
            data=None,
            error=f"Method '{method_name}' not found in any of the provided files"
        )
