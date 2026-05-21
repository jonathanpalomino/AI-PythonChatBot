# =============================================================================
# src/tools/plsql_doc_generator_tool.py
# =============================================================================
"""
Tool para generar documentación Markdown de paquetes PL/SQL.

Extiende BaseTool y se auto-descubre via ToolDiscovery.

Acciones soportadas:
  - generate_single : genera un .md para un único archivo PL/SQL.
  - generate_batch  : genera .md para una lista de archivos.
  - analyze_only    : devuelve el análisis como JSON (sin escribir archivo);
                      ideal para consumo directo por API o LLM.

Parámetros de entrada:
  action                 (str, requerido) — acción a ejecutar.
  file_path              (str)            — ruta al archivo .sql/.pls/… (single).
  file_paths             (array)          — lista de rutas (batch).
  content                (str)            — contenido PL/SQL directo (alternativa a file_path).
  output_dir             (str)            — directorio de salida (default: ./markdown_out).
  known_schemas          (array)          — schemas del proyecto para heurística del detector.
  include_trace_packages (boolean)        — incluir paquetes de traza (default: false).
  trace_packages         (array)          — paquetes de traza a excluir (default: ["TRAZAS"]).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.document_loaders.plsql_package_analyzer import (
    MarkdownDocGenerator,
    PackageAnalysisResult,
    PlsqlPackageAnalyzer,
)
from src.tools.base_tool import BaseTool, ToolCategory, ToolParameter, ToolResult
from src.utils.logger import get_logger


class PlsqlDocGeneratorTool(BaseTool):
    """
    Generador de documentación Markdown para paquetes PL/SQL.

    Analiza el SPEC y el BODY de un paquete para mapear los subprogramas
    públicos y sus invocaciones internas, produciendo un fichero .md
    estructurado con subtítulos por procedimiento/función y listas de llamadas.
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        # Eliminamos la instancia estática del generador para soportar configuraciones dinámicas por ejecución
        super().__init__()

    # =========================================================================
    # Metadatos de la Tool
    # =========================================================================

    @property
    def name(self) -> str:
        return 'plsql_doc_generator'

    @property
    def description(self) -> str:
        return (
            'Genera documentación Markdown estructurada para paquetes PL/SQL. '
            'Analiza el especificador (SPEC) y el cuerpo (BODY) de uno o varios '
            'paquetes Oracle PL/SQL, mapea los subprogramas públicos '
            '(PROCEDURE / FUNCTION) y sus invocaciones internas, y produce '
            'ficheros .md con subtítulos por subprograma y listas de llamadas. '
            'Soporta lectura de archivos locales o contenido PL/SQL directo.'
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.DOCUMENT

    @property
    def enabled_by_default(self) -> bool:
        return False

    @property
    def llm_hint(self) -> Optional[str]:
        return (
            "Se dispone de la tool 'plsql_doc_generator' para analizar y documentar "
            "paquetes Oracle PL/SQL. Úsala cuando el usuario pida documentar, "
            "mapear o analizar la estructura interna (procedimientos, funciones, "
            "llamadas internas) de un paquete PL/SQL."
        )

    # =========================================================================
    # Parámetros
    # =========================================================================

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name='action',
                type='string',
                description=(
                    'Acción a ejecutar: '
                    '"generate_single" para un archivo, '
                    '"generate_batch" para varios archivos, '
                    '"analyze_only" para obtener el análisis como JSON sin escribir archivos.'
                ),
                required=True,
                enum=['generate_single', 'generate_batch', 'analyze_only'],
                example='generate_single',
            ),
            ToolParameter(
                name='file_path',
                type='string',
                description=(
                    'Ruta absoluta o relativa al archivo PL/SQL '
                    '(.sql, .pls, .pck, .pkb, .pks, …). '
                    'Usado con "generate_single" y "analyze_only" cuando no se provee "content".'
                ),
                required=False,
                example='C:/proyectos/my_pkg.sql',
            ),
            ToolParameter(
                name='file_paths',
                type='array',
                description=(
                    'Lista de rutas a archivos PL/SQL. '
                    'Usado con "generate_batch".'
                ),
                required=False,
                example=['C:/proyectos/pkg_a.sql', 'C:/proyectos/pkg_b.sql'],
            ),
            ToolParameter(
                name='content',
                type='string',
                description=(
                    'Contenido PL/SQL directamente como texto. '
                    'Alternativa a "file_path" para uso desde LLM o API '
                    'sin acceso a sistema de archivos. '
                    'El paquete debe incluir tanto el SPEC como el BODY.'
                ),
                required=False,
                example=(
                    'CREATE OR REPLACE PACKAGE PKG_TEST AS\n'
                    '  PROCEDURE PROC_A(p_id NUMBER);\n'
                    'END PKG_TEST;\n'
                    'CREATE OR REPLACE PACKAGE BODY PKG_TEST AS\n'
                    '  PROCEDURE PROC_A(p_id NUMBER) IS\n'
                    '  BEGIN NULL; END PROC_A;\n'
                    'END PKG_TEST;'
                ),
            ),
            ToolParameter(
                name='output_dir',
                type='string',
                description=(
                    'Directorio de salida donde se escribirán los ficheros .md. '
                    'Se crea automáticamente si no existe. '
                    'Default: "./markdown_out" relativo al directorio de trabajo.'
                ),
                required=False,
                default='./markdown_out',
                example='C:/proyectos/docs',
            ),
            ToolParameter(
                name='known_schemas',
                type='array',
                description=(
                    'Lista de schemas de base de datos del proyecto '
                    '(ej: ["POS", "TRON2000"]). '
                    'Mejora la precisión del detector de llamadas al distinguir '
                    'schemas reales de alias de tabla.'
                ),
                required=False,
                default=[],
                example=['POS', 'TRON2000'],
            ),
            ToolParameter(
                name='include_trace_packages',
                type='boolean',
                description=(
                    'Si se deben incluir las invocaciones a paquetes de traza/log '
                    'en la documentación generada. Default: false.'
                ),
                required=False,
                default=False,
                example=False,
            ),
            ToolParameter(
                name='trace_packages',
                type='array',
                description=(
                    'Nombres de paquetes de traza a excluir cuando '
                    '"include_trace_packages" es false. Default: ["TRAZAS"].'
                ),
                required=False,
                default=['TRAZAS'],
                example=['TRAZAS', 'PKG_LOG'],
            ),
            ToolParameter(
                name='use_obsidian_links',
                type='boolean',
                description=(
                    'Si se deben generar enlaces en formato Obsidian [[PKG#SUB]] '
                    'en lugar de texto plano. Default: false.'
                ),
                required=False,
                default=False,
                example=True,
            ),
            ToolParameter(
                name='synonym_map',
                type='object',
                description=(
                    'Diccionario de sinónimos de Oracle (mapeo de nombre a objeto real). '
                    'Ej: {"EM_K_BATCH_POLIZA": "EM_K_BATCH_POLIZA_MPE"}. '
                    'Útil para que los backlinks de Obsidian apunten al archivo correcto.'
                ),
                required=False,
                default={},
                example={'EM_K_BATCH_POLIZA': 'EM_K_BATCH_POLIZA_MPE'},
            ),
            ToolParameter(
                name='synonyms_file',
                type='string',
                description=(
                    'Ruta a un archivo .json con los sinónimos. '
                    'Formato: {"SINONIMO": "OBJETO"} o [{"synonym": "...", "target": "..."}].'
                ),
                required=False,
                example='C:/proyectos/synonyms.json',
            ),
            ToolParameter(
                name='known_subprograms_file',
                type='string',
                description=(
                    'Ruta a un archivo .json con la lista de subprogramas conocidos '
                    'del proyecto para detección más precisa. '
                    'Formato: ["PACKAGE.PROCEDURE", "PACKAGE.FUNCTION", ...]'
                ),
                required=False,
                example='C:/proyectos/known_subprograms.json',
            ),
            ToolParameter(
                name='build_subprogram_index',
                type='boolean',
                description=(
                    'Si es true, analiza automáticamente todos los archivos para '
                    'construir el índice de subprogramas conocidos antes de procesar.'
                ),
                required=False,
                default=False,
                example=True,
            ),
            ToolParameter(
                name='save_subprogram_index',
                type='string',
                description=(
                    'Ruta donde guardar el índice de subprogramas construido '
                    'automáticamente para reutilización futura.'
                ),
                required=False,
                example='C:/proyectos/subprogram_index.json',
            ),
        ]

    # =========================================================================
    # Intents (para modo agente LLM)
    # =========================================================================

    def get_intent_definitions(self) -> Dict[str, Any]:
        return {
            'generate_plsql_docs': {
                'description': 'Generar documentación markdown de un paquete PL/SQL',
                'action_name': 'generate_single',
                'requires_target': True,
                'target_patterns': [
                    r'(?:paquete|package|fichero|archivo)\s+([\w$#./-]+\.(?:sql|pls|pck|pkb|pks|plsql|bdy|spc))',
                    r'"([\w$#./-]+\.(?:sql|pls|pck|pkb|pks|plsql|bdy|spc))"',
                ],
                'examples': [
                    'genera la documentación del paquete PL/SQL',
                    'documenta el package Oracle',
                    'crea el markdown del paquete',
                    'genera docs para el fichero .sql',
                    'analiza y documenta el package body',
                ],
                'default_params': {},
            },
            'analyze_plsql_package': {
                'description': (
                    'Analizar la estructura e invocaciones internas de un paquete PL/SQL '
                    'devolviendo el resultado como JSON'
                ),
                'action_name': 'analyze_only',
                'requires_target': False,
                'target_patterns': [],
                'examples': [
                    'analiza el paquete PL/SQL',
                    'qué procedimientos tiene el package',
                    'muéstrame las llamadas internas del paquete',
                    'lista las funciones del package body',
                    'inspecciona la estructura del paquete Oracle',
                ],
                'default_params': {},
            },
        }

    def params_from_intent(self, intent_result: Any) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            'action': intent_result.intent_def.action_name,
        }
        if intent_result.target:
            params['file_path'] = intent_result.target
        if intent_result.intent_def.default_params:
            params.update(intent_result.intent_def.default_params)
        return params

    # =========================================================================
    # Ejecución principal
    # =========================================================================

    async def execute(self, **kwargs) -> ToolResult:
        """Ejecuta la acción de documentación solicitada."""
        # Extraer parámetros con valores por defecto
        action = kwargs.get('action')
        file_path = kwargs.get('file_path')
        file_paths = kwargs.get('file_paths')
        content = kwargs.get('content')
        output_dir = kwargs.get('output_dir', './markdown_out')
        known_schemas = kwargs.get('known_schemas')
        include_trace_packages = kwargs.get('include_trace_packages', False)
        trace_packages = kwargs.get('trace_packages')
        use_obsidian_links = kwargs.get('use_obsidian_links', False)
        synonym_map = kwargs.get('synonym_map')
        synonyms_file = kwargs.get('synonyms_file')
        known_subprograms_file = kwargs.get('known_subprograms_file')
        build_subprogram_index = kwargs.get('build_subprogram_index', False)
        """Ejecuta la acción de documentación solicitada."""

        # Cargar sinónimos desde fichero si existe
        final_synonyms = synonym_map or {}
        if synonyms_file:
            try:
                s_path = Path(synonyms_file)
                if s_path.exists():
                    with s_path.open('r', encoding='utf-8') as f:
                        file_data = json.load(f)
                        # Normalizar (MarkdownDocGenerator.normalize lo hará después,
                        # pero aquí simplemente cargamos el dato crudo para mezclarlo si es dict)
                        if isinstance(file_data, dict):
                            final_synonyms.update(file_data)
                        elif isinstance(file_data, list):
                            # Si es lista, la pasamos tal cual al generador
                            final_synonyms = file_data
                else:
                    self.logger.warning(f"[plsql_doc_generator] No se encontró synonyms_file: {synonyms_file}")
            except Exception as e:
                self.logger.error(f"[plsql_doc_generator] Error cargando synonyms_file: {e}")

        # Cargar índice de subprogramas conocidos
        known_subprograms = []
        if known_subprograms_file:
            try:
                sp_path = Path(known_subprograms_file)
                if sp_path.exists():
                    with sp_path.open('r', encoding='utf-8') as f:
                        file_data = json.load(f)
                        if isinstance(file_data, list):
                            known_subprograms = file_data
                        else:
                            self.logger.warning(f"[plsql_doc_generator] known_subprograms_file debe contener una lista de strings")
                else:
                    self.logger.warning(f"[plsql_doc_generator] No se encontró known_subprograms_file: {known_subprograms_file}")
            except Exception as e:
                self.logger.error(f"[plsql_doc_generator] Error cargando known_subprograms_file: {e}")

        # Construir índice automáticamente si se solicita
        if build_subprogram_index:
            try:
                # Determinar qué archivos analizar para el índice
                files_for_index = []
                if file_paths:
                    files_for_index = file_paths
                elif file_path:
                    files_for_index = [file_path]
                elif action == 'generate_batch' and file_paths:
                    files_for_index = file_paths

                if files_for_index:
                    known_subprograms = await self._build_subprogram_index(files_for_index)
                    self.logger.info(f"[plsql_doc_generator] Construido índice automático con {len(known_subprograms)} subprogramas")
                else:
                    self.logger.warning("[plsql_doc_generator] No hay archivos para construir índice automático")
            except Exception as e:
                self.logger.error(f"[plsql_doc_generator] Error construyendo índice automático: {e}")

        # Usar analizador ANTLR si está disponible, sino usar el regular mejorado
        try:
            from src.document_loaders.plsql_package_analyzer_antlr import PlsqlPackageAnalyzerANTLR
            analyzer = PlsqlPackageAnalyzerANTLR(
                known_schemas=known_schemas or [],
                include_trace_packages=include_trace_packages,
                trace_packages=trace_packages or ['TRAZAS'],
                known_subprograms=known_subprograms,
            )
            self.logger.info("[plsql_doc_generator] Usando analizador ANTLR (precisión 100%)")
        except ImportError:
            # Fallback al analizador regular mejorado
            from src.document_loaders.plsql_package_analyzer import PlsqlPackageAnalyzer
            analyzer = PlsqlPackageAnalyzer(
                known_schemas=known_schemas or [],
                include_trace_packages=include_trace_packages,
                trace_packages=trace_packages or ['TRAZAS'],
                known_subprograms=known_subprograms,
            )
            self.logger.info("[plsql_doc_generator] Usando analizador regex mejorado (precisión ~95%)")

        if action == 'generate_single':
            return await self._execute_generate_single(
                analyzer, file_path, content, output_dir, use_obsidian_links, final_synonyms
            )
        elif action == 'generate_batch':
            return await self._execute_generate_batch(
                analyzer, file_paths or [], output_dir, use_obsidian_links, final_synonyms, known_schemas
            )
        elif action == 'analyze_only':
            return await self._execute_analyze_only(
                analyzer, file_path, content
            )
        else:
            return ToolResult(
                success=False,
                data=None,
                error=f"Acción desconocida: '{action}'. "
                      "Usa 'generate_single', 'generate_batch' o 'analyze_only'.",
            )

    # =========================================================================
    # Acciones específicas
    # =========================================================================

    async def _execute_generate_single(
        self,
        analyzer: PlsqlPackageAnalyzer,
        file_path: Optional[str],
        content: Optional[str],
        output_dir: str,
        use_obsidian_links: bool = False,
        synonym_map: Optional[Dict[str, str] | List] = None,
    ) -> ToolResult:
        """Genera un único fichero .md."""
        if not file_path and not content:
            return ToolResult(
                success=False,
                data=None,
                error="Se requiere 'file_path' o 'content' para la acción 'generate_single'.",
            )

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: (
                    analyzer.analyze_file(file_path)
                    if file_path
                    else analyzer.analyze_content(content or "")
                ),
            )

            output_path = self._write_markdown(result, output_dir, use_obsidian_links, synonym_map, analyzer.known_schemas)
            self.logger.info(f"[plsql_doc_generator] Generado: {output_path}")

            return ToolResult(
                success=True,
                data={
                    'package_name': result.package_name,
                    'output_file': str(output_path),
                    'procedures': result.procedures,
                    'functions': result.functions,
                    'warnings': result.warnings,
                },
                metadata={
                    'action': 'generate_single',
                    'subprogram_count': len(result.procedures) + len(result.functions),
                },
            )

        except Exception as exc:
            self.logger.error(
                f"[plsql_doc_generator] Error en generate_single: {exc}",
                exc_info=True,
            )
            return ToolResult(
                success=False,
                data=None,
                error=str(exc),
            )

    async def _execute_generate_batch(
        self,
        analyzer: PlsqlPackageAnalyzer,
        file_paths: List[str],
        output_dir: str,
        use_obsidian_links: bool = False,
        synonym_map: Optional[Dict[str, str] | List] = None,
        known_schemas: Optional[List[str]] = None,
    ) -> ToolResult:
        """Genera ficheros .md para varios archivos PL/SQL."""
        if not file_paths:
            return ToolResult(
                success=False,
                data=None,
                error="Se requiere 'file_paths' con al menos un archivo para 'generate_batch'.",
            )

        generated: List[Dict[str, Any]] = []
        errors: List[Dict[str, str]] = []

        for fp in file_paths:
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda f=fp: analyzer.analyze_file(f)
                )
                output_path = self._write_markdown(result, output_dir, use_obsidian_links, synonym_map, analyzer.known_schemas)
                self.logger.info(f"[plsql_doc_generator] Generado: {output_path}")
                generated.append({
                    'package_name': result.package_name,
                    'output_file': str(output_path),
                    'procedures': result.procedures,
                    'functions': result.functions,
                    'warnings': result.warnings,
                })
            except Exception as exc:
                self.logger.error(
                    f"[plsql_doc_generator] Error procesando '{fp}': {exc}",
                    exc_info=True,
                )
                errors.append({'file_path': fp, 'error': str(exc)})

        return ToolResult(
            success=len(errors) == 0,
            data={
                'generated': generated,
                'errors': errors,
                'total': len(file_paths),
                'success_count': len(generated),
                'error_count': len(errors),
            },
            metadata={
                'action': 'generate_batch',
                'output_dir': output_dir,
            },
        )

    async def _execute_analyze_only(
        self,
        analyzer: PlsqlPackageAnalyzer,
        file_path: Optional[str],
        content: Optional[str],
    ) -> ToolResult:
        """Devuelve el análisis como JSON sin escribir ficheros."""
        if not file_path and not content:
            return ToolResult(
                success=False,
                data=None,
                error="Se requiere 'file_path' o 'content' para la acción 'analyze_only'.",
            )

        try:
            result: PackageAnalysisResult = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: (
                    analyzer.analyze_file(file_path)
                    if file_path
                    else analyzer.analyze_content(content or "")
                ),
            )

            return ToolResult(
                success=True,
                data=result.to_dict(),
                metadata={
                    'action': 'analyze_only',
                    'package_name': result.package_name,
                    'subprogram_count': len(result.procedures) + len(result.functions),
                },
            )

        except Exception as exc:
            self.logger.error(
                f"[plsql_doc_generator] Error en analyze_only: {exc}",
                exc_info=True,
            )
            return ToolResult(
                success=False,
                data=None,
                error=str(exc),
            )

    # =========================================================================
    # Utilidades privadas
    # =========================================================================

    def _write_markdown(self, result: PackageAnalysisResult, output_dir: str, use_obsidian_links: bool = False, synonym_map: Optional[Dict[str, str] | List] = None, known_schemas: Optional[Set[str]] = None) -> Path:
        """Genera el Markdown y lo escribe en el directorio de salida."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f'{result.package_name}.md'
        md_generator = MarkdownDocGenerator(use_obsidian_links=use_obsidian_links, synonym_map=synonym_map, known_schemas=known_schemas)
        markdown = md_generator.generate(result)
        output_path.write_text(markdown, encoding='utf-8')
        return output_path

    # =========================================================================
    # Utilidades para índice de subprogramas
    # =========================================================================

    async def _build_subprogram_index(self, file_paths: List[str]) -> List[str]:
        """
        Construye automáticamente un índice de subprogramas conocidos
        analizando todos los archivos proporcionados.
        """
        if not file_paths:
            return []

        # Usar el mejor analyzer disponible (ANTLR si está disponible)
        try:
            from src.document_loaders.plsql_package_analyzer_antlr import PlsqlPackageAnalyzerANTLR
            temp_analyzer = PlsqlPackageAnalyzerANTLR(
                known_schemas=[],  # No necesitamos schemas para indexar
                include_trace_packages=True,  # Incluir todo para indexar
                trace_packages=[],
            )
            self.logger.debug("[plsql_doc_generator] Usando ANTLR para construir índice")
        except ImportError:
            # Fallback al analizador regular
            from src.document_loaders.plsql_package_analyzer import PlsqlPackageAnalyzer
            temp_analyzer = PlsqlPackageAnalyzer(
                known_schemas=[],  # No necesitamos schemas para indexar
                include_trace_packages=True,  # Incluir todo para indexar
                trace_packages=[],
            )
            self.logger.debug("[plsql_doc_generator] Usando regex para construir índice")

        known_subprograms = set()

        for fp in file_paths:
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda f=fp: temp_analyzer.analyze_file(f)
                )

                pkg_name = result.package_name

                # Agregar procedures públicas
                for proc in result.procedures:
                    known_subprograms.add(f"{pkg_name}.{proc}")

                # Agregar functions públicas
                for func in result.functions:
                    known_subprograms.add(f"{pkg_name}.{func}")

                # Agregar subprogramas privados (útiles para llamadas internas)
                for priv in result.private_subprograms:
                    known_subprograms.add(f"{pkg_name}.{priv}")

            except Exception as e:
                self.logger.warning(f"[plsql_doc_generator] Error analizando {fp} para índice: {e}")

        return sorted(list(known_subprograms))

    # =========================================================================
    # Formateo para LLM
    # =========================================================================

    def format_output(self, result: ToolResult) -> str:
        """Formatea el resultado para consumo del LLM."""
        if not result.success:
            return f'❌ Error en plsql_doc_generator: {result.error}'

        data = result.data
        action = result.metadata.get('action', '')

        if action == 'generate_single':
            pkg = data.get('package_name', '?')
            procs = len(data.get('procedures', []))
            funcs = len(data.get('functions', []))
            out_file = data.get('output_file', '?')
            warnings = data.get('warnings', [])
            lines = [
                f'✅ Documentación generada para el paquete **{pkg}**',
                f'📄 Fichero: `{out_file}`',
                f'📋 {procs} procedimiento(s), {funcs} función(es) documentados',
            ]
            if warnings:
                lines.append(f'⚠️ Advertencias: {"; ".join(warnings)}')
            return '\n'.join(lines)

        elif action == 'generate_batch':
            total = data.get('total', 0)
            ok = data.get('success_count', 0)
            ko = data.get('error_count', 0)
            lines = [
                f'✅ Batch completado: {ok}/{total} paquetes documentados',
            ]
            if ko:
                lines.append(f'❌ {ko} error(es) durante el batch')
            for gen in data.get('generated', []):
                lines.append(
                    f'  • {gen["package_name"]} → `{gen["output_file"]}`'
                )
            return '\n'.join(lines)

        elif action == 'analyze_only':
            pkg = data.get('package_name', '?')
            procs = data.get('procedures', [])
            funcs = data.get('functions', [])
            calls_map = data.get('calls_by_subprogram', {})
            lines = [
                f'🔍 Análisis del paquete **{pkg}**',
                f'📋 {len(procs)} procedimiento(s): {", ".join(procs) or "ninguno"}',
                f'📋 {len(funcs)} función(es): {", ".join(funcs) or "ninguna"}',
            ]
            for sp_name, sp_data in calls_map.items():
                sp_calls = sp_data.get('calls', [])
                if sp_calls:
                    lines.append(
                        f'  • {sp_name} invoca: {", ".join(sp_calls)}'
                    )
            return '\n'.join(lines)

        return str(data)
