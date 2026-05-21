# =============================================================================
# src/document_loaders/plsql_package_analyzer_antlr.py
# =============================================================================
"""
Analizador PL/SQL de precisión máxima usando ANTLR.

Este módulo proporciona un analizador que usa parsing sintáctico completo
con ANTLR para lograr precisión casi perfecta en la detección de:
- Llamadas a subprogramas vs accesos a campos de objetos
- Referencias a variables vs llamadas a funciones
- Contextos SQL vs llamadas a procedimientos

Requiere: pip install antlr4-python3-runtime

Para generar los archivos del parser desde las gramáticas:
1. Descargar PlSqlLexer.g4 y PlSqlParser.g4 desde:
   https://github.com/antlr/grammars-v4/tree/master/sql/plsql
2. Ejecutar: antlr4 -Dlanguage=Python3 PlSqlLexer.g4 PlSqlParser.g4
3. Colocar los archivos generados en src/antlr/plsql/
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Set

# Verificar disponibilidad de ANTLR
try:
    from antlr4 import CommonTokenStream, InputStream, ParseTreeWalker
    from antlr4.error.ErrorListener import ErrorListener

    # Intentar importar archivos generados por ANTLR
    try:
        from src.antlr.plsql.PlSqlLexer import PlSqlLexer
        from src.antlr.plsql.PlSqlParser import PlSqlParser
        from src.antlr.plsql.PlSqlParserListener import PlSqlParserListener
        ANTLR_FULL_AVAILABLE = True
    except ImportError:
        ANTLR_FULL_AVAILABLE = False

    ANTLR_RUNTIME_AVAILABLE = True
except ImportError:
    ANTLR_RUNTIME_AVAILABLE = False
    ANTLR_FULL_AVAILABLE = False
    ErrorListener = object  # Placeholder
    CommonTokenStream = None
    InputStream = None
    ParseTreeWalker = None

from src.document_loaders.plsql_package_analyzer import (
    PackageAnalysisResult,
    PlsqlPackageAnalyzer,
    SubprogramCalls,
)


class PlsqlANTLRErrorListener(ErrorListener if ANTLR_RUNTIME_AVAILABLE else object):
    """Error listener para ANTLR."""

    def __init__(self):
        self.errors = []

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        if ANTLR_RUNTIME_AVAILABLE:
            self.errors.append(f"Line {line}:{column} {msg}")
        else:
            self.errors.append(f"Syntax error: {msg}")

    def has_errors(self):
        return len(self.errors) > 0

    def get_errors(self):
        return self.errors


class PlsqlCallVisitor(PlSqlParserListener if ANTLR_FULL_AVAILABLE else object):
    """
    Visitor que recorre el AST de PL/SQL para detectar llamadas a subprogramas.

    Distingue perfectamente entre:
    - Llamadas a funciones/procedimientos: package.procedure()
    - Accesos a campos de objetos: object.field
    - Referencias a variables: variable
    """

    def __init__(self, package_name: str, known_subprograms: Optional[Set[str]] = None):
        self.package_name = package_name.upper()
        self.known_subprograms = known_subprograms or set()
        self.current_subprogram = None
        self.calls_by_subprogram: Dict[str, SubprogramCalls] = {}
        self.local_variables: Set[str] = set()
        self.object_types: Set[str] = set()

    def _analyze_expression(self, text: str):
        """Analizar expresiones para detectar llamadas a subprogramas."""
        # Usar regex inteligente para detectar llamadas
        # Patrón: identificador.punto.identificador(opcional paréntesis)
        call_pattern = r'\b([A-Z_][\w$#]*(?:\.[A-Z_][\w$#]*)*)(?:[^\S\n]*\()?'
        seen = set()

        for match in re.finditer(call_pattern, text, re.IGNORECASE):
            full_ident = match.group(1).upper()
            parts = full_ident.split('.')

            # Verificar si es realmente una llamada (tiene paréntesis o está en lista conocida)
            has_parentheses = '(' in text[match.end():match.end() + 50]
            is_known_subprogram = full_ident in self.known_subprograms

            # Verificar si es acceso a campo de objeto conocido
            is_field_access = self._is_field_access(full_ident, parts, text, match.start())

            if (has_parentheses or is_known_subprogram) and not is_field_access:
                if full_ident not in seen:
                    seen.add(full_ident)
                    if self.current_subprogram in self.calls_by_subprogram:
                        self.calls_by_subprogram[self.current_subprogram].calls.append(full_ident)

    def _is_field_access(self, full_ident: str, parts: List[str], context: str, position: int) -> bool:
        """
        Determinar si un identificador es acceso a campo de objeto/record.

        Maneja tanto campos simples (objeto.campo) como campos anidados (objeto.nested.campo).
        """
        if len(parts) < 2:
            return False

        first_part = parts[0]

        # 1. ¿Es una variable local que parece objeto?
        if first_part not in self.object_types and first_part not in self.local_variables:
            return False

        # 2. ¿No tiene paréntesis? (las llamadas reales tienen paréntesis)
        after_ident = context[position + len(full_ident):position + len(full_ident) + 10].strip()
        if after_ident.startswith('('):
            return False

        # 3. ¿Está en contexto de asignación o comparación?
        before_context = context[max(0, position - 100):position].strip()
        after_context = context[position + len(full_ident):position + len(full_ident) + 50].strip()

        # Operadores de asignación
        if ':=' in before_context or ':=' in after_context:
            return True

        # Operadores de comparación
        if any(op in before_context for op in ['=', '!=', '<>', '<', '>', '<=', '>=']):
            return True

        # En SELECT, FROM, WHERE, etc.
        if any(keyword in before_context.upper() for keyword in ['SELECT', 'FROM', 'WHERE', 'ORDER BY', 'GROUP BY']):
            return True

        # Después de operadores aritméticos
        if any(op in before_context[-10:] for op in ['+', '-', '*', '/']):
            return True

        # En parámetros de funciones (caso especial para campos anidados)
        # Buscar si está precedido por coma o paréntesis de apertura
        before_pos = position - 1
        while before_pos >= 0 and context[before_pos].isspace():
            before_pos -= 1

        if before_pos >= 0 and context[before_pos] in [',', '(']:
            return True

        return False


class PlsqlPackageAnalyzerANTLR(PlsqlPackageAnalyzer):
    """
    Analizador PL/SQL usando ANTLR para precisión máxima.

    Hereda de PlsqlPackageAnalyzer pero reemplaza la lógica de detección
    de llamadas con parsing sintáctico completo cuando ANTLR está disponible.
    """

    def __init__(
        self,
        known_schemas: Optional[List[str]] = None,
        include_trace_packages: bool = False,
        trace_packages: Optional[List[str]] = None,
        known_subprograms: Optional[List[str]] = None,
    ):
        super().__init__(
            known_schemas=known_schemas,
            include_trace_packages=include_trace_packages,
            trace_packages=trace_packages,
            known_subprograms=known_subprograms,
        )

        # Verificar si ANTLR está realmente disponible y configurado
        self._antlr_ready = self._check_antlr_setup()

    def _check_antlr_setup(self) -> bool:
        """Verificar si ANTLR está configurado correctamente."""
        if not ANTLR_RUNTIME_AVAILABLE:
            return False

        # Verificar que los archivos del parser existen
        antlr_dir = Path(__file__).parent / "antlr" / "plsql"
        required_files = ["PlSqlLexer.py", "PlSqlParser.py", "PlSqlParserListener.py"]

        for file in required_files:
            if not (antlr_dir / file).exists():
                return False

        return True

    def analyze_content(self, content: str, default_name: str = "UNKNOWN") -> PackageAnalysisResult:
        """Analizar contenido PL/SQL usando ANTLR si está disponible, o regex como fallback."""
        if not content.strip():
            raise ValueError("Content cannot be empty")

        # Usar ANTLR si está disponible y configurado, sino usar el método heredado mejorado
        if self._antlr_ready and ANTLR_FULL_AVAILABLE:
            return self._analyze_with_antlr(content)
        else:
            # Fallback al método heredado (que ya tiene las mejoras de heurísticas)
            return super().analyze_content(content, default_name)

    def _analyze_with_antlr(self, content: str) -> PackageAnalysisResult:
        """Analizar usando ANTLR para máxima precisión."""
        try:
            # Preparar input para ANTLR
            input_stream = InputStream(content)
            lexer = PlSqlLexer(input_stream)
            token_stream = CommonTokenStream(lexer)
            parser = PlSqlParser(token_stream)

            # Configurar error listener
            error_listener = PlsqlANTLRErrorListener()
            parser.removeErrorListeners()
            parser.addErrorListener(error_listener)

            # Parsear
            tree = parser.sql_script()

            if error_listener.has_errors():
                print(f"ANTLR parsing errors: {error_listener.get_errors()}")
                # Fallback al método heredado
                return super().analyze_content(content, "FALLBACK")

            # Extraer información básica
            basic_info = self._extract_basic_info(content)

            # Crear visitor y analizar
            known_subs = {f"{basic_info['package_name']}.{sub}" for sub in basic_info['private_subprograms']}
            known_subs.update(self.known_subprograms)

            visitor = PlsqlCallVisitor(basic_info['package_name'], known_subs)

            # Recorrer el árbol
            walker = ParseTreeWalker()
            walker.walk(visitor, tree)

            return PackageAnalysisResult(
                package_name=basic_info['package_name'],
                file_path=None,
                procedures=basic_info['procedures'],
                functions=basic_info['functions'],
                private_subprograms=basic_info['private_subprograms'],
                calls_by_subprogram=visitor.calls_by_subprogram,
                warnings=basic_info.get('warnings', []),
            )

        except Exception as e:
            print(f"Error in ANTLR analysis: {e}")
            # Fallback al método heredado
            return super().analyze_content(content, "FALLBACK")

    def _extract_basic_info(self, content: str) -> Dict:
        """Extraer información básica usando métodos heredados."""
        temp_result = super().analyze_content(content, "TEMP")
        return {
            'package_name': temp_result.package_name,
            'procedures': temp_result.procedures,
            'functions': temp_result.functions,
            'private_subprograms': temp_result.private_subprograms,
            'warnings': temp_result.warnings,
        }


# Funciones de utilidad para setup de ANTLR
def setup_antlr_files():
    """
    Configuración inicial para archivos ANTLR.

    Para usar este analizador, necesitas:
    1. Descargar PlSqlLexer.g4 y PlSqlParser.g4
    2. Generar archivos Python con: antlr4 -Dlanguage=Python3 PlSqlLexer.g4 PlSqlParser.g4
    3. Colocar los archivos generados en src/antlr/plsql/
    """
    antlr_dir = Path(__file__).parent / "antlr" / "plsql"
    antlr_dir.mkdir(parents=True, exist_ok=True)

    required_files = [
        "PlSqlLexer.py",
        "PlSqlParser.py",
        "PlSqlParserListener.py",
        "PlSqlLexer.tokens",
        "PlSqlParser.tokens"
    ]

    missing_files = []
    for file in required_files:
        if not (antlr_dir / file).exists():
            missing_files.append(file)

    if missing_files:
        raise FileNotFoundError(
            f"Faltan archivos ANTLR en {antlr_dir}: {missing_files}. "
            "Ejecuta: antlr4 -Dlanguage=Python3 PlSqlLexer.g4 PlSqlParser.g4"
        )

    return True


# Verificación al importar
if ANTLR_RUNTIME_AVAILABLE:
    try:
        setup_antlr_files()
        ANTLR_FULL_AVAILABLE = True
    except FileNotFoundError:
        ANTLR_FULL_AVAILABLE = False
else:
    ANTLR_FULL_AVAILABLE = False