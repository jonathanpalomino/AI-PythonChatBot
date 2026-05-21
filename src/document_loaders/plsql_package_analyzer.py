# =============================================================================
# src/document_loaders/plsql_package_analyzer.py
# =============================================================================
"""
Motor de análisis semántico para paquetes PL/SQL.

Port Python del reader_package_logic.js (v5 — definitive).

Detecta subprogramas públicos declarados en el SPEC y mapea las
invocaciones internas dentro de cada bloque del BODY, generando
documentación Markdown estructurada.

Arquitectura de detección (ídem JS v5):
  ① Un único regex captura el identificador completo (dotted) antes de "("
     → cada sitio de invocación produce exactamente un match.
  ② [^\S\n]* entre ident y "(" — no permite newlines
     → evita falsa adyacencia  B.COD_CIA\n(SELECT ...)
  ③ Lookbehind de contexto INSERT INTO
     → INSERT INTO schema.tabla(cols) NO es una llamada.
  ④ is_likely_pkg_or_schema() para el segmento líder en llamadas de 2 partes
     → filtra alias de tabla de 1-3 letras (A, B, TB1, …).
  ⑤ IGNORE_TOKENS: tipos de datos, keywords SQL/PL-SQL y
     funciones built-in de Oracle que aparecen seguidos de "(".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set


# =============================================================================
# § 1  Conjuntos de exclusión
# =============================================================================

IGNORE_PACKAGES: Set[str] = {
    'DBMS_OUTPUT', 'DBMS_UTILITY', 'DBMS_SQL', 'DBMS_LOB', 'DBMS_PIPE',
    'DBMS_ALERT', 'DBMS_SCHEDULER', 'DBMS_STATS', 'DBMS_METADATA',
    'DBMS_SESSION', 'DBMS_LOCK', 'DBMS_APPLICATION_INFO', 'DBMS_CRYPTO',
    'UTL_FILE', 'UTL_HTTP', 'UTL_SMTP', 'UTL_RAW', 'UTL_ENCODE', 'UTL_I18N',
    'SYS', 'STANDARD', 'UTL_INADDR','TRN_K_GLOBAL',
}

IGNORE_TOKENS: Set[str] = {
    # Tipos de datos Oracle / PL/SQL
    'NUMBER', 'VARCHAR2', 'VARCHAR', 'CHAR', 'NCHAR', 'NVARCHAR2', 'NVARCHAR',
    'INTEGER', 'INT', 'SMALLINT', 'PLS_INTEGER', 'BINARY_INTEGER',
    'BINARY_FLOAT', 'BINARY_DOUBLE', 'FLOAT', 'REAL', 'DOUBLE', 'PRECISION',
    'DEC', 'DECIMAL', 'BOOLEAN', 'DATE', 'TIMESTAMP', 'INTERVAL',
    'CLOB', 'BLOB', 'NCLOB', 'RAW', 'LONG', 'BFILE', 'XMLTYPE',
    'ANYDATA', 'ANYTYPE',
    'NATURAL', 'NATURALN', 'POSITIVE', 'POSITIVEN', 'SIGNTYPE',
    'SIMPLE_INTEGER', 'SIMPLE_FLOAT', 'SIMPLE_DOUBLE',
    # Conversión
    'TO_CHAR', 'TO_NUMBER', 'TO_DATE', 'TO_CLOB', 'TO_NCLOB', 'TO_BLOB',
    'TO_YMINTERVAL', 'TO_DSINTERVAL', 'TO_TIMESTAMP', 'TO_TIMESTAMP_TZ',
    'TO_BINARY_FLOAT', 'TO_BINARY_DOUBLE',
    # Null / condicional
    'NVL', 'NVL2', 'COALESCE', 'NULLIF', 'LNNVL', 'DECODE', 'NANVL',
    'LLIF', 'ULLIF', 'NLIF', 'NNVL', 'NNVL2',
    # Cadena
    'TRIM', 'LTRIM', 'RTRIM', 'UPPER', 'LOWER', 'INITCAP',
    'SUBSTR', 'SUBSTRB', 'INSTR', 'INSTRB', 'LENGTH', 'LENGTHB',
    'REPLACE', 'TRANSLATE', 'LPAD', 'RPAD', 'CONCAT', 'CHR', 'ASCII',
    'REGEXP_REPLACE', 'REGEXP_SUBSTR', 'REGEXP_INSTR',
    'REGEXP_LIKE', 'REGEXP_COUNT',
    # Matemáticas
    'ABS', 'CEIL', 'FLOOR', 'MOD', 'ROUND', 'TRUNC', 'SIGN', 'POWER',
    'SQRT', 'EXP', 'LN', 'LOG', 'REMAINDER', 'BITAND',
    # Agregación / analíticas
    'SUM', 'COUNT', 'AVG', 'MAX', 'MIN', 'MEDIAN', 'STDDEV', 'VARIANCE',
    'LISTAGG', 'COLLECT', 'XMLAGG', 'WM_CONCAT',
    'RANK', 'DENSE_RANK', 'ROW_NUMBER', 'NTILE',
    'LAG', 'LEAD', 'FIRST_VALUE', 'LAST_VALUE',
    'RATIO_TO_REPORT', 'PERCENT_RANK', 'CUME_DIST',
    # Fecha / hora
    'SYSDATE', 'SYSTIMESTAMP', 'CURRENT_DATE', 'CURRENT_TIMESTAMP',
    'LOCALTIMESTAMP', 'NUMTOYMINTERVAL', 'NUMTODSINTERVAL',
    'MONTHS_BETWEEN', 'ADD_MONTHS', 'LAST_DAY', 'NEXT_DAY',
    'EXTRACT', 'TZ_OFFSET',
    # Misc Oracle built-in
    'SYS_GUID', 'SYS_CONTEXT', 'USERENV',
    'GREATEST', 'LEAST', 'VSIZE', 'DUMP', 'ORA_HASH',
    'HEXTORAW', 'RAWTOHEX', 'CHARTOROWID', 'ROWIDTOCHAR',
    'ROWNUM', 'ROWID',
    'CAST', 'TREAT', 'MULTISET',
    'XMLELEMENT', 'XMLFOREST', 'XMLROOT', 'XMLQUERY',
    'XMLTABLE', 'XMLSEQUENCE',
    # Keywords SQL que preceden a "("
    'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'MERGE', 'EXECUTE',
    'FROM', 'WHERE', 'HAVING', 'INTO', 'VALUES', 'SET',
    'ON', 'USING', 'RETURNING',
    'GROUP', 'ORDER', 'WITHIN', 'OVER', 'PARTITION', 'KEEP',
    'AND', 'OR', 'NOT', 'IN', 'EXISTS', 'ANY', 'ALL', 'SOME',
    'BETWEEN', 'LIKE', 'OVERLAPS',
    'BY', 'AS', 'AT', 'OF', 'TO',
    'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER', 'CROSS', 'FULL',
    'NATURAL', 'UNION', 'INTERSECT', 'MINUS', 'EXCEPT',
    'PIVOT', 'UNPIVOT', 'DISTINCT', 'UNIQUE', 'SAMPLE',
    'CONNECT', 'START', 'PRIOR', 'NOCYCLE',
    'BULK', 'COLLECT', 'LIMIT', 'FORALL',
    # Keywords PL/SQL estructurales
    'IF', 'ELSIF', 'THEN', 'ELSE', 'WHEN', 'CASE', 'MATCHED',
    'LOOP', 'WHILE', 'FOR', 'CURSOR',
    'BEGIN', 'END', 'IS', 'RETURN', 'RAISE',
    'OPEN', 'FETCH', 'CLOSE',
    'EXCEPTION', 'OTHERS',
    'TABLE', 'RECORD', 'VARRAY', 'TYPE', 'SUBTYPE', 'REF',
    # Métodos de colección
    'FIRST', 'LAST', 'NEXT', 'EXTEND', 'DELETE', 'OBJECT', 'UNDER',
    # Excepciones predefinidas
    'NO_DATA_FOUND', 'TOO_MANY_ROWS', 'DUP_VAL_ON_INDEX',
    'VALUE_ERROR', 'INVALID_NUMBER', 'ZERO_DIVIDE',
    'CURSOR_ALREADY_OPEN', 'INVALID_CURSOR',
    # Atributos / Métodos de secuencia y otros
    'NEXTVAL', 'CURRVAL',
    # Proyectos específicos (Trazas/Logs)
    'MPE_P_TRAZA5',
}


# =============================================================================
# § 2  Data classes de resultado
# =============================================================================

@dataclass
class SubprogramCalls:
    """Invocaciones detectadas para un único subprograma."""
    name: str
    kind: str  # "PROCEDURE" | "FUNCTION"
    calls: List[str] = field(default_factory=list)


@dataclass
class PackageAnalysisResult:
    """Resultado del análisis de un paquete PL/SQL."""
    package_name: str
    file_path: Optional[str]
    procedures: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    private_subprograms: List[str] = field(default_factory=list)
    calls_by_subprogram: Dict[str, SubprogramCalls] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'package_name': self.package_name,
            'file_path': self.file_path,
            'procedures': self.procedures,
            'functions': self.functions,
            'private_subprograms': self.private_subprograms,
            'calls_by_subprogram': {
                k: {'name': v.name, 'kind': v.kind, 'calls': v.calls}
                for k, v in self.calls_by_subprogram.items()
            },
            'warnings': self.warnings,
        }


# =============================================================================
# § 3  Limpieza de texto
# =============================================================================

class PlsqlTextCleaner:
    """
    Limpia texto PL/SQL para análisis de llamadas o análisis estructural.

    Existe en dos modos:
    - clean()     : preserva saltos de línea (para detección de llamadas).
    - clean_full(): colapsa todo el espacio (para detección estructural).
    """

    _RE_TOKENIZER = re.compile(
        r"(--[^\r\n]*)|(/\*[\s\S]*?\*/)|('(?:''|[^'])*')"
    )
    _RE_HORIZONTAL_WS = re.compile(r'[^\S\n]+')
    _RE_NL_WS = re.compile(r'[ \t]*\n[ \t]*')
    _RE_MULTI_NL = re.compile(r'\n{3,}')

    @classmethod
    def neutralize(cls, text: str) -> str:
        """
        Produce una versión del texto donde comentarios y strings son espacios,
        preservando el largo total para búsqueda de índices.
        """
        def replacer(m):
            # group 1: line comment, group 2: block comment, group 3: string literal
            if m.group(1):
                return ' ' * len(m.group(1))
            elif m.group(2):
                return ' ' * len(m.group(2))
            elif m.group(3):
                return ' ' * len(m.group(3))
            return m.group(0)

        return cls._RE_TOKENIZER.sub(replacer, text)

    @classmethod
    def clean(cls, text: str) -> str:
        """Limpieza para detección de llamadas — preserva saltos de línea."""
        def replacer(m):
            if m.group(1):
                return ''
            elif m.group(2):
                return ' '
            elif m.group(3):
                return "''"
            return m.group(0)

        s = cls._RE_TOKENIZER.sub(replacer, text)
        s = cls._RE_HORIZONTAL_WS.sub(' ', s)
        s = cls._RE_NL_WS.sub('\n', s)
        s = cls._RE_MULTI_NL.sub('\n\n', s)
        return s.strip()

    @classmethod
    def clean_full(cls, text: str) -> str:
        """Limpieza para análisis estructural — colapsa todo el espacio."""
        s = cls.clean(text)
        s = re.sub(r'\n', ' ', s)
        s = re.sub(r'\s+', ' ', s)
        return s.strip()


# =============================================================================
# § 4  Detección de llamadas
# =============================================================================

class PlsqlCallDetector:
    """
    Detecta invocaciones a subprogramas dentro de un bloque PL/SQL.

    Porta la lógica del § 9 del reader_package_logic.js v5.
    """

    # Regex central: identificador dotted completo
    # Opcionalmente seguido de "("
    _CALL_RE = re.compile(
        r'\b([A-Z_][\w$#]*(?:\.[A-Z_][\w$#]*)*)(?:[^\S\n]*\()?',
        re.IGNORECASE
    )

    def __init__(
        self,
        pkg_name: str,
        subprogram_index: Set[str],
        declared_cursors: Set[str],
        local_types: Set[str],
        known_schemas: Optional[Set[str]] = None,
        include_trace_packages: bool = False,
        trace_packages: Optional[Set[str]] = None,
        all_local_subprograms: Optional[Set[str]] = None,
        # Nueva funcionalidad: índice global de subprogramas conocidos
        known_subprograms: Optional[Set[str]] = None,
    ):
        self.pkg_name = pkg_name.upper()
        self.subprogram_index = subprogram_index
        self.all_local_subprograms = all_local_subprograms or subprogram_index
        self.declared_cursors = declared_cursors
        self.local_types = local_types
        self.known_schemas: Set[str] = known_schemas or set()
        self.include_trace_packages = include_trace_packages
        self.trace_set: Set[str] = trace_packages or {'TRAZAS'}

        # Índice global de subprogramas conocidos para validación precisa
        self.known_subprograms: Set[str] = known_subprograms or set()
        # Convertir a mayúsculas para comparaciones consistentes
        self.known_subprograms = {s.upper() for s in self.known_subprograms}

    def _is_likely_pkg_or_schema(self, name: str) -> bool:
        """
        Heurística: ¿el primer segmento parece un paquete/schema o un alias de tabla?

        Reglas (en orden):
          1. Está en known_schemas                 → schema real.
          2. No empieza con prefijos de variable   → excluye G_, V_, L_, etc.
          3. Contiene guion bajo "_"               → convención Oracle de packages.
          4. Longitud > 4                          → improbable alias de tabla.
        """
        if name in self.known_schemas:
            return True
            
        upper_name = name.upper()
        # Common variable prefixes in PL/SQL that are NOT packages
        # G_ (global), V_ (variable), L_ (local), P_ (parameter), C_ (cursor), R_ (record)
        if upper_name.startswith(('V_', 'L_', 'LV_', 'G_', 'C_', 'R_', 'P_', 'PV_', 'TB_', 'RG_', 'REC_', 'REG_')):
            return False

        if '_' in name:
            return True
        if len(name) > 4:
            return True
        return False

    def _is_object_field_access(self, text: str, match_start: int, full_ident: str) -> bool:
        """
        Determina si un identificador dotted es un acceso a campo de objeto/registro
        en lugar de una llamada a subprograma.

        Analiza el contexto sintáctico para distinguir:
        - Asignaciones: obj.field := value
        - Parámetros: procedure(obj.field)
        - Expresiones: obj.field + 1, IF obj.field = 'X'
        - Accesos directos: obj.field
        """
        parts = full_ident.split('.')
        if len(parts) != 2:
            return False  # Solo analizar identificadores de 2 partes por ahora

        # Si el identificador tiene paréntesis al final, definitivamente NO es field access
        after_match = text[match_start:match_start + len(full_ident) + 10].strip()
        if after_match[len(full_ident):].strip().startswith('('):
            return False

        # Buscar el contexto inmediatamente después del identificador
        after_full_match = text[match_start + len(full_ident):match_start + len(full_ident) + 50].strip()

        # Patrón más robusto: identificador seguido de := o = (con posibles espacios)
        if after_full_match.startswith(':=') or after_full_match.startswith('='):
            return True

        # Buscar contexto antes (últimos 200 caracteres para más contexto)
        before_match = text[max(0, match_start - 200):match_start]

        # 1. Asignaciones y operaciones (más robusto)
        # Buscar patrones como: variable :=, variable =, variable +, etc.
        if re.search(r'\b' + re.escape(parts[0]) + r'\.' + re.escape(parts[1]) + r'\s*[:=<>!+\-*/]\s*$', before_match):
            return True

        # 2. En contextos de comparación/condición
        lines_context = before_match.split('\n')[-5:]  # Más líneas de contexto
        context_str = ' '.join(lines_context).strip()

        # IF/ELSIF/WHEN conditions
        if re.search(r'\b(IF|ELSIF|WHEN)\s+.*\b' + re.escape(full_ident) + r'\s*[=<>!]', context_str, re.IGNORECASE):
            return True

        # 3. En SELECT/UPDATE/INSERT statements
        if re.search(r'\b(SELECT|UPDATE|INSERT|SET)\s+.*\b' + re.escape(full_ident) + r'\b', context_str, re.IGNORECASE):
            return True

        # 4. En expresiones de función (como parámetros)
        # Buscar si está dentro de paréntesis de llamada a función
        open_parens_before = before_match.count('(')
        close_parens_before = before_match.count(')')
        if open_parens_before > close_parens_before:
            return True

        # 5. Heurística adicional: si la primera parte parece un objeto/registro conocido
        if self._is_known_object_type(parts[0]):
            # Para objetos conocidos, asumir que .field es acceso a campo
            # A menos que esté seguido de paréntesis (lo cual ya verificamos arriba)
            return True

        return False

        # Si el identificador tiene paréntesis al final, definitivamente NO es field access
        after_match = text[match_start:match_start + len(full_ident) + 10].strip()
        if after_match[len(full_ident):].strip().startswith('('):
            return False

        # Buscar el contexto inmediatamente después del identificador
        after_full_match = text[match_start + len(full_ident):match_start + len(full_ident) + 50]

        # Patrón específico: identificador seguido inmediatamente de := o =
        if after_full_match.strip().startswith(':=') or after_full_match.strip().startswith('='):
            return True

        # Buscar el contexto antes del identificador para otros patrones
        before_match = text[max(0, match_start - 100):match_start]

        # Si está precedido por operadores de comparación o aritméticos
        if re.search(r'[+\-*/<>!=]\s*$', before_match.strip()):
            return True

        # Si está en un contexto de IF/WHEN con comparación
        lines_before = before_match.split('\n')[-3:]  # Últimas 3 líneas
        context_before = ' '.join(lines_before).strip()
        if re.search(r'\b(IF|WHEN|ELSIF)\s+.*' + re.escape(full_ident) + r'\s*[=<>!]', context_before, re.IGNORECASE):
            return True

        # Si está en un SELECT o UPDATE como campo
        if re.search(r'\b(SELECT|UPDATE|SET)\s+.*' + re.escape(full_ident) + r'\b', context_before, re.IGNORECASE):
            return True

        return False

    def _is_known_object_type(self, identifier: str) -> bool:
        """
        Determina si un identificador parece ser un objeto/registro conocido.
        Basado en patrones comunes en el código analizado.
        """
        upper_id = identifier.upper()

        # Prefijos comunes de objetos/registros
        object_prefixes = ['LO_', 'PO_', 'O_', 'L_', 'P_', 'REC_']

        if any(upper_id.startswith(prefix) for prefix in object_prefixes):
            return True

        # Tipos conocidos del sistema
        known_objects = ['LO_RETENCION', 'LO_LIQUIDACION', 'LO_SINIESTRO', 'O_RETENCIONES_S']

        return upper_id in known_objects

    def detect(self, text: str) -> List[str]:
        """
        Devuelve la lista de invocaciones únicas detectadas en `text`.

        `text` debe ser el bloque RAW (sin limpiar) de un único subprograma.
        La limpieza se aplica internamente para preservar los saltos de línea.
        """
        cleaned = PlsqlTextCleaner.clean(text)
        calls: List[str] = []
        seen: Set[str] = set()

        def add_call(name: str):
            if name not in seen:
                seen.add(name)
                calls.append(name)

        # 1) Detect calls using common regex (parameterized or not)
        for m in self._CALL_RE.finditer(cleaned):
            full_ident = m.group(1).upper()
            parts = full_ident.split('.')

            # Excluir %TYPE y %ROWTYPE (declaraciones de variables)
            post = cleaned[m.end():m.end()+15].lstrip().upper()
            if post.startswith('%TYPE') or post.startswith('%ROWTYPE'):
                continue

            # ③ Contexto de SQL (Tablas/Vistas): ignorar si precede FROM, JOIN, UPDATE, INTO, TABLE
            pre = cleaned[max(0, m.start() - 40):m.start()].rstrip().upper()
            if re.search(r'\b(FROM|JOIN|UPDATE|INTO|TABLE)$', pre):
                continue

            # Verificación inteligente: el índice conocido es opcional pero mejorativo
            if self.known_subprograms and full_ident in self.known_subprograms:
                # Está en el índice conocido - definitivamente es una llamada válida
                pass  # Continuar procesando normalmente
            elif self._is_object_field_access(cleaned, m.start(), full_ident):
                # Parece ser acceso a campo de objeto - excluir
                continue

            p0 = parts[0]

            # --- Caso 1: schema.package.subprogram (3+ partes) ---------------
            if len(parts) >= 3:
                if any(p in IGNORE_TOKENS for p in parts) or p0 in IGNORE_PACKAGES:
                    continue
                if not self.include_trace_packages and p0 in self.trace_set:
                    continue

                # Strip known schema prefix (insensible a mayúsculas)
                if p0.upper() in self.known_schemas:
                    add_call(f'{parts[1]}.{parts[2]}')
                else:
                    add_call(f'{parts[0]}.{parts[1]}.{parts[2]}')

            # --- Caso 2: package.subprogram (2 partes) -----------------------
            elif len(parts) == 2:
                p1 = parts[1]

                if any(p in IGNORE_TOKENS for p in parts) or p0 in IGNORE_PACKAGES:
                    continue
                if p0 in self.declared_cursors or p0 in self.local_types:
                    continue
                if not self.include_trace_packages and p0 in self.trace_set:
                    continue
                if not self._is_likely_pkg_or_schema(p0):
                    continue

                # Strip known schema prefix (insensible a mayúsculas)
                if p0.upper() in self.known_schemas:
                    add_call(parts[1])
                else:
                    add_call(f'{parts[0]}.{parts[1]}')

            # --- Caso 3: identificador standalone ----------------------------
            else:
                if p0 in IGNORE_TOKENS or p0 in IGNORE_PACKAGES:
                    continue
                if p0 in self.declared_cursors or p0 in self.local_types:
                    continue
                if not self.include_trace_packages and p0 in self.trace_set:
                    continue

                # Check if it is a local subprogram (internal call)
                if p0 in self.all_local_subprograms:
                    add_call(f'{self.pkg_name}.{p0}')
                elif m.group(0).rstrip()[-1] == '(':
                    # Only add standalone external identifiers if they have parentheses
                    if len(p0) >= 3 and self._is_likely_pkg_or_schema(p0):
                        add_call(p0)

        return calls


# =============================================================================
# § 5  Extracción de metadatos del SPEC / BODY
# =============================================================================

class _PlsqlStructureExtractor:
    """Utilidades estáticas para extraer piezas estructurales de texto PL/SQL."""

    _RE_PKG_NAME = re.compile(
        r'CREATE\s+(?:OR\s+REPLACE\s+)?PACKAGE\s+(?:BODY\s+)?'
        r'(?:[\w$#]+\.)?(?:"?(?P<name>[\w$#]+)"?).*?\b(?:AS|IS)\b',
        re.IGNORECASE | re.DOTALL,
    )
    _RE_SPEC_HDR = re.compile(
        r'CREATE\s+(?:OR\s+REPLACE\s+)?PACKAGE\s+(?!BODY\b)'
        r'(?:[\w$#]+\.)?"?[\w$#]+"?.*?\b(?:AS|IS)\b',
        re.IGNORECASE | re.DOTALL,
    )
    _RE_BODY_HDR = re.compile(
        r'CREATE\s+(?:OR\s+REPLACE\s+)?PACKAGE\s+BODY\s+'
        r'(?:[\w$#]+\.)?"?[\w$#]+"?.*?\b(?:AS|IS)\b',
        re.IGNORECASE | re.DOTALL,
    )
    _RE_SPEC_HDR = re.compile(
        r'CREATE\s+(?:OR\s+REPLACE\s+)?PACKAGE\s+(?!BODY\b)'
        r'(?:[\w$#]+\.)?"?[\w$#]+"?.*?\b(?:AS|IS)\b',
        re.IGNORECASE | re.DOTALL,
    )
    _RE_BODY_HDR = re.compile(
        r'CREATE\s+(?:OR\s+REPLACE\s+)?PACKAGE\s+BODY\s+'
        r'(?:[\w$#]+\.)?"?[\w$#]+"?.*?\b(?:AS|IS)\b',
        re.IGNORECASE | re.DOTALL,
    )
    _RE_END_BLOCK = re.compile(
        r'END\s+(?:[\w$#]+\s*)?;',
        re.IGNORECASE,
    )

    @classmethod
    def package_name(cls, content: str) -> Optional[str]:
        neutralized = PlsqlTextCleaner.neutralize(content)
        m = cls._RE_PKG_NAME.search(neutralized)
        return m.group('name').upper() if m else None

    @classmethod
    def spec_block(cls, content: str) -> str:
        """
        Extrae el contenido del SPEC del paquete.

        Estrategia: busca el header del SPEC en el texto previo al
        PACKAGE BODY (si existe), luego toma hasta el ÚLTIMO END word;
        de esa sección. Funciona para SPECs con cualquier cierre
        (END;, END PKG_NAME;, etc.).
        """
        # Use neutralized text for searching to avoid issues with large comments
        neutralized = PlsqlTextCleaner.neutralize(content)
        body_hdr_m = cls._RE_BODY_HDR.search(neutralized)
        search_area = content[:body_hdr_m.start()] if body_hdr_m else content

        # Search in neutralized text but map back to original
        search_area_neutralized = PlsqlTextCleaner.neutralize(search_area)
        spec_hdr_m = cls._RE_SPEC_HDR.search(search_area_neutralized)
        if not spec_hdr_m:
            return ''

        # Extract from original content using the same positions
        after_hdr = search_area[spec_hdr_m.end():]
        ends = list(cls._RE_END_BLOCK.finditer(after_hdr))
        if not ends:
            return after_hdr
        return after_hdr[:ends[-1].start()]

    @classmethod
    def body_block(cls, content: str) -> str:
        """
        Extrae el contenido del BODY del paquete.

        Estrategia: busca el header del BODY, luego toma hasta el
        ÚLTIMO END word; del fichero. En un cuerpo de paquete con N
        subprogramas habrá N END xxx; — el último es el cierre del paquete.
        """
        # Use neutralized text for searching to avoid issues with large comments
        neutralized = PlsqlTextCleaner.neutralize(content)
        body_hdr_m = cls._RE_BODY_HDR.search(neutralized)
        if not body_hdr_m:
            return ''

        # Extract from original content using the same positions
        after_hdr = content[body_hdr_m.end():]
        ends = list(cls._RE_END_BLOCK.finditer(after_hdr))
        if not ends:
            return after_hdr
        return after_hdr[:ends[-1].start()]

    @classmethod
    def subprograms_from_spec(cls, spec_raw: str) -> Dict[str, List[str]]:
        """Extrae procedures y functions declarados en el SPEC (fuente de verdad)."""
        if not spec_raw:
            return {'procedures': [], 'functions': []}
        c = PlsqlTextCleaner.clean_full(spec_raw)
        procs: List[str] = []
        funcs: List[str] = []
        for m in re.finditer(r'\bPROCEDURE\s+([A-Z_][\w$#]*)\b', c, re.IGNORECASE):
            procs.append(m.group(1).upper())
        for m in re.finditer(r'\bFUNCTION\s+([A-Z_][\w$#]*)\b', c, re.IGNORECASE):
            funcs.append(m.group(1).upper())
        return {
            'procedures': list(dict.fromkeys(procs)),
            'functions': list(dict.fromkeys(funcs)),
        }

    @classmethod
    def declared_cursors(cls, body_raw: str) -> Set[str]:
        """Cursores declarados en el body — no son invocaciones externas."""
        c = PlsqlTextCleaner.clean_full(body_raw or '')
        cursors: Set[str] = set()
        for m in re.finditer(r'\bCURSOR\s+([A-Z_][\w$#]*)\b', c, re.IGNORECASE):
            cursors.add(m.group(1).upper())
        return cursors

    @classmethod
    def local_type_names(cls, spec_raw: str, body_raw: str) -> Set[str]:
        """TYPE names definidos en SPEC o BODY — no son llamadas."""
        combined = PlsqlTextCleaner.clean_full(
            (spec_raw or '') + ' ' + (body_raw or '')
        )
        types: Set[str] = set()
        for m in re.finditer(r'\bTYPE\s+([A-Z_][\w$#]*)\s+IS\b', combined, re.IGNORECASE):
            types.add(m.group(1).upper())
        return types

    @classmethod
    def split_body_into_blocks(cls, body_raw: str) -> List[Dict]:
        """
        Divide el BODY en un bloque por cada subprograma.

        Usa texto limpio para detectar cabeceras, pero extrae slices del
        texto original para que el detector trabaje con el fuente íntegro.
        """
        if not body_raw:
            return []

        # Usamos texto neutralizado (comentarios -> espacios) para encontrar
        # los offsets reales dentro de body_raw.
        neutralized = PlsqlTextCleaner.neutralize(body_raw)

        indices: List[Dict] = []

        # Con parámetros: PROCEDURE|FUNCTION nombre(
        re1 = re.compile(
            r'\b(PROCEDURE|FUNCTION)\s+([A-Z_][\w$#]*)\s*\(',
            re.IGNORECASE,
        )
        # Sin parámetros: PROCEDURE|FUNCTION nombre AS|IS
        re2 = re.compile(
            r'\b(PROCEDURE|FUNCTION)\s+([A-Z_][\w$#]*)\s+(?:AS|IS)\b',
            re.IGNORECASE,
        )

        for m in re1.finditer(neutralized):
            indices.append({'start': m.start(), 'name': m.group(2).upper()})
        for m in re2.finditer(neutralized):
            indices.append({'start': m.start(), 'name': m.group(2).upper()})

        indices.sort(key=lambda x: x['start'])

        # Deduplicar: misma posición o mismo nombre a distancia ≤ 10 chars
        deduped: List[Dict] = []
        for idx in indices:
            prev = deduped[-1] if deduped else None
            if prev and prev['start'] == idx['start']:
                continue
            if prev and prev['name'] == idx['name'] and abs(idx['start'] - prev['start']) <= 10:
                continue
            deduped.append(idx)

        blocks: List[Dict] = []
        for i, cur in enumerate(deduped):
            name_esc = re.escape(cur['name'])
            # Buscar el final explícito "END <nombre>;" en el texto neutralizado
            end_match = re.search(r'\bEND\s+' + name_esc + r'\s*;', neutralized[cur['start']:], re.IGNORECASE)
            
            if end_match:
                end = cur['start'] + end_match.end()
            else:
                end = deduped[i + 1]['start'] if i + 1 < len(deduped) else len(body_raw)
                
            blocks.append({
                'name': cur['name'],
                'text': body_raw[cur['start']:end],
            })

        return blocks


# =============================================================================
# § 6  Orquestador principal
# =============================================================================

class PlsqlPackageAnalyzer:
    """
    Analiza el contenido de un paquete PL/SQL y produce un PackageAnalysisResult.

    Puede recibir la ruta del archivo o su contenido directamente.
    """

    def __init__(
        self,
        known_schemas: Optional[List[str]] = None,
        include_trace_packages: bool = False,
        trace_packages: Optional[List[str]] = None,
        # Nueva funcionalidad: índice global de subprogramas para detección precisa
        known_subprograms: Optional[List[str]] = None,
    ):
        self.known_schemas: Set[str] = {
            s.upper() for s in (known_schemas or [])
        }
        self.include_trace_packages = include_trace_packages
        self.trace_packages: Set[str] = {
            s.upper() for s in (trace_packages or ['TRAZAS'])
        }
        # Índice global de subprogramas conocidos
        self.known_subprograms: Set[str] = {
            s.upper() for s in (known_subprograms or [])
        }

    def analyze_file(self, file_path: str | Path) -> PackageAnalysisResult:
        """Lee un archivo PL/SQL y lo analiza."""
        path = Path(file_path)
        content: Optional[str] = None

        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                content = path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            result = PackageAnalysisResult(
                package_name=path.stem.upper(),
                file_path=str(path),
            )
            result.warnings.append(f"No se pudo decodificar el archivo: {path}")
            return result

        result = self.analyze_content(content)
        result.file_path = str(path)
        return result

    def analyze_content(
        self,
        content: str,
        default_name: str = 'PACKAGE_UNKNOWN',
    ) -> PackageAnalysisResult:
        """Analiza el contenido PL/SQL de un paquete."""
        extractor = _PlsqlStructureExtractor

        pkg_name = extractor.package_name(content) or default_name.upper()
        spec_raw = extractor.spec_block(content)
        body_raw = extractor.body_block(content)

        result = PackageAnalysisResult(
            package_name=pkg_name,
            file_path=None,
        )

        if not spec_raw and not body_raw:
            result.warnings.append("No se encontró SPEC ni BODY en el contenido.")
            return result

        subprograms = extractor.subprograms_from_spec(spec_raw)
        result.procedures = subprograms['procedures']
        result.functions = subprograms['functions']

        subprogram_index: Set[str] = set(result.procedures) | set(result.functions)

        declared_cursors = extractor.declared_cursors(body_raw)
        local_types = extractor.local_type_names(spec_raw, body_raw)

        body_blocks = extractor.split_body_into_blocks(body_raw)
        all_local_subprograms = subprogram_index | {b['name'] for b in body_blocks}
        result.private_subprograms = sorted(list({b['name'] for b in body_blocks} - subprogram_index))

        detector = PlsqlCallDetector(
            pkg_name=pkg_name,
            subprogram_index=subprogram_index,
            declared_cursors=declared_cursors,
            local_types=local_types,
            known_schemas=self.known_schemas,
            include_trace_packages=self.include_trace_packages,
            trace_packages=self.trace_packages,
            all_local_subprograms=all_local_subprograms,
            known_subprograms=self.known_subprograms,
        )

        for block in body_blocks:
            name: str = block['name']
            # Permitir mapear las llamadas de subprogramas públicos y privados
            raw_calls = detector.detect(block['text'])
            self_ref = f'{pkg_name}.{name}'
            filtered = [c for c in raw_calls if c != self_ref]
            kind = 'PROCEDURE' if name in result.procedures else 'FUNCTION' if name in result.functions else 'PRIVATE_SUBPROGRAM'
            
            if name in result.calls_by_subprogram:
                # Agregado para soportar SOBRECARGA: combina llamadas de distintas instancias
                existing_calls = result.calls_by_subprogram[name].calls
                combined = sorted(list(set(existing_calls + filtered)))
                result.calls_by_subprogram[name].calls = combined
            else:
                result.calls_by_subprogram[name] = SubprogramCalls(
                    name=name,
                    kind=kind,
                    calls=filtered,
                )

        # Subprogramas del SPEC sin bloque en el BODY
        for sp in subprogram_index:
            if sp not in result.calls_by_subprogram:
                kind = 'PROCEDURE' if sp in result.procedures else 'FUNCTION'
                result.calls_by_subprogram[sp] = SubprogramCalls(
                    name=sp,
                    kind=kind,
                    calls=[],
                )
                result.warnings.append(
                    f"Subprograma '{sp}' declarado en SPEC pero sin bloque en BODY."
                )

        return result


# =============================================================================
# § 7  Generador de Markdown
# =============================================================================

class MarkdownDocGenerator:
    """Convierte un PackageAnalysisResult en documentación Markdown."""

    def __init__(
        self,
        use_obsidian_links: bool = False,
        synonym_map: Optional[Dict[str, str] | List] = None,
        known_schemas: Optional[Set[str]] = None,
    ):
        self.use_obsidian_links = use_obsidian_links
        self.synonym_map = self._normalize_synonym_map(synonym_map)
        # Asegurar que known_schemas sean mayúsculas para comparaciones consistentes
        self.known_schemas = {s.upper() for s in (known_schemas or [])}

    @staticmethod
    def _normalize_synonym_map(synonym_data: Optional[Dict[str, str] | List]) -> Dict[str, str]:
        """Normaliza el mapeo de sinónimos desde dict o list."""
        if not synonym_data:
            return {}
        
        normalized = {}
        if isinstance(synonym_data, dict):
            normalized = {k.upper(): v.upper() for k, v in synonym_data.items()}
        elif isinstance(synonym_data, list):
            for item in synonym_data:
                if isinstance(item, list) and len(item) >= 2:
                    normalized[str(item[0]).upper()] = str(item[1]).upper()
                elif isinstance(item, dict):
                    # Soporta {"synonym": "...", "target": "..."} o {"name": "...", "target": "..."}
                    s = item.get('synonym') or item.get('name')
                    t = item.get('target') or item.get('object')
                    if s and t:
                        normalized[str(s).upper()] = str(t).upper()
                    else:
                        # De lo contrario, tratar todas las llaves del dict como mapeos directos
                        for k, v in item.items():
                            normalized[str(k).upper()] = str(v).upper()
        return normalized

    def generate(self, result: PackageAnalysisResult) -> str:
        """Genera el Markdown completo para un paquete analizado."""
        pkg = result.package_name
        lines: List[str] = [
            f'---',
            f'tags:',
            f'  - Programa-PLSQL',
            f'---',
            f'# {pkg}',
            '',
            f'El objeto {pkg} es el encargado de controlar la lógica.',
        ]

        if result.procedures:
            lines.append('')
            lines.append('## Procedimientos')
            for name in result.procedures:
                lines.append('')
                lines.append(self._format_subprogram(name, result))

        if result.functions:
            lines.append('')
            lines.append('## Funciones')
            for name in result.functions:
                lines.append('')
                lines.append(self._format_subprogram(name, result))

        if getattr(result, 'private_subprograms', None):
            lines.append('')
            lines.append('## Procedimientos Internos')
            for name in result.private_subprograms:
                lines.append('')
                lines.append(self._format_subprogram(name, result))

        return '\n'.join(lines).strip() + '\n'

    def _strip_known_schema(self, full_name: str) -> str:
        """Elimina el prefijo si coincide con un esquema conocido (ej. TRON2000.PKG -> PKG)."""
        if '.' in full_name:
            parts = full_name.split('.')
            if parts[0].upper() in self.known_schemas:
                return '.'.join(parts[1:])
        return full_name

    def _format_subprogram(self, name: str, result: PackageAnalysisResult) -> str:
        """Formatea un bloque ### para un subprograma."""
        sub = result.calls_by_subprogram.get(name)
        calls = sub.calls if sub else []
        if self.use_obsidian_links:
            formatted_calls = []
            for c in calls:
                if '.' in c:
                    pkg_part, sub_part = c.rsplit('.', 1)
                    # Resolve synonym if exists
                    pkg_part = self.synonym_map.get(pkg_part.upper(), pkg_part)
                    # Limpiar esquema si el target del sinónimo o el original lo tiene
                    pkg_part = self._strip_known_schema(pkg_part)
                    formatted_calls.append(f'[[{pkg_part}#{sub_part}]]')
                else:
                    # Possible standalone synonym?
                    res_c = self.synonym_map.get(c.upper(), c)
                    # Limpiar esquema si el sinónimo lo convirtió en SCHEMA.OBJETO
                    res_c = self._strip_known_schema(res_c)
                    if '.' in res_c:
                        # Si sigue teniendo un punto después de stripping, usar formato PKG#SUB
                        formatted_calls.append(f'[[{res_c.replace(".", "#")}]]')
                    else:
                        formatted_calls.append(f'[[{res_c}]]')
            calls_md = '\n'.join(f'- {fc}' for fc in formatted_calls)
        else:
            # Aplicar sinónimos incluso sin obsidian links
            formatted_calls = []
            for c in calls:
                if '.' in c:
                    pkg_part, sub_part = c.rsplit('.', 1)
                    # Resolve synonym if exists
                    pkg_part = self.synonym_map.get(pkg_part.upper(), pkg_part)
                    # Limpiar esquema si el target del sinónimo o el original lo tiene
                    pkg_part = self._strip_known_schema(pkg_part)
                    formatted_calls.append(f'{pkg_part}.{sub_part}')
                else:
                    # Possible standalone synonym?
                    res_c = self.synonym_map.get(c.upper(), c)
                    # Limpiar esquema si el sinónimo lo convirtió en SCHEMA.OBJETO
                    res_c = self._strip_known_schema(res_c)
                    formatted_calls.append(res_c)
            calls_md = '\n'.join(f'- {fc}' for fc in formatted_calls)

        calls_md = calls_md if calls else '- (sin invocaciones detectadas)'
        return (
            f'### {name}\n'
            f'Invoca a los siguientes programas:\n'
            f'{calls_md}'
        )
