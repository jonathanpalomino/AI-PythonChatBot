# =============================================================================
# src/document_loaders/sql_plsql_loader.py
# =============================================================================
"""
Loader para archivos SQL (.sql) y PL/SQL con detección inteligente de bloques
"""
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.config.constants import CODE_CHUNK_SIZE
# Importar desde el archivo base_loader.py
from .base_loader import BaseDocumentLoader, DocumentSection, ProcessedDocument

# Expresiones regulares para identificar bloques de PL/SQL
PLSQL_BLOCK_START = re.compile(
    r'^\s*(?:CREATE|CREATE\s+OR\s+REPLACE)\s+'
    r'(PACKAGE\s+BODY|PACKAGE|FUNCTION|PROCEDURE|TYPE\s+BODY|TYPE|TRIGGER|VIEW|MATERIALIZED\s+VIEW|SEQUENCE|TABLE|INDEX)\s+'
    r'([\w\.]+)',  # Captura nombres con schema: owner.object_name
    re.IGNORECASE | re.MULTILINE
)


class PlsqlBlockParser:
    """
    Parser inteligente para bloques PL/SQL.
    Detecta estructuras anidadas usando conteo de BEGIN/END.
    """

    def __init__(self, content: str):
        self.content = content
        self.lines = content.split('\n')
        self.pos = 0

    def find_block_end(self, start_pos: int, block_type: str, block_name: Optional[str] = None) -> int:
        """
        Encuentra el END que cierra un bloque PL/SQL.

        Estrategia:
        1. Para PACKAGE/PACKAGE BODY: busca específicamente END <nombre_package>;
        2. Para PROCEDURE/FUNCTION standalone: cuenta BEGIN/END anidados
        3. Para otros DDL (TABLE, INDEX, etc.): busca el primer ';'

        Args:
            start_pos: Posición inicial del bloque
            block_type: Tipo de bloque (PACKAGE_BODY, PROCEDURE, etc.)
            block_name: Nombre del bloque

        Returns:
            Posición del carácter después del ';' que cierra el bloque
        """
        search_text = self.content[start_pos:]
        block_type_clean = block_type.replace('_', ' ')

        # Para PACKAGE o PACKAGE BODY, buscar END <nombre>;
        if 'PACKAGE' in block_type:
            # Extraer solo el nombre del paquete (sin schema si lo hubiera)
            simple_name = block_name.split('.')[-1] if block_name else None

            if simple_name:
                # Buscar patrón END <nombre>; (case insensitive)
                end_pattern = re.compile(
                    rf'\bEND\s+{re.escape(simple_name)}\s*;',
                    re.IGNORECASE | re.DOTALL
                )
                match = end_pattern.search(search_text)
                if match:
                    return start_pos + match.end()

        # Para PROCEDURE/FUNCTION/TRIGGER, buscar con conteo de BEGIN/END
        if block_type in ('PROCEDURE', 'FUNCTION', 'TRIGGER'):
            depth = 0
            current_pos = 0

            begin_pattern = re.compile(r'\bBEGIN\b', re.IGNORECASE)
            end_pattern = re.compile(r'\bEND\b', re.IGNORECASE)

            # Buscar primer BEGIN
            begin_match = begin_pattern.search(search_text)
            if begin_match:
                depth = 1
                current_pos = begin_match.end()

                # Buscar ENDs correspondientes
                while depth > 0 and current_pos < len(search_text):
                    next_begin = begin_pattern.search(search_text, current_pos)
                    next_end = end_pattern.search(search_text, current_pos)

                    if next_end is None:
                        break

                    # Determinar qué llega primero
                    if next_begin and next_begin.start() < next_end.start():
                        depth += 1
                        current_pos = next_begin.end()
                    else:
                        depth -= 1
                        if depth == 0:
                            # Buscar el ; que cierra este END
                            semicolon_pos = search_text.find(';', next_end.end())
                            if semicolon_pos != -1:
                                return start_pos + semicolon_pos + 1
                        current_pos = next_end.end()

        # Para DDL simples (TABLE, INDEX, VIEW, etc.), buscar primer ';'
        semicolon_match = re.search(r';', search_text)
        if semicolon_match:
            return start_pos + semicolon_match.end()

        # Por defecto, retornar el final del contenido
        return len(self.content)


class SqlPlsqlLoader(BaseDocumentLoader):
    """Carga y procesa archivos SQL y PL/SQL con detección inteligente de bloques"""

    def __init__(self):
        super().__init__()
        # Se añaden extensiones comunes para archivos SQL/PLSQL
        self.supported_extensions = {
            '.sql', '.pls', '.pks', '.pkb', '.prc', '.fnc', '.plsql',
            '.pck', '.spc', '.bdy', '.typ', '.tps', '.tpb'
        }

    def load(self, file_path: Path, original_filename: str = None) -> ProcessedDocument:
        """Carga un archivo SQL/PLSQL"""
        # Se reutiliza la lógica de carga con múltiples encodings
        content = None
        encoding_used = 'utf-8'

        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                encoding_used = encoding
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            raise ValueError(f"No se pudo decodificar el archivo {file_path}")

        # Extraer metadatos y secciones
        metadata = self._extract_metadata(content, encoding_used)
        sections = self.extract_sections(content)

        # Generar contenido completo
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
            metadata=metadata,
            recommended_chunk_size=CODE_CHUNK_SIZE  # Large chunks to preserve complete PL/SQL units
        )

    def _extract_metadata(self, content: str, encoding: str) -> Dict[str, Any]:
        """Extrae metadatos básicos del contenido SQL/PLSQL"""
        lines = content.split('\n')
        # Contar sentencias SQL/PLSQL (aproximación: número de ';' en líneas no vacías)
        statement_count = content.count(';')

        return {
            'line_count': len(lines),
            'char_count': len(content),
            'non_empty_lines': len([l for l in lines if l.strip()]),
            'encoding': encoding,
            'estimated_statement_count': statement_count
        }

    def extract_sections(self, content: str) -> List[DocumentSection]:
        """
        Extrae secciones de código SQL/PLSQL usando detección inteligente.

        Estrategia:
        1. Detectar bloques PL/SQL (PACKAGE BODY, PROCEDURE standalone, FUNCTION standalone, etc.)
        2. Para PACKAGE BODY: extraer procedures/functions individuales como secciones separadas
        3. Agrupar sentencias SQL simples en secciones más grandes (~1000 chars)
        """
        sections = []
        parser = PlsqlBlockParser(content)
        remaining_content = content.strip()
        current_position = 0
        section_number = 1

        while current_position < len(content):
            remaining = content[current_position:].strip()
            if not remaining:
                break

            match = PLSQL_BLOCK_START.search(remaining)

            if match:
                # 1. Procesar contenido anterior como sentencias SQL simples
                pre_block_content = remaining[:match.start()].strip()
                if pre_block_content:
                    self._add_grouped_statements(
                        pre_block_content, sections, start_num=section_number
                    )
                    section_number = len(sections) + 1

                # 2. Extraer bloque PL/SQL completo
                block_type = match.group(1).upper().replace(' ', '_')
                block_name = match.group(2)
                block_start_in_remaining = match.start()

                # Usar el parser inteligente para encontrar el final
                block_end_in_content = parser.find_block_end(
                    current_position + match.start(),
                    block_type,
                    block_name
                )

                block_content = content[current_position + block_start_in_remaining:block_end_in_content].strip()

                # 3. Si es PACKAGE BODY, extraer procedures/functions individuales
                if block_type == 'PACKAGE_BODY':
                    # Extraer unidades internas (procedures, functions)
                    internal_sections = self._extract_package_internals(
                        block_content,
                        block_name,
                        section_number
                    )
                    if internal_sections:
                        sections.extend(internal_sections)
                        section_number = len(sections) + 1
                    else:
                        # Si no se encontraron internos, crear sección del package completo
                        sections.append(DocumentSection(
                            title=f"{block_type}: {block_name}",
                            content=block_content,
                            level=1,
                            metadata={'type': block_type, 'name': block_name, 'section_number': section_number}
                        ))
                        section_number += 1
                else:
                    # Para otros tipos (PROCEDURE standalone, FUNCTION standalone, etc.)
                    sections.append(DocumentSection(
                        title=f"{block_type}: {block_name}",
                        content=block_content,
                        level=1,
                        metadata={'type': block_type, 'name': block_name, 'section_number': section_number}
                    ))
                    section_number += 1

                # Avanzar la posición
                current_position = block_end_in_content

            else:
                # No hay más bloques PL/SQL, procesar el resto como SQL simple
                self._add_grouped_statements(
                    remaining, sections, start_num=section_number
                )
                break

        return sections

    def _extract_package_internals(
        self,
        package_content: str,
        package_name: str,
        start_section_num: int
    ) -> List[DocumentSection]:
        """
        Extrae procedures y functions individuales de un package body.

        Estrategia simple:
        1. Encontrar todos los PROCEDURE/FUNCTION
        2. El contenido de cada uno va desde su inicio hasta el inicio del siguiente
           (o hasta el END del package si es el último)

        Args:
            package_content: Contenido completo del package body
            package_name: Nombre del package
            start_section_num: Número inicial de sección

        Returns:
            Lista de secciones, una por cada procedure/function
        """
        sections = []
        section_num = start_section_num

        # Patrones para detectar inicio de PROCEDURE/FUNCTION dentro del package
        subprogram_pattern = re.compile(
            r'^\s*(PROCEDURE|FUNCTION)\s+([\w]+)',
            re.IGNORECASE | re.MULTILINE
        )

        # Buscar todas las ocurrencias
        matches = list(subprogram_pattern.finditer(package_content))

        if not matches:
            return []  # No se encontraron subprogramas

        # Encontrar el final del package (END <package_name>;)
        package_end_pattern = re.compile(
            rf'\bEND\s+{re.escape(package_name)}\s*;',
            re.IGNORECASE | re.DOTALL
        )
        package_end_match = package_end_pattern.search(package_content)
        package_end_pos = package_end_match.start() if package_end_match else len(package_content)

        # Extraer cada subprograma usando límites: desde su inicio hasta el inicio del siguiente
        for i, match in enumerate(matches):
            subprogram_type = match.group(1).upper()
            subprogram_name = match.group(2)
            subprogram_start = match.start()

            # El final es el inicio del siguiente subprograma, o el END del package
            if i + 1 < len(matches):
                subprogram_end = matches[i + 1].start()
            else:
                subprogram_end = package_end_pos

            subprogram_content = package_content[subprogram_start:subprogram_end].strip()

            sections.append(DocumentSection(
                title=f"{subprogram_type}: {package_name}.{subprogram_name}",
                content=subprogram_content,
                level=2,  # Level 2 porque es interno al package
                metadata={
                    'type': subprogram_type,
                    'name': subprogram_name,
                    'parent_package': package_name,
                    'section_number': section_num
                }
            ))
            section_num += 1

        return sections

    def _add_grouped_statements(
        self,
        content: str,
        sections: List[DocumentSection],
        start_num: int,
        target_size: int = 1000
    ):
        """
        Agrupa sentencias SQL simples en secciones más grandes.

        En lugar de crear una sección por cada ';', agrupa sentencias
        hasta alcanzar un tamaño objetivo para evitar chunks demasiado pequeños.

        Args:
            content: Contenido SQL a agrupar
            sections: Lista de secciones donde añadir
            start_num: Número de sección inicial
            target_size: Tamaño objetivo de caracteres por sección
        """
        if not content.strip():
            return

        # Dividir por punto y coma
        statements = re.split(r';\s*\n*', content)

        current_section_num = start_num
        accumulated_statements = []
        accumulated_size = 0

        for stmt in statements:
            stmt = stmt.strip()
            # Skip empty or trivial statements
            if not stmt or stmt in ('/', '--', '/*', '*/'):
                continue

            # Añadir a acumulador
            accumulated_statements.append(stmt)
            accumulated_size += len(stmt)

            # Si alcanzamos el tamaño objetivo, crear sección
            if accumulated_size >= target_size:
                self._create_statement_section(
                    accumulated_statements,
                    sections,
                    current_section_num
                )
                current_section_num += 1
                accumulated_statements = []
                accumulated_size = 0

        # Crear sección final con lo que quede
        if accumulated_statements:
            self._create_statement_section(
                accumulated_statements,
                sections,
                current_section_num
            )

    def _create_statement_section(
        self,
        statements: List[str],
        sections: List[DocumentSection],
        section_num: int
    ):
        """Crea una sección a partir de un grupo de sentencias SQL."""
        if not statements:
            return

        # Usar la primera línea del primer statement como título
        first_line = statements[0].split('\n')[0].strip()
        if len(statements) > 1:
            title = f"{first_line}... (+{len(statements)-1} more)"
        else:
            title = first_line if len(first_line) < 100 else first_line[:97] + "..."

        # Reconstruir contenido con punto y coma
        content = ';\n\n'.join(statements) + ';'

        sections.append(DocumentSection(
            title=title,
            content=content,
            level=2,
            metadata={
                'type': 'SQL_STATEMENTS_GROUP',
                'statement_count': len(statements),
                'section_number': section_num
            }
        ))

    def _generate_full_content(self, sections: List[DocumentSection]) -> str:
        """Genera el contenido completo con títulos de sección"""
        full_content = []

        for section in sections:
            # Usar Markdown para el título de la sección
            if section.level == 1:
                 full_content.append(f"## {section.title}")
            elif section.level == 2:
                 full_content.append(f"### {section.title}")

            # Añadir el contenido del código
            full_content.append(section.content)
            full_content.append("\n" + "="*20 + "\n") # Separador visual entre secciones

        return '\n'.join(full_content).strip()
