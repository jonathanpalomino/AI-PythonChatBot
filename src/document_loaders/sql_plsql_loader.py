# =============================================================================
# src/document_loaders/sql_plsql_loader.py
# =============================================================================
"""
Loader para archivos SQL (.sql) y PL/SQL
"""
from pathlib import Path
from typing import List, Dict, Any
import re

# Importar desde el archivo base_loader.py (asumiendo que está en el mismo nivel o ruta)
from .base_loader import BaseDocumentLoader, DocumentSection, ProcessedDocument 

# Expresiones regulares para identificar bloques de PL/SQL
PLSQL_BLOCK_START = re.compile(
    r'^\s*(?:CREATE|CREATE\s+OR\s+REPLACE)\s+'
    r'(PACKAGE|PACKAGE\s+BODY|FUNCTION|PROCEDURE|TYPE|TRIGGER|VIEW|MATERIALIZED\s+VIEW|SEQUENCE|TABLE|INDEX)\s+'
    r'(\w+\.?\w+)', 
    re.IGNORECASE | re.MULTILINE | re.DOTALL
)

class SqlPlsqlLoader(BaseDocumentLoader):
    """Carga y procesa archivos SQL y PL/SQL"""

    def __init__(self):
        super().__init__()
        # Se añaden extensiones comunes para archivos SQL/PLSQL
        self.supported_extensions = {'.sql', '.pls', '.pks', '.pkb', '.prc', '.fnc','.plsql'}

    def load(self, file_path: Path, original_filename: str = None) -> ProcessedDocument:
        """Carga un archivo SQL/PLSQL"""
        # Se reutiliza la lógica de carga con múltiples encodings del TextLoader
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

        # Generar contenido completo (opcional: reconstruir con títulos, similar a TextLoader)
        full_content = self._generate_full_content(sections)

        # Convertir a ruta relativa (reutilizando la lógica de TextLoader)
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
            metadata=metadata
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
        Extrae secciones de código SQL/PLSQL.
        Prioriza la extracción de bloques PL/SQL. El resto se divide por ';'.
        """
        sections = []
        remaining_content = content.strip()
        current_position = 0
        section_number = 1

        while remaining_content:
            match = PLSQL_BLOCK_START.search(remaining_content)

            if match:
                # 1. Procesar contenido anterior como secciones SQL simples
                pre_block_content = remaining_content[:match.start()].strip()
                self._process_simple_statements(
                    pre_block_content, sections, start_num=section_number
                )
                section_number = len(sections) + 1
                
                # 2. Encontrar el final del bloque PL/SQL
                block_type = match.group(1).upper()
                block_name = match.group(2)
                block_start_index = match.start()
                
                # Buscar el ';' que cierra la definición del bloque. 
                # Esto es heurístico y podría fallar en casos complejos de sintaxis.
                end_match = re.search(r';\s*(?:/\*|--|\n|$)', remaining_content[match.end():], re.IGNORECASE)
                
                if end_match:
                    block_end_index = match.end() + end_match.start() + 1
                    block_content = remaining_content[block_start_index:block_end_index].strip()
                    remaining_content = remaining_content[block_end_index:].strip()
                else:
                    # Si no encuentra el final, trata el resto del archivo como el bloque
                    block_content = remaining_content[block_start_index:].strip()
                    remaining_content = ""

                # 3. Crear la sección del bloque PL/SQL
                sections.append(DocumentSection(
                    title=f"{block_type.replace(' ', '_')}: {block_name}",
                    content=block_content,
                    level=1,
                    metadata={'type': block_type, 'name': block_name, 'section_number': section_number}
                ))
                section_number += 1
                
            else:
                # Si no se encuentran más bloques PL/SQL, procesar el resto como sentencias SQL simples
                self._process_simple_statements(
                    remaining_content, sections, start_num=section_number
                )
                remaining_content = "" # Termina el bucle

        return sections

    def _process_simple_statements(self, content: str, sections: List[DocumentSection], start_num: int):
        """Divide el contenido restante por punto y coma (;) como sentencias SQL simples"""
        if not content:
            return

        # Limpiar comentarios y dividir por ';'
        # Simplificación: No maneja delimitadores '$$' o 'SET SERVEROUTPUT ON;'
        statements = re.split(r';\s*\n*', content) 
        
        current_section_num = start_num
        
        for stmt in statements:
            stmt = stmt.strip()
            if not stmt:
                continue

            # Usar la primera línea como título si es corta, sino un título genérico
            first_line = stmt.split('\n')[0].strip()
            title = f"SQL_Statement_{current_section_num}"
            
            if len(first_line) < 100:
                 title = first_line # Usar el inicio de la sentencia como título

            sections.append(DocumentSection(
                title=title,
                content=stmt + ';', # Añadir el ';' de vuelta para claridad
                level=2, # Nivel inferior a los bloques PL/SQL
                metadata={'type': 'SQL_STATEMENT', 'section_number': current_section_num}
            ))
            current_section_num += 1


    def _generate_full_content(self, sections: List[DocumentSection]) -> str:
        """Genera el contenido completo con títulos de sección (similar a TextLoader)"""
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