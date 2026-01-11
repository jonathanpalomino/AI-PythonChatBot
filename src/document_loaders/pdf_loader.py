# =============================================================================
# src/document_loaders/pdf_loader.py
# =============================================================================
"""
Loader para documentos PDF
"""
import re
from pathlib import Path
from typing import List, Dict, Any

import PyPDF2

from .base_loader import BaseDocumentLoader, DocumentSection, ProcessedDocument


class PDFLoader(BaseDocumentLoader):
    """Carga y procesa documentos PDF"""

    def __init__(self):
        super().__init__()
        self.supported_extensions = {'.pdf'}

    def load(self, file_path: Path, original_filename: str = None) -> ProcessedDocument:
        """Carga un documento PDF"""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)

            # Extraer metadata
            metadata = self._extract_metadata(pdf_reader)

            # Extraer contenido completo de todas las páginas
            full_text = self._extract_all_text(pdf_reader)

            # Extraer secciones basadas en el contenido
            sections = self.extract_sections(full_text)

            # Generar contenido completo desde las secciones
            content = self._generate_full_content(sections)

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
                content=content,
                sections=sections,
                metadata=metadata
            )

    def _extract_metadata(self, pdf_reader: PyPDF2.PdfReader) -> Dict[str, Any]:
        """Extrae metadata del documento PDF"""
        metadata = {
            'author': '',
            'title': '',
            'subject': '',
            'creator': '',
            'producer': '',
            'created': '',
            'modified': '',
            'pages': len(pdf_reader.pages)
        }

        if pdf_reader.metadata:
            metadata['author'] = pdf_reader.metadata.get('/Author', '') or ''
            metadata['title'] = pdf_reader.metadata.get('/Title', '') or ''
            metadata['subject'] = pdf_reader.metadata.get('/Subject', '') or ''
            metadata['creator'] = pdf_reader.metadata.get('/Creator', '') or ''
            metadata['producer'] = pdf_reader.metadata.get('/Producer', '') or ''

            # Fechas de creación y modificación
            created = pdf_reader.metadata.get('/CreationDate', '')
            if created:
                metadata['created'] = self._parse_pdf_date(created)

            modified = pdf_reader.metadata.get('/ModDate', '')
            if modified:
                metadata['modified'] = self._parse_pdf_date(modified)

        return metadata

    def _parse_pdf_date(self, date_str: str) -> str:
        """Convierte fecha PDF (D:YYYYMMDDHHmmSS) a formato ISO"""
        try:
            # Formato típico: D:20230615120000+00'00'
            if date_str.startswith('D:'):
                date_str = date_str[2:]

            # Extraer componentes
            if len(date_str) >= 14:
                year = date_str[0:4]
                month = date_str[4:6]
                day = date_str[6:8]
                hour = date_str[8:10]
                minute = date_str[10:12]
                second = date_str[12:14]
                return f"{year}-{month}-{day}T{hour}:{minute}:{second}"
        except:
            pass

        return str(date_str)

    def _extract_all_text(self, pdf_reader: PyPDF2.PdfReader) -> str:
        """Extrae todo el texto del PDF"""
        all_text = []

        for page_num, page in enumerate(pdf_reader.pages, 1):
            try:
                text = page.extract_text()
                if text.strip():
                    # Agregar marcador de página para referencia
                    all_text.append(f"[PÁGINA {page_num}]")
                    all_text.append(text.strip())
                    all_text.append("")  # Línea en blanco entre páginas
            except Exception as e:
                print(f"Error extrayendo texto de página {page_num}: {e}")
                continue

        return '\n'.join(all_text)

    def extract_sections(self, content: str) -> List[DocumentSection]:
        """
        Extrae secciones del contenido basándose en patrones comunes de encabezados.
        Detecta:
        - Títulos en mayúsculas
        - Títulos numerados (1. 2. 3. o 1.1, 1.2, etc.)
        - Líneas cortas seguidas de contenido (probables títulos)
        """
        sections = []
        lines = content.split('\n')

        current_section = None
        current_content = []
        current_level = 0
        pre_heading_content = []

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Skip marcadores de página
            if line.startswith('[PÁGINA'):
                i += 1
                continue

            # Skip líneas vacías
            if not line:
                i += 1
                continue

            # Detectar posibles encabezados
            is_heading, level, clean_title = self._is_heading(line, i, lines)

            if is_heading:
                # Guardar contenido antes del primer heading
                if current_section is None and pre_heading_content:
                    intro_content = '\n\n'.join(pre_heading_content).strip()
                    if intro_content:
                        sections.append(DocumentSection(
                            title="Introduction",
                            content=intro_content,
                            level=0,
                            metadata={}
                        ))
                    pre_heading_content = []

                # Guardar sección anterior
                if current_section:
                    section_content = '\n\n'.join(current_content).strip()
                    if section_content:
                        sections.append(DocumentSection(
                            title=current_section,
                            content=section_content,
                            level=current_level,
                            metadata={}
                        ))

                # Nueva sección
                current_section = clean_title
                current_level = level
                current_content = []
            else:
                # Agregar contenido a la sección actual
                if current_section:
                    current_content.append(line)
                else:
                    pre_heading_content.append(line)

            i += 1

        # Guardar contenido previo si no hay headings
        if current_section is None and pre_heading_content:
            intro_content = '\n\n'.join(pre_heading_content).strip()
            if intro_content:
                sections.append(DocumentSection(
                    title="Document Content",
                    content=intro_content,
                    level=0,
                    metadata={}
                ))

        # Última sección
        if current_section:
            section_content = '\n\n'.join(current_content).strip()
            if section_content:
                sections.append(DocumentSection(
                    title=current_section,
                    content=section_content,
                    level=current_level,
                    metadata={}
                ))

        return sections

    def _is_heading(self, line: str, line_idx: int, all_lines: List[str]) -> tuple:
        """
        Determina si una línea es un encabezado.
        Retorna: (es_heading, nivel, título_limpio)
        """
        # Patrón 1: Numeración estilo 1. 2. 3. o 1.1, 1.2, etc.
        numbered_pattern = r'^(\d+(?:\.\d+)*)\.\s+(.+)$'
        match = re.match(numbered_pattern, line)
        if match:
            numbering = match.group(1)
            title = match.group(2)
            level = numbering.count('.') + 1
            return True, level, title.strip()

        # Patrón 2: Línea completamente en mayúsculas (mínimo 3 palabras)
        if line.isupper() and len(line.split()) >= 2 and len(line) < 100:
            return True, 1, line.strip()

        # Patrón 3: Línea corta (<80 chars) seguida de contenido más largo
        # (posible título)
        if len(line) < 80 and not line.endswith(('.', ',', ';', ':')):
            # Verificar si la siguiente línea es más larga (contenido)
            if line_idx + 1 < len(all_lines):
                next_line = all_lines[line_idx + 1].strip()
                if next_line and len(next_line) > len(line) * 1.5:
                    # Verificar que no sea una lista o viñeta
                    if not re.match(r'^[-•*]\s', line) and not re.match(r'^\d+\)', line):
                        # Verificar que tenga características de título
                        # (primera letra mayúscula, no termina en puntuación)
                        if line[0].isupper() and not line.endswith((',', ';')):
                            return True, 2, line.strip()

        return False, 0, line

    def _generate_full_content(self, sections: List[DocumentSection]) -> str:
        """Genera el contenido completo del documento desde las secciones"""
        full_content = []

        for section in sections:
            # Agregar título de sección con formato markdown
            header_prefix = '#' * (section.level if section.level > 0 else 1)
            full_content.append(f"{header_prefix} {section.title}")
            full_content.append(section.content)
            full_content.append("")  # Línea en blanco entre secciones

        return '\n'.join(full_content).strip()
