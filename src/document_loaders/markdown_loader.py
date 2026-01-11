# =============================================================================
# src/document_loaders/markdown_loader.py
# =============================================================================
"""
Loader para archivos Markdown con soporte Obsidian
"""
import re
from pathlib import Path
from typing import List, Dict

import yaml

from .base_loader import BaseDocumentLoader, DocumentSection, ProcessedDocument


class MarkdownLoader(BaseDocumentLoader):
    """Carga y procesa archivos Markdown de Obsidian"""

    def __init__(self):
        super().__init__()
        self.supported_extensions = {'.md'}

    def load(self, file_path: Path, original_filename: str = None) -> ProcessedDocument:
        """Carga un archivo Markdown"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extraer metadata PRIMERO
        metadata = self._extract_metadata(content)

        # Extraer secciones (pasando metadata para enriquecerlas)
        sections = self.extract_sections(content, metadata)

        # Generar contenido completo desde las secciones
        full_content = self._generate_full_content(sections, metadata)

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
            metadata=metadata
        )

    def _extract_metadata(self, content: str) -> Dict:
        """Extrae metadata del documento (tags, frontmatter)"""
        metadata = {
            'tags': [],
            'frontmatter': {}
        }

        # Frontmatter YAML (debe ir PRIMERO)
        frontmatter = self._extract_frontmatter(content)
        if frontmatter:
            metadata['frontmatter'] = frontmatter

            # GENÉRICO: Promover campos de primer nivel a metadata directa
            # Esto permite acceso rápido sin navegar el dict anidado
            for key, value in frontmatter.items():
                # Solo promover valores simples (str, int, float, bool)
                if isinstance(value, (str, int, float, bool)):
                    metadata[f'fm_{key}'] = value  # Prefijo 'fm_' para distinguir origen

        # Tags inline (estilo Obsidian)
        tags = re.findall(r'#[\w-]+', content)
        metadata['tags'] = [tag[1:] for tag in tags]

        # GENÉRICO: Extraer título del documento (primer header de nivel 1)
        # Esto mejora la contextualización para RAG
        document_title = self._extract_document_title(content)
        if document_title:
            metadata['document_title'] = document_title

        return metadata

    def _extract_frontmatter(self, content: str) -> Dict:
        """Extrae frontmatter YAML si existe"""
        match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
        if not match:
            return {}

        yaml_content = match.group(1)

        try:
            # Usar PyYAML para parsear correctamente la estructura
            frontmatter = yaml.safe_load(yaml_content)
            return frontmatter if isinstance(frontmatter, dict) else {}
        except yaml.YAMLError as e:
            # Si falla el parsing, intentar método manual simplificado
            from src.utils.logger import get_logger
            logger = get_logger(__name__)
            logger.warning(f"Error parsing YAML frontmatter: {e}. Using fallback parser.")

            # Fallback: parsing manual básico
            frontmatter = {}
            current_key = None
            current_list = []

            for line in yaml_content.split('\n'):
                line = line.strip()
                if not line:
                    continue

                # Detectar clave: valor
                if ':' in line and not line.startswith('-'):
                    # Guardar lista anterior si existe
                    if current_key and current_list:
                        frontmatter[current_key] = current_list
                        current_list = []

                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()

                    if value:  # Tiene valor inline
                        frontmatter[key] = value
                        current_key = None
                    else:  # Es una lista
                        current_key = key

                # Detectar elemento de lista
                elif line.startswith('-') and current_key:
                    item = line[1:].strip()
                    current_list.append(item)

            # Guardar última lista
            if current_key and current_list:
                frontmatter[current_key] = current_list

            return frontmatter

    def _extract_document_title(self, content: str) -> str:
        """Extrae el título del documento desde el primer header de nivel 1 (#)

        Busca el primer header # ignorando el frontmatter YAML.
        Retorna el texto del header sin el símbolo #.

        Returns:
            Título del documento o cadena vacía si no se encuentra
        """
        # Remover frontmatter primero
        content_without_frontmatter = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)

        # Buscar primer header de nivel 1 (# titulo)
        # Permitir espacios opcionales antes del #
        match = re.search(r'^\s*#\s+(.+)$', content_without_frontmatter, re.MULTILINE)

        if match:
            title = match.group(1).strip()
            # Limpiar tags inline si hay en el título
            title = re.sub(r'#[\w-]+', '', title).strip()
            return title

        return ""

    def extract_sections(self, content: str, metadata: Dict = None) -> List[DocumentSection]:
        """Extrae secciones basadas en headers markdown"""
        sections = []

        # Remover frontmatter del contenido
        content_without_frontmatter = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)
        lines = content_without_frontmatter.split('\n')

        current_section = None
        current_content = []
        current_level = 0
        pre_heading_content = []

        from src.utils.logger import get_logger
        logger = get_logger(__name__)

        logger.debug(f"Extracting sections from content length: {len(content)}")

        def enrich_section_metadata(section_meta: Dict) -> Dict:
            """Agrega campos clave del frontmatter a la metadata de la sección"""
            if metadata and 'frontmatter' in metadata:
                # Copiar solo campos simples del frontmatter
                for key, value in metadata['frontmatter'].items():
                    if isinstance(value, (str, int, float, bool)):
                        section_meta[f'doc_{key}'] = value  # Prefijo 'doc_' = del documento
            return section_meta

        for line in lines:
            # Detectar headers (##, ###, ####) - Permitir espacios iniciales
            header_match = re.match(r'^\s*(#{1,6})\s+(.+)$', line)

            if header_match:
                # Guardar contenido antes del primer header
                if current_section is None and pre_heading_content:
                    intro_content = '\n'.join(pre_heading_content).strip()
                    if intro_content:
                        section_metadata = {'section_type': 'introduction'}
                        section_metadata = enrich_section_metadata(section_metadata)

                        sections.append(DocumentSection(
                            title="Introduction",
                            content=intro_content,
                            level=0,
                            metadata=section_metadata
                        ))
                    pre_heading_content = []

                # Guardar sección anterior
                if current_section:
                    section_content = '\n'.join(current_content).strip()
                    if section_content:
                        section_meta = self._extract_section_metadata(
                            current_section,
                            section_content
                        )
                        section_meta = enrich_section_metadata(section_meta)

                        sections.append(DocumentSection(
                            title=current_section,
                            content=section_content,
                            level=current_level,
                            metadata=section_meta
                        ))

                # Nueva sección
                current_level = len(header_match.group(1))
                current_section = header_match.group(2).strip()
                current_content = []
            else:
                # Agregar línea a contenido actual o pre-heading
                if current_section:
                    current_content.append(line)
                else:
                    pre_heading_content.append(line)

        # Guardar contenido previo si no hay headers
        if current_section is None and pre_heading_content:
            intro_content = '\n'.join(pre_heading_content).strip()
            if intro_content:
                section_metadata = {'section_type': 'document'}
                section_metadata = enrich_section_metadata(section_metadata)

                sections.append(DocumentSection(
                    title="Document Content",
                    content=intro_content,
                    level=0,
                    metadata=section_metadata
                ))

        # Última sección
        if current_section:
            section_content = '\n'.join(current_content).strip()
            if section_content:
                section_meta = self._extract_section_metadata(
                    current_section,
                    section_content
                )
                section_meta = enrich_section_metadata(section_meta)

                sections.append(DocumentSection(
                    title=current_section,
                    content=section_content,
                    level=current_level,
                    metadata=section_meta
                ))

        # FALLBACK: Si no se detectaron secciones pero hay contenido, tratar todo como una sección
        if not sections and content.strip():
            logger.warning(
                "No sections detected using headers. Treating entire file as single section.")
            section_metadata = {'section_type': 'general'}
            section_metadata = enrich_section_metadata(section_metadata)

            sections.append(DocumentSection(
                title="General",
                content=content_without_frontmatter.strip(),
                level=1,
                metadata=section_metadata
            ))

        logger.info(f"Extracted {len(sections)} sections from markdown")
        return sections

    def _extract_section_metadata(self, title: str, content: str) -> Dict:
        """Extrae metadata específica de una sección (ej: método HTTP, endpoint)"""
        metadata = {}

        # Método HTTP
        method_match = re.search(r'\b(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\b',
                                 title, re.IGNORECASE)
        if method_match:
            metadata['method'] = method_match.group(1).upper()

        # Endpoint/ruta API
        endpoint_match = re.search(r'`([\/\w\-{}]+)`', content)
        if endpoint_match:
            metadata['endpoint'] = endpoint_match.group(1)

        # Contexto funcional
        context_match = re.search(r'Contexto:\s*(\w+)', content, re.IGNORECASE)
        if context_match:
            metadata['context'] = context_match.group(1)

        return metadata

    def _generate_full_content(self, sections: List[DocumentSection], metadata: Dict = None) -> str:
        """Genera el contenido completo INCLUYENDO información clave del frontmatter"""
        full_content = []

        # GENÉRICO: Agregar campos importantes del frontmatter al inicio
        if metadata and 'frontmatter' in metadata:
            fm = metadata['frontmatter']

            # Lista de campos que típicamente son "identificadores principales"
            # Estos aparecerán primero y con mayor prominencia
            priority_fields = ['tabla', 'id', 'name', 'title', 'code', 'identifier', 'table']

            # Campos de metadatos secundarios útiles
            secondary_fields = ['owner', 'tipo', 'type', 'categoria', 'category',
                                'sistema', 'system', 'status', 'version']

            # Agregar campos prioritarios primero
            for field in priority_fields:
                if field in fm and isinstance(fm[field], (str, int, float, bool)):
                    label = field.replace('_', ' ').title()
                    full_content.append(f"**{label}**: {fm[field]}")

            # Agregar campos secundarios
            secondary_added = []
            for field in secondary_fields:
                if field in fm and isinstance(fm[field], (str, int, float, bool)):
                    label = field.replace('_', ' ').title()
                    secondary_added.append(f"{label}: {fm[field]}")

            if secondary_added:
                full_content.append(" | ".join(secondary_added))

            if full_content:  # Si agregamos algo, poner línea en blanco
                full_content.append("")

        # Resto de las secciones
        for section in sections:
            # Agregar título de sección con formato markdown
            header_prefix = '#' * (section.level if section.level > 0 else 1)
            full_content.append(f"{header_prefix} {section.title}")
            full_content.append(section.content)
            full_content.append("")  # Línea en blanco entre secciones

        return '\n'.join(full_content).strip()

    # src/document_loaders/markdown_loader.py
    # AGREGAR estos métodos a la clase MarkdownLoader existente

    def load_with_obsidian_context(
        self,
        file_path: Path,
        graph: Dict,
        original_filename: str = None
    ) -> ProcessedDocument:
        """
        Carga un archivo Markdown con contexto de grafo Obsidian

        Args:
            file_path: Ruta al archivo
            graph: Grafo de Obsidian (resultado de ObsidianGraphBuilder)
            original_filename: Nombre original del archivo

        Returns:
            ProcessedDocument enriquecido con metadata de Obsidian
        """
        # Cargar documento normalmente
        doc = self.load(file_path, original_filename)

        # Enriquecer con metadata de grafo
        note_name = file_path.stem

        if note_name in graph:
            graph_metadata = graph[note_name]["metadata"]

            # Agregar metadata de Obsidian
            doc.metadata.update({
                "obsidian_outgoing_links": graph_metadata["outgoing"],
                "obsidian_incoming_links": graph_metadata["incoming"],
                "obsidian_link_count": graph_metadata["link_count"],
                "obsidian_is_hub": graph_metadata["is_hub"],
                "obsidian_is_index": graph_metadata["is_index"],
                "obsidian_note_type": graph_metadata["note_type"],
                "obsidian_tags": graph_metadata["tags"],
                "obsidian_embeds": graph_metadata["embeds"]
            })

            self.logger.info(
                f"Enriched document with Obsidian metadata",
                extra={
                    "note": note_name,
                    "note_type": graph_metadata["note_type"],
                    "links": graph_metadata["link_count"]
                }
            )

        return doc
