# =============================================================================
# src/document_loaders/obsidian_note_adapter.py
# Obsidian-specific adapter over generic MarkdownLoader
# =============================================================================
"""
Adapter que agrega comportamiento específico de Obsidian sobre MarkdownLoader genérico.
Sigue el patrón Adapter para separar responsabilidades.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.document_loaders.base_loader import ProcessedDocument
from src.document_loaders.graph_types import (
    GraphEdge, GraphNode, EdgeType,
    QdrantChunkMetadata
)
from src.document_loaders.markdown_loader import MarkdownLoader
from src.document_loaders.obsidian_detector import ObsidianDetector, ObsidianContext
from src.document_loaders.obsidian_graph import ObsidianGraphBuilder
from src.utils.logger import get_logger


@dataclass
class WikiLink:
    """Representa un wikilink de Obsidian"""
    target: str                     # Nota destino
    display_text: Optional[str]     # Texto mostrado (si usa |)
    section: Optional[str]          # Sección (#heading)
    block_id: Optional[str]         # Block reference (^block-id)
    line_number: int                # Línea donde aparece
    context: str = ""               # Contexto textual alrededor


@dataclass
class Transclusion:
    """Representa una transclusion (![[nota]])"""
    target: str
    section: Optional[str] = None
    line_number: int = 0
    resolved_content: Optional[str] = None


@dataclass
class BlockRef:
    """Representa una block reference (^block-id)"""
    block_id: str
    content: str
    line_number: int


@dataclass
class ObsidianCallout:
    """Representa un callout de Obsidian (> [!TYPE])"""
    callout_type: str  # NOTE, WARNING, TIP, etc.
    title: Optional[str]
    content: str
    foldable: bool = False


@dataclass
class ObsidianProcessedNote:
    """
    Nota de Obsidian procesada con toda la metadata enriquecida.

    Diseñada para ser RAG-ready: contiene chunks + graph edges.
    """
    # Documento base del loader genérico
    base_document: ProcessedDocument

    # Features específicas de Obsidian
    wikilinks: List[WikiLink] = field(default_factory=list)
    backlinks: List[str] = field(default_factory=list)
    transclusions: List[Transclusion] = field(default_factory=list)
    block_refs: List[BlockRef] = field(default_factory=list)
    callouts: List[ObsidianCallout] = field(default_factory=list)

    # Graph metadata
    graph_edges: List[GraphEdge] = field(default_factory=list)
    graph_node: Optional[GraphNode] = None

    # Vault context
    vault_root: Optional[Path] = None
    vault_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def note_id(self) -> str:
        """ID de la nota (filename sin extensión)"""
        return Path(self.base_document.file_path).stem

    def to_qdrant_chunks(self) -> List[Dict[str, Any]]:
        """
        Convierte a chunks listos para indexar en Qdrant.

        Returns:
            Lista de dicts con formato: {content, metadata}
        """
        chunks = []

        # Extraer tags de múltiples fuentes
        all_tags = []

        # 1. Tags del frontmatter YAML
        frontmatter = self.base_document.metadata.get('frontmatter', {})
        if isinstance(frontmatter, dict):
            frontmatter_tags = frontmatter.get('tags', [])
            # Asegurar que sea lista
            if isinstance(frontmatter_tags, str):
                all_tags.append(frontmatter_tags)
            elif isinstance(frontmatter_tags, list):
                all_tags.extend(frontmatter_tags)

        # 2. Tags inline del metadata (extraídos del contenido)
        inline_tags = self.base_document.metadata.get('tags', [])
        if isinstance(inline_tags, list):
            all_tags.extend(inline_tags)
        elif isinstance(inline_tags, str):
            all_tags.append(inline_tags)

        # Deduplicar y limpiar
        all_tags = list(set(tag.strip() for tag in all_tags if tag))

        # Extraer aliases del frontmatter
        aliases = frontmatter.get('aliases', []) if isinstance(frontmatter, dict) else []
        if isinstance(aliases, str):
            aliases = [aliases]

        from src.utils.date_utils import get_current_utc_iso
        indexed_at = get_current_utc_iso()

        total_sections = len(self.base_document.sections)
        for i, section in enumerate(self.base_document.sections):
            # Crear metadata enriquecida
            chunk_metadata = QdrantChunkMetadata(
                source_note=self.note_id,
                file_path=self.base_document.file_path,
                section=section.title,
                outgoing_links=[link.target for link in self.wikilinks],
                incoming_links=self.backlinks,
                tags=all_tags,  # Usar tags combinados
                aliases=aliases,
                note_type=self.graph_node.note_type.value if self.graph_node else 'atomic',
                is_hub=self.graph_node.is_hub if self.graph_node else False,
                is_index=self.graph_node.is_index if self.graph_node else False,
                chunk_index=i,
                total_chunks=total_sections,
                indexed_at=indexed_at,
                extra_metadata={
                    **section.metadata,
                    'frontmatter': frontmatter  # Incluir todo el frontmatter
                }
            )

            chunks.append({
                "content": section.content,
                "metadata": chunk_metadata.to_qdrant_payload()
            })

        return chunks


class ObsidianNoteAdapter:
    """
    Adapter que agrega funcionalidad Obsidian sobre MarkdownLoader genérico.

    Responsabilidades:
    - Extracción de wikilinks, transclusions, block-refs
    - Resolución de aliases
    - Parseo de callouts
    - Generación de graph edges para RAG
    """

    def __init__(
        self,
        vault_root: Optional[Path] = None,
        markdown_loader: Optional[MarkdownLoader] = None
    ):
        """
        Args:
            vault_root: Raíz del vault Obsidian (opcional, se puede detectar)
            markdown_loader: Instancia de MarkdownLoader (opcional, se crea una por defecto)
        """
        self.vault_root = vault_root
        self.loader = markdown_loader or MarkdownLoader()
        self.detector = ObsidianDetector()
        self.graph_builder = ObsidianGraphBuilder()
        self.logger = get_logger(__name__)

        # Cache de notas ya cargadas
        self._notes_cache: Dict[str, ObsidianProcessedNote] = {}

    async def load_note(
        self,
        note_path: Path,
        include_graph: bool = True,
        resolve_transclusions: bool = True,
        vault_context: Optional[ObsidianContext] = None
    ) -> ObsidianProcessedNote:
        """
        Carga una nota de Obsidian con toda la metadata enriquecida.

        Args:
            note_path: Ruta a la nota .md
            include_graph: Incluir metadata de grafo (edges, backlinks)
            resolve_transclusions: Resolver ![[transclusions]] inline
            vault_context: Contexto de vault (opcional, se detecta automáticamente)

        Returns:
            ObsidianProcessedNote con metadata completa
        """
        # 1. Detectar vault context si no se provee
        if vault_context is None and self.vault_root is None:
            vault_context = self.detector.detect(note_path)
            if vault_context.is_obsidian:
                self.vault_root = vault_context.vault_root

        # 2. Cargar documento base con loader genérico
        base_doc = self.loader.load(note_path)

        # 3. Extraer features de Obsidian del contenido
        raw_content = base_doc.content

        wikilinks = self._extract_wikilinks(raw_content)
        transclusions = self._extract_transclusions(raw_content)
        block_refs = self._extract_block_refs(raw_content)
        callouts = self._extract_callouts(raw_content)

        # 4. Resolver transclusions si se requiere
        if resolve_transclusions and self.vault_root:
            for transcl in transclusions:
                await self._resolve_transclusion(transcl)

        # 5. Generar graph edges
        graph_edges = self._generate_graph_edges(note_path, wikilinks, transclusions)

        # 6. Crear nodo de grafo
        graph_node = self._create_graph_node(note_path, wikilinks, base_doc.metadata)

        # 7. Obtener backlinks si include_graph
        backlinks = []
        if include_graph and self.vault_root:
            # Los backlinks se calculan después de escanear todo el vault
            # Por ahora dejamos vacío, se llenará con build_graph()
            pass

        # 8. Crear ObsidianProcessedNote
        obsidian_note = ObsidianProcessedNote(
            base_document=base_doc,
            wikilinks=wikilinks,
            backlinks=backlinks,
            transclusions=transclusions,
            block_refs=block_refs,
            callouts=callouts,
            graph_edges=graph_edges,
            graph_node=graph_node,
            vault_root=self.vault_root,
            vault_metadata=vault_context.vault_config if vault_context else {}
        )

        # Cache
        self._notes_cache[obsidian_note.note_id] = obsidian_note

        self.logger.info(
            f"Loaded Obsidian note",
            extra={
                "note": obsidian_note.note_id,
                "wikilinks": len(wikilinks),
                "transclusions": len(transclusions),
                "callouts": len(callouts)
            }
        )

        return obsidian_note

    async def load_vault(
        self,
        vault_path: Path,
        note_names: Optional[List[str]] = None,
        include_graph: bool = True
    ) -> List[ObsidianProcessedNote]:
        """
        Carga vault completo o subset de notas.

        Args:
            vault_path: Ruta al vault
            note_names: Lista de nombres de notas (sin .md), None = todas
            include_graph: Incluir graph bidireccional

        Returns:
            Lista de notas procesadas
        """
        # Detectar vault
        context = self.detector.detect(vault_path)

        if not context.is_obsidian:
            self.logger.warning(f"Path {vault_path} is not an Obsidian vault")
            return []

        self.vault_root = context.vault_root

        # Filtrar archivos si se especificaron note_names
        if note_names:
            files = []
            for name in note_names:
                matching = list(context.vault_root.rglob(f"{name}.md"))
                files.extend(matching)
        else:
            files = context.files

        # Cargar todas las notas
        notes = []
        for file_path in files:
            note = await self.load_note(
                file_path,
                include_graph=include_graph,
                vault_context=context
            )
            notes.append(note)

        # Si include_graph, construir grafo bidireccional
        if include_graph:
            await self._build_bidirectional_graph(notes)

        self.logger.info(f"Loaded {len(notes)} notes from vault {vault_path}")

        return notes

    def _extract_wikilinks(self, content: str) -> List[WikiLink]:
        """
        Extrae [[wikilinks]] del contenido.

        Soporta:
        - [[nota]]
        - [[nota|display text]]
        - [[nota#section]]
        - [[nota#^block-id]]
        """
        wikilinks = []

        # Pattern: [[target|display?#section?^block?]]
        pattern = r'\[\[([^\]|#^]+)(?:\|([^\]#^]+))?(?:#([^\]^]+))?(?:\^([^\]]+))?\]\]'

        for line_num, line in enumerate(content.split('\n'), 1):
            for match in re.finditer(pattern, line):
                target = match.group(1).strip()
                display_text = match.group(2).strip() if match.group(2) else None
                section = match.group(3).strip() if match.group(3) else None
                block_id = match.group(4).strip() if match.group(4) else None

                # Contexto: 50 caracteres antes y después
                start = max(0, match.start() - 50)
                end = min(len(line), match.end() + 50)
                context = line[start:end]

                wikilinks.append(WikiLink(
                    target=target,
                    display_text=display_text,
                    section=section,
                    block_id=block_id,
                    line_number=line_num,
                    context=context
                ))

        return wikilinks

    def _extract_transclusions(self, content: str) -> List[Transclusion]:
        """Extrae ![[transclusions]] del contenido"""
        transclusions = []

        # Pattern: ![[nota#section?]]
        pattern = r'!\[\[([^\]#]+)(?:#([^\]]+))?\]\]'

        for line_num, line in enumerate(content.split('\n'), 1):
            for match in re.finditer(pattern, line):
                target = match.group(1).strip()
                section = match.group(2).strip() if match.group(2) else None

                transclusions.append(Transclusion(
                    target=target,
                    section=section,
                    line_number=line_num
                ))

        return transclusions

    def _extract_block_refs(self, content: str) -> List[BlockRef]:
        """Extrae block references (^block-id) del contenido"""
        block_refs = []

        # Pattern: texto ^block-id al final de línea
        pattern = r'^(.+)\s+\^([\w-]+)\s*$'

        for line_num, line in enumerate(content.split('\n'), 1):
            match = re.match(pattern, line)
            if match:
                content_text = match.group(1).strip()
                block_id = match.group(2).strip()

                block_refs.append(BlockRef(
                    block_id=block_id,
                    content=content_text,
                    line_number=line_num
                ))

        return block_refs

    def _extract_callouts(self, content: str) -> List[ObsidianCallout]:
        """
        Extrae callouts de Obsidian del contenido.

        Format: > [!TYPE] Title (opcional)
                > Content
        """
        callouts = []
        lines = content.split('\n')

        i = 0
        while i < len(lines):
            line = lines[i]

            # Detectar inicio de callout: > [!TYPE]
            match = re.match(r'^\s*>\s*\[!(\w+)\]([+-])?\s*(.*)$', line)
            if match:
                callout_type = match.group(1).upper()
                foldable = match.group(2) is not None
                title = match.group(3).strip() if match.group(3) else None

                # Capturar contenido (líneas siguientes que empiezan con >)
                content_lines = []
                i += 1
                while i < len(lines) and lines[i].strip().startswith('>'):
                    content_line = lines[i].strip()[1:].strip()  # Remover >
                    content_lines.append(content_line)
                    i += 1

                callouts.append(ObsidianCallout(
                    callout_type=callout_type,
                    title=title,
                    content='\n'.join(content_lines),
                    foldable=foldable
                ))
            else:
                i += 1

        return callouts

    async def _resolve_transclusion(self, transclusion: Transclusion) -> None:
        """Resuelve una transclusion cargando el contenido referenciado"""
        if not self.vault_root:
            return

        # Buscar archivo
        target_paths = list(self.vault_root.rglob(f"{transclusion.target}.md"))

        if not target_paths:
            self.logger.debug(f"Transclusion target not found: {transclusion.target}")
            return

        # Cargar contenido
        target_path = target_paths[0]
        target_doc = self.loader.load(target_path)

        # Si hay sección específica, buscarla
        if transclusion.section:
            for section in target_doc.sections:
                if section.title == transclusion.section:
                    transclusion.resolved_content = section.content
                    return
        else:
            # Transcluir documento completo
            transclusion.resolved_content = target_doc.content

    def _generate_graph_edges(
        self,
        note_path: Path,
        wikilinks: List[WikiLink],
        transclusions: List[Transclusion]
    ) -> List[GraphEdge]:
        """Genera graph edges desde wikilinks y transclusions"""
        edges = []
        source_note = note_path.stem

        # Edges desde wikilinks
        for link in wikilinks:
            edges.append(GraphEdge(
                source_note=source_note,
                target_note=link.target,
                edge_type=EdgeType.WIKILINK,
                section=link.section,
                block_id=link.block_id,
                line_number=link.line_number,
                context=link.context,
                metadata={"display_text": link.display_text}
            ))

        # Edges desde transclusions
        for transcl in transclusions:
            edges.append(GraphEdge(
                source_note=source_note,
                target_note=transcl.target,
                edge_type=EdgeType.TRANSCLUSION,
                section=transcl.section,
                line_number=transcl.line_number,
                metadata={}
            ))

        return edges

    def _create_graph_node(
        self,
        note_path: Path,
        wikilinks: List[WikiLink],
        metadata: Dict[str, Any]
    ) -> GraphNode:
        """Crea un GraphNode desde metadata de la nota"""
        note_id = note_path.stem

        # Extraer aliases del frontmatter
        aliases = metadata.get('frontmatter', {}).get('aliases', [])
        if isinstance(aliases, str):
            aliases = [aliases]

        # Extraer tags
        tags = metadata.get('tags', [])

        # Outgoing links desde wikilinks
        outgoing = [link.target for link in wikilinks]

        node = GraphNode(
            note_id=note_id,
            note_name=note_path.stem,
            file_path=note_path,
            aliases=aliases,
            tags=tags,
            outgoing_links=outgoing,
            incoming_links=[],  # Se llenará con build_bidirectional_graph
            metadata=metadata
        )

        # Clasificar tipo de nota
        node.note_type = node.classify_note_type()

        return node

    async def _build_bidirectional_graph(self, notes: List[ObsidianProcessedNote]) -> None:
        """
        Construye grafo bidireccional resolviendo backlinks.

        Actualiza los backlinks de cada nota basándose en los outgoing links de otras.
        """
        # Crear índice: note_id -> ObsidianProcessedNote
        notes_index = {note.note_id: note for note in notes}

        # Resolver backlinks
        for note in notes:
            for link in note.wikilinks:
                target = link.target

                # Si el target existe en nuestras notas, agregar backlink
                if target in notes_index:
                    target_note = notes_index[target]
                    if note.note_id not in target_note.backlinks:
                        target_note.backlinks.append(note.note_id)

                    # También actualizar incoming_links en el graph_node
                    if target_note.graph_node:
                        if note.note_id not in target_note.graph_node.incoming_links:
                            target_note.graph_node.incoming_links.append(note.note_id)

        # Reclasificar todos los nodos con la info actualizada
        for note in notes:
            if note.graph_node:
                note.graph_node.note_type = note.graph_node.classify_note_type()

        self.logger.info(f"Built bidirectional graph for {len(notes)} notes")
