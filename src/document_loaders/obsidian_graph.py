# src/document_loaders/obsidian_graph.py
"""
Obsidian Graph Builder
Construye el grafo bidireccional de relaciones entre notas
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from src.utils.logger import get_logger

# ✅ MEJORA 1: Compilar regex una sola vez (module-level)
WIKILINK_PATTERN = re.compile(r"\[\[(.*?)\]\]")
ALIAS_PATTERN = re.compile(r"\[\[([^\|\]]+)\|([^\]]+)\]\]")
TAG_PATTERN = re.compile(r"(?:^|\s)#([\w-]+)")
EMBED_PATTERN = re.compile(r"!\[\[([^\]]+)\]\]")
CODE_BLOCK_PATTERN = re.compile(r"```.*?```", flags=re.DOTALL)
INLINE_CODE_PATTERN = re.compile(r"`[^`]+`")


@dataclass
class NoteMetadata:
    """Metadata de una nota individual"""
    name: str
    path: Path
    outgoing: List[str] = field(default_factory=list)
    incoming: List[str] = field(default_factory=list)
    aliases: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    embeds: List[str] = field(default_factory=list)
    link_count: int = 0
    is_hub: bool = False
    is_index: bool = False


class ObsidianGraphBuilder:
    """Construye el grafo de relaciones de una vault de Obsidian"""

    # ✅ MEJORA 2: Constantes configurables
    HUB_THRESHOLD = 5  # Incoming links para ser considerado hub
    INDEX_THRESHOLD = 10  # Outgoing links para ser considerado index
    ATOMIC_THRESHOLD = 3  # Total links para ser considerado atomic

    def __init__(self):
        self.logger = get_logger(__name__)
        self.notes: Dict[str, NoteMetadata] = {}
        self.graph: Dict[str, Dict[str, List[str]]] = {}
        self.alias_map: Dict[str, str] = {}

    def scan_vault(self, vault_path: Path) -> Dict[str, NoteMetadata]:
        """
        Escanea vault completo

        Args:
            vault_path: Ruta raíz del vault

        Returns:
            Dict de notas procesadas
        """
        md_files = list(vault_path.rglob("*.md"))

        # Filtrar archivos de sistema
        md_files = [
            f for f in md_files
            if not any(p.startswith(".") for p in f.relative_to(vault_path).parts)
        ]

        self.logger.info(f"Scanning vault",
                         extra={"vault": str(vault_path), "notes_count": len(md_files)})

        # Primera pasada: indexar todas las notas
        for mdfile in md_files:
            self._process_note(mdfile, vault_path)

        # Segunda pasada: construir grafo bidireccional
        self.build_bidirectional_graph()

        self.logger.info(f"Indexed {len(self.notes)} notes")
        return self.notes

    def scan_files(self, files: List[Path], vault_root: Optional[Path] = None) -> Dict[
        str, NoteMetadata]:
        """
        Escanea archivos específicos en contexto de vault

        Args:
            files: Lista de archivos a procesar
            vault_root: Raíz del vault (opcional)

        Returns:
            Dict de notas procesadas
        """
        self.logger.info(f"Scanning specific files", extra={"files_count": len(files)})

        for mdfile in files:
            self._process_note(mdfile, vault_root)

        # Si hay vault, escanear notas referenciadas
        if vault_root:
            self._scan_referenced_notes(vault_root)

        self.build_bidirectional_graph()

        self.logger.info(f"Indexed {len(self.notes)} notes (including references)")
        return self.notes

    def _process_note(self, filepath: Path, vault_root: Optional[Path]):
        """
        Procesa una nota individual y extrae sus links

        Args:
            file_path: Ruta al archivo .md
            vault_root: Raíz del vault (puede ser None)
        """
        try:
            note_name = filepath.stem

            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # ✅ MEJORA 5: Usar regex pre-compilados
            outgoing_links = self._extract_wikilinks(content)
            aliases = self._extract_aliases(content)
            tags = self._extract_tags(content)
            embeds = self._extract_embeds(content)

            metadata = NoteMetadata(
                name=note_name,
                path=filepath,
                outgoing=outgoing_links,
                incoming=[],  # Se llenará en segunda pasada
                aliases=aliases,
                tags=tags,
                embeds=embeds,
                link_count=len(outgoing_links)
            )

            self.notes[note_name] = metadata

            # Registrar aliases
            for alias, target in aliases.items():
                self.alias_map[alias] = target

            self.logger.debug(
                f"Processed note: {note_name}",
                extra={
                    "outgoing_links": len(outgoing_links),
                    "tags": len(tags),
                    "embeds": len(embeds)
                }
            )

        except Exception as e:
            self.logger.error(f"Error processing note {filepath}: {e}", exc_info=True)

    def _scan_referenced_notes(self, vault_root: Path):
        """
        Escanea notas referenciadas pero no incluidas en scan inicial

        Args:
            vault_root: Raíz del vault
        """
        all_referenced: Set[str] = set()

        # Recolectar todas las referencias
        for note_data in self.notes.values():
            all_referenced.update(note_data.outgoing)

        # Buscar notas referenciadas que no estén en self.notes
        missing = all_referenced - set(self.notes.keys())

        if missing:
            self.logger.info(
                f"Found referenced notes not in initial scan",
                extra={"missing_count": len(missing)}
            )

            for note_name in missing:
                possible_paths = list(vault_root.rglob(f"{note_name}.md"))
                if possible_paths:
                    self.logger.debug(f"Found referenced note: {note_name}")
                    self._process_note(possible_paths[0], vault_root)
                else:
                    self.logger.debug(f"Referenced note not found: {note_name}")

    def build_bidirectional_graph(self) -> Dict[str, Dict]:
        """
        Segunda pasada: resolver backlinks y construir grafo bidireccional

        Returns:
            Dict con estructura: {nota: {"in": [...], "out": [...], "metadata": ...}}
        """
        self.logger.info("Building bidirectional graph")

        # Inicializar grafo
        for note_name in self.notes.keys():
            self.graph[note_name] = {"in": [], "out": []}

        # Resolver links y construir grafo
        for note_name, note_data in self.notes.items():
            # Procesar outgoing links
            for target in note_data.outgoing:
                # Resolver alias si existe
                resolved = self._resolve_link(target)

                # Agregar a outgoing
                if resolved not in self.graph[note_name]["out"]:
                    self.graph[note_name]["out"].append(resolved)

                # Agregar backlink al target
                if resolved in self.graph:
                    if note_name not in self.graph[resolved]["in"]:
                        self.graph[resolved]["in"].append(note_name)
                else:
                    # Nota referenciada no existe en vault
                    self.logger.debug(f"Broken link: {note_name} -> {resolved}")
                    self.graph[note_name]["out"].append(f"[BROKEN] {resolved}")

        # Actualizar metadata en cada nota
        for note_name, note_data in self.notes.items():
            incoming_count = len(self.graph[note_name]["in"])
            outgoing_count = len(self.graph[note_name]["out"])

            # Actualizar incoming links en NoteMetadata
            note_data.incoming = self.graph[note_name]["in"]
            note_data.link_count = incoming_count + outgoing_count
            note_data.is_hub = incoming_count >= self.HUB_THRESHOLD
            note_data.is_index = outgoing_count >= self.INDEX_THRESHOLD

            # ✅ MEJORA 6: Agregar metadata al grafo
            self.graph[note_name]["metadata"] = {
                "outgoing": self.graph[note_name]["out"],
                "incoming": self.graph[note_name]["in"],
                "link_count": note_data.link_count,
                "is_hub": note_data.is_hub,
                "is_index": note_data.is_index,
                "tags": note_data.tags,
                "embeds": note_data.embeds,
                "note_type": self._classify_note_type(incoming_count, outgoing_count)
            }

        self.logger.info(
            "Graph construction complete",
            extra={
                "notes": len(self.graph),
                "hubs": sum(1 for n in self.notes.values() if n.is_hub),
                "indexes": sum(1 for n in self.notes.values() if n.is_index)
            }
        )
        return self.graph

    def _resolve_link(self, link: str) -> str:
        """Resuelve un link, manejando aliases"""
        if link in self.alias_map:
            return self.alias_map[link]
        return link

    def _classify_note_type(self, incoming_count: int, outgoing_count: int) -> str:
        """Clasifica tipo de nota según sus conexiones"""
        total = incoming_count + outgoing_count

        if incoming_count >= self.HUB_THRESHOLD:
            return "hub"
        elif outgoing_count >= self.INDEX_THRESHOLD:
            return "index"
        elif total <= self.ATOMIC_THRESHOLD:
            return "atomic"
        else:
            return "bridge"

    # =========================================================================
    # ✅ MEJORA 7: Métodos de extracción optimizados
    # =========================================================================

    def _extract_wikilinks(self, content: str) -> List[str]:
        """Extrae wikilinks usando regex pre-compilado"""
        matches = WIKILINK_PATTERN.findall(content)
        links = [match.strip() for match in matches if match.strip()]
        return list(set(links))  # Deduplicar

    def _extract_aliases(self, content: str) -> Dict[str, str]:
        """Extrae aliases usando regex pre-compilado"""
        matches = ALIAS_PATTERN.findall(content)
        aliases = {}
        for target, alias in matches:
            target = target.strip()
            alias = alias.strip()
            if target and alias:
                aliases[alias] = target
        return aliases

    def _extract_tags(self, content: str) -> List[str]:
        """Extrae tags del contenido"""
        # Remover bloques de código
        content_no_code = CODE_BLOCK_PATTERN.sub("", content)
        content_no_code = INLINE_CODE_PATTERN.sub("", content_no_code)

        # Extraer tags
        tags = TAG_PATTERN.findall(content_no_code)
        return list(set(tags))

    def _extract_embeds(self, content: str) -> List[str]:
        """Extrae ![[embeds]] usando regex pre-compilado"""
        embeds = EMBED_PATTERN.findall(content)
        return list(set(embeds))
