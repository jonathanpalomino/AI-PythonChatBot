# src/document_loaders/obsidian_graph.py
"""
Obsidian Graph Builder
Construye el grafo bidireccional de relaciones entre notas
"""
import re
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

from src.utils.logger import get_logger


@dataclass
class NoteMetadata:
    """Metadata de una nota individual"""
    name: str
    path: Path
    outgoing: List[str]  # Links que esta nota menciona
    incoming: List[str]  # Backlinks (notas que mencionan a esta)
    aliases: Dict[str, str]  # {alias: nota_real}
    tags: List[str]
    embeds: List[str]  # ![[embeds]]
    link_count: int
    is_hub: bool  # True si tiene muchos incoming links
    is_index: bool  # True si tiene muchos outgoing links


class ObsidianGraphBuilder:
    """
    Construye el grafo de relaciones de una vault de Obsidian
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        self.notes: Dict[str, NoteMetadata] = {}  # {nombre_nota: metadata}
        self.graph: Dict[str, Dict[str, List[str]]] = {}  # {nota: {"in": [], "out": []}}
        self._alias_map: Dict[str, str] = {}  # {alias: nota_real}

    async def scan_vault(self, vault_path: Path) -> Dict[str, NoteMetadata]:
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
            if not any(p.startswith('.') for p in f.relative_to(vault_path).parts)
        ]

        self.logger.info(
            f"Scanning vault",
            extra={
                "vault": str(vault_path),
                "notes_count": len(md_files)
            }
        )

        # Primera pasada: indexar todas las notas
        for md_file in md_files:
            await self._process_note(md_file, vault_path)

        self.logger.info(f"Indexed {len(self.notes)} notes")

        return self.notes

    async def scan_files(
        self,
        files: List[Path],
        vault_root: Optional[Path]
    ) -> Dict[str, NoteMetadata]:
        """
        Escanea archivos específicos en contexto de vault

        Args:
            files: Lista de archivos a procesar
            vault_root: Raíz del vault (opcional)

        Returns:
            Dict de notas procesadas
        """
        self.logger.info(
            f"Scanning specific files",
            extra={"files_count": len(files)}
        )

        for md_file in files:
            await self._process_note(md_file, vault_root)

        # Si hay vault_root, también escanear notas referenciadas
        if vault_root:
            await self._scan_referenced_notes(vault_root)

        self.logger.info(f"Indexed {len(self.notes)} notes (including references)")

        return self.notes

    async def _process_note(self, file_path: Path, vault_root: Optional[Path]):
        """
        Procesa una nota individual y extrae sus links

        Args:
            file_path: Ruta al archivo .md
            vault_root: Raíz del vault (puede ser None)
        """
        try:
            note_name = file_path.stem

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extraer diferentes tipos de links y metadata
            outgoing_links = self._extract_wikilinks(content)
            aliases = self._extract_aliases(content)
            tags = self._extract_tags(content)
            embeds = self._extract_embeds(content)

            # Crear metadata
            metadata = NoteMetadata(
                name=note_name,
                path=file_path,
                outgoing=outgoing_links,
                incoming=[],  # Se llenará en segunda pasada
                aliases=aliases,
                tags=tags,
                embeds=embeds,
                link_count=len(outgoing_links),
                is_hub=False,  # Se calculará después
                is_index=False  # Se calculará después
            )

            self.notes[note_name] = metadata

            # Registrar aliases
            for alias, target in aliases.items():
                self._alias_map[alias] = target

            self.logger.debug(
                f"Processed note: {note_name}",
                extra={
                    "outgoing_links": len(outgoing_links),
                    "tags": len(tags),
                    "embeds": len(embeds)
                }
            )

        except Exception as e:
            self.logger.error(
                f"Error processing note {file_path}: {e}",
                exc_info=True
            )

    async def _scan_referenced_notes(self, vault_root: Path):
        """
        Escanea notas referenciadas pero no incluidas en scan inicial

        Args:
            vault_root: Raíz del vault
        """
        all_referenced: Set[str] = set()

        # Recolectar todas las referencias
        for note_data in self.notes.values():
            all_referenced.update(note_data.outgoing)

        # Buscar notas referenciadas que no están en self.notes
        missing = all_referenced - set(self.notes.keys())

        if missing:
            self.logger.info(
                f"Found referenced notes not in initial scan",
                extra={"missing_count": len(missing)}
            )

            for note_name in missing:
                # Buscar archivo en vault
                possible_paths = list(vault_root.rglob(f"{note_name}.md"))

                if possible_paths:
                    self.logger.debug(f"Found referenced note: {note_name}")
                    await self._process_note(possible_paths[0], vault_root)
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
            self.graph[note_name] = {
                "in": [],
                "out": [],
                "metadata": {}
            }

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
            note_data.is_hub = incoming_count > 5
            note_data.is_index = outgoing_count > 10

            # Agregar metadata al grafo
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
        """
        Resuelve un link, manejando aliases

        Args:
            link: Nombre del link (puede ser alias)

        Returns:
            Nombre real de la nota
        """
        # Primero intentar resolver como alias
        if link in self._alias_map:
            return self._alias_map[link]

        # Si no es alias, retornar como está
        return link

    def _classify_note_type(self, incoming_count: int, outgoing_count: int) -> str:
        """
        Clasifica tipo de nota según sus conexiones

        Types:
        - "hub": Muchos incoming (>5)
        - "index": Muchos outgoing (>10)
        - "atomic": Pocos links (<3 total)
        - "bridge": Balancea incoming/outgoing

        Args:
            incoming_count: Número de backlinks
            outgoing_count: Número de links salientes

        Returns:
            Tipo de nota
        """
        total = incoming_count + outgoing_count

        if incoming_count > 5:
            return "hub"
        elif outgoing_count > 10:
            return "index"
        elif total < 3:
            return "atomic"
        else:
            return "bridge"

    # =========================================================================
    # Extracción de Links y Metadata
    # =========================================================================

    def _extract_wikilinks(self, content: str) -> List[str]:
        """
        Extrae [[wikilinks]] del contenido

        Patterns:
        - [[nota]]
        - [[nota|alias]]
        - [[nota#section]]
        - [[nota#^block]]

        Args:
            content: Contenido markdown

        Returns:
            Lista de nombres de notas referenciadas
        """
        links = []

        # Pattern: [[nota]] o [[nota|alias]] o [[nota#section]]
        pattern = r'\[\[([^\]|#]+)(?:[#|][^\]]+)?\]\]'
        matches = re.findall(pattern, content)

        for match in matches:
            target = match.strip()
            if target:
                links.append(target)

        # Retornar únicos
        return list(set(links))

    def _extract_aliases(self, content: str) -> Dict[str, str]:
        """
        Extrae aliases de wikilinks

        Pattern: [[nota_real|alias_mostrado]]

        Args:
            content: Contenido markdown

        Returns:
            Dict {alias: nota_real}
        """
        aliases = {}

        # Pattern: [[target|alias]]
        pattern = r'\[\[([^\]|]+)\|([^\]]+)\]\]'
        matches = re.findall(pattern, content)

        for target, alias in matches:
            target = target.strip()
            alias = alias.strip()

            if target and alias:
                aliases[alias] = target

        return aliases

    def _extract_tags(self, content: str) -> List[str]:
        """
        Extrae #tags del contenido

        Args:
            content: Contenido markdown

        Returns:
            Lista de tags (sin el #)
        """
        # Pattern: #tag (pero no en código)
        # Buscar tags que no estén en bloques de código

        # Remover bloques de código
        content_no_code = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
        content_no_code = re.sub(r'`[^`]+`', '', content_no_code)

        # Extraer tags
        pattern = r'(?:^|\s)#([\w-]+)'
        tags = re.findall(pattern, content_no_code)

        return list(set(tags))

    def _extract_embeds(self, content: str) -> List[str]:
        """
        Extrae ![[embeds]] del contenido

        Args:
            content: Contenido markdown

        Returns:
            Lista de archivos embebidos
        """
        # Pattern: ![[archivo]]
        pattern = r'!\[\[([^\]]+)\]\]'
        embeds = re.findall(pattern, content)

        return list(set(embeds))
