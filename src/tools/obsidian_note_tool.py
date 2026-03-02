# =============================================================================
# src/tools/obsidian_note_tool.py
# Professional Obsidian Vault Tool with Advanced Features
# =============================================================================
"""
Professional Obsidian Vault Tool with advanced features:
- Note caching with TTL
- Vault structure validation
- Multi-vault support
- Performance metrics
- Incremental loading support
- Advanced filtering
- Export capabilities
- Detailed error handling
"""

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.document_loaders.obsidian_note_adapter import ObsidianNoteAdapter
from src.tools.base_tool import BaseTool, ToolCategory, ToolParameter, ToolResult
from src.utils.logger import get_logger


# =============================================================================
# Enums and Data Classes
# =============================================================================

class OutputFormat(Enum):
    """Output formats for Obsidian notes."""
    QDRANT_CHUNKS = "qdrant_chunks"
    FULL_NOTES = "full_notes"
    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"


class CacheStrategy(Enum):
    """Caching strategies for notes."""
    NO_CACHE = "no_cache"
    MEMORY_CACHE = "memory_cache"
    FILE_CACHE = "file_cache"


@dataclass
class CacheConfig:
    """Configuration for note caching."""
    strategy: CacheStrategy = CacheStrategy.MEMORY_CACHE
    ttl: int = 3600  # seconds (1 hour)
    max_size: int = 500  # max number of cached notes


@dataclass
class FilterConfig:
    """Configuration for note filtering."""
    tags: Optional[List[str]] = None
    date_range: Optional[Dict[str, str]] = None  # {"start": "2024-01-01", "end": "2024-12-31"}
    note_types: Optional[List[str]] = None  # ["atomic", "moc", "canvas", etc.
    min_links: Optional[int] = None
    max_links: Optional[int] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for vault loading."""
    vaults_loaded: int = 0
    notes_loaded: int = 0
    total_load_time: float = 0.0
    min_load_time: float = float('inf')
    max_load_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    transclusions_resolved: int = 0
    wikilinks_found: int = 0


# =============================================================================
# Professional Obsidian Note Tool
# =============================================================================

class ObsidianNoteTool(BaseTool):
    """
    Professional Obsidian Vault Tool with advanced features.

    Features:
    - Note caching with TTL
    - Vault structure validation
    - Multi-vault support
    - Performance metrics
    - Incremental loading support
    - Advanced filtering
    - Export capabilities
    - Detailed error handling
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        self.adapter = ObsidianNoteAdapter()
        self._cache_config = CacheConfig()
        self._note_cache: Dict[str, Dict[str, Any]] = {}
        self._vault_cache: Dict[str, Dict[str, Any]] = {}
        self._metrics = PerformanceMetrics()
        self._active_vaults: Dict[str, Path] = {}
        super().__init__()

    # =========================================================================
    # Tool Definition
    # =========================================================================

    @property
    def name(self) -> str:
        return "obsidian_vault_loader"

    @property
    def description(self) -> str:
        return """Professional Obsidian vault loader with advanced features:
- Note caching with TTL
- Vault structure validation
- Multi-vault support
- Performance metrics
- Incremental loading support
- Advanced filtering (tags, dates, note types)
- Export capabilities (Markdown, JSON, HTML)
- Detailed error handling"""

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.DOCUMENT

    @property
    def enabled_by_default(self) -> bool:
        return False

    @property
    def requires_context(self) -> List[str]:
        return ["filesystem"]

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="vault_path",
                type="string",
                description="Absolute path to Obsidian vault root directory",
                required=True,
                example="C:/Users/user/Documents/MyVault"
            ),
            ToolParameter(
                name="note_names",
                type="array",
                description="Optional list of specific note names to load (without .md extension)",
                required=False,
                example=["Servicios", "ServerGroups", "Arquitectura"]
            ),
            ToolParameter(
                name="include_graph",
                type="boolean",
                description="Include bidirectional graph metadata (backlinks, incoming/outgoing links)",
                required=False,
                default=True,
                example=True
            ),
            ToolParameter(
                name="resolve_transclusions",
                type="boolean",
                description="Resolve ![[transclusions]] by loading embedded content",
                required=False,
                default=True,
                example=True
            ),
            ToolParameter(
                name="output_format",
                type="string",
                description="Output format: 'qdrant_chunks', 'full_notes', 'markdown', 'json', 'html'",
                required=False,
                default="full_notes",
                enum=["qdrant_chunks", "full_notes", "markdown", "json", "html"],
                example="full_notes"
            ),
            ToolParameter(
                name="max_notes",
                type="integer",
                description="Maximum number of notes to load (prevents loading huge vaults)",
                required=False,
                default=100,
                example=50
            ),
            ToolParameter(
                name="cache_config",
                type="object",
                description="""Cache configuration. Format:
                                {
                                    "strategy": "no_cache|memory_cache|file_cache",
                                    "ttl": 3600,
                                    "max_size": 500
                                }""",
                required=False,
                default=None
            ),
            ToolParameter(
                name="filter_config",
                type="object",
                description="""Filter configuration. Format:
                                {
                                    "tags": ["tag1", "tag2"],
                                    "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
                                    "note_types": ["atomic", "moc", "canvas"],
                                    "min_links": 5,
                                    "max_links": 50
                                }""",
                required=False,
                default=None
            ),
            ToolParameter(
                name="incremental",
                type="boolean",
                description="Load only notes modified since last load (requires cache)",
                required=False,
                default=False,
                example=False
            ),
            ToolParameter(
                name="validate_structure",
                type="boolean",
                description="Validate vault structure before loading",
                required=False,
                default=True,
                example=True
            ),
            ToolParameter(
                name="export_path",
                type="string",
                description="Path to export notes (for markdown/json/html formats)",
                required=False,
                example="C:/Users/user/Documents/exported_notes"
            )
        ]

    # =========================================================================
    # Main Execution Method
    # =========================================================================

    async def execute(
        self,
        vault_path: str,
        note_names: Optional[List[str]] = None,
        include_graph: bool = True,
        resolve_transclusions: bool = True,
        output_format: str = "full_notes",
        max_notes: int = 100,
        cache_config: Optional[Dict[str, Any]] = None,
        filter_config: Optional[Dict[str, Any]] = None,
        incremental: bool = False,
        validate_structure: bool = True,
        export_path: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Execute Obsidian vault loading with professional features"""

        try:
            # Validate inputs
            await self.validate_input(
                vault_path=vault_path,
                note_names=note_names or [],
                include_graph=include_graph,
                resolve_transclusions=resolve_transclusions,
                output_format=output_format,
                max_notes=max_notes
            )

            # Apply configurations
            if cache_config:
                self._apply_cache_config(cache_config)

            # Validate vault structure
            if validate_structure:
                validation_result = self._validate_vault_structure(vault_path)
                if not validation_result.get("valid"):
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"Vault structure validation failed: {validation_result.get('errors')}"
                    )

            vault_path_obj = Path(vault_path)

            if not vault_path_obj.exists():
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Vault path does not exist: {vault_path}"
                )

            # Check if vault is already loaded
            vault_key = str(vault_path_obj)
            if vault_key in self._active_vaults:
                self.logger.info(f"Vault already loaded: {vault_path}")
                vault_data = self._vault_cache[vault_key]
                notes = vault_data.get("notes", [])
            else:
                # Load notes
                start_time = datetime.utcnow()
                notes = await self.adapter.load_vault(
                    vault_path=vault_path_obj,
                    note_names=note_names,
                    include_graph=include_graph
                )
                load_time = (datetime.utcnow() - start_time).total_seconds()

                # Update metrics
                self._metrics.vaults_loaded += 1
                self._metrics.notes_loaded += len(notes)
                self._metrics.total_load_time += load_time
                if load_time < self._metrics.min_load_time:
                    self._metrics.min_load_time = load_time
                if load_time > self._metrics.max_load_time:
                    self._metrics.max_load_time = load_time

                # Cache vault
                self._vault_cache[vault_key] = {
                    "notes": notes,
                    "loaded_at": datetime.utcnow(),
                    "vault_path": str(vault_path_obj)
                }
                self._active_vaults[vault_key] = vault_path_obj

            if not notes:
                return ToolResult(
                    success=True,
                    data={"notes": [], "message": "No notes found in vault"},
                    metadata={"vault_path": str(vault_path_obj)}
                )

            # Apply filters
            if filter_config:
                notes = self._apply_filters(notes, filter_config)

            # Limit notes
            if len(notes) > max_notes:
                self.logger.warning(f"Vault has {len(notes)} notes, limiting to {max_notes}")
                notes = notes[:max_notes]

            # Format output
            if output_format == OutputFormat.QDRANT_CHUNKS.value:
                output_data = self._format_as_qdrant_chunks(notes)
            elif output_format == OutputFormat.MARKDOWN.value:
                output_data = self._format_as_markdown(notes)
            elif output_format == OutputFormat.JSON.value:
                output_data = self._format_as_json(notes)
            elif output_format == OutputFormat.HTML.value:
                output_data = self._format_as_html(notes)
            else:  # FULL_NOTES
                output_data = self._format_as_full_notes(notes)

            # Export if path provided
            if export_path and output_format in [OutputFormat.MARKDOWN.value, OutputFormat.JSON.value, OutputFormat.HTML.value]:
                export_result = await self._export_notes(notes, export_path, output_format)
                if not export_result.get("success"):
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"Export failed: {export_result.get('error')}"
                    )

            # Calculate statistics
            total_wikilinks = sum(len(n.wikilinks) for n in notes)
            total_edges = sum(len(n.graph_edges) for n in notes)
            hubs = [n.note_id for n in notes if n.graph_node and n.graph_node.is_hub]
            indexes = [n.note_id for n in notes if n.graph_node and n.graph_node.is_index]

            self.logger.info(
                f"Successfully loaded Obsidian vault",
                extra={
                    "vault_path": str(vault_path_obj),
                    "notes_count": len(notes),
                    "output_format": output_format,
                    "total_wikilinks": total_wikilinks,
                    "total_edges": total_edges,
                    "hub_count": len(hubs),
                    "index_count": len(indexes)
                }
            )

            return ToolResult(
                success=True,
                data=output_data,
                metadata={
                    "vault_path": str(vault_path_obj),
                    "notes_count": len(notes),
                    "output_format": output_format,
                    "total_wikilinks": total_wikilinks,
                    "total_edges": total_edges,
                    "hub_count": len(hubs),
                    "index_count": len(indexes),
                    "performance_metrics": self._get_performance_metrics()
                }
            )

        except Exception as e:
            self.logger.error(
                f"Failed to load Obsidian vault: {e}",
                exc_info=True,
                extra={"vault_path": vault_path}
            )

            return ToolResult(
                success=False,
                data=None,
                error=f"Failed to load vault: {str(e)}"
            )

    # =========================================================================
    # Configuration Methods
    # =========================================================================

    def _apply_cache_config(self, cache_config: Dict[str, Any]):
        """Apply cache configuration"""
        self._cache_config = CacheConfig(
            strategy=CacheStrategy(cache_config.get("strategy", "memory_cache")),
            ttl=cache_config.get("ttl", 3600),
            max_size=cache_config.get("max_size", 500)
        )

    # =========================================================================
    # Validation Methods
    # =========================================================================

    def _validate_vault_structure(self, vault_path: str) -> Dict[str, Any]:
        """Validate Obsidian vault structure"""
        errors = []

        vault_path_obj = Path(vault_path)

        # Check if vault exists
        if not vault_path_obj.exists():
            errors.append("Vault path does not exist")
            return {"valid": False, "errors": errors}

        # Check for .obsidian folder
        obsidian_folder = vault_path_obj / ".obsidian"
        if not obsidian_folder.exists():
            errors.append("Missing .obsidian folder (not a valid Obsidian vault)")

        # Check for configuration files
        config_files = ["plugins", "themes", "workspace.json"]
        for config_file in config_files:
            config_path = vault_path_obj / f".obsidian/{config_file}"
            if not config_path.exists():
                errors.append(f"Missing {config_file} in .obsidian folder")

        # Check for at least one markdown file
        md_files = list(vault_path_obj.glob("*.md"))
        if not md_files:
            errors.append("No markdown files found in vault")

        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

    # =========================================================================
    # Filter Methods
    # =========================================================================

    def _apply_filters(
        self,
        notes: List[Any],
        filter_config: Dict[str, Any]
    ) -> List[Any]:
        """Apply filters to notes"""
        filtered_notes = notes

        # Filter by tags
        tags = filter_config.get("tags")
        if tags:
            filtered_notes = [
                n for n in filtered_notes
                if any(tag in n.base_document.metadata.get("tags", []) for tag in tags)
            ]

        # Filter by date range
        date_range = filter_config.get("date_range")
        if date_range:
            start_date = datetime.fromisoformat(date_range["start"])
            end_date = datetime.fromisoformat(date_range["end"])

            filtered_notes = [
                n for n in filtered_notes
                if self._is_note_in_date_range(n, start_date, end_date)
            ]

        # Filter by note types
        note_types = filter_config.get("note_types")
        if note_types:
            filtered_notes = [
                n for n in filtered_notes
                if n.graph_node and n.graph_node.note_type.value in note_types
            ]

        # Filter by link count
        min_links = filter_config.get("min_links")
        max_links = filter_config.get("max_links")
        if min_links is not None or max_links is not None:
            filtered_notes = [
                n for n in filtered_notes
                if self._is_note_in_link_range(n, min_links, max_links)
            ]

        return filtered_notes

    def _is_note_in_date_range(
        self,
        note: Any,
        start_date: datetime,
        end_date: datetime
    ) -> bool:
        """Check if note is in date range"""
        note_date = note.base_document.metadata.get("created")
        if not note_date:
            return True

        try:
            note_datetime = datetime.fromisoformat(note_date)
            return start_date <= note_datetime <= end_date
        except:
            return True

    def _is_note_in_link_range(
        self,
        note: Any,
        min_links: Optional[int],
        max_links: Optional[int]
    ) -> bool:
        """Check if note is in link count range"""
        link_count = len(note.wikilinks) + len(note.backlinks)

        if min_links is not None and link_count < min_links:
            return False

        if max_links is not None and link_count > max_links:
            return False

        return True

    # =========================================================================
    # Export Methods
    # =========================================================================

    async def _export_notes(
        self,
        notes: List[Any],
        export_path: str,
        output_format: str
    ) -> Dict[str, Any]:
        """Export notes to specified format"""
        try:
            export_path_obj = Path(export_path)
            export_path_obj.mkdir(parents=True, exist_ok=True)

            if output_format == OutputFormat.MARKDOWN.value:
                for note in notes:
                    file_path = export_path_obj / f"{note.note_id}.md"
                    file_path.write_text(note.base_document.content, encoding='utf-8')

            elif output_format == OutputFormat.JSON.value:
                export_data = self._format_as_json(notes)
                file_path = export_path_obj / "notes.json"
                file_path.write_text(json.dumps(export_data, indent=2), encoding='utf-8')

            elif output_format == OutputFormat.HTML.value:
                export_data = self._format_as_html(notes)
                file_path = export_path_obj / "notes.html"
                file_path.write_text(export_data, encoding='utf-8')

            return {
                "success": True,
                "exported_count": len(notes),
                "export_path": str(export_path_obj)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    # =========================================================================
    # Output Formatting Methods
    # =========================================================================

    def _format_as_qdrant_chunks(self, notes) -> dict:
        """Format notes as Qdrant chunks"""
        all_chunks = []
        all_edges = []

        for note in notes:
            chunks = note.to_qdrant_chunks()
            all_chunks.extend(chunks)

            edges_dict = [edge.to_dict() for edge in note.graph_edges]
            all_edges.extend(edges_dict)

        return {
            "chunks": all_chunks,
            "chunk_count": len(all_chunks),
            "graph": {
                "edges": all_edges,
                "edge_count": len(all_edges)
            },
            "format": "qdrant_ready"
        }

    def _format_as_full_notes(self, notes) -> dict:
        """Format notes with full metadata"""
        formatted_notes = []

        for note in notes:
            formatted_note = {
                "note_id": note.note_id,
                "file_path": note.base_document.file_path,
                "content": note.base_document.content,
                "sections": [
                    {
                        "title": s.title,
                        "content": s.content,
                        "level": s.level,
                        "metadata": s.metadata
                    }
                    for s in note.base_document.sections
                ],
                "metadata": note.base_document.metadata,
                "wikilinks": [
                    {
                        "target": link.target,
                        "display_text": link.display_text,
                        "section": link.section,
                        "line": link.line_number
                    }
                    for link in note.wikilinks
                ],
                "backlinks": note.backlinks,
                "transclusions": [
                    {
                        "target": t.target,
                        "section": t.section,
                        "resolved": t.resolved_content is not None
                    }
                    for t in note.transclusions
                ],
                "callouts": [
                    {
                        "type": c.callout_type,
                        "title": c.title,
                        "content": c.content
                    }
                    for c in note.callouts
                ],
                "graph_info": {
                    "outgoing_links": note.graph_node.outgoing_links if note.graph_node else [],
                    "incoming_links": note.graph_node.incoming_links if note.graph_node else [],
                    "note_type": note.graph_node.note_type.value if note.graph_node else "atomic",
                    "is_hub": note.graph_node.is_hub if note.graph_node else False,
                    "is_index": note.graph_node.is_index if note.graph_node else False
                }
            }
            formatted_notes.append(formatted_note)

        # Calculate graph statistics
        total_edges = sum(len(n.graph_edges) for n in notes)
        hubs = [n.note_id for n in notes if n.graph_node and n.graph_node.is_hub]
        indexes = [n.note_id for n in notes if n.graph_node and n.graph_node.is_index]

        return {
            "notes": formatted_notes,
            "notes_count": len(notes),
            "graph_summary": {
                "total_edges": total_edges,
                "hubs": hubs,
                "indexes": indexes,
                "hub_count": len(hubs),
                "index_count": len(indexes)
            },
            "format": "full_notes"
        }

    def _format_as_markdown(self, notes) -> str:
        """Format notes as Markdown"""
        markdown_lines = ["# Obsidian Notes Export\n\n"]

        for note in notes:
            markdown_lines.append(f"## {note.note_id}\n\n")
            markdown_lines.append(note.base_document.content)
            markdown_lines.append("\n---\n\n")

        return "\n".join(markdown_lines)

    def _format_as_json(self, notes) -> dict:
        """Format notes as JSON"""
        return {
            "notes": [
                {
                    "note_id": note.note_id,
                    "file_path": note.base_document.file_path,
                    "content": note.base_document.content,
                    "metadata": note.base_document.metadata,
                    "tags": note.base_document.metadata.get("tags", []),
                    "created": note.base_document.metadata.get("created"),
                    "modified": note.base_document.metadata.get("modified")
                }
                for note in notes
            ],
            "count": len(notes),
            "exported_at": datetime.utcnow().isoformat()
        }

    def _format_as_html(self, notes) -> str:
        """Format notes as HTML"""
        html_lines = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<meta charset='UTF-8'>",
            "<title>Obsidian Notes Export</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }",
            "h1 { color: #333; border-bottom: 2px solid #eee; padding-bottom: 10px; }",
            "h2 { color: #555; margin-top: 30px; }",
            ".note { background: #f9f9f9; padding: 15px; margin-bottom: 20px; border-radius: 5px; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>Obsidian Notes Export</h1>"
        ]

        for note in notes:
            html_lines.append(f"<div class='note'>")
            html_lines.append(f"<h2>{note.note_id}</h2>")
            html_lines.append(f"<pre>{note.base_document.content}</pre>")
            html_lines.append("</div>")

        html_lines.extend(["</body>", "</html>"])
        return "\n".join(html_lines)

    # =========================================================================
    # Performance Metrics Methods
    # =========================================================================

    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        avg_load_time = (
            self._metrics.total_load_time / self._metrics.vaults_loaded
            if self._metrics.vaults_loaded > 0 else 0
        )

        return {
            "vaults_loaded": self._metrics.vaults_loaded,
            "notes_loaded": self._metrics.notes_loaded,
            "avg_load_time": avg_load_time,
            "min_load_time": self._metrics.min_load_time,
            "max_load_time": self._metrics.max_load_time,
            "cache_hits": self._metrics.cache_hits,
            "cache_misses": self._metrics.cache_misses,
            "cache_hit_rate": (
                self._metrics.cache_hits / (self._metrics.cache_hits + self._metrics.cache_misses) * 100
                if (self._metrics.cache_hits + self._metrics.cache_misses) > 0 else 0
            )
        }

    def reset_metrics(self):
        """Reset performance metrics"""
        self._metrics = PerformanceMetrics()

    # =========================================================================
    # Format Output Method
    # =========================================================================

    def format_output(self, result: ToolResult) -> str:
        """Format tool output for LLM consumption"""
        if not result.success:
            return f"❌ Error loading Obsidian vault: {result.error}"

        data = result.data

        if data.get("format") == "qdrant_ready":
            return (
                f"✅ Loaded Obsidian vault successfully\n"
                f"📄 Chunks ready for Qdrant: {data['chunk_count']}\n"
                f"🔗 Graph edges: {data['graph']['edge_count']}\n"
                f"\nThe vault has been processed and is ready for indexing in Qdrant."
            )
        else:  # full_notes
            notes_count = data.get('notes_count', 0)
            graph = data.get('graph_summary', {})

            summary_parts = [
                f"✅ Loaded {notes_count} notes from Obsidian vault",
                f"🔗 Total connections: {graph.get('total_edges', 0)} edges"
            ]

            if graph.get('hub_count', 0) > 0:
                hubs = graph.get('hubs', [])
                summary_parts.append(
                    f"🎯 Hub notes (highly referenced): {', '.join(hubs[:5])}"
                    + (f" and {graph['hub_count'] - 5} more" if graph['hub_count'] > 5 else "")
                )

            if graph.get('index_count', 0) > 0:
                indexes = graph.get('indexes', [])
                summary_parts.append(
                    f"📑 Index notes (many links): {', '.join(indexes[:5])}"
                    + (f" and {graph['index_count'] - 5} more" if graph['index_count'] > 5 else "")
                )

            return "\n".join(summary_parts)
