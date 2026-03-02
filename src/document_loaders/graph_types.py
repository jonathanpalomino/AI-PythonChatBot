# =============================================================================
# src/document_loaders/graph_types.py
# Shared types for knowledge graph representation (RAG-ready for Qdrant)
# =============================================================================
"""
Tipos compartidos para representación de grafos de conocimiento.
Diseñado para indexación en Qdrant con metadata de relaciones.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
from pathlib import Path


class EdgeType(str, Enum):
    """Tipos de aristas en el grafo de conocimiento"""
    WIKILINK = "wikilink"           # [[nota]] - Enlace estándar
    TRANSCLUSION = "transclusion"   # ![[nota]] - Contenido embebido
    BLOCK_REF = "block_ref"         # ^block-id - Referencia a bloque
    TAG = "tag"                     # #tag - Relación por tag compartido
    ALIAS = "alias"                 # Alias en frontmatter
    PARENT_CHILD = "parent_child"   # Relación jerárquica de carpetas


class NoteType(str, Enum):
    """Clasificación de notas según conectividad en el grafo"""
    HUB = "hub"           # Muchos incoming links (>5) - Concepto central
    INDEX = "index"       # Muchos outgoing links (>10) - Nota índice/MOC
    ATOMIC = "atomic"     # Pocos links (<3 total) - Nota atómica
    BRIDGE = "bridge"     # Balance de incoming/outgoing - Conectora


@dataclass
class GraphEdge:
    """
    Arista en el grafo de conocimiento.
    
    Representa una conexión direccional entre dos notas.
    """
    source_note: str                    # Nota origen
    target_note: str                    # Nota destino
    edge_type: EdgeType                 # Tipo de relación
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata opcional según tipo de edge
    section: Optional[str] = None       # Sección donde aparece el link
    block_id: Optional[str] = None      # ID de bloque (para block_ref)
    line_number: Optional[int] = None   # Línea donde aparece
    context: Optional[str] = None       # Contexto textual alrededor del link

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización"""
        return {
            "source": self.source_note,
            "target": self.target_note,
            "type": self.edge_type.value,
            "section": self.section,
            "block_id": self.block_id,
            "line_number": self.line_number,
            "context": self.context,
            **self.metadata
        }


@dataclass
class GraphNode:
    """
    Nodo en el grafo de conocimiento.
    
    Representa una nota con sus características y conectividad.
    """
    note_id: str                        # ID único (normalmente filename sin extensión)
    note_name: str                      # Nombre display
    file_path: Path                     # Ruta al archivo
    aliases: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    # Conectividad (computed)
    outgoing_links: List[str] = field(default_factory=list)
    incoming_links: List[str] = field(default_factory=list)
    
    # Clasificación (computed)
    note_type: NoteType = NoteType.ATOMIC
    is_hub: bool = False
    is_index: bool = False
    
    # Metadata adicional
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def link_count(self) -> int:
        """Total de links (incoming + outgoing)"""
        return len(self.outgoing_links) + len(self.incoming_links)
    
    def classify_note_type(self) -> NoteType:
        """
        Clasifica el tipo de nota según conectividad.
        
        Returns:
            NoteType clasificado
        """
        incoming_count = len(self.incoming_links)
        outgoing_count = len(self.outgoing_links)
        total = incoming_count + outgoing_count
        
        if incoming_count > 5:
            self.is_hub = True
            return NoteType.HUB
        elif outgoing_count > 10:
            self.is_index = True
            return NoteType.INDEX
        elif total < 3:
            return NoteType.ATOMIC
        else:
            return NoteType.BRIDGE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización"""
        return {
            "note_id": self.note_id,
            "note_name": self.note_name,
            "file_path": str(self.file_path),
            "aliases": self.aliases,
            "tags": self.tags,
            "outgoing_links": self.outgoing_links,
            "incoming_links": self.incoming_links,
            "link_count": self.link_count,
            "note_type": self.note_type.value,
            "is_hub": self.is_hub,
            "is_index": self.is_index,
            **self.metadata
        }


@dataclass
class KnowledgeGraph:
    """
    Grafo de conocimiento completo.
    
    Contiene nodos (notas) y aristas (relaciones).
    """
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: List[GraphEdge] = field(default_factory=list)
    
    def add_node(self, node: GraphNode) -> None:
        """Agrega un nodo al grafo"""
        self.nodes[node.note_id] = node
    
    def add_edge(self, edge: GraphEdge) -> None:
        """Agrega una arista al grafo"""
        self.edges.append(edge)
        
        # Actualizar listas de links en nodos
        if edge.source_note in self.nodes:
            source_node = self.nodes[edge.source_note]
            if edge.target_note not in source_node.outgoing_links:
                source_node.outgoing_links.append(edge.target_note)
        
        if edge.target_note in self.nodes:
            target_node = self.nodes[edge.target_note]
            if edge.source_note not in target_node.incoming_links:
                target_node.incoming_links.append(edge.source_note)
    
    def classify_all_nodes(self) -> None:
        """Clasifica todos los nodos según su conectividad"""
        for node in self.nodes.values():
            node.note_type = node.classify_note_type()
    
    def get_node(self, note_id: str) -> Optional[GraphNode]:
        """Obtiene un nodo por ID"""
        return self.nodes.get(note_id)
    
    def get_neighbors(self, note_id: str, direction: str = "both") -> List[str]:
        """
        Obtiene vecinos de un nodo.
        
        Args:
            note_id: ID de la nota
            direction: "in", "out", o "both"
            
        Returns:
            Lista de IDs de notas vecinas
        """
        node = self.get_node(note_id)
        if not node:
            return []
        
        if direction == "in":
            return node.incoming_links
        elif direction == "out":
            return node.outgoing_links
        else:  # both
            return list(set(node.incoming_links + node.outgoing_links))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización"""
        return {
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "edges": [edge.to_dict() for edge in self.edges],
            "node_count": len(self.nodes),
            "edge_count": len(self.edges)
        }


@dataclass
class QdrantChunkMetadata:
    """
    Metadata enriquecida para chunks indexados en Qdrant.
    
    Incluye información de grafo para queries contextuales.
    """
    # Identificación del documento
    source_note: str
    file_path: str
    section: str
    
    # Información de grafo
    outgoing_links: List[str] = field(default_factory=list)
    incoming_links: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    
    # Clasificación
    note_type: str = NoteType.ATOMIC.value
    is_hub: bool = False
    is_index: bool = False
    
    # Jerarquía/contexto
    heading_path: Optional[List[str]] = None  # ["H1", "H2", "H3"]
    parent_section: Optional[str] = None

    # Indexación
    chunk_index: int = 0
    total_chunks: int = 1
    indexed_at: Optional[str] = None

    # Metadata adicional de frontmatter u otras fuentes
    extra_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_qdrant_payload(self) -> Dict[str, Any]:
        """
        Convierte a formato de payload de Qdrant.
        
        Returns:
            Dict compatible con Qdrant metadata
        """
        payload = {
            "source_note": self.source_note,
            "file": self.file_path,
            "section": self.section,
            "outgoing_links": self.outgoing_links,
            "incoming_links": self.incoming_links,
            "tags": self.tags,
            "aliases": self.aliases,
            "note_type": self.note_type,
            "is_hub": self.is_hub,
            "is_index": self.is_index,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "indexed_at": self.indexed_at,
        }
        
        if self.heading_path:
            payload["heading_path"] = self.heading_path
        
        if self.parent_section:
            payload["parent_section"] = self.parent_section
        
        # Merge extra metadata
        payload.update(self.extra_metadata)
        
        return payload
