# =============================================================================
# src/services/obsidian/graph_utils.py
# Utilidades para procesamiento de chunks Obsidian y generación de Mermaid
# =============================================================================
"""
Funciones utilitarias para trabajar con chunks de Obsidian en RAG:
- Agrupar chunks por nota
- Generar código Mermaid para visualización
"""
from typing import Dict, List, Any, Set, Tuple, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


def group_chunks_by_note(chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Agrupa chunks de Qdrant por source_note y genera contexto por nota.

    Args:
        chunks: Lista de chunks devueltos por Qdrant con estructura:
            {
                'id': int,
                'payload': {
                    'source_note': str,
                    'file': str,
                    'section': str,
                    'content': str,
                    'outgoing_links': List[str],
                    'incoming_links': List[str],
                    'tags': List[str],
                    'note_type': str,
                    'is_hub': bool,
                    'is_index': bool,
                    'doc_title': str,
                    'doc_type': str,
                    'doc_subtype': str,
                    'doc_system': str,
                    'doc_owner': str,
                    'frontmatter': dict,
                    ...
                },
                'score': float (opcional)
            }

    Returns:
        Dict con estructura:
            {
                'note_id': {
                    'title': str,
                    'file': str,
                    'tags': List[str],
                    'context': str,  # Texto unificado de chunks
                    'sections': List[str],  # Secciones únicas
                    'outgoing_links': List[str],
                    'incoming_links': List[str],
                    'note_type': str,
                    'is_hub': bool,
                    'is_index': bool,
                    'doc_metadata': dict,  # doc_type, doc_system, etc.
                    'chunk_count': int,
                    'avg_score': float
                },
                ...
            }
    """
    notes = {}

    for chunk in chunks:
        payload = chunk.get('payload', {})
        note_id = payload.get('source_note')

        if not note_id:
            logger.warning(f"Chunk without source_note: {chunk.get('id')}")
            continue

        if note_id not in notes:
            # Extraer metadata del frontmatter
            frontmatter = payload.get('frontmatter', {})

            notes[note_id] = {
                'title': payload.get('doc_title') or frontmatter.get('title') or note_id,
                'file': payload.get('file', ''),
                'tags': payload.get('tags', []),
                'outgoing_links': payload.get('outgoing_links', []),
                'incoming_links': payload.get('incoming_links', []),
                'note_type': payload.get('note_type', 'atomic'),
                'is_hub': payload.get('is_hub', False),
                'is_index': payload.get('is_index', False),
                'doc_metadata': {
                    'type': payload.get('doc_type'),
                    'subtype': payload.get('doc_subtype'),
                    'system': payload.get('doc_system'),
                    'owner': payload.get('doc_owner'),
                    'severity': payload.get('doc_severity'),
                    'updated': frontmatter.get('updated')
                },
                'chunks': [],
                'sections': set(),
                'scores': []
            }

        # Agregar chunk y score
        section = payload.get('section', '')
        content = payload.get('content', '')
        score = chunk.get('score', 0.0)

        notes[note_id]['chunks'].append({
            'content': content,
            'section': section,
            'score': score
        })

        if section:
            notes[note_id]['sections'].add(section)

        notes[note_id]['scores'].append(score)

    # Unificar chunks por nota
    for note_id, note_data in notes.items():
        # Ordenar chunks por score (descendente)
        note_data['chunks'].sort(key=lambda x: x['score'], reverse=True)

        # Tomar top 3-5 chunks más relevantes
        top_chunks = note_data['chunks'][:5]

        # Construir contexto unificado (agrupar por sección si existe)
        context_parts = []
        seen_sections = set()

        for chunk in top_chunks:
            section = chunk['section']
            content = chunk['content']

            if section and section not in seen_sections:
                context_parts.append(f"### {section}\n{content}")
                seen_sections.add(section)
            elif not section:
                context_parts.append(content)

        note_data['context'] = '\n\n'.join(context_parts)
        note_data['sections'] = list(note_data['sections'])
        note_data['chunk_count'] = len(note_data['chunks'])
        note_data['avg_score'] = (
            sum(note_data['scores']) / len(note_data['scores'])
            if note_data['scores'] else 0.0
        )

        # Limpiar datos temporales
        del note_data['chunks']
        del note_data['scores']

    logger.info(
        f"Grouped {sum(n['chunk_count'] for n in notes.values())} chunks into {len(notes)} notes",
        extra={'total_chunks': sum(n['chunk_count'] for n in notes.values()), 'notes': len(notes)}
    )

    return notes


def generate_mermaid(
    notes: Dict[str, Dict[str, Any]],
    max_nodes: int = 10,
    max_edges_per_node: int = 5,
    vault_name: Optional[str] = None,
    include_metadata: bool = True
) -> Optional[str]:
    """
    Genera código Mermaid (flowchart) para las notas usadas en la respuesta.

    Args:
        notes: Dict de notas agrupadas (output de group_chunks_by_note)
        max_nodes: Máximo de nodos a incluir en el grafo (default: 10)
        max_edges_per_node: Máximo de enlaces salientes por nodo (default: 5)
        vault_name: Nombre del vault para enlaces clicables (opcional)
        include_metadata: Incluir tooltips con metadata (tags, type, etc.)

    Returns:
        Código Mermaid en formato string (con ```mermaid wrapper), o None si < 2 notas
    """
    if not notes or len(notes) < 2:
        logger.debug("Not enough notes for Mermaid graph (need at least 2)")
        return None

    lines = ["```mermaid", "graph TD"]

    # Ordenar notas por relevancia (avg_score) y limitar
    sorted_notes = sorted(
        notes.items(),
        key=lambda x: x[1]['avg_score'],
        reverse=True
    )[:max_nodes]

    note_ids = [note_id for note_id, _ in sorted_notes]
    note_set = set(note_ids)

    # 1) Definición de nodos con estilos según tipo
    for note_id in note_ids:
        data = notes[note_id]

        # Sanitizar ID para Mermaid
        safe_id = _sanitize_mermaid_id(note_id)

        # Título del nodo (truncar si es muy largo)
        title = data['title']
        if len(title) > 40:
            title = title[:37] + '...'

        # Escapar comillas
        safe_label = title.replace('"', '\\"')

        # Agregar metadata al label si include_metadata=True
        if include_metadata:
            doc_type = data['doc_metadata'].get('type', '')
            doc_system = data['doc_metadata'].get('system', '')
            if doc_type and doc_system:
                safe_label = f"{safe_label}\\n[{doc_type} - {doc_system}]"

        # Determinar forma del nodo según tipo de nota
        node_shape = _get_node_shape(data)

        # Determinar clase CSS según características
        node_class = _get_node_class(data)

        lines.append(f'    {safe_id}{node_shape[0]}"{safe_label}"{node_shape[1]}:::{node_class}')

    # 2) Aristas usando outgoing_links (solo a notas del subconjunto)
    seen_edges: Set[Tuple[str, str]] = set()

    for note_id in note_ids:
        data = notes[note_id]
        safe_src = _sanitize_mermaid_id(note_id)

        # Filtrar outgoing_links:
        # - Solo notas (excluir imágenes .png, .jpg, etc.)
        # - Solo si están en el subconjunto de notas
        outgoing = [
            link for link in data.get('outgoing_links', [])
            if link in note_set  # Está en nuestro subconjunto
        ]

        # Limitar a max_edges_per_node
        for target in outgoing[:max_edges_per_node]:
            safe_dst = _sanitize_mermaid_id(target)
            edge_key = (safe_src, safe_dst)

            # Evitar duplicados
            if edge_key in seen_edges:
                continue

            seen_edges.add(edge_key)

            # Tipo de flecha según relación
            arrow_style = _get_arrow_style(notes[note_id], notes[target])
            lines.append(f"    {safe_src} {arrow_style} {safe_dst}")

    # 3) Estilos CSS para diferentes tipos de notas
    lines.extend([
        "",
        "    %% Estilos de nodos",
        "    classDef hub fill:#ff9999,stroke:#cc0000,stroke-width:3px,color:#000",
        "    classDef index fill:#99ccff,stroke:#0066cc,stroke-width:2px,color:#000",
        "    classDef runbook fill:#ffcc99,stroke:#ff8800,stroke-width:2px,color:#000",
        "    classDef guide fill:#99ff99,stroke:#00cc00,stroke-width:2px,color:#000",
        "    classDef reference fill:#cc99ff,stroke:#8800cc,stroke-width:2px,color:#000",
        "    classDef default fill:#e0e0e0,stroke:#888,stroke-width:1px,color:#000"
    ])

    # 4) Enlaces clicables (opcional, si se provee vault_name)
    if vault_name:
        lines.append("")
        lines.append("    %% Enlaces clicables")
        for note_id in note_ids:
            safe_id = _sanitize_mermaid_id(note_id)
            # URL para abrir en Obsidian
            # Formato: obsidian://open?vault=VaultName&file=NoteName
            obsidian_url = f"obsidian://open?vault={vault_name}&file={note_id}"
            lines.append(f'    click {safe_id} "{obsidian_url}" _blank')

    lines.append("```")

    mermaid_code = '\n'.join(lines)
    logger.info(
        f"Generated Mermaid graph",
        extra={
            'nodes': len(note_ids),
            'edges': len(seen_edges),
            'has_vault_links': vault_name is not None
        }
    )

    return mermaid_code


# =============================================================================
# Funciones auxiliares privadas
# =============================================================================

def _sanitize_mermaid_id(node_id: str) -> str:
    """Sanitiza un ID de nota para Mermaid (alfanumérico + underscore)."""
    safe = node_id.replace('-', '_').replace(' ', '_').replace('.', '_')
    safe = ''.join(c for c in safe if c.isalnum() or c == '_')
    return safe or 'node'  # Fallback si queda vacío


def _get_node_shape(note_data: Dict[str, Any]) -> Tuple[str, str]:
    """
    Determina la forma del nodo según el tipo de documento.

    Returns:
        Tuple (opening, closing) para el nodo Mermaid.
        Ejemplos:
        - Runbook: [", "]  (rectángulo)
        - Guide: (", ")  (rectángulo redondeado)
        - Reference: {", "}  (rombo)
        - Default: [", "]
    """
    doc_type = note_data.get('doc_metadata', {}).get('type', '').lower()

    if doc_type == 'runbook':
        return ('[', ']')  # Rectángulo
    elif doc_type == 'guide':
        return ('([', '])')  # Stadium (rectángulo redondeado)
    elif doc_type == 'reference':
        return ('{{', '}}')  # Hexágono
    else:
        return ('[', ']')  # Default: rectángulo


def _get_node_class(note_data: Dict[str, Any]) -> str:
    """
    Determina la clase CSS del nodo según sus características.

    Prioridad:
    1. is_hub → hub
    2. is_index → index
    3. doc_type (runbook, guide, reference)
    4. default
    """
    if note_data.get('is_hub'):
        return 'hub'
    elif note_data.get('is_index'):
        return 'index'

    doc_type = note_data.get('doc_metadata', {}).get('type', '').lower()

    if doc_type == 'runbook':
        return 'runbook'
    elif doc_type == 'guide':
        return 'guide'
    elif doc_type == 'reference':
        return 'reference'
    else:
        return 'default'


def _get_arrow_style(source_note: Dict[str, Any], target_note: Dict[str, Any]) -> str:
    """
    Determina el estilo de flecha según la relación entre notas.

    Returns:
        String con el estilo de flecha Mermaid.
        Ejemplos:
        - -->  (normal)
        - ==>  (gruesa, para relaciones importantes)
        - -.->  (punteada, para referencias débiles)
    """
    # Si source es runbook y target también → flecha gruesa
    source_type = source_note.get('doc_metadata', {}).get('type', '').lower()
    target_type = target_note.get('doc_metadata', {}).get('type', '').lower()

    if source_type == 'runbook' and target_type == 'runbook':
        return '==>'  # Flecha gruesa
    elif source_type == 'reference' or target_type == 'reference':
        return '-.->|ref|'  # Flecha punteada con label
    else:
        return '-->'  # Flecha normal
