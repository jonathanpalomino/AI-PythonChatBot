# =============================================================================
# src/services/obsidian/rag_enhancer.py
# Enriquecimiento de respuestas RAG con contexto Obsidian
# =============================================================================
"""
Funciones para enriquecer respuestas RAG con:
- Agrupación de chunks por nota
- Generación de grafo Mermaid
- Metadata de fuentes
"""
from typing import Dict, List, Any, Optional
from src.services.obsidian.graph_utils import group_chunks_by_note, generate_mermaid
from src.utils.logger import get_logger

logger = get_logger(__name__)


def enhance_obsidian_response(
    chunks: List[Dict[str, Any]],
    llm_response: Dict[str, Any],
    min_notes_for_graph: int = 2,
    vault_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Enriquece una respuesta RAG con metadata de Obsidian.

    Agrega:
    - Notas agrupadas por source_note
    - Lista de fuentes (note_id + título + metadata)
    - Código Mermaid si hay ≥2 notas relacionadas

    Args:
        chunks: Chunks devueltos por Qdrant
        llm_response: Respuesta del LLM con estructura:
            {
                'content': str,
                'metadata': dict,
                'model': str,
                'provider': str,
                ...
            }
        min_notes_for_graph: Mínimo de notas para generar grafo (default: 2)
        vault_name: Nombre del vault para enlaces clicables (opcional)

    Returns:
        llm_response actualizado con:
            llm_response['metadata']['obsidian'] = {
                'notes_used': Dict[note_id, metadata],
                'sources': List[Dict],
                'mermaid_graph': str (opcional),
                'stats': Dict
            }
    """
    if not chunks:
        logger.debug("No chunks provided, skipping Obsidian enhancement")
        return llm_response

    # 1) Agrupar chunks por nota
    notes = group_chunks_by_note(chunks)

    # 2) Generar lista de fuentes para frontend (ordenada por relevancia)
    sources = []
    for note_id, note_data in notes.items():
        sources.append({
            'note_id': note_id,
            'title': note_data['title'],
            'file': note_data['file'],
            'tags': note_data['tags'],
            'sections': note_data['sections'],
            'note_type': note_data['note_type'],
            'doc_type': note_data['doc_metadata'].get('type'),
            'doc_system': note_data['doc_metadata'].get('system'),
            'doc_owner': note_data['doc_metadata'].get('owner'),
            'chunk_count': note_data['chunk_count'],
            'relevance': round(note_data['avg_score'], 3),
            'is_hub': note_data['is_hub'],
            'is_index': note_data['is_index']
        })

    # Ordenar fuentes por relevancia (avg_score)
    sources.sort(key=lambda x: x['relevance'], reverse=True)

    # 3) Generar Mermaid si hay suficientes notas
    mermaid_code = None
    if len(notes) >= min_notes_for_graph:
        mermaid_code = generate_mermaid(
            notes,
            vault_name=vault_name,
            include_metadata=True
        )

    # 4) Agregar metadata al response
    if 'metadata' not in llm_response:
        llm_response['metadata'] = {}

    llm_response['metadata']['obsidian'] = {
        'notes_used': notes,
        'sources': sources,
        'mermaid_graph': mermaid_code,
        'stats': {
            'total_chunks': len(chunks),
            'notes_count': len(notes),
            'has_graph': mermaid_code is not None,
            'hub_notes': sum(1 for n in notes.values() if n['is_hub']),
            'index_notes': sum(1 for n in notes.values() if n['is_index'])
        }
    }

    logger.info(
        f"Enhanced response with Obsidian context",
        extra={
            'notes_count': len(notes),
            'has_mermaid': mermaid_code is not None,
            'hub_count': llm_response['metadata']['obsidian']['stats']['hub_notes']
        }
    )

    return llm_response
