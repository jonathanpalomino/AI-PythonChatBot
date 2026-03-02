# =============================================================================
# src/services/chat/response_formatter.py
# Response Formatter - Response formatting logic (REFACTORED from chat_orchestrator.py)
# =============================================================================
"""
ResponseFormatter: Responsable de formatear respuestas del LLM.

Responsabilidades (SRP):
- Formatear respuestas para el cliente
- Extraer y formatear metadata
- Construir respuestas de error amigables
- Formatear sources y referencias
- Validar respuestas en modo estricto
- Extraer metadata de RAG
"""
import os
import re
from typing import Dict, Any, List, Optional

from src.providers.manager import ChatResponse
from src.utils.logger import get_logger


class ResponseFormatter:
    """
    Responsable de formatear respuestas del LLM.
    Separa la lógica de formateo de la orquestación.
    
    Migrado desde ChatOrchestrator para cumplir SRP.
    """

    def __init__(self):
        self.logger = get_logger(__name__)

    # =============================================================================
    # Response Formatting
    # =============================================================================

    def format_response(
        self,
        response: ChatResponse,
        tools_executed: List[str] = None,
        rag_metadata: Dict[str, Any] = None,
        mode: str = "agent"
    ) -> ChatResponse:
        """
        Format the final response with metadata.
        
        Args:
            response: Raw LLM response
            tools_executed: List of executed tools
            rag_metadata: RAG metadata if available
            mode: Processing mode (agent/manual)
            
        Returns:
            Formatted ChatResponse
        """
        # Ensure metadata dict exists
        if response.metadata is None:
            response.metadata = {}
        
        # Add tools executed
        if tools_executed:
            response.metadata["tools_executed"] = tools_executed
        
        # Add mode
        response.metadata["mode"] = mode
        
        # Add RAG metadata
        if rag_metadata:
            response.metadata["rag_metadata"] = rag_metadata
            
            # Add sources to response content
            sources_text = self.format_sources_text(rag_metadata)
            if sources_text and response.content:
                response.content += sources_text
        
        return response

    def format_empty_response(
        self,
        original_response: ChatResponse,
        thinking_content: Optional[str] = None
    ) -> ChatResponse:
        """
        Format response when LLM returns empty content.
        
        Args:
            original_response: Original response with empty content
            thinking_content: Thinking content if available
            
        Returns:
            ChatResponse with friendly message
        """
        if thinking_content:
            message = "[El modelo solo generó razonamiento interno sin respuesta final. Por favor, intenta reformular tu pregunta.]"
        else:
            message = "[El modelo no generó una respuesta. Por favor, intenta de nuevo.]"
        
        return ChatResponse(
            content=message,
            model=original_response.model,
            provider=original_response.provider,
            tokens_used=original_response.tokens_used,
            cost=original_response.cost,
            metadata=original_response.metadata or {}
        )

    # =============================================================================
    # Sources Formatting (Migrado desde ChatOrchestrator)
    # =============================================================================

    def format_sources_text(self, rag_metadata: Dict[str, Any]) -> str:
        """
        Formatea la lista de fuentes encontradas en el RAG.
        Extrae solo el nombre base de los archivos para limpieza.
        
        Migrado desde ChatOrchestrator._format_sources_text()
        """
        if not rag_metadata or not rag_metadata.get('files'):
            return ""

        sources = rag_metadata['files']
        if not sources:
            return ""

        # Limpiar y deduplicar nombres de archivo (nombres base únicamente)
        clean_sources = []
        seen = set()
        for f in sources:
            if not f:
                continue
            base = os.path.basename(f)
            if base not in seen:
                clean_sources.append(base)
                seen.add(base)

        if not clean_sources:
            return ""

        output = "\n\n**Fuentes:**"
        for s in clean_sources:
            output += f"\n- {s}"

        # Agregar sección de Elementos Relacionados (outgoing_links)
        related = rag_metadata.get('related_elements')
        if related:
            # Filtrar vacíos y deduplicar
            clean_related = sorted(list({str(r).strip() for r in related if r}))
            if clean_related:
                output += "\n\n**Elementos relacionados:**"
                for r in clean_related:
                    output += f"\n- {r}"

        return output

    def _format_sources_text(
        self,
        rag_metadata: Dict[str, Any]
    ) -> Optional[str]:
        """
        Format sources from RAG metadata (alias for compatibility).
        
        Args:
            rag_metadata: RAG metadata dict
            
        Returns:
            Formatted sources string or None
        """
        sources = rag_metadata.get("sources", [])
        if not sources:
            return None
        
        parts = ["\n\n---\n**Fuentes consultadas:**"]
        
        for i, source in enumerate(sources, 1):
            filename = source.get("filename", "Unknown")
            score = source.get("score", 0)
            parts.append(f"{i}. {filename} (relevancia: {score:.2f})")
        
        return "\n".join(parts)

    def format_sources_list(
        self,
        rag_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Format RAG results as sources list.
        
        Args:
            rag_results: RAG search results
            
        Returns:
            List of source dicts
        """
        sources = []
        seen_files = set()
        
        for result in rag_results:
            filename = result.get("file", "Unknown")
            
            if filename not in seen_files:
                sources.append({
                    "filename": filename,
                    "score": result.get("score", 0),
                    "chunk_index": result.get("chunk_index")
                })
                seen_files.add(filename)
        
        return sources

    # =============================================================================
    # RAG Metadata Extraction (Migrado desde ChatOrchestrator)
    # =============================================================================

    def extract_rag_metadata(
        self,
        rag_context: Any,
        chunks: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Extract metadata from RAG results.
        If chunks are provided, extract directly from structured data (more reliable).
        Otherwise, parse from formatted string context.
        
        Migrado desde ChatOrchestrator._extract_rag_metadata()
        """
        metadata = {
            'files': [],
            'symbols': [],
            'languages': set(),
            'sources': [],
            'total_chunks': 0,
            'avg_score': 0.0
        }

        # PRIORIDAD 1: Extraer directo de chunks (más confiable)
        if chunks:
            scores = []
            seen_sources = set()
            
            for chunk in chunks:
                # Extraer filename desde metadata
                filename = chunk.get('metadata', {}).get('file') or chunk.get('file') or chunk.get('filePath')
                if filename and filename not in metadata['files']:
                    metadata['files'].append(filename)

                    # Detectar lenguaje por extensión
                    ext = filename.split('.')[-1].lower() if '.' in filename else None
                    if ext:
                        metadata['languages'].add(ext)

                # Extraer outgoing_links para "Elementos relacionados"
                links = chunk.get('metadata', {}).get('outgoing_links') or chunk.get('outgoing_links')
                if links and isinstance(links, list):
                    if 'related_elements' not in metadata:
                        metadata['related_elements'] = set()
                    metadata['related_elements'].update(links)
                
                # Extract score
                score = chunk.get('score', 0)
                if score:
                    scores.append(score)
                
                # Build sources list
                if filename and filename not in seen_sources:
                    metadata['sources'].append({
                        "filename": filename,
                        "score": score
                    })
                    seen_sources.add(filename)

            metadata['total_chunks'] = len(chunks)
            if scores:
                metadata['avg_score'] = sum(scores) / len(scores)

            self.logger.debug(
                f"Extracted metadata from chunks",
                extra={
                    "files": metadata['files'],
                    "languages": list(metadata['languages']),
                    "chunk_count": len(chunks)
                }
            )

        # FALLBACK: Parsear del string (si chunks no tienen metadata)
        if rag_context and isinstance(rag_context, str):
            try:
                # Extraer filenames del contexto formateado
                file_patterns = [
                    r"File:\s*([^\n]+)",
                    r"Source:\s*([^\n]+)",
                    r"\[([^\]]+\.(py|js|ts|md|txt|json|yaml|yml|toml|rs|go|java|cpp|c|h|hpp))\]",
                    r"From\s+([^\n]+\.(py|js|ts|md|txt|json|yaml|yml|toml|rs|go|java|cpp|c|h|hpp))",
                ]

                for pattern in file_patterns:
                    matches = re.findall(pattern, rag_context, re.IGNORECASE)
                    for match in matches:
                        if isinstance(match, tuple):
                            filename = match[0].strip()
                        else:
                            filename = match.strip()

                        if filename and filename not in metadata['files']:
                            metadata['files'].append(filename)

                            # Detectar lenguaje
                            ext = filename.split('.')[-1].lower() if '.' in filename else None
                            if ext:
                                metadata['languages'].add(ext)

            except Exception as e:
                self.logger.warning(f"Error extracting RAG metadata: {e}")

        # FINAL: Convertir sets a listas para serialización JSON
        if isinstance(metadata.get('languages'), set):
            metadata['languages'] = sorted(list(metadata['languages']))

        if isinstance(metadata.get('related_elements'), set):
            metadata['related_elements'] = sorted(list(metadata['related_elements']))

        return metadata

    # =============================================================================
    # Error Formatting
    # =============================================================================

    def format_error_response(
        self,
        error: Exception,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Format error for client response.
        
        Args:
            error: Exception that occurred
            context: Additional context
            
        Returns:
            Error dict for response
        """
        error_msg = str(error)
        
        # Check for specific error types
        if "model_decommissioned" in error_msg.lower():
            return {
                "error": "model_decommissioned",
                "message": "El modelo seleccionado ya no está disponible (descontinuado por el proveedor).",
                "suggestion": "Por favor, selecciona otro modelo en la configuración."
            }
        
        if "429" in error_msg or "rate_limit" in error_msg.lower():
            return {
                "error": "rate_limit",
                "message": "El proveedor de IA está saturado temporalmente (Rate Limit 429).",
                "suggestion": "Espera unos segundos e intenta de nuevo."
            }
        
        if "context_length_exceeded" in error_msg.lower():
            return {
                "error": "context_too_long",
                "message": "El mensaje es demasiado largo para el modelo seleccionado.",
                "suggestion": "Reduce el tamaño del mensaje o selecciona un modelo con mayor contexto."
            }
        
        # Generic error
        return {
            "error": "unknown",
            "message": f"Error: {error_msg}",
            "suggestion": "Intenta de nuevo o contacta soporte si el problema persiste."
        }

    # =============================================================================
    # Tool Results Formatting
    # =============================================================================

    def format_tool_results_for_llm(
        self,
        tool_results: List[Dict[str, Any]]
    ) -> str:
        """
        Format tool results for LLM consumption.
        
        Args:
            tool_results: List of tool result dicts
            
        Returns:
            Formatted string for LLM
        """
        parts = []
        
        for result in tool_results:
            tool_name = result.get("tool_name", "unknown")
            success = result.get("success", False)
            content = result.get("result", "")
            
            if success:
                parts.append(f"**{tool_name}**:\n{content}")
            else:
                parts.append(f"**{tool_name}** (Error):\n{content}")
        
        return "\n\n---\n\n".join(parts)

    # =============================================================================
    # Strict Mode Validation
    # =============================================================================

    def validate_strict_response(
        self,
        response: ChatResponse,
        context_parts: List[str]
    ) -> ChatResponse:
        """
        Validate response in strict mode.
        
        Migrado desde ChatOrchestrator._validate_strict_response()
        """
        # Check if response has sources
        if not context_parts:
            response.metadata["warning"] = "No sources available for verification"
            response.metadata["confidence_score"] = 0.3
        else:
            # Simple check: does response reference the context?
            # More sophisticated validation could be added
            response.metadata["confidence_score"] = 0.8

        return response

    # =============================================================================
    # Streaming Response Building
    # =============================================================================

    def build_response_from_stream(
        self,
        full_content: str,
        thinking_content: Optional[str],
        metadata: Dict[str, Any]
    ) -> ChatResponse:
        """
        Build ChatResponse from accumulated stream data.
        
        Args:
            full_content: Accumulated content
            thinking_content: Accumulated thinking content
            metadata: Final metadata
            
        Returns:
            ChatResponse object
        """
        return ChatResponse(
            content=full_content,
            model=metadata.get("model", ""),
            provider=metadata.get("provider", ""),
            tokens_used=metadata.get("tokens_used"),
            cost=metadata.get("cost"),
            thinking_content=thinking_content,
            metadata=metadata
        )
