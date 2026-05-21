# =============================================================================
# src/tools/rag_tool.py
# RAG Search Tool - Professional Edition
# =============================================================================
"""
Herramienta para realizar búsquedas RAG (Generación Aumentada por Recuperación) sobre colecciones de Qdrant.
"""

import asyncio
from asyncio import Lock
from typing import List, Dict, Any, Optional

from ollama import AsyncClient as OllamaClient
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.config.constants import DEFAULT_RAG_LIMIT, DEFAULT_SCORE_THRESHOLD
from src.config.settings import settings, get_qdrant_config
from src.repositories import FileRepository
from src.services.embedding.embedding_service import get_embedding_service
from src.services.search.hybrid_search import BM25Index, HybridSearchFusion
from src.services.search.reranker import CrossEncoderReranker
from src.tools.base_tool import BaseTool, ToolCategory, ToolParameter, ToolResult
from src.tools.rag_tool_utils import (
    InputValidator,
    MetricsCollector,
    EmbeddingCache,
    PerformanceTimer,
    ValidationError,
    NetworkError,
    TimeoutError,
    DatabaseError,
    EmbeddingError,
    ValidationResult,
    SearchMetrics,
)
from src.utils.health_checker import HealthCheckResult
from src.utils.health_checker import HealthChecker
from src.utils.logger import get_logger


class RAGTool(BaseTool):
    """Herramienta para búsqueda en colecciones de Qdrant."""

    def __init__(self):
        # Initialize clients
        qdrant_config = get_qdrant_config()
        self.qdrant = AsyncQdrantClient(**qdrant_config)
        self.ollama = OllamaClient(host=settings.OLLAMA_BASE_URL)
        self.logger = get_logger(__name__)

        # Re-ranking and hybrid search (lazy loading)
        self._reranker = None
        self._bm25_indexes = {}  # Cache: collection_name -> BM25Index
        self._bm25_locks = {}    # Lock per collection to prevent race conditions
        self._fusion = HybridSearchFusion()

        # Professional utilities
        self._validator = InputValidator()
        self._metrics_collector = MetricsCollector()
        self._embedding_cache = EmbeddingCache(
            capacity=getattr(settings, 'EMBEDDING_CACHE_CAPACITY', 1000),
            ttl_seconds=getattr(settings, 'EMBEDDING_CACHE_TTL', 3600)
        )
        self._health_checker = HealthChecker()
        self._performance_timer = PerformanceTimer()
        self._main_lock = Lock()  # Main lock for thread-safe operations

        # Configuration constants
        self._max_scroll_iterations = getattr(settings, 'MAX_SCROLL_ITERATIONS', 50)
        self._scroll_timeout = getattr(settings, 'SCROLL_TIMEOUT', 10.0)
        self._max_retries = getattr(settings, 'RAG_MAX_RETRIES', 3)
        self._retry_delay = getattr(settings, 'RAG_RETRY_DELAY', 1.0)

        super().__init__()

    # Definición de la Herramienta

    @property
    def name(self) -> str:
        return "rag_search"

    @property
    def description(self) -> str:
        return "Busca información relevante en colecciones de documentos usando búsqueda semántica."

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.RAG

    @property
    def enabled_by_default(self) -> bool:
        return True

    @property
    def requires_context(self) -> List[str]:
        return ["qdrant"]

    @property
    def required_dependencies(self) -> List[str]:
        """Dependencias de infraestructura que esta tool necesita."""
        return ["qdrant"]

    @property
    def file_dependent_actions(self) -> set:
        """Acciones que requieren archivos para ejecutarse."""
        # RAG siempre requiere archivos o colección
        return {"search", "full_document", "semantic_search"}

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="Consulta de búsqueda para encontrar documentación relevante.",
                required=False,
                example="Cómo configurar la herramienta HTTP"
            ),
            ToolParameter(
                name="collections",
                type="array",
                description="Lista de nombres de colecciones en las que buscar.",
                required=False,
                example=["documentation", "api_guide"]
            ),
            ToolParameter(
                name="k",
                type="integer",
                description=f"Número de resultados a retornar (por defecto: {DEFAULT_RAG_LIMIT}).",
                required=False,
                default=DEFAULT_RAG_LIMIT,
                example=DEFAULT_RAG_LIMIT
            ),
            ToolParameter(
                name="score_threshold",
                type="number",
                description=f"Puntaje mínimo de similitud (0.0-1.0, por defecto: {DEFAULT_SCORE_THRESHOLD}).",
                required=False,
                default=DEFAULT_SCORE_THRESHOLD,
                example=DEFAULT_SCORE_THRESHOLD
            ),
            ToolParameter(
                name="filters",
                type="object",
                description="Filtros adicionales (ej., {\"method\": \"GET\", \"context\": \"NWT\"}).",
                required=False,
                default={},
                example={"method": "GET", "context": "NWT"}
            ),
            ToolParameter(
                name="enable_rerank",
                type="boolean",
                description="Habilitar re-ranking con cross-encoder (por defecto: False).",
                required=False,
                default=False,
                example=False
            ),
            ToolParameter(
                name="rerank_top_k",
                type="integer",
                description="Número de resultados tras el re-ranking (por defecto: el mismo que k).",
                required=False,
                default=None,
                example=5
            ),
            ToolParameter(
                name="search_mode",
                type="string",
                description="Modo de búsqueda: 'semantic', 'lexical', 'hybrid', o 'full_document' (por defecto: 'semantic').",
                required=False,
                default="semantic",
                example="semantic"
            ),
            ToolParameter(
                name="hybrid_alpha",
                type="number",
                description="Peso para búsqueda semántica vs léxica en modo híbrido (0.0-1.0, por defecto: 0.5).",
                required=False,
                default=0.5,
                example=0.5
            ),
            ToolParameter(
                name="enable_parent_retrieval",
                type="boolean",
                description="Habilitar recuperación de documentos padre (por defecto: False).",
                required=False,
                default=False,
                example=False
            ),
            ToolParameter(
                name="parent_mode",
                type="string",
                description="Modo de recuperación padre: 'full_parent' o 'windowed' (por defecto: 'full_parent').",
                required=False,
                default="full_parent",
                example="full_parent"
            ),
            ToolParameter(
                name="embedding_model",
                type="string",
                description="Modelo de embedding a usar para la generación de la consulta.",
                required=False,
                default=None,
                example="nomic-embed-text"
            ),
            ToolParameter(
                name="enable_contextual_retrieval",
                type="boolean",
                description="Habilitar recuperación contextual para mejorar la calidad de búsqueda.",
                required=False,
                default=settings.ENABLE_CONTEXTUAL_RETRIEVAL,
                example=True
            ),
            ToolParameter(
                name="context_generation_model",
                type="string",
                description="Modelo a usar para generar descripciones de contexto.",
                required=False,
                default=settings.CONTEXT_GENERATION_MODEL,
                example="qwen2.5:3b"
            ),
        ]

    # Contratos declarativos v2.0 — ToolSelector / IntentRouter

    @property
    def requires_intent_classification(self) -> bool:
        """
        RAGTool no necesita enrutamiento interno complejo ya que su acción principal es la búsqueda.
        """
        return False

    def get_intent_definitions(self) -> Dict[str, Any]:
        """
        Retorna las definiciones de intención para RAGTool.
        """
        from src.services.intent.config import get_intents_by_registered_tool
        intents = get_intents_by_registered_tool("rag_search")
        return {
            i.name: {
                "description": i.description,
                "action_name": i.action_name,
                "requires_target": i.requires_target,
                "target_patterns": i.target_patterns,
                "examples": i.examples_es + i.examples_en,
                "default_params": i.default_params,
                "confidence_threshold": i.confidence_threshold,
            }
            for i in intents
        }

    # Ejecución

    async def get_full_document_content(
        self,
        file_id: str,
        collection_name: str = "documentation"
    ) -> Optional[str]:
        """
        Recupera el contenido completo de un documento concatenando todos sus fragmentos de Qdrant.

        Args:
            file_id: UUID del archivo.
            collection_name: Nombre de la colección.

        Returns:
            Cadena de contenido concatenado o None si no se encuentra.
        """
        try:
            # Filter by file_id
            scroll_filter = Filter(
                must=[
                    FieldCondition(key="file_id", match=MatchValue(value=file_id))
                ]
            )

            all_chunks = []
            next_page_offset = None

            # Scroll through all points for this file
            while True:
                records, next_page_offset = await self.qdrant.scroll(
                    collection_name=collection_name,
                    scroll_filter=scroll_filter,
                    limit=100,
                    offset=next_page_offset,
                    with_payload=True
                )
                all_chunks.extend(records)
                if next_page_offset is None:
                    break

            if not all_chunks:
                self.logger.warning(f"No chunks found for file_id {file_id} in {collection_name}")
                return None

            # Sort by chunk_index to ensure correct order
            # Handle cases where chunk_index might be missing or None
            valid_chunks = [
                c for c in all_chunks
                if c.payload and isinstance(c.payload.get('chunk_index'), int)
            ]

            # If we have valid indexed chunks, sort them
            if valid_chunks:
                valid_chunks.sort(key=lambda x: x.payload['chunk_index'])
                full_content = "\n".join([c.payload.get('content', '') for c in valid_chunks])
            else:
                # Fallback: try to sort by payload ID or just join (less reliable)
                self.logger.warning(f"Chunks for {file_id} missing chunk_index, joining arbitrarily")
                full_content = "\n".join([c.payload.get('content', '') for c in all_chunks if c.payload])

            self.logger.info(f"Reconstructed document {file_id} from {len(all_chunks)} chunks")
            return full_content

        except Exception as e:
            self.logger.error(f"Error reconstructing document {file_id}: {e}")
            return None

    async def execute(
        self,
        query: str,
        collections: List[str],
        k: int = 5,
        score_threshold: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
        enable_rerank: bool = False,
        rerank_top_k: Optional[int] = None,
        search_mode: str = "semantic",
        hybrid_alpha: float = 0.5,
        enable_parent_retrieval: bool = False,
        parent_mode: str = "full_parent",
        embedding_model: Optional[str] = None,
        enable_contextual_retrieval: bool = settings.ENABLE_CONTEXTUAL_RETRIEVAL,
        context_generation_model: str = settings.CONTEXT_GENERATION_MODEL,
        file_repo: Optional[FileRepository] = None,
        # Parámetros adicionales aceptados pero no usados por RAG (para compatibilidad con fast-path)
        action: Optional[str] = None,
        target: Optional[str] = None,
    ) -> ToolResult:
        """
        Ejecuta la búsqueda RAG con re-ranking opcional y búsqueda híbrida.

        Args:
            query: Consulta de búsqueda.
            collections: Nombres de las colecciones en las que buscar.
            k: Número de resultados.
            score_threshold: Umbral mínimo de puntaje.
            filters: Filtros opcionales de metadatos.
            enable_rerank: Habilitar re-ranking con cross-encoder.
            rerank_top_k: Número de resultados tras el re-ranking.
            search_mode: 'semantic', 'lexical', 'hybrid' o 'full_document'.
            hybrid_alpha: Peso para semántica vs léxica (0.0=léxica, 1.0=semántica).
            enable_parent_retrieval: Habilitar recuperación de documentos padre.
            parent_mode: Modo de recuperación padre ('full_parent' o 'windowed').
            embedding_model: Modelo de embedding a usar.
            enable_contextual_retrieval: Habilitar recuperación contextual.
            context_generation_model: Modelo para la generación de contexto.
            file_repo: Repositorio de archivos opcional.

        Returns:
            ToolResult con los resultados de la búsqueda.
        """
        # Create timer for metrics
        timer = self._performance_timer.create_timer()
        timer.start()

        # Sanitize query
        sanitized_query = self._validator.sanitize_query(query)

        # Professional validation
        validation_result = self._validator.validate_all(
            query=sanitized_query,
            collections=collections,
            k=k,
            score_threshold=score_threshold,
            filters=filters,
            search_mode=search_mode,
            hybrid_alpha=hybrid_alpha,
            parent_mode=parent_mode
        )

        if not validation_result.is_valid:
            error_msg = f"Validation failed: {'; '.join(validation_result.errors)}"
            self.logger.error(error_msg)
            return ToolResult(success=False, data=None, error=error_msg)

        # Log warnings if any
        if validation_result.warnings:
            for warning in validation_result.warnings:
                self.logger.warning(f"Validation warning: {warning}")

        try:
            # Inyectar file_repo
            if file_repo:
                self.file_repo = file_repo

            # Obtener modelo de embedding (sin db)
            if not embedding_model:
                embedding_model = settings.EMBEDDING_MODEL

            # Validate inputs using base class method
            await self.validate_input(
                query=sanitized_query,
                collections=collections,
                k=k,
                score_threshold=score_threshold,
                filters=filters or {},
                enable_rerank=enable_rerank,
                search_mode=search_mode,
                hybrid_alpha=hybrid_alpha,
                enable_contextual_retrieval=enable_contextual_retrieval,
                context_generation_model=context_generation_model,
            )

            self.logger.info(
                "Executing RAG search",
                extra={
                    "query": sanitized_query,
                    "query_length": len(sanitized_query),
                    "collections": collections,
                    "k": k,
                    "score_threshold": score_threshold,
                    "filters": filters,
                    "enable_rerank": enable_rerank,
                    "rerank_top_k": rerank_top_k,
                    "search_mode": search_mode,
                    "hybrid_alpha": hybrid_alpha,
                    "enable_parent_retrieval": enable_parent_retrieval,
                    "parent_mode": parent_mode,
                    "embedding_model": embedding_model,
                    "enable_contextual_retrieval": enable_contextual_retrieval,
                    "context_generation_model": context_generation_model,
                },
            )

            # --- FULL DOCUMENT RETRIEVAL MODE ---
            if search_mode == "full_document":
                if not filters or ('file' not in filters and 'file_id' not in filters):
                    return ToolResult(
                        success=False,
                        data=None,
                        error="For 'full_document' mode, you must provide a 'file' or 'file_id' in filters."
                    )

                # Fetch full document content from specified collections
                # Iterate through collections until the document is found
                last_error = None
                for collection in collections:
                    try:
                        full_doc_result = await self._fetch_full_document(collection, filters)
                        # If content is found, return successfully
                        if full_doc_result.get("content"):
                            # Record metrics
                            execution_time = timer.stop()
                            metric = SearchMetrics(
                                query_length=len(sanitized_query),
                                collections_searched=len(collections),
                                total_results=1,
                                execution_time_ms=execution_time,
                                strategy_used="full_document"
                            )
                            self._metrics_collector.add_metric(metric)

                            return ToolResult(
                                success=True,
                                data={"chunks": [full_doc_result], "count": 1},
                                metadata={
                                    "mode": "full_document",
                                    "file": full_doc_result.get("file"),
                                    "collection": collection,
                                    "execution_time_ms": execution_time
                                }
                            )
                    except Exception as e:
                        last_error = str(e)
                        self.logger.warning(
                            f"Failed to fetch full document from collection '{collection}': {e}")

                # If we get here, the document was not found in any collection
                execution_time = timer.stop()
                metric = SearchMetrics(
                    query_length=len(sanitized_query),
                    collections_searched=len(collections),
                    total_results=0,
                    execution_time_ms=execution_time,
                    strategy_used="full_document"
                )
                self._metrics_collector.add_metric(metric)

                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Document not found in any of the specified collections: {collections}. Last error: {last_error}"
                )

            # Enhanced search with fallback strategies for better recall
            all_results = await self._enhanced_search_with_fallback(
                query=sanitized_query,
                collections=collections,
                k=k,
                score_threshold=score_threshold,
                filters=filters,
                search_mode=search_mode,
                hybrid_alpha=hybrid_alpha,
                embedding_model=embedding_model
            )

            if not all_results:
                execution_time = timer.stop()
                metric = SearchMetrics(
                    query_length=len(sanitized_query),
                    collections_searched=len(collections),
                    total_results=0,
                    execution_time_ms=execution_time,
                    strategy_used="no_results"
                )
                self._metrics_collector.add_metric(metric)

                self.logger.info(
                    "No relevant results found", extra={"collections_searched": collections}
                )
                return ToolResult(
                    success=True,
                    data={"chunks": [], "message": "No relevant results found"},
                    metadata={
                        "collections_searched": collections,
                        "execution_time_ms": execution_time
                    },
                )

            # Apply re-ranking if enabled
            rerank_time = 0.0
            if enable_rerank and all_results:
                self.logger.info("Applying cross-encoder re-ranking")
                if self._reranker is None:
                    self._reranker = CrossEncoderReranker(
                        model_name=settings.RERANK_MODEL,
                        batch_size=settings.RERANK_BATCH_SIZE,
                        device=settings.RERANK_DEVICE,
                    )

                rerank_timer = self._performance_timer.create_timer()
                rerank_timer.start()
                top_k_rerank = rerank_top_k or k
                all_results = self._reranker.rerank(sanitized_query, all_results, top_k_rerank)
                rerank_time = rerank_timer.stop()

                self.logger.info(
                    "Re-ranking completed",
                    extra={"output_count": len(all_results), "rerank_time_ms": rerank_time}
                )
            else:
                # Sort by score and limit (if not re-ranked)
                score_key = "rrf_score" if search_mode == "hybrid" else "bm25_score" if search_mode == "lexical" else "score"
                all_results.sort(key=lambda x: x.get(score_key, 0), reverse=True)
                all_results = all_results[:k]

            # Apply parent document retrieval if enabled
            if enable_parent_retrieval and all_results:
                self.logger.info("Applying parent document retrieval")
                all_results = self._expand_to_parents(all_results, parent_mode)
                self.logger.info(
                    "Parent retrieval completed",
                    extra={"output_count": len(all_results)}
                )

            # Calculate average score
            score_key = "rerank_score" if enable_rerank else "rrf_score" if search_mode == "hybrid" else "bm25_score" if search_mode == "lexical" else "score"
            avg_score = (
                sum(r.get(score_key, 0) for r in all_results) / len(
                    all_results) if all_results else 0
            )

            # Stop timer and record metrics
            execution_time = timer.stop()
            metric = SearchMetrics(
                query_length=len(sanitized_query),
                collections_searched=len(collections),
                total_results=len(all_results),
                execution_time_ms=execution_time,
                rerank_time_ms=rerank_time,
                avg_score=avg_score,
                strategy_used=search_mode
            )
            self._metrics_collector.add_metric(metric)

            self.logger.info(
                "RAG search completed successfully",
                extra={
                    "results_count": len(all_results),
                    "avg_score": avg_score,
                    "collections_searched": collections,
                    "search_mode": search_mode,
                    "rerank_applied": enable_rerank,
                    "execution_time_ms": execution_time,
                },
            )

            return ToolResult(
                success=True,
                data={"chunks": all_results, "count": len(all_results)},
                metadata={
                    "collections_searched": collections,
                    "avg_score": avg_score,
                    "search_mode": search_mode,
                    "rerank_applied": enable_rerank,
                    "execution_time_ms": execution_time,
                },
            )
        except ValidationError as e:
            execution_time = timer.stop()
            self.logger.error(f"Validation error: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))
        except NetworkError as e:
            execution_time = timer.stop()
            self.logger.error(f"Network error: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=f"Network error: {str(e)}")
        except TimeoutError as e:
            execution_time = timer.stop()
            self.logger.error(f"Timeout error: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=f"Timeout error: {str(e)}")
        except DatabaseError as e:
            execution_time = timer.stop()
            self.logger.error(f"Database error: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=f"Database error: {str(e)}")
        except EmbeddingError as e:
            execution_time = timer.stop()
            self.logger.error(f"Embedding error: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=f"Embedding error: {str(e)}")
        except Exception as e:
            execution_time = timer.stop()
            self.logger.error(
                f"RAG search failed: {e}", exc_info=True,
                extra={"query": sanitized_query[:100], "collections": collections}
            )
            return ToolResult(success=False, data=None, error=str(e))

    # Métodos Auxiliares

    async def _generate_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """
        Genera el embedding para el texto dado usando caché.

        Args:
            text: Texto para generar el embedding.
            model: Nombre opcional del modelo de embedding.

        Returns:
            Lista de vectores de embedding.

        Raises:
            EmbeddingError: Si la generación del embedding falla.
        """
        try:
            # Check cache first
            cached_embedding = self._embedding_cache.get(text, model)
            if cached_embedding is not None:
                self.logger.debug(f"Cache hit for embedding: {text[:50]}...")
                return cached_embedding

            # Generate embedding
            embedding_service = await get_embedding_service()
            embedding = await embedding_service.generate_embedding(text, model=model)

            # Cache the result
            self._embedding_cache.put(text, embedding, model)

            return embedding
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}", exc_info=True)
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}", details={"text": text[:100], "model": model})

    async def _search_collection(
        self,
        collection_name: str,
        query_vector: List[float],
        k: int,
        score_threshold: float,
        filters: Optional[Dict[str, Any]],
    ) -> List[Dict]:
        """Busca en una única colección y retorna diccionarios de resultados estandarizados."""
        # Build Qdrant filter if provided
        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            if conditions:
                qdrant_filter = Filter(must=conditions)

        # Perform search in Qdrant collection using the modern query_points API
        search_results = await self.qdrant.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=k,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
        )

        # Convert hits to standardized result dicts
        # query_points returns a QueryResponse object with a .points attribute
        results: List[Dict] = []
        for hit in search_results.points:
            results.append({
                "file": hit.payload.get("file", "unknown"),
                "section": hit.payload.get("section", ""),
                "content": hit.payload.get("content", ""),
                "score": hit.score,
                "collection": collection_name,
                "metadata": {
                    k: v for k, v in hit.payload.items() if k not in ["file", "section", "content"]
                },
            })

        # Log number of hits retrieved
        self.logger.debug(
            f"Search collection '{collection_name}' returned {len(results)} hits",
            extra={"collection": collection_name, "hits": len(results)},
        )
        return results

    async def _search_lexical(
        self,
        collection_name: str,
        query: str,
        k: int,
    ) -> List[Dict]:
        """Busca usando búsqueda léxica BM25."""
        try:
            # Load or get cached BM25 index
            bm25_index = await self._get_bm25_index(collection_name)  # ← Await

            if bm25_index is None:
                self.logger.warning(
                    f"No BM25 index found for collection '{collection_name}'",
                    extra={"collection": collection_name}
                )
                return []

            # Perform BM25 search
            results = bm25_index.search(query, k)
            return results

        except Exception as e:
            self.logger.error(
                f"Lexical search failed: {e}",
                exc_info=True,
                extra={"collection": collection_name}
            )
            return []

    async def _search_hybrid(
        self,
        collection_name: str,
        query: str,
        query_vector: List[float],
        k: int,
        score_threshold: float,
        alpha: float,
        filters: Optional[Dict[str, Any]],
    ) -> List[Dict]:
        """Busca usando un enfoque híbrido (semántica + léxica)."""
        try:
            # Perform semantic search
            semantic_results = await self._search_collection(
                collection_name=collection_name,
                query_vector=query_vector,
                k=k * 2,  # Get more results for fusion
                score_threshold=score_threshold,
                filters=filters,
            )

            # Perform lexical search
            lexical_results = await self._search_lexical(
                collection_name=collection_name,
                query=query,
                k=k * 2,  # Get more results for fusion
            )

            # Fuse results using RRF
            fused_results = self._fusion.reciprocal_rank_fusion(
                semantic_results=semantic_results,
                lexical_results=lexical_results,
                alpha=alpha,
            )

            return fused_results[:k]

        except Exception as e:
            self.logger.error(
                f"Hybrid search failed: {e}",
                exc_info=True,
                extra={"collection": collection_name}
            )
            # Fallback to semantic search
            return await self._search_collection(
                collection_name=collection_name,
                query_vector=query_vector,
                k=k,
                score_threshold=score_threshold,
                filters=filters,
            )

    async def _get_bm25_index(self, collection_name: str) -> Optional[BM25Index]:
        """
        Obtiene o carga el índice BM25 para una colección (seguro para hilos).
        """
        # Fast path: check cache without lock
        if collection_name in self._bm25_indexes:
            return self._bm25_indexes[collection_name]

        # Ensure lock exists for this collection
        if collection_name not in self._bm25_locks:
            self._bm25_locks[collection_name] = Lock()

        # Acquire lock for this collection
        async with self._bm25_locks[collection_name]:
            # Double-check: another coroutine might have loaded it
            if collection_name in self._bm25_indexes:
                return self._bm25_indexes[collection_name]

            # Load from disk in separate thread (blocking I/O)
            try:
                bm25_index = await asyncio.to_thread(
                    BM25Index.load,
                    directory=str(settings.BM25_INDEX_DIR),
                    collection_name=collection_name
                )

                if bm25_index:
                    self._bm25_indexes[collection_name] = bm25_index
                    self.logger.debug(
                        f"BM25 index loaded for collection '{collection_name}'",
                        extra={"collection": collection_name}
                    )
                else:
                    self.logger.warning(
                        f"BM25 index not found on disk for '{collection_name}'",
                        extra={"collection": collection_name}
                    )

                return bm25_index

            except Exception as e:
                self.logger.error(
                    f"Failed to load BM25 index for '{collection_name}': {e}",
                    exc_info=True,
                    extra={"collection": collection_name}
                )
                return None

    def _expand_to_parents(self, results: List[Dict], parent_mode: str = "full_parent") -> List[
        Dict]:
        """
        Expande fragmentos hijos a documentos padre.

        Args:
            results: Lista de resultados de búsqueda (fragmentos hijos).
            parent_mode: 'full_parent' o 'windowed'.

        Returns:
            Lista de documentos padre con resultados deduplicados.
        """
        if not results:
            return results

        # Check if results have parent metadata
        has_parent_metadata = any(r.get('metadata', {}).get('parent_id') for r in results)

        if not has_parent_metadata:
            self.logger.debug("No parent metadata found, returning original chunks")
            return results

        # Group results by parent_id
        parent_groups: Dict[str, List[Dict]] = {}
        orphan_results = []

        for result in results:
            metadata = result.get('metadata', {})
            parent_id = metadata.get('parent_id')

            if parent_id:
                if parent_id not in parent_groups:
                    parent_groups[parent_id] = []
                parent_groups[parent_id].append(result)
            else:
                orphan_results.append(result)

        # Create parent documents
        parent_results = []

        for parent_id, child_chunks in parent_groups.items():
            # Get best score among child chunks
            best_child = max(child_chunks, key=lambda x: x.get('score', 0))
            best_score = best_child.get('score', 0)

            # Get parent content
            metadata = child_chunks[0].get('metadata', {})
            parent_content = metadata.get('parent_content')
            parent_title = metadata.get('parent_title')

            if parent_content:
                parent_result = {
                    'file': best_child.get('file'),
                    'section': parent_title or best_child.get('section', ''),
                    'content': parent_content,  # Full parent content
                    'score': best_score,
                    'collection': best_child.get('collection'),
                    'metadata': {
                        **metadata,
                        'is_parent': True,
                        'child_count': len(child_chunks),
                        'child_scores': [c.get('score', 0) for c in child_chunks]
                    }
                }
                parent_results.append(parent_result)
            else:
                parent_results.extend(child_chunks)

        parent_results.extend(orphan_results)
        parent_results.sort(key=lambda x: x.get('score', 0), reverse=True)

        self.logger.info(
            f"Parent retrieval: {len(results)} chunks -> {len(parent_results)} parents",
            extra={'original_count': len(results), 'parent_count': len(parent_results)}
        )

        return parent_results

    async def _enhanced_search_with_fallback(
        self,
        query: str,
        collections: List[str],
        k: int,
        score_threshold: float,
        filters: Optional[Dict[str, Any]],
        search_mode: str,
        hybrid_alpha: float,
        embedding_model: Optional[str]
    ) -> List[Dict]:
        """
        Búsqueda mejorada con múltiples estrategias de fallback para mejorar la recuperación.
        """
        all_results = []

        # Strategy 1: Standard search with original parameters
        self.logger.debug("Strategy 1: Standard search")
        for collection_name in collections:
            try:
                results = await self._search_collection_with_mode(
                    collection_name=collection_name,
                    query=query,
                    k=k,
                    score_threshold=score_threshold,
                    filters=filters,
                    search_mode=search_mode,
                    hybrid_alpha=hybrid_alpha,
                    embedding_model=embedding_model
                )
                all_results.extend(results)
            except Exception as e:
                self.logger.error(f"Strategy 1 failed for {collection_name}: {e}")

        if all_results:
            # Apply metadata enrichment for better results
            all_results = self._enrich_results_with_content(all_results, query)

            self.logger.debug(f"Strategy 1 found {len(all_results)} results")
            return all_results

        # Strategy 2: Progressive threshold reduction for better recall
        self.logger.debug("Strategy 2: Progressive threshold reduction")
        thresholds_to_try = []
        current_threshold = score_threshold

        # Try progressively lower thresholds down to 0.3
        while current_threshold > 0.3:
            current_threshold = max(0.3, current_threshold - 0.1)  # Reduce by 0.1 each step
            thresholds_to_try.append(round(current_threshold, 1))

        # Remove duplicates and ensure we don't go below 0.3
        thresholds_to_try = list(dict.fromkeys(thresholds_to_try))

        for threshold in thresholds_to_try:
            self.logger.debug(f"Strategy 2: Trying threshold {threshold}")
            for collection_name in collections:
                try:
                    results = await self._search_collection_with_mode(
                        collection_name=collection_name,
                        query=query,
                        k=k * 2,  # Get more results
                        score_threshold=threshold,
                        filters=filters,
                        search_mode=search_mode,
                        hybrid_alpha=hybrid_alpha,
                        embedding_model=embedding_model
                    )
                    all_results.extend(results)
                    if results:  # If we found results with this threshold, break
                        break
                except Exception as e:
                    self.logger.error(f"Strategy 2 failed for {collection_name} at threshold {threshold}: {e}")

            if all_results:  # If we found results with any threshold, break
                break

        if all_results:
            # Apply metadata enrichment for better results
            all_results = self._enrich_results_with_content(all_results, query)

            self.logger.debug(f"Strategy 2 found {len(all_results)} results with threshold {threshold}")
            # Sort and limit
            all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            return all_results[:k]

        # Strategy 3: Query expansion for better semantic matching
        self.logger.debug("Strategy 3: Query expansion for better semantic matching")
        expanded_queries = self._expand_query_universal(query)

        for expanded_query in expanded_queries:
            for collection_name in collections:
                try:
                    results = await self._search_collection_with_mode(
                        collection_name=collection_name,
                        query=expanded_query,
                        k=k,
                        score_threshold=score_threshold,
                        filters=filters,
                        search_mode=search_mode,
                        hybrid_alpha=hybrid_alpha,
                        embedding_model=embedding_model
                    )
                    all_results.extend(results)
                    if results:  # If we found results with this expansion, break
                        break
                except Exception as e:
                    self.logger.error(f"Strategy 3 failed for {collection_name}: {e}")

            if all_results:  # If we found results with any expansion, break
                break

        if all_results:
            # Apply metadata enrichment for better results
            all_results = self._enrich_results_with_content(all_results, query)

            self.logger.debug(f"Strategy 3 found {len(all_results)} results")
            all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            return all_results[:k]

        # Strategy 4: Hybrid search with lexical filtering for keyword matching
        self.logger.debug("Strategy 4: Hybrid search with lexical filtering")
        for collection_name in collections:
            try:
                results = await self._search_with_lexical_filtering(
                    collection_name=collection_name,
                    query=query,
                    k=k,
                    score_threshold=score_threshold,
                    filters=filters,
                    embedding_model=embedding_model
                )
                all_results.extend(results)
            except Exception as e:
                self.logger.error(f"Strategy 4 failed for {collection_name}: {e}")

        if all_results:
            # Apply metadata enrichment for better results
            all_results = self._enrich_results_with_content(all_results, query)

            self.logger.debug(f"Strategy 4 found {len(all_results)} results")
            all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            return all_results[:k]

        self.logger.debug("No results found with any strategy")
        return []

    def _enrich_results_with_content(self, results: List[Dict], query: str) -> List[Dict]:
        """
        Enriquece los resultados basados en metadatos con resultados basados en contenido si los metadatos son insuficientes.
        """
        if not results:
            return results

        # With improved metadata validation during ingestion, most results should have good metadata
        # This method now serves as a safety net for legacy documents
        good_metadata_results = []
        poor_metadata_results = []
        content_enriched_results = []

        for result in results:
            metadata = result.get('metadata', {})
            doc_author = metadata.get('doc_author', '').lower()

            # Check if metadata has meaningful information (not generic like "autor")
            has_meaningful_metadata = (
                doc_author and
                doc_author != 'autor' and
                doc_author != 'author' and
                len(doc_author) > 3 and
                not doc_author.isspace()
            )

            if has_meaningful_metadata:
                good_metadata_results.append(result)
            else:
                poor_metadata_results.append(result)

        # For results with poor metadata, prioritize content-based results
        if poor_metadata_results:
            # Sort poor metadata results by content quality
            for result in poor_metadata_results:
                content = result.get('content', '').lower()

                # Check if content contains meaningful information
                has_meaningful_content = self._content_has_meaningful_info(content, query)

                if has_meaningful_content:
                    content_enriched_results.append(result)

        # Prioritize: 1) Good metadata, 2) Content-enriched, 3) Poor metadata
        prioritized_results = good_metadata_results + content_enriched_results + poor_metadata_results

        if poor_metadata_results:
            self.logger.debug(
                f"Metadata enrichment applied: {len(good_metadata_results)} good metadata + "
                f"{len(content_enriched_results)} content-enriched + {len(poor_metadata_results)} poor metadata"
            )

        return prioritized_results

    async def _fetch_full_document(self, collection_name: str, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Obtiene todos los fragmentos de un archivo y reconstruye el documento completo.

        Args:
            collection_name: Nombre de la colección de Qdrant.
            filters: Diccionario que contiene 'file' o 'file_id'.

        Returns:
            Diccionario que representa el documento completo con contenido combinado.
        """
        try:
            qdrant_filter = Filter(must=[
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in filters.items()
            ])

            all_points = []
            next_offset = None
            iteration = 0

            # Scroll through all points with timeout protection
            while iteration < self._max_scroll_iterations:
                try:
                    scroll_result, next_offset = await asyncio.wait_for(
                        self.qdrant.scroll(
                            collection_name=collection_name,
                            scroll_filter=qdrant_filter,
                            limit=100,
                            offset=next_offset,
                            with_payload=True,
                            with_vectors=False
                        ),
                        timeout=self._scroll_timeout
                    )

                    all_points.extend(scroll_result)
                    iteration += 1

                    if next_offset is None:
                        break

                except asyncio.TimeoutError:
                    self.logger.warning(
                        f"Scroll timeout at iteration {iteration} for collection '{collection_name}'",
                        extra={
                            "collection": collection_name,
                            "points_collected": len(all_points),
                            "iteration": iteration
                        }
                    )
                    raise TimeoutError(
                        f"Scroll timeout after {iteration} iterations for collection '{collection_name}'",
                        details={"collection": collection_name, "points_collected": len(all_points)}
                    )

            # Log if max iterations reached
            if iteration >= self._max_scroll_iterations:
                self.logger.warning(
                    f"Reached max iterations ({self._max_scroll_iterations}) during scroll",
                    extra={
                        "collection": collection_name,
                        "points_collected": len(all_points)
                    }
                )

            if not all_points:
                return {
                    "content": "",
                    "message": "No content found for the specified file.",
                    "file": filters.get("file", "unknown")
                }

            # Sort by chunk_index (handle missing values)
            sorted_points = sorted(
                all_points,
                key=lambda p: p.payload.get('chunk_index', 0)
            )

            # Concatenate content
            full_content = "\n\n".join([p.payload.get('content', '') for p in sorted_points])

            # Construct result
            first_payload = sorted_points[0].payload
            return {
                "file": first_payload.get("file", filters.get("file", "unknown")),
                "content": full_content,
                "score": 1.0,
                "collection": collection_name,
                "metadata": {
                    "reconstructed": True,
                    "total_chunks": len(sorted_points),
                    "iterations": iteration,
                    "original_metadata": {
                        k: v for k, v in first_payload.items()
                        if k not in ['content', 'chunk_index']
                    }
                }
            }

        except TimeoutError:
            raise
        except Exception as e:
            self.logger.error(
                f"Error fetching full document: {e}",
                exc_info=True,
                extra={"collection": collection_name, "filters": filters}
            )
            raise DatabaseError(
                f"Failed to fetch full document from collection '{collection_name}': {str(e)}",
                details={"collection": collection_name, "filters": filters}
            )

    def _content_has_meaningful_info(self, content: str, query: str) -> bool:
        """
        Verifica si el contenido incluye información relevante relacionada con la consulta.
        """
        import re

        # Convert query to keywords for matching
        query_keywords = self._extract_keywords(query)

        # Check if content contains query keywords
        content_has_keywords = any(keyword in content for keyword in query_keywords)

        # Check for meaningful patterns in content (not just generic terms)
        meaningful_patterns = [
            r'[a-z]+\s+[a-z]+',  # Two consecutive words (potential names, terms)
            r'\d{2,}/\d{2}/\d{4}',  # Date patterns
            r'[A-Z][a-z]+\s+[A-Z][a-z]+',  # Capitalized names
            r'\b[a-z]{4,}\b',  # Words longer than 3 characters
        ]

        has_meaningful_patterns = any(
            re.search(pattern, content, re.IGNORECASE) for pattern in meaningful_patterns
        )

        # Check for specific document patterns that indicate meaningful content
        document_patterns = [
            r'preparaci[oó]n.*[a-z]{3,}',  # preparación + word
            r'versi[oó]n.*[a-z]{3,}',     # versión + word
            r'control.*cambios.*[a-z]{3,}', # control de cambios + word
            r'autor.*[a-z]{3,}',          # autor + word
        ]

        has_document_patterns = any(
            re.search(pattern, content, re.IGNORECASE) for pattern in document_patterns
        )

        # Content is meaningful if it has query keywords OR meaningful patterns
        return content_has_keywords or has_meaningful_patterns or has_document_patterns

    def _expand_query_universal(self, query: str) -> List[str]:
        """
        Expansión universal de consultas para mejorar la coincidencia semántica.
        """
        expansions = []
        query_lower = query.lower()

        # Common expansion patterns for better semantic matching
        expansions.extend([
            f"{query} del documento",
            f"{query} del archivo",
            f"{query} en el documento",
            f"{query} en el archivo",
            f"qué es {query}",
            f"qué significa {query}",
            f"qué contiene {query}",
            f"qué incluye {query}",
        ])

        # Specific expansions based on query patterns
        if any(word in query_lower for word in ['quien', 'quién', 'quien es', 'quién es']):
            expansions.extend([
                f"quién es {query}",
                f"quién creó {query}",
                f"quién preparó {query}",
                f"quién hizo {query}",
                f"quién escribió {query}",
            ])

        if any(word in query_lower for word in ['qué', 'que', 'qué es', 'que es']):
            expansions.extend([
                f"qué es {query}",
                f"qué significa {query}",
                f"qué contiene {query}",
                f"qué incluye {query}",
                f"qué representa {query}",
            ])

        if any(word in query_lower for word in ['cómo', 'como', 'cómo es', 'como es']):
            expansions.extend([
                f"cómo es {query}",
                f"cómo funciona {query}",
                f"cómo se hace {query}",
                f"cómo se utiliza {query}",
                f"cómo se implementa {query}",
            ])

        if any(word in query_lower for word in ['dónde', 'donde', 'dónde está', 'donde está']):
            expansions.extend([
                f"dónde está {query}",
                f"dónde se encuentra {query}",
                f"dónde se ubica {query}",
                f"dónde aparece {query}",
            ])

        # Remove duplicates and return (keep original order)
        return list(dict.fromkeys(expansions))

    async def _search_collection_with_mode(
        self,
        collection_name: str,
        query: str,
        k: int,
        score_threshold: float,
        filters: Optional[Dict[str, Any]],
        search_mode: str,
        hybrid_alpha: float,
        embedding_model: Optional[str]
    ) -> List[Dict]:
        """Busca en la colección con el modo especificado."""
        if search_mode == "hybrid":
            # Generate query vector for hybrid search
            query_vector = await self._generate_embedding(query, model=embedding_model)
            return await self._search_hybrid(
                collection_name=collection_name,
                query=query,
                query_vector=query_vector,
                k=k,
                score_threshold=score_threshold,
                alpha=hybrid_alpha,
                filters=filters,
            )
        elif search_mode == "lexical":
            return await self._search_lexical(
                collection_name=collection_name,
                query=query,
                k=k,
            )
        else:  # semantic (default)
            query_vector = await self._generate_embedding(query, model=embedding_model)
            return await self._search_collection(
                collection_name=collection_name,
                query_vector=query_vector,
                k=k,
                score_threshold=score_threshold,
                filters=filters,
            )

    async def _search_with_lexical_filtering(
        self,
        collection_name: str,
        query: str,
        k: int,
        score_threshold: float,
        filters: Optional[Dict[str, Any]],
        embedding_model: Optional[str]
    ) -> List[Dict]:
        """
        Búsqueda híbrida universal con filtrado léxico.
        """
        # Generate query vector
        query_vector = await self._generate_embedding(query, model=embedding_model)

        # Get semantic results with lower threshold
        semantic_results = await self._search_collection(
            collection_name=collection_name,
            query_vector=query_vector,
            k=k * 3,  # Get more results for filtering
            score_threshold=0.3,  # Lower threshold for hybrid
            filters=filters,
        )

        # Extract keywords from query for lexical filtering
        query_keywords = self._extract_keywords(query)

        # Filter for results that contain query keywords
        filtered_results = []

        for hit in semantic_results:
            content = hit.get('content', '').lower()
            section = hit.get('section', '').lower()

            # Check if content or section contains any query keywords
            content_matches = any(keyword in content for keyword in query_keywords)
            section_matches = any(keyword in section for keyword in query_keywords)

            if content_matches or section_matches:
                filtered_results.append(hit)

        # Sort by score and limit
        filtered_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return filtered_results[:k]

    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extrae palabras clave significativas de la consulta para el filtrado léxico.
        """
        import re

        # Convert to lowercase and remove special characters
        query_clean = re.sub(r'[^\w\s]', ' ', query.lower())

        # Split into words and filter
        words = query_clean.split()

        # Remove common stop words (Spanish and English)
        stop_words = {
            'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
            'es', 'son', 'está', 'están', 'fue', 'fueron', 'ser', 'estar',
            'en', 'de', 'del', 'con', 'para', 'por', 'sin', 'sobre',
            'y', 'o', 'u', 'e', 'a', 'al', 'a la', 'a los', 'a las',
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'without',
            'and', 'or', 'but', 'if', 'what', 'who', 'how', 'where', 'when',
            'why', 'which', 'this', 'that', 'these', 'those', 'my', 'your'
        }

        # Keep only meaningful words (length > 2 and not stop words)
        keywords = [
            word for word in words
            if len(word) > 2 and word not in stop_words
        ]

        # Return unique keywords
        return list(dict.fromkeys(keywords))

    # Métodos de API Profesional

    async def health_check(self) -> HealthCheckResult:
        """
        Realiza una verificación de salud integral de los componentes de la herramienta RAG.

        Returns:
            HealthCheckResult con el estado de todos los componentes.
        """
        embedding_service = await get_embedding_service()
        return await self._health_checker.perform_health_check(
            qdrant_client=self.qdrant,
            embedding_service=embedding_service,
            bm25_indexes=self._bm25_indexes
        )

    def get_metrics(self, count: int = 100) -> List[SearchMetrics]:
        """
        Obtiene las métricas de búsqueda recientes.

        Args:
            count: Número de métricas recientes a retornar.

        Returns:
            Lista de métricas de búsqueda recientes.
        """
        return self._metrics_collector.get_recent_metrics(count)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Obtiene un resumen de todas las métricas.

        Returns:
            Diccionario con el resumen de métricas.
        """
        return {
            "average_execution_time_ms": self._metrics_collector.get_average_execution_time(),
            "success_rate": self._metrics_collector.get_success_rate(),
            "cache_hit_rate": self._metrics_collector.get_cache_hit_rate(),
            "average_score": self._metrics_collector.get_average_score(),
            "strategy_distribution": self._metrics_collector.get_strategy_distribution(),
            "total_searches": len(self._metrics_collector._metrics),
        }

    def clear_metrics(self) -> None:
        """Limpia todas las métricas recolectadas."""
        self._metrics_collector.clear_metrics()

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de la caché de embeddings.

        Returns:
            Diccionario con estadísticas de caché.
        """
        return self._embedding_cache.get_stats()

    def clear_cache(self) -> None:
        """Limpia la caché de embeddings."""
        self._embedding_cache.clear()

    async def validate_collections_exist(self, collections: List[str]) -> ValidationResult:
        """
        Valida que las colecciones especificadas existan en Qdrant.

        Args:
            collections: Lista de nombres de colecciones a validar.

        Returns:
            ValidationResult con el resultado de la validación.
        """
        errors = []
        warnings = []

        try:
            collections_info = await self.qdrant.get_collections()
            existing_collections = {col.name for col in collections_info.collections}

            for collection in collections:
                if collection not in existing_collections:
                    errors.append(f"Collection '{collection}' does not exist")

            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings
            )
        except Exception as e:
            self.logger.error(f"Failed to validate collections: {e}", exc_info=True)
            return ValidationResult(
                is_valid=False,
                errors=[f"Failed to validate collections: {str(e)}"],
                warnings=warnings
            )

    async def get_collection_stats(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene estadísticas para una colección específica.

        Args:
            collection_name: Nombre de la colección.

        Returns:
            Diccionario con estadísticas de la colección o None si no existe.
        """
        try:
            collection_info = await self.qdrant.get_collection(collection_name)
            return {
                "name": collection_name,
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "segments_count": collection_info.segments_count,
                "status": collection_info.status,
                "optimizer_status": collection_info.optimizer_status,
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection stats for '{collection_name}': {e}", exc_info=True)
            return None

    async def list_collections(self) -> List[str]:
        """
        List all available collections in Qdrant.

        Returns:
            List of collection names
        """
        try:
            collections_info = await self.qdrant.get_collections()
            return [col.name for col in collections_info.collections]
        except Exception as e:
            self.logger.error(f"Failed to list collections: {e}", exc_info=True)
            return []
