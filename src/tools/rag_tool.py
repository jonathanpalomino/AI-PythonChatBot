# =============================================================================
# src/tools/rag_tool.py
# RAG Search Tool
# =============================================================================
"""
Tool for performing Retrieval-Augmented Generation (RAG) searches over Qdrant collections.
"""

from typing import List, Dict, Any, Optional

from ollama import AsyncClient as OllamaClient
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import settings, get_qdrant_config
from src.services.embedding_service import EmbeddingService
from src.services.hybrid_search import BM25Index, HybridSearchFusion
from src.services.model_service import model_service
from src.services.reranker import CrossEncoderReranker
from src.tools.base_tool import BaseTool, ToolCategory, ToolParameter, ToolResult
from src.utils.logger import get_logger


class RAGTool(BaseTool):
    """Tool for searching in Qdrant collections"""

    def __init__(self):
        # Initialize clients
        qdrant_config = get_qdrant_config()
        self.qdrant = AsyncQdrantClient(**qdrant_config)
        self.ollama = OllamaClient(host=settings.OLLAMA_BASE_URL)
        self.logger = get_logger(__name__)

        # Re-ranking and hybrid search (lazy loading)
        self._reranker = None
        self._bm25_indexes = {}  # Cache: collection_name -> BM25Index
        self._fusion = HybridSearchFusion()

        super().__init__()

    # =============================================================================
    # Tool Definition
    # =============================================================================

    @property
    def name(self) -> str:
        return "rag_search"

    @property
    def description(self) -> str:
        return "Search for relevant information in documentation collections using semantic search"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.RAG

    @property
    def enabled_by_default(self) -> bool:
        return True

    @property
    def requires_context(self) -> List[str]:
        return ["qdrant"]

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="Search query to find relevant documentation",
                required=False,
                example="How to configure the HTTP tool"
            ),
            ToolParameter(
                name="collections",
                type="array",
                description="List of collection names to search in",
                required=False,
                example=["documentation", "api_guide"]
            ),
            ToolParameter(
                name="k",
                type="integer",
                description="Number of results to return (default: 5)",
                required=False,
                default=5,
                example=5
            ),
            ToolParameter(
                name="score_threshold",
                type="number",
                description="Minimum similarity score (0.0-1.0, default: 0.5)",
                required=False,
                default=0.5,
                example=0.5
            ),
            ToolParameter(
                name="filters",
                type="object",
                description="Additional filters (e.g., {\"method\": \"GET\", \"context\": \"NWT\"})",
                required=False,
                default={},
                example={"method": "GET", "context": "NWT"}
            ),
            ToolParameter(
                name="enable_rerank",
                type="boolean",
                description="Enable re-ranking with cross-encoder (default: False)",
                required=False,
                default=False,
                example=False
            ),
            ToolParameter(
                name="rerank_top_k",
                type="integer",
                description="Number of results after re-ranking (default: same as k)",
                required=False,
                default=None,
                example=5
            ),
            ToolParameter(
                name="search_mode",
                type="string",
                description="Search mode: 'semantic', 'lexical', or 'hybrid' (default: 'semantic')",
                required=False,
                default="semantic",
                example="semantic"
            ),
            ToolParameter(
                name="hybrid_alpha",
                type="number",
                description="Weight for semantic vs lexical in hybrid mode (0.0-1.0, default: 0.5)",
                required=False,
                default=0.5,
                example=0.5
            ),
            ToolParameter(
                name="enable_parent_retrieval",
                type="boolean",
                description="Enable parent document retrieval (default: False)",
                required=False,
                default=False,
                example=False
            ),
            ToolParameter(
                name="parent_mode",
                type="string",
                description="Parent retrieval mode: 'full_parent' or 'windowed' (default: 'full_parent')",
                required=False,
                default="full_parent",
                example="full_parent"
            ),
            ToolParameter(
                name="embedding_model",
                type="string",
                description="Embedding model to use for query generation",
                required=False,
                default=None,
                example="nomic-embed-text"
            ),
            ToolParameter(
                name="enable_contextual_retrieval",
                type="boolean",
                description="Enable contextual retrieval for improved search quality",
                required=False,
                default=settings.ENABLE_CONTEXTUAL_RETRIEVAL,
                example=True
            ),
            ToolParameter(
                name="context_generation_model",
                type="string",
                description="Model to use for generating context descriptions",
                required=False,
                default=settings.CONTEXT_GENERATION_MODEL,
                example="qwen2.5:3b"
            ),
        ]

    # =============================================================================
    # Execution
    # =============================================================================

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
        db: Optional[AsyncSession] = None,
    ) -> ToolResult:
        """Execute RAG search with optional re-ranking and hybrid search.

        Args:
            query: Search query.
            collections: Collection names to search.
            k: Number of results.
            score_threshold: Minimum score threshold.
            filters: Optional metadata filters.
            enable_rerank: Enable cross-encoder re-ranking.
            rerank_top_k: Number of results after re-ranking (defaults to k).
            search_mode: 'semantic', 'lexical', or 'hybrid'.
            hybrid_alpha: Weight for semantic vs lexical (0.0=lexical, 1.0=semantic).
            enable_parent_retrieval: Enable parent document retrieval.
            parent_mode: Parent retrieval mode ('full_parent' or 'windowed').
            embedding_model: Embedding model to use.
            enable_contextual_retrieval: Enable contextual retrieval.
            context_generation_model: Model for context generation.
            db: Database session for querying model metadata.
        Returns:
            ToolResult with search results.
        """
        try:
            # Validate inputs
            await self.validate_input(
                query=query,
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
                    "query": query,
                    "query_length": len(query),
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

            # Enhanced search with fallback strategies for better recall
            all_results = await self._enhanced_search_with_fallback(
                query=query,
                collections=collections,
                k=k,
                score_threshold=score_threshold,
                filters=filters,
                search_mode=search_mode,
                hybrid_alpha=hybrid_alpha,
                embedding_model=embedding_model,
                db=db
            )

            if not all_results:
                self.logger.info(
                    "No relevant results found", extra={"collections_searched": collections}
                )
                return ToolResult(
                    success=True,
                    data={"chunks": [], "message": "No relevant results found"},
                    metadata={"collections_searched": collections},
                )

            # Apply re-ranking if enabled
            if enable_rerank and all_results:
                self.logger.info("Applying cross-encoder re-ranking")
                if self._reranker is None:
                    self._reranker = CrossEncoderReranker(
                        model_name=settings.RERANK_MODEL,
                        batch_size=settings.RERANK_BATCH_SIZE,
                        device=settings.RERANK_DEVICE,
                    )

                top_k_rerank = rerank_top_k or k
                all_results = self._reranker.rerank(query, all_results, top_k_rerank)
                self.logger.info(
                    "Re-ranking completed",
                    extra={"output_count": len(all_results)}
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

            self.logger.info(
                "RAG search completed successfully",
                extra={
                    "results_count": len(all_results),
                    "avg_score": avg_score,
                    "collections_searched": collections,
                    "search_mode": search_mode,
                    "rerank_applied": enable_rerank,
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
                },
            )
        except Exception as e:
            self.logger.error(
                f"RAG search failed: {e}", exc_info=True,
                extra={"query": query[:100], "collections": collections}
            )
            return ToolResult(success=False, data=None, error=str(e))

    # =============================================================================
    # Helper Methods
    # =============================================================================

    async def _generate_embedding(self, text: str, db: Optional[AsyncSession] = None, model: Optional[str] = None) -> List[
        float]:
        """Generate embedding for text using model from database."""
        embedding_service = EmbeddingService(db)
        return await embedding_service.generate_embedding(text, db, model=model)

    async def _search_collection(
        self,
        collection_name: str,
        query_vector: List[float],
        k: int,
        score_threshold: float,
        filters: Optional[Dict[str, Any]],
    ) -> List[Dict]:
        """Search in a single collection and return standardized result dicts."""
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

    def _search_lexical(
        self,
        collection_name: str,
        query: str,
        k: int,
    ) -> List[Dict]:
        """Search using BM25 lexical search."""
        try:
            # Load or get cached BM25 index
            bm25_index = self._get_bm25_index(collection_name)

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
        """Search using hybrid approach (semantic + lexical)."""
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
            lexical_results = self._search_lexical(
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

    def _get_bm25_index(self, collection_name: str) -> Optional[BM25Index]:
        """Get or load BM25 index for a collection."""
        # Check cache
        if collection_name in self._bm25_indexes:
            return self._bm25_indexes[collection_name]

        # Try to load from disk
        bm25_index = BM25Index.load(
            directory=str(settings.BM25_INDEX_DIR),
            collection_name=collection_name
        )

        if bm25_index:
            self._bm25_indexes[collection_name] = bm25_index

        return bm25_index

    def _expand_to_parents(self, results: List[Dict], parent_mode: str = "full_parent") -> List[
        Dict]:
        """
        Expand child chunks to parent documents.

        Args:
            results: List of search results (child chunks)
            parent_mode: 'full_parent' or 'windowed'

        Returns:
            List of parent documents with deduplicated results
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
        embedding_model: Optional[str],
        db: Optional[AsyncSession]
    ) -> List[Dict]:
        """
        Enhanced search with multiple fallback strategies for better recall.
        Universal solution that works for any type of query that has low semantic similarity.
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
                    embedding_model=embedding_model,
                    db=db
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
                        embedding_model=embedding_model,
                        db=db
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
                        embedding_model=embedding_model,
                        db=db
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
                    embedding_model=embedding_model,
                    db=db
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
        Enrique metadata-based results with content-based results when metadata is insufficient.
        
        This addresses the issue where metadata has generic values (like "autor") but
        the actual information is in the document content. Since we now validate metadata
        during ingestion, this method is primarily for backward compatibility with existing
        documents that may have generic metadata values.
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

    def _content_has_meaningful_info(self, content: str, query: str) -> bool:
        """
        Check if content contains meaningful information related to the query.
        Universal solution that works for any type of query without hardcoding.
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
        Universal query expansion that works for any type of query.
        Adds common context terms that improve semantic matching.
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
        embedding_model: Optional[str],
        db: Optional[AsyncSession]
    ) -> List[Dict]:
        """Search collection with specified mode"""
        if search_mode == "hybrid":
            # Generate query vector for hybrid search
            query_vector = await self._generate_embedding(query, db=db, model=embedding_model)
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
            return self._search_lexical(
                collection_name=collection_name,
                query=query,
                k=k,
            )
        else:  # semantic (default)
            query_vector = await self._generate_embedding(query, db=db, model=embedding_model)
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
        embedding_model: Optional[str],
        db: Optional[AsyncSession]
    ) -> List[Dict]:
        """
        Universal hybrid search with lexical filtering for keyword matching.
        Works for any type of query by using the query terms themselves for filtering.
        """
        # Generate query vector
        query_vector = await self._generate_embedding(query, db=db, model=embedding_model)
        
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
        Extract meaningful keywords from query for lexical filtering.
        This is a universal approach that works for any type of query.
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
