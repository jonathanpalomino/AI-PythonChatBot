# =============================================================================
# src/services/reranker.py
# Re-ranking Service for RAG Results
# =============================================================================
"""
Service for re-ranking RAG search results using cross-encoder models.
Cross-encoders provide more accurate relevance scores by processing query-document
pairs together, unlike bi-encoders which encode them separately.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from src.utils.logger import get_logger


class Reranker(ABC):
    """Abstract base class for re-ranking implementations."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Re-rank search results based on query relevance.

        Args:
            query: Search query.
            results: List of search results with 'content' and 'score' fields.
            top_k: Number of top results to return.

        Returns:
            Re-ranked list of results, limited to top_k.
        """
        pass


class CrossEncoderReranker(Reranker):
    """Re-ranker using sentence-transformers cross-encoder models."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        device: Optional[str] = None
    ):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace model name for cross-encoder.
            batch_size: Batch size for inference.
            device: Device to run model on ('cuda', 'cpu', or None for auto).
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None  # Lazy loading
        self.logger = get_logger(__name__)

    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self.logger.info(
                    f"Loading cross-encoder model: {self.model_name}",
                    extra={"model_name": self.model_name}
                )

                self._model = CrossEncoder(
                    self.model_name,
                    device=self.device,
                    max_length=512
                )

                self.logger.info(
                    "Cross-encoder model loaded successfully",
                    extra={"model_name": self.model_name}
                )

            except Exception as e:
                self.logger.error(
                    f"Failed to load cross-encoder model: {e}",
                    exc_info=True,
                    extra={"model_name": self.model_name}
                )
                raise

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Re-rank results using cross-encoder.

        Args:
            query: Search query.
            results: List of search results with 'content' field.
            top_k: Number of top results to return.

        Returns:
            Re-ranked list of results with updated 'rerank_score' field.
        """
        if not results:
            return results

        try:
            # Ensure model is loaded
            self._load_model()

            self.logger.debug(
                f"Re-ranking {len(results)} results for query",
                extra={"num_results": len(results), "query_length": len(query)}
            )

            # Prepare query-document pairs
            pairs = []
            for result in results:
                content = result.get("content", "")
                pairs.append([query, content])

            # Get cross-encoder scores in batches
            rerank_scores = self._model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False
            )

            # Add rerank scores to results
            for i, result in enumerate(results):
                result["rerank_score"] = float(rerank_scores[i])
                result["original_score"] = result.get("score", 0.0)

            # Sort by rerank score (descending)
            reranked = sorted(
                results,
                key=lambda x: x["rerank_score"],
                reverse=True
            )

            # Limit to top_k
            reranked = reranked[:top_k]

            self.logger.info(
                f"Re-ranking completed, returning top {len(reranked)} results",
                extra={
                    "input_count": len(results),
                    "output_count": len(reranked),
                    "top_score": reranked[0]["rerank_score"] if reranked else None
                }
            )

            return reranked

        except Exception as e:
            self.logger.error(
                f"Re-ranking failed: {e}",
                exc_info=True,
                extra={"num_results": len(results)}
            )
            # Fallback to original results if re-ranking fails
            self.logger.warning("Falling back to original ranking")
            return results[:top_k]


class NoOpReranker(Reranker):
    """Pass-through reranker that doesn't modify results (for testing/fallback)."""

    def __init__(self):
        self.logger = get_logger(__name__)

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Return results unchanged, limited to top_k."""
        self.logger.debug("Using NoOpReranker - no re-ranking applied")
        return results[:top_k]
