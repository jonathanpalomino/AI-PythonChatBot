# =============================================================================
# src/services/hybrid_search.py
# Hybrid Search Service - Combining Semantic and Lexical Search
# =============================================================================
"""
Service for hybrid search combining semantic (dense vectors) and lexical (BM25) search.
Uses Reciprocal Rank Fusion (RRF) to merge results from both approaches.
"""

import os
import pickle
from typing import List, Dict, Any, Optional

from rank_bm25 import BM25Okapi

from src.utils.logger import get_logger


class BM25Index:
    """BM25 index for lexical search over document collections."""

    def __init__(
        self,
        collection_name: str,
        k1: float = 1.5,
        b: float = 0.75
    ):
        """
        Initialize BM25 index.

        Args:
            collection_name: Name of the collection.
            k1: BM25 k1 parameter (term frequency saturation).
            b: BM25 b parameter (length normalization).
        """
        self.collection_name = collection_name
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.documents = []
        self.tokenized_corpus = []
        self.logger = get_logger(__name__)

    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        """
        Build BM25 index from documents.

        Args:
            documents: List of documents with 'content' field.
        """
        try:
            self.logger.info(
                f"Building BM25 index for collection: {self.collection_name}",
                extra={"collection": self.collection_name, "doc_count": len(documents)}
            )

            self.documents = documents

            # Tokenize documents
            self.tokenized_corpus = [
                self._tokenize(doc.get("content", ""))
                for doc in documents
            ]

            # Build BM25 index
            self.bm25 = BM25Okapi(
                self.tokenized_corpus,
                k1=self.k1,
                b=self.b
            )

            self.logger.info(
                f"BM25 index built successfully",
                extra={"collection": self.collection_name, "doc_count": len(documents)}
            )

        except Exception as e:
            self.logger.error(
                f"Failed to build BM25 index: {e}",
                exc_info=True,
                extra={"collection": self.collection_name}
            )
            raise

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search using BM25.

        Args:
            query: Search query.
            k: Number of results to return.

        Returns:
            List of results with BM25 scores.
        """
        if self.bm25 is None:
            self.logger.warning(
                "BM25 index not built, returning empty results",
                extra={"collection": self.collection_name}
            )
            return []

        try:
            # Tokenize query
            tokenized_query = self._tokenize(query)

            # Get BM25 scores
            scores = self.bm25.get_scores(tokenized_query)

            # Get top-k document indices
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:k]

            # Build results
            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include documents with positive scores
                    doc = self.documents[idx].copy()
                    doc["bm25_score"] = float(scores[idx])
                    doc["lexical_rank"] = len(results) + 1
                    results.append(doc)

            self.logger.debug(
                f"BM25 search returned {len(results)} results",
                extra={
                    "collection": self.collection_name,
                    "query_length": len(query),
                    "results_count": len(results)
                }
            )

            return results

        except Exception as e:
            self.logger.error(
                f"BM25 search failed: {e}",
                exc_info=True,
                extra={"collection": self.collection_name}
            )
            return []

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization (can be enhanced with proper NLP preprocessing).

        Args:
            text: Text to tokenize.

        Returns:
            List of tokens.
        """
        # Simple tokenization: lowercase and split by whitespace/punctuation
        import re
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def save(self, directory: str) -> None:
        """
        Save BM25 index to disk.

        Args:
            directory: Directory to save index.
        """
        try:
            os.makedirs(directory, exist_ok=True)
            filepath = os.path.join(directory, f"{self.collection_name}.pkl")

            data = {
                "collection_name": self.collection_name,
                "k1": self.k1,
                "b": self.b,
                "documents": self.documents,
                "tokenized_corpus": self.tokenized_corpus,
            }

            with open(filepath, "wb") as f:
                pickle.dump(data, f)

            self.logger.info(
                f"BM25 index saved to {filepath}",
                extra={"collection": self.collection_name, "filepath": filepath}
            )

        except Exception as e:
            self.logger.error(
                f"Failed to save BM25 index: {e}",
                exc_info=True,
                extra={"collection": self.collection_name}
            )

    @classmethod
    def load(cls, directory: str, collection_name: str) -> Optional["BM25Index"]:
        """
        Load BM25 index from disk.

        Args:
            directory: Directory containing index.
            collection_name: Name of collection to load.

        Returns:
            BM25Index instance or None if not found.
        """
        logger = get_logger(__name__)
        try:
            filepath = os.path.join(directory, f"{collection_name}.pkl")

            if not os.path.exists(filepath):
                logger.debug(
                    f"BM25 index file not found: {filepath}",
                    extra={"collection": collection_name}
                )
                return None

            with open(filepath, "rb") as f:
                data = pickle.load(f)

            index = cls(
                collection_name=data["collection_name"],
                k1=data["k1"],
                b=data["b"]
            )
            index.documents = data["documents"]
            index.tokenized_corpus = data["tokenized_corpus"]

            # Rebuild BM25 from tokenized corpus
            index.bm25 = BM25Okapi(
                index.tokenized_corpus,
                k1=index.k1,
                b=index.b
            )

            logger.info(
                f"BM25 index loaded from {filepath}",
                extra={"collection": collection_name, "doc_count": len(index.documents)}
            )

            return index

        except Exception as e:
            logger.error(
                f"Failed to load BM25 index: {e}",
                exc_info=True,
                extra={"collection": collection_name}
            )
            return None


class HybridSearchFusion:
    """Fusion strategies for combining semantic and lexical search results."""

    def __init__(self):
        self.logger = get_logger(__name__)

    def reciprocal_rank_fusion(
        self,
        semantic_results: List[Dict[str, Any]],
        lexical_results: List[Dict[str, Any]],
        alpha: float = 0.5,
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).

        RRF formula: score(d) = alpha * (1/(k + rank_semantic(d))) + (1-alpha) * (1/(k + rank_lexical(d)))

        Args:
            semantic_results: Results from semantic search.
            lexical_results: Results from lexical (BM25) search.
            alpha: Weight for semantic vs lexical (0.0 = all lexical, 1.0 = all semantic).
            k: Constant for RRF formula (default: 60).

        Returns:
            Fused and sorted results.
        """
        try:
            self.logger.debug(
                "Starting reciprocal rank fusion",
                extra={
                    "semantic_count": len(semantic_results),
                    "lexical_count": len(lexical_results),
                    "alpha": alpha
                }
            )

            # Create a mapping of document ID to combined score
            # Use content hash or file+section as document ID
            doc_scores = {}
            doc_data = {}

            # Process semantic results
            for rank, result in enumerate(semantic_results, start=1):
                doc_id = self._get_doc_id(result)
                semantic_score = alpha * (1.0 / (k + rank))

                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                    doc_data[doc_id] = result.copy()

                doc_scores[doc_id] += semantic_score
                doc_data[doc_id]["semantic_rank"] = rank
                doc_data[doc_id]["semantic_score"] = result.get("score", 0.0)

            # Process lexical results
            for rank, result in enumerate(lexical_results, start=1):
                doc_id = self._get_doc_id(result)
                lexical_score = (1.0 - alpha) * (1.0 / (k + rank))

                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                    doc_data[doc_id] = result.copy()

                doc_scores[doc_id] += lexical_score
                doc_data[doc_id]["lexical_rank"] = rank
                doc_data[doc_id]["bm25_score"] = result.get("bm25_score", 0.0)

            # Build fused results
            fused_results = []
            for doc_id, rrf_score in doc_scores.items():
                result = doc_data[doc_id]
                result["rrf_score"] = rrf_score
                result["fusion_method"] = "rrf"
                fused_results.append(result)

            # Sort by RRF score (descending)
            fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)

            self.logger.info(
                "Reciprocal rank fusion completed",
                extra={
                    "input_semantic": len(semantic_results),
                    "input_lexical": len(lexical_results),
                    "output_count": len(fused_results),
                    "top_score": fused_results[0]["rrf_score"] if fused_results else None
                }
            )

            return fused_results

        except Exception as e:
            self.logger.error(
                f"Reciprocal rank fusion failed: {e}",
                exc_info=True
            )
            # Fallback to semantic results
            return semantic_results

    def _get_doc_id(self, result: Dict[str, Any]) -> str:
        """
        Generate a unique document ID from result.

        Args:
            result: Search result dictionary.

        Returns:
            Document ID string.
        """
        # Use file + section as ID, or fall back to content hash
        file = result.get("file", "")
        section = result.get("section", "")

        if file or section:
            return f"{file}::{section}"

        # Fallback: hash of content
        content = result.get("content", "")
        return str(hash(content))
