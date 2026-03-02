# =============================================================================
# src/services/intent/cache.py
# Intent Classification Cache - LRU + TTL
# =============================================================================
"""
Cache inteligente para resultados de intent classification.

Características:
- LRU (Least Recently Used) eviction
- TTL (Time To Live) configurable
- Cache key basado en query + context hash
- Thread-safe para entornos async

✅ FIX: Import condicional para evitar circular import con router.py
"""

import hashlib
from datetime import datetime, timedelta
from collections import OrderedDict
from typing import Optional, Dict, Any, TYPE_CHECKING

from src.utils.logger import get_logger

# ✅ Import condicional: solo para type hints, no en runtime
if TYPE_CHECKING:
    from src.services.intent.router import IntentResult


class IntentCache:
    """
    Cache LRU + TTL para resultados de intent classification.

    Evita re-clasificar queries idénticas dentro del TTL window.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize intent cache.

        Args:
            max_size: Máximo número de entries en cache (LRU)
            ttl_seconds: Time to live en segundos (default: 1 hora)
        """
        self._cache: OrderedDict[str, tuple[Any, datetime]] = OrderedDict()
        self.max_size = max_size
        self.ttl = timedelta(seconds=ttl_seconds)
        self.logger = get_logger(__name__)

        # Stats
        self._hits = 0
        self._misses = 0

        self.logger.info(
            f"IntentCache initialized: max_size={max_size}, ttl={ttl_seconds}s"
        )

    def get(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional["IntentResult"]:
        """
        Obtiene intent result del cache si existe y no expiró.

        Args:
            query: User query
            context: Optional context dict (file_ids, conversation_id, etc.)

        Returns:
            IntentResult si hit, None si miss o expired
        """
        cache_key = self._generate_key(query, context)

        # Check if key exists
        if cache_key not in self._cache:
            self._misses += 1
            self.logger.debug(f"Cache MISS: {query[:50]}...")
            return None

        result, timestamp = self._cache[cache_key]

        # Check TTL expiration
        age = datetime.now() - timestamp
        if age > self.ttl:
            del self._cache[cache_key]
            self._misses += 1
            self.logger.debug(
                f"Cache EXPIRED: {query[:50]}... (age: {age.total_seconds():.1f}s)"
            )
            return None

        # Cache HIT - move to end (LRU: mark as recently used)
        self._cache.move_to_end(cache_key)
        self._hits += 1

        self.logger.debug(
            f"Cache HIT: {query[:50]}... (age: {age.total_seconds():.1f}s)"
        )

        return result

    def set(
        self,
        query: str,
        result: "IntentResult",
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Guarda intent result en cache.

        Args:
            query: User query
            result: IntentResult to cache
            context: Optional context dict
        """
        cache_key = self._generate_key(query, context)

        # LRU eviction: remove oldest if at capacity
        if len(self._cache) >= self.max_size:
            evicted_key = next(iter(self._cache))
            self._cache.popitem(last=False)
            self.logger.debug(f"Cache EVICTION (LRU): key={evicted_key[:16]}...")

        # Store with timestamp
        self._cache[cache_key] = (result, datetime.now())

        self.logger.debug(
            f"Cache SET: {query[:50]}... (intent: {result.intent_name})"
        )

    def _generate_key(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate deterministic cache key from query + context.

        Args:
            query: User query (normalized)
            context: Optional context dict

        Returns:
            MD5 hash of query + sorted context items
        """
        key_parts = [query.strip().lower()]

        if context:
            # Sort context items for deterministic hashing
            context_items = sorted(context.items())
            context_str = str(context_items)
            key_parts.append(context_str)

        combined = "|".join(key_parts)
        return hashlib.md5(combined.encode('utf-8')).hexdigest()

    def clear(self):
        """Clear all cached intents and reset stats"""
        size = len(self._cache)
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        self.logger.info(f"Cache cleared: {size} entries removed")

    def invalidate(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Invalidate specific cached intent.

        Args:
            query: User query to invalidate
            context: Optional context dict
        """
        cache_key = self._generate_key(query, context)
        if cache_key in self._cache:
            del self._cache[cache_key]
            self.logger.debug(f"Cache invalidated: {query[:50]}...")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with size, hit rate, miss rate, etc.
        """
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0
        miss_rate = (self._misses / total_requests * 100) if total_requests > 0 else 0.0

        return {
            "current_size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": int(self.ttl.total_seconds()),
            "hits": self._hits,
            "misses": self._misses,
            "total_requests": total_requests,
            "hit_rate_percent": round(hit_rate, 2),
            "miss_rate_percent": round(miss_rate, 2),
            "utilization_percent": round(len(self._cache) / self.max_size * 100, 2)
        }

    def print_stats(self):
        """Print cache statistics to logger"""
        stats = self.get_stats()
        self.logger.info(
            f"Intent Cache Stats: "
            f"Size={stats['current_size']}/{stats['max_size']}, "
            f"Hit Rate={stats['hit_rate_percent']:.1f}%, "
            f"Total Requests={stats['total_requests']}"
        )


# =============================================================================
# Global Cache Instance
# =============================================================================

_intent_cache: Optional[IntentCache] = None


def get_intent_cache(
    max_size: int = 1000,
    ttl_seconds: int = 3600
) -> IntentCache:
    """
    Get or create global intent cache instance (singleton pattern).

    Args:
        max_size: Maximum cache size (only used on first call)
        ttl_seconds: TTL in seconds (only used on first call)

    Returns:
        Global IntentCache instance
    """
    global _intent_cache

    if _intent_cache is None:
        _intent_cache = IntentCache(max_size=max_size, ttl_seconds=ttl_seconds)

    return _intent_cache


def reset_intent_cache():
    """Reset global intent cache (useful for testing)"""
    global _intent_cache
    if _intent_cache:
        _intent_cache.clear()
    _intent_cache = None
