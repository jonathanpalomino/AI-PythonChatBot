# src/utils/cache.py
"""
Capa centralizada de caché para todo el backend.
- LRU en memoria por defecto
- TTL configurable
- Stats centralizadas + Prometheus-ready
- Thread/async safe
"""
import asyncio
from typing import Any, Optional, Callable, Dict, TypeVar
from collections import OrderedDict
from threading import Lock
import time
import threading
from functools import wraps
from dataclasses import dataclass
from prometheus_client import Histogram, Counter  # pip install prometheus-client

T = TypeVar('T')


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0


class CentralizedLRUCache:
    """
    LRU Cache centralizado con:
    - Thread-safe (Lock)
    - TTL por entry
    - Stats globales
    - Prometheus metrics
    - Configuración global
    """

    _instance = None
    _lock = threading.Lock()

    DEFAULT_MAX_SIZE = 500
    DEFAULT_TTL = 3600  # 1h

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True

        # Caches por namespace: {namespace: OrderedDict}
        self._caches: Dict[str, OrderedDict] = {}
        self._locks: Dict[str, Lock] = {}
        self._ttls: Dict[str, float] = {}
        self._max_sizes: Dict[str, int] = {}

        # Stats por namespace
        self.stats: Dict[str, CacheStats] = {}

        # Prometheus metrics
        self.hit_histogram = Histogram('cache_hit_latency_seconds', 'Cache hit latency')
        self.miss_counter = Counter('cache_misses_total', 'Cache misses')

    def create_cache(
        self,
        namespace: str,
        max_size: int = DEFAULT_MAX_SIZE,
        ttl: Optional[float] = DEFAULT_TTL
    ):
        """Crea/configura un caché por namespace"""
        with self._lock:
            self._caches[namespace] = OrderedDict()
            self._locks[namespace] = Lock()
            self._max_sizes[namespace] = max_size
            self._ttls[namespace] = ttl or self.DEFAULT_TTL
            self.stats[namespace] = CacheStats()

    def get(self, namespace: str, key: str) -> Optional[Any]:
        """Get thread-safe con TTL check"""
        if namespace not in self._caches:
            self.create_cache(namespace)

        cache_lock = self._locks[namespace]
        with cache_lock:
            cache = self._caches[namespace]

            if key not in cache:
                self.stats[namespace].misses += 1
                self.miss_counter.inc()
                return None

            timestamp, value = cache[key]
            if time.time() - timestamp > self._ttls[namespace]:
                # TTL expired
                del cache[key]
                self.stats[namespace].evictions += 1
                return None

            # Move to end (LRU)
            cache.move_to_end(key)
            self.stats[namespace].hits += 1
            self.stats[namespace].size = len(cache)
            return value

    def set(self, namespace: str, key: str, value: Any, ttl: Optional[float] = None):
        """Set con eviction automática"""
        if namespace not in self._caches:
            self.create_cache(namespace)

        cache_lock = self._locks[namespace]
        with cache_lock:
            cache = self._caches[namespace]
            timestamp = time.time()

            cache[key] = (timestamp, value)
            cache.move_to_end(key)

            # Evict si necesario
            while len(cache) > self._max_sizes[namespace]:
                old_key, _ = cache.popitem(last=False)
                self.stats[namespace].evictions += 1

            self.stats[namespace].size = len(cache)

    def clear(self, namespace: str):
        """Limpia caché específico"""
        if namespace in self._caches:
            with self._locks[namespace]:
                self._caches[namespace].clear()
                self.stats[namespace].size = 0

    def get_stats(self, namespace: Optional[str] = None) -> Dict:
        """Stats por namespace o global"""
        if namespace:
            return {namespace: self.stats.get(namespace, CacheStats()).__dict__}

        return {
            ns: stat.__dict__
            for ns, stat in self.stats.items()
        }

    def get_all_items(self, namespace: str) -> Dict[str, Any]:
        """Retorna todos los items del namespace (sin timestamp)"""
        if namespace not in self._caches:
            return {}
        with self._locks[namespace]:
            return {k: v for k, (ts, v) in self._caches[namespace].items()}

    def delete(self, namespace: str, key: str):
        """Elimina key específica"""
        if namespace in self._caches:
            with self._locks[namespace]:
                self._caches[namespace].pop(key, None)
                self.stats[namespace].size = len(self._caches[namespace])

# Singleton global
cache_manager = CentralizedLRUCache()


# Decorator para métodos
def cached(namespace: str, ttl: Optional[float] = None, max_size: int = 500):
    """Decorator: @cached('dimensions')"""

    def decorator(func: Callable) -> Callable:
        cache_manager.create_cache(namespace, max_size, ttl)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Key from args/kwargs (simple hash)
            key = f"{func.__name__}:{hash((args, frozenset(kwargs.items())))}"

            cached_value = cache_manager.get(namespace, key)
            if cached_value is not None:
                return cached_value

            result = await func(*args, **kwargs)
            cache_manager.set(namespace, key, result, ttl)
            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            key = f"{func.__name__}:{hash((args, frozenset(kwargs.items())))}"

            cached_value = cache_manager.get(namespace, key)
            if cached_value is not None:
                return cached_value

            result = func(*args, **kwargs)
            cache_manager.set(namespace, key, result, ttl)
            return result

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator
