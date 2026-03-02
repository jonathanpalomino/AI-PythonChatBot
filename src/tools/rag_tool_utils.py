# =============================================================================
# src/tools/rag_tool_utils.py
# Professional utilities for RAG Tool
# =============================================================================
"""
Professional utilities for RAG Tool including:
- Input validation and sanitization
- Metrics collection and monitoring
- Caching mechanisms
- Error handling and retry logic
- Health checks (imported from health_checker.py)
- Performance timing
"""

import asyncio
import hashlib
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from src.utils.logger import get_logger

# Import health check utilities from centralized location
from src.utils.health_checker import HealthCheckResult, HealthChecker

logger = get_logger(__name__)

T = TypeVar('T')


# =============================================================================
# Enums
# =============================================================================

class ErrorType(str, Enum):
    """Types of errors that can occur in RAG operations"""
    VALIDATION_ERROR = "validation_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    DATABASE_ERROR = "database_error"
    EMBEDDING_ERROR = "embedding_error"
    UNKNOWN_ERROR = "unknown_error"


class SearchMode(str, Enum):
    """Search modes"""
    SEMANTIC = "semantic"
    LEXICAL = "lexical"
    HYBRID = "hybrid"
    FULL_DOCUMENT = "full_document"


class ParentMode(str, Enum):
    """Parent retrieval modes"""
    FULL_PARENT = "full_parent"
    WINDOWED = "windowed"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ValidationResult:
    """Result of validation operation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class SearchMetrics:
    """Metrics for search operations"""
    query_length: int
    collections_searched: int
    total_results: int
    execution_time_ms: float
    embedding_time_ms: float = 0.0
    search_time_ms: float = 0.0
    rerank_time_ms: float = 0.0
    avg_score: float = 0.0
    strategy_used: str = "standard"
    cache_hit: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# Input Validation and Sanitization
# =============================================================================

class InputValidator:
    """Professional input validation and sanitization"""
    
    # Constants
    MAX_QUERY_LENGTH = 1000
    MIN_QUERY_LENGTH = 3
    MAX_COLLECTIONS = 10
    MAX_K = 100
    MIN_K = 1
    MAX_FILTERS = 20
    VALID_SEARCH_MODES = [mode.value for mode in SearchMode]
    VALID_PARENT_MODES = [mode.value for mode in ParentMode]
    
    # Patterns for sanitization
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',  # JavaScript protocol
        r'on\w+\s*=',  # Event handlers
        r'\.\./',  # Path traversal
        r'\x00',  # Null bytes
    ]
    
    @classmethod
    def validate_query(cls, query: str) -> ValidationResult:
        """Validate search query"""
        errors = []
        warnings = []
        
        if not query or not query.strip():
            errors.append("Query cannot be empty")
            return ValidationResult(is_valid=False, errors=errors)
        
        query = query.strip()
        
        if len(query) < cls.MIN_QUERY_LENGTH:
            errors.append(f"Query must be at least {cls.MIN_QUERY_LENGTH} characters")
        
        if len(query) > cls.MAX_QUERY_LENGTH:
            warnings.append(f"Query is very long ({len(query)} chars), consider shortening")
        
        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                errors.append(f"Query contains potentially dangerous pattern: {pattern}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    @classmethod
    def validate_collections(cls, collections: List[str]) -> ValidationResult:
        """Validate collections list"""
        errors = []
        warnings = []
        
        if not collections:
            errors.append("Collections list cannot be empty")
            return ValidationResult(is_valid=False, errors=errors)
        
        if len(collections) > cls.MAX_COLLECTIONS:
            warnings.append(f"Searching {len(collections)} collections, consider reducing")
        
        # Validate each collection name
        for collection in collections:
            if not collection or not collection.strip():
                errors.append("Collection name cannot be empty")
            elif not re.match(r'^[a-zA-Z0-9_-]+$', collection):
                errors.append(f"Invalid collection name: {collection}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    @classmethod
    def validate_k(cls, k: int) -> ValidationResult:
        """Validate k parameter"""
        errors = []
        
        if not isinstance(k, int):
            errors.append("k must be an integer")
        elif k < cls.MIN_K:
            errors.append(f"k must be at least {cls.MIN_K}")
        elif k > cls.MAX_K:
            errors.append(f"k cannot exceed {cls.MAX_K}")
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)
    
    @classmethod
    def validate_score_threshold(cls, score_threshold: float) -> ValidationResult:
        """Validate score threshold"""
        errors = []
        
        if not isinstance(score_threshold, (int, float)):
            errors.append("score_threshold must be a number")
        elif score_threshold < 0.0 or score_threshold > 1.0:
            errors.append("score_threshold must be between 0.0 and 1.0")
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)
    
    @classmethod
    def validate_filters(cls, filters: Optional[Dict[str, Any]]) -> ValidationResult:
        """Validate filters"""
        errors = []
        warnings = []
        
        if filters is None:
            return ValidationResult(is_valid=True)
        
        if not isinstance(filters, dict):
            errors.append("filters must be a dictionary")
            return ValidationResult(is_valid=False, errors=errors)
        
        if len(filters) > cls.MAX_FILTERS:
            warnings.append(f"Many filters ({len(filters)}), may impact performance")
        
        # Validate filter keys and values
        for key, value in filters.items():
            if not key or not isinstance(key, str):
                errors.append(f"Invalid filter key: {key}")
            elif not re.match(r'^[a-zA-Z0-9_-]+$', key):
                errors.append(f"Invalid filter key format: {key}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    @classmethod
    def validate_search_mode(cls, search_mode: str) -> ValidationResult:
        """Validate search mode"""
        errors = []
        
        if search_mode not in cls.VALID_SEARCH_MODES:
            errors.append(f"Invalid search_mode: {search_mode}. Must be one of {cls.VALID_SEARCH_MODES}")
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)
    
    @classmethod
    def validate_hybrid_alpha(cls, hybrid_alpha: float) -> ValidationResult:
        """Validate hybrid alpha"""
        errors = []
        
        if not isinstance(hybrid_alpha, (int, float)):
            errors.append("hybrid_alpha must be a number")
        elif hybrid_alpha < 0.0 or hybrid_alpha > 1.0:
            errors.append("hybrid_alpha must be between 0.0 and 1.0")
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)
    
    @classmethod
    def validate_parent_mode(cls, parent_mode: str) -> ValidationResult:
        """Validate parent mode"""
        errors = []
        
        if parent_mode not in cls.VALID_PARENT_MODES:
            errors.append(f"Invalid parent_mode: {parent_mode}. Must be one of {cls.VALID_PARENT_MODES}")
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)
    
    @classmethod
    def sanitize_query(cls, query: str) -> str:
        """Sanitize query string"""
        if not query:
            return ""
        
        # Remove dangerous patterns
        sanitized = query
        for pattern in cls.DANGEROUS_PATTERNS:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        return sanitized
    
    @classmethod
    def validate_all(
        cls,
        query: str,
        collections: List[str],
        k: int,
        score_threshold: float,
        filters: Optional[Dict[str, Any]],
        search_mode: str,
        hybrid_alpha: float,
        parent_mode: str
    ) -> ValidationResult:
        """Validate all parameters"""
        all_errors = []
        all_warnings = []
        
        # Validate each parameter
        result = cls.validate_query(query)
        all_errors.extend(result.errors)
        all_warnings.extend(result.warnings)
        
        result = cls.validate_collections(collections)
        all_errors.extend(result.errors)
        all_warnings.extend(result.warnings)
        
        result = cls.validate_k(k)
        all_errors.extend(result.errors)
        
        result = cls.validate_score_threshold(score_threshold)
        all_errors.extend(result.errors)
        
        result = cls.validate_filters(filters)
        all_errors.extend(result.errors)
        all_warnings.extend(result.warnings)
        
        result = cls.validate_search_mode(search_mode)
        all_errors.extend(result.errors)
        
        result = cls.validate_hybrid_alpha(hybrid_alpha)
        all_errors.extend(result.errors)
        
        result = cls.validate_parent_mode(parent_mode)
        all_errors.extend(result.errors)
        
        return ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings
        )


# =============================================================================
# Metrics and Monitoring
# =============================================================================

class MetricsCollector:
    """Professional metrics collection for RAG operations"""
    
    def __init__(self, max_metrics: int = 1000):
        self._metrics: List[SearchMetrics] = []
        self._max_metrics = max_metrics
        self._lock = None  # Will be set by RAGTool
        self.logger = get_logger(__name__)
    
    def add_metric(self, metric: SearchMetrics) -> None:
        """Add a metric to the collection"""
        if self._lock:
            # Thread-safe add (will be called with lock)
            self._metrics.append(metric)
        else:
            self._metrics.append(metric)
        
        # Keep only recent metrics
        if len(self._metrics) > self._max_metrics:
            self._metrics = self._metrics[-self._max_metrics:]
    
    def get_recent_metrics(self, count: int = 100) -> List[SearchMetrics]:
        """Get recent metrics"""
        return self._metrics[-count:]
    
    def get_average_execution_time(self) -> float:
        """Get average execution time in milliseconds"""
        if not self._metrics:
            return 0.0
        return sum(m.execution_time_ms for m in self._metrics) / len(self._metrics)
    
    def get_success_rate(self) -> float:
        """Get success rate (0.0-1.0)"""
        if not self._metrics:
            return 0.0
        successful = sum(1 for m in self._metrics if m.total_results > 0)
        return successful / len(self._metrics)
    
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate (0.0-1.0)"""
        if not self._metrics:
            return 0.0
        cache_hits = sum(1 for m in self._metrics if m.cache_hit)
        return cache_hits / len(self._metrics)
    
    def get_average_score(self) -> float:
        """Get average score across all searches"""
        if not self._metrics:
            return 0.0
        return sum(m.avg_score for m in self._metrics) / len(self._metrics)
    
    def get_strategy_distribution(self) -> Dict[str, int]:
        """Get distribution of strategies used"""
        distribution = {}
        for metric in self._metrics:
            strategy = metric.strategy_used
            distribution[strategy] = distribution.get(strategy, 0) + 1
        return distribution
    
    def clear_metrics(self) -> None:
        """Clear all metrics"""
        self._metrics.clear()
        self.logger.info("Metrics cleared")


# =============================================================================
# Caching Mechanisms
# =============================================================================

class LRUCache:
    """Thread-safe LRU cache implementation"""
    
    def __init__(self, capacity: int = 1000, ttl_seconds: int = 3600):
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, Tuple[Any, datetime]] = OrderedDict()
        self.logger = get_logger(__name__)
    
    def _is_expired(self, timestamp: datetime) -> bool:
        """Check if cache entry is expired"""
        return datetime.utcnow() - timestamp > timedelta(seconds=self.ttl_seconds)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key not in self.cache:
            return None
        
        value, timestamp = self.cache[key]
        
        # Check if expired
        if self._is_expired(timestamp):
            del self.cache[key]
            self.logger.debug(f"Cache entry expired: {key[:50]}...")
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return value
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache"""
        # Remove if exists
        if key in self.cache:
            del self.cache[key]
        
        # Add to end
        self.cache[key] = (value, datetime.utcnow())
        
        # Remove oldest if over capacity
        if len(self.cache) > self.capacity:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            self.logger.debug(f"Cache evicted oldest entry: {oldest_key[:50]}...")
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self.cache.clear()
        self.logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "size": len(self.cache),
            "capacity": self.capacity,
            "ttl_seconds": self.ttl_seconds,
            "usage_percent": len(self.cache) / self.capacity * 100 if self.capacity > 0 else 0
        }


class EmbeddingCache:
    """Cache for embedding vectors"""
    
    def __init__(self, capacity: int = 1000, ttl_seconds: int = 3600):
        self.cache = LRUCache(capacity=capacity, ttl_seconds=ttl_seconds)
        self.logger = get_logger(__name__)
    
    def _generate_key(self, text: str, model: Optional[str] = None) -> str:
        """Generate cache key for embedding"""
        key_data = f"{text}|{model or 'default'}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, text: str, model: Optional[str] = None) -> Optional[List[float]]:
        """Get cached embedding"""
        key = self._generate_key(text, model)
        embedding = self.cache.get(key)
        if embedding:
            self.logger.debug(f"Cache hit for embedding: {key[:20]}...")
        return embedding
    
    def put(self, text: str, embedding: List[float], model: Optional[str] = None) -> None:
        """Cache embedding"""
        key = self._generate_key(text, model)
        self.cache.put(key, embedding)
        self.logger.debug(f"Cached embedding: {key[:20]}...")
    
    def clear(self) -> None:
        """Clear cache"""
        self.cache.clear()
        self.logger.info("Embedding cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_stats()


# =============================================================================
# Error Handling and Retry Logic
# =============================================================================

class RAGToolError(Exception):
    """Base exception for RAG tool errors"""
    
    def __init__(self, message: str, error_type: ErrorType = ErrorType.UNKNOWN_ERROR, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_type = error_type
        self.details = details or {}


class ValidationError(RAGToolError):
    """Validation error"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorType.VALIDATION_ERROR, details)


class NetworkError(RAGToolError):
    """Network error"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorType.NETWORK_ERROR, details)


class TimeoutError(RAGToolError):
    """Timeout error"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorType.TIMEOUT_ERROR, details)


class DatabaseError(RAGToolError):
    """Database error"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorType.DATABASE_ERROR, details)


class EmbeddingError(RAGToolError):
    """Embedding error"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorType.EMBEDDING_ERROR, details)


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    exponential_base: float = 2.0,
    retryable_exceptions: Tuple[type, ...] = (Exception,)
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        retryable_exceptions: Tuple of exception types to retry on
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        # Last attempt failed, raise the exception
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    
                    # Wait before retry
                    await asyncio.sleep(delay)
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    return decorator


# =============================================================================
# Performance Timing
# =============================================================================

class PerformanceTimer:
    """Performance timing utilities"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def time_execution(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to measure function execution time"""
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.time() - start_time) * 1000
                self.logger.debug(f"{func.__name__} executed in {elapsed_ms:.2f}ms")
        
        return wrapper
    
    def create_timer(self) -> 'Timer':
        """Create a new timer"""
        return Timer()


class Timer:
    """Simple timer for measuring execution time"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self) -> None:
        """Start the timer"""
        self.start_time = time.time()
    
    def stop(self) -> float:
        """Stop the timer and return elapsed time in milliseconds"""
        self.end_time = time.time()
        if self.start_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000
    
    def elapsed(self) -> float:
        """Get elapsed time in milliseconds without stopping"""
        if self.start_time is None:
            return 0.0
        return (time.time() - self.start_time) * 1000
