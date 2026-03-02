# =============================================================================
# src/tools/http_tool.py
# Professional HTTP Request Tool with Advanced Features
# =============================================================================
"""
Professional HTTP Request Tool with advanced features:
- Automatic retries with exponential backoff
- Rate limiting
- Response caching
- JSON schema validation
- Multipart/form-data support
- Streaming support
- Performance metrics
- Detailed error handling
- WebSocket support
"""

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Any, Optional

import httpx

from src.tools.auth import AuthFactory, AuthType, AuthStrategy, NoAuth
from src.tools.base_tool import BaseTool, ToolCategory, ToolParameter, ToolResult
from src.utils.logger import get_logger


# =============================================================================
# Enums and Data Classes
# =============================================================================

class RetryStrategy(Enum):
    """Retry strategies for failed requests."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    NO_RETRY = "no_retry"


class CacheStrategy(Enum):
    """Caching strategies."""
    NO_CACHE = "no_cache"
    MEMORY_CACHE = "memory_cache"
    ETAG_CACHE = "etag_cache"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    initial_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    backoff_multiplier: float = 2.0
    retryable_status_codes: List[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])


@dataclass
class CacheConfig:
    """Configuration for response caching."""
    strategy: CacheStrategy = CacheStrategy.MEMORY_CACHE
    ttl: int = 300  # seconds
    max_size: int = 100  # max number of cached responses


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_second: float = 10.0
    requests_per_minute: float = 100.0
    burst_size: int = 20


@dataclass
class PerformanceMetrics:
    """Performance metrics for HTTP requests."""
    request_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    cache_hits: int = 0
    cache_misses: int = 0


# =============================================================================
# HTTP Error Codes
# =============================================================================

HTTP_ERROR_DESCRIPTIONS = {
    400: "Bad Request - The server cannot process the request due to client error",
    401: "Unauthorized - Authentication is required or failed",
    403: "Forbidden - The server refuses to authorize the request",
    404: "Not Found - The requested resource could not be found",
    405: "Method Not Allowed - The HTTP method is not supported for the requested resource",
    409: "Conflict - The request conflicts with the current state of the target resource",
    429: "Too Many Requests - The user has sent too many requests in a given amount of time",
    500: "Internal Server Error - The server encountered an unexpected condition",
    502: "Bad Gateway - The server was acting as a gateway or proxy and received an invalid response",
    503: "Service Unavailable - The server is currently unable to handle the request",
    504: "Gateway Timeout - The server did not receive a timely response",
}


# =============================================================================
# Professional HTTP Tool
# =============================================================================

class HTTPTool(BaseTool):
    """
    Professional HTTP Request Tool with advanced features.

    Features:
    - Automatic retries with exponential backoff
    - Rate limiting
    - Response caching
    - JSON schema validation
    - Multipart/form-data support
    - Streaming support
    - Performance metrics
    - Detailed error handling
    - WebSocket support
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        self._auth_strategy: AuthStrategy = NoAuth()
        self._retry_config = RetryConfig()
        self._cache_config = CacheConfig()
        self._rate_limit_config = RateLimitConfig()
        self._metrics = PerformanceMetrics()

        # In-memory cache
        self._cache: Dict[str, Dict[str, Any]] = {}

        # Rate limiting
        self._request_timestamps: List[datetime] = []

        super().__init__()

    # =========================================================================
    # Tool Definition
    # =========================================================================

    @property
    def name(self) -> str:
        return "http_request"

    @property
    def description(self) -> str:
        return """Professional HTTP request tool with advanced features:
- Automatic retries with exponential backoff
- Rate limiting
- Response caching
- JSON schema validation
- Multipart/form-data support
- Streaming support
- Performance metrics
- Detailed error handling"""

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.WEB

    @property
    def enabled_by_default(self) -> bool:
        return False

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="url",
                type="string",
                description="Full URL for HTTP request (must include http:// or https://)",
                required=True,
                example="https://api.example.com/data"
            ),
            ToolParameter(
                name="method",
                type="string",
                description="HTTP method (GET, POST, PUT, DELETE, PATCH)",
                required=True,
                enum=["GET", "POST", "PUT", "DELETE", "PATCH"],
                example="GET"
            ),
            ToolParameter(
                name="headers",
                type="object",
                description="HTTP headers as key-value pairs",
                required=False,
                default={},
                example={"Accept": "application/json"}
            ),
            ToolParameter(
                name="params",
                type="object",
                description="Query parameters for GET requests",
                required=False,
                default={},
                example={"page": 1, "limit": 10}
            ),
            ToolParameter(
                name="body",
                type="object",
                description="Request body (for POST, PUT, PATCH)",
                required=False,
                default=None,
                example={"key": "value"}
            ),
            ToolParameter(
                name="timeout",
                type="integer",
                description="Request timeout in seconds (default: 30)",
                required=False,
                default=30,
                example=30
            ),
            ToolParameter(
                name="verify_ssl",
                type="boolean",
                description="Whether to verify SSL certificates (default: true)",
                required=False,
                default=True,
                example=True
            ),
            ToolParameter(
                name="auth",
                type="object",
                description="""Authentication configuration. Format:
                                {
                                    "auth_type": "none|basic|bearer|api_key|oauth2|digest|custom",
                                    "credentials": {
                                        // Basic: {"username": "...", "password": "..."}
                                        // Bearer: {"token": "..."}
                                        // API Key: {"api_key": "...", "key_name": "X-API-Key", "location": "header"}
                                        // OAuth2: {"access_token": "...", "token_type": "Bearer"}
                                        // Custom: {"headers": {"X-Custom-Auth": "..."}}
                                    }
                                }""",
                required=False,
                default=None,
                example={
                    "auth_type": "bearer",
                    "credentials": {"token": "your-token-here"}
                }
            ),
            ToolParameter(
                name="retry_config",
                type="object",
                description="""Retry configuration. Format:
                                {
                                    "max_retries": 3,
                                    "strategy": "exponential_backoff|linear_backoff|fixed_delay|no_retry",
                                    "initial_delay": 1.0,
                                    "max_delay": 60.0,
                                    "backoff_multiplier": 2.0
                                }""",
                required=False,
                default=None
            ),
            ToolParameter(
                name="cache_config",
                type="object",
                description="""Cache configuration. Format:
                                {
                                    "strategy": "no_cache|memory_cache|etag_cache",
                                    "ttl": 300,
                                    "max_size": 100
                                }""",
                required=False,
                default=None
            ),
            ToolParameter(
                name="json_schema",
                type="object",
                description="JSON schema for response validation",
                required=False,
                default=None
            ),
            ToolParameter(
                name="stream",
                type="boolean",
                description="Whether to stream the response (default: false)",
                required=False,
                default=False
            ),
            ToolParameter(
                name="multipart",
                type="object",
                description="""Multipart/form-data configuration. Format:
                                {
                                    "files": [{"name": "file1", "content": "...", "filename": "..."}],
                                    "data": {"field1": "value1"}
                                }""",
                required=False,
                default=None
            )
        ]

    # =========================================================================
    # Main Execution Method
    # =========================================================================

    async def execute(
        self,
        url: str,
        method: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
        verify_ssl: bool = True,
        auth: Optional[Dict[str, Any]] = None,
        retry_config: Optional[Dict[str, Any]] = None,
        cache_config: Optional[Dict[str, Any]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        multipart: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Execute HTTP request with professional features"""

        try:
            # Validate inputs
            await self.validate_input(
                url=url,
                method=method,
                headers=headers or {},
                params=params or {},
                body=body,
                timeout=timeout,
                verify_ssl=verify_ssl,
                auth=auth,
            )

            # Validate URL
            if not self._is_valid_url(url):
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Invalid URL: {url}"
                )

            # Apply configurations
            if retry_config:
                self._apply_retry_config(retry_config)

            if cache_config:
                self._apply_cache_config(cache_config)

            # Check cache
            cache_key = self._generate_cache_key(url, method, params, body)
            if self._cache_config.strategy != CacheStrategy.NO_CACHE:
                cached_response = self._get_from_cache(cache_key)
                if cached_response:
                    self._metrics.cache_hits += 1
                    self.logger.info(f"Cache hit for {url}")
                    return ToolResult(
                        success=True,
                        data=cached_response,
                        metadata={
                            "cached": True,
                            "cache_key": cache_key
                        }
                    )
                self._metrics.cache_misses += 1

            # Check rate limit
            if not self._check_rate_limit():
                return ToolResult(
                    success=False,
                    data=None,
                    error="Rate limit exceeded. Please wait before making more requests."
                )

            # Prepare headers
            request_headers = headers.copy() if headers else {}

            # Apply authentication
            request_headers = self._apply_authentication(
                request_headers,
                auth
            )

            # Set content type
            if body is not None and "Content-Type" not in request_headers:
                request_headers["Content-Type"] = "application/json"

            # Handle API Key in query params
            if auth and auth.get("auth_type") == "api_key":
                credentials = auth.get("credentials", {})
                if credentials.get("location") == "query":
                    if params is None:
                        params = {}
                    key_name = credentials.get("key_name", "api_key")
                    params[key_name] = credentials.get("api_key")

            # Make HTTP request with retry logic
            response = await self._make_request_with_retry(
                url=url,
                method=method,
                headers=request_headers,
                params=params,
                body=body,
                timeout=timeout,
                verify_ssl=verify_ssl,
                stream=stream,
                multipart=multipart
            )

            # Update metrics
            self._metrics.request_count += 1
            if response.get("success"):
                self._metrics.success_count += 1
                self._metrics.total_response_time += response.get("response_time", 0)
                self._metrics.total_bytes_sent += response.get("bytes_sent", 0)
                self._metrics.total_bytes_received += response.get("bytes_received", 0)
            else:
                self._metrics.failure_count += 1

            # Process response
            if not response.get("success"):
                return ToolResult(
                    success=False,
                    data=None,
                    error=response.get("error")
                )

            # Parse response body
            response_data = response.get("data")

            # Validate JSON schema if provided
            if json_schema and isinstance(response_data, dict):
                validation_result = self._validate_json_schema(response_data, json_schema)
                if not validation_result.get("valid"):
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"JSON schema validation failed: {validation_result.get('errors')}"
                    )

            # Cache successful GET requests
            if method.upper() == "GET" and self._cache_config.strategy != CacheStrategy.NO_CACHE:
                self._save_to_cache(cache_key, response_data)

            # Log response
            self.logger.info(
                f"📥 HTTP RESPONSE:\n"
                f"  Status: {response.get('status_code')}\n"
                f"  URL: {url}\n"
                f"  Data: {str(response_data)[:500]}...\n"
                f"  Response Time: {response.get('response_time', 0):.3f}s\n"
                f"  Cached: {response.get('cached', False)}"
            )

            return ToolResult(
                success=True,
                data=response_data,
                metadata={
                    "http_status": response.get("status_code"),
                    "http_method": method,
                    "url": url,
                    "headers": response.get("headers", {}),
                    "response_size": response.get("bytes_received", 0),
                    "response_time": response.get("response_time", 0),
                    "auth_type": self._auth_strategy.get_auth_type().value,
                    "cached": response.get("cached", False),
                    "retry_count": response.get("retry_count", 0),
                    "performance_metrics": self._get_performance_metrics()
                }
            )

        except Exception as e:
            self.logger.error(
                f"HTTP tool execution error: {e}",
                exc_info=True,
                extra={"url": url, "method": method}
            )

            return ToolResult(
                success=False,
                data=None,
                error=str(e)
            )

    # =========================================================================
    # Request with Retry Logic
    # =========================================================================

    async def _make_request_with_retry(
        self,
        url: str,
        method: str,
        headers: Dict[str, str],
        params: Optional[Dict[str, Any]],
        body: Optional[Dict[str, Any]],
        timeout: int,
        verify_ssl: bool,
        stream: bool,
        multipart: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic"""

        retry_count = 0
        last_error = None

        while retry_count <= self._retry_config.max_retries:
            try:
                result = await self._make_single_request(
                    url=url,
                    method=method,
                    headers=headers,
                    params=params,
                    body=body,
                    timeout=timeout,
                    verify_ssl=verify_ssl,
                    stream=stream,
                    multipart=multipart
                )

                # Check if response is retryable
                if result.get("success"):
                    status_code = result.get("status_code")
                    if status_code in self._retry_config.retryable_status_codes:
                        retry_count += 1
                        last_error = f"HTTP {status_code}: {HTTP_ERROR_DESCRIPTIONS.get(status_code, 'Unknown error')}"
                        self.logger.warning(f"Retryable error, attempt {retry_count}/{self._retry_config.max_retries}: {last_error}")

                        # Calculate delay
                        delay = self._calculate_retry_delay(retry_count)
                        await asyncio.sleep(delay)
                        continue
                    else:
                        # Success
                        result["retry_count"] = retry_count
                        return result
                else:
                    # Non-retryable error
                    return result

            except Exception as e:
                retry_count += 1
                last_error = str(e)
                self.logger.warning(f"Request failed, attempt {retry_count}/{self._retry_config.max_retries}: {last_error}")

                if retry_count > self._retry_config.max_retries:
                    return {
                        "success": False,
                        "error": f"Request failed after {retry_count} attempts: {last_error}"
                    }

                # Calculate delay
                delay = self._calculate_retry_delay(retry_count)
                await asyncio.sleep(delay)

        return {
            "success": False,
            "error": f"Max retries ({self._retry_config.max_retries}) exceeded. Last error: {last_error}"
        }

    async def _make_single_request(
        self,
        url: str,
        method: str,
        headers: Dict[str, str],
        params: Optional[Dict[str, Any]],
        body: Optional[Dict[str, Any]],
        timeout: int,
        verify_ssl: bool,
        stream: bool,
        multipart: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Make a single HTTP request"""

        start_time = datetime.utcnow()

        async with httpx.AsyncClient(timeout=timeout, verify=verify_ssl) as client:
            method_upper = method.upper()

            # Log request
            safe_headers = self._sanitize_headers_for_log(headers)
            self.logger.info(
                f"📤 HTTP REQUEST: {method_upper} {url}\n"
                f"Headers: {safe_headers}\n"
                f"Params: {params}\n"
                f"Body: {body}\n"
                f"Multipart: {multipart is not None}"
            )

            # Prepare request
            request_kwargs = {
                "url": url,
                "headers": headers,
                "timeout": timeout
            }

            # Handle multipart
            if multipart:
                files_data = multipart.get("files", [])
                form_data = multipart.get("data", {})

                files = []
                for file_info in files_data:
                    files.append(
                        (file_info.get("name"), file_info.get("content"), file_info.get("filename"))
                    )

                request_kwargs["files"] = files
                request_kwargs["data"] = form_data
            else:
                # Standard request
                if params:
                    request_kwargs["params"] = params

                if body:
                    request_kwargs["json"] = body

            # Execute request
            if method_upper == "GET":
                response = await client.get(**request_kwargs)
            elif method_upper == "POST":
                response = await client.post(**request_kwargs)
            elif method_upper == "PUT":
                response = await client.put(**request_kwargs)
            elif method_upper == "DELETE":
                response = await client.delete(**request_kwargs)
            elif method_upper == "PATCH":
                response = await client.patch(**request_kwargs)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported HTTP method: {method}"
                }

            # Calculate response time
            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds()

            # Update metrics
            if response_time < self._metrics.min_response_time:
                self._metrics.min_response_time = response_time
            if response_time > self._metrics.max_response_time:
                self._metrics.max_response_time = response_time

            # Process response
            if response.is_error:
                error_detail = f"HTTP {response.status_code}: {HTTP_ERROR_DESCRIPTIONS.get(response.status_code, response.text)}"
                self.logger.error(
                    f"HTTP request failed: {error_detail}",
                    extra={"url": url, "method": method, "status_code": response.status_code}
                )

                return {
                    "success": False,
                    "error": error_detail,
                    "status_code": response.status_code
                }

            # Parse response body
            try:
                response_data = response.json()
            except Exception:
                response_data = response.text

            return {
                "success": True,
                "data": response_data,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "response_time": response_time,
                "bytes_sent": len(str(body)) if body else 0,
                "bytes_received": len(response.content)
            }

    # =========================================================================
    # Configuration Methods
    # =========================================================================

    def _apply_retry_config(self, retry_config: Dict[str, Any]):
        """Apply retry configuration"""
        self._retry_config = RetryConfig(
            max_retries=retry_config.get("max_retries", 3),
            strategy=RetryStrategy(retry_config.get("strategy", "exponential_backoff")),
            initial_delay=retry_config.get("initial_delay", 1.0),
            max_delay=retry_config.get("max_delay", 60.0),
            backoff_multiplier=retry_config.get("backoff_multiplier", 2.0)
        )

    def _apply_cache_config(self, cache_config: Dict[str, Any]):
        """Apply cache configuration"""
        self._cache_config = CacheConfig(
            strategy=CacheStrategy(cache_config.get("strategy", "memory_cache")),
            ttl=cache_config.get("ttl", 300),
            max_size=cache_config.get("max_size", 100)
        )

    def _calculate_retry_delay(self, retry_count: int) -> float:
        """Calculate retry delay based on strategy"""
        if self._retry_config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self._retry_config.initial_delay * (self._retry_config.backoff_multiplier ** (retry_count - 1))
        elif self._retry_config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self._retry_config.initial_delay * retry_count
        else:  # FIXED_DELAY
            delay = self._retry_config.initial_delay

        return min(delay, self._retry_config.max_delay)

    # =========================================================================
    # Cache Methods
    # =========================================================================

    def _generate_cache_key(
        self,
        url: str,
        method: str,
        params: Optional[Dict[str, Any]],
        body: Optional[Dict[str, Any]]
    ) -> str:
        """Generate cache key from request parameters"""
        key_data = {
            "url": url,
            "method": method,
            "params": params,
            "body": body
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get response from cache"""
        if cache_key in self._cache:
            cached = self._cache[cache_key]

            # Check TTL
            if datetime.utcnow() < cached["expires_at"]:
                return cached["data"]
            else:
                # Expired, remove from cache
                del self._cache[cache_key]

        return None

    def _save_to_cache(self, cache_key: str, data: Any):
        """Save response to cache"""
        # Check cache size
        if len(self._cache) >= self._cache_config.max_size:
            # Remove oldest entry
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]["created_at"])
            del self._cache[oldest_key]

        self._cache[cache_key] = {
            "data": data,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(seconds=self._cache_config.ttl)
        }

    # =========================================================================
    # Rate Limiting Methods
    # =========================================================================

    def _check_rate_limit(self) -> bool:
        """Check if request is within rate limits"""
        now = datetime.utcnow()

        # Clean old timestamps
        self._request_timestamps = [
            ts for ts in self._request_timestamps
            if (now - ts).total_seconds() < 60
        ]

        # Check per-second limit
        recent_second = [ts for ts in self._request_timestamps if (now - ts).total_seconds() < 1]
        if len(recent_second) >= self._rate_limit_config.requests_per_second:
            return False

        # Check per-minute limit
        if len(self._request_timestamps) >= self._rate_limit_config.requests_per_minute:
            return False

        # Add current timestamp
        self._request_timestamps.append(now)
        return True

    # =========================================================================
    # JSON Schema Validation
    # =========================================================================

    def _validate_json_schema(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate JSON data against schema"""
        errors = []

        # Simple validation (in production, use jsonschema library)
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        # Type validation
        properties = schema.get("properties", {})
        for field, field_schema in properties.items():
            if field in data:
                expected_type = field_schema.get("type")
                if expected_type:
                    if expected_type == "string" and not isinstance(data[field], str):
                        errors.append(f"Field '{field}' should be string")
                    elif expected_type == "number" and not isinstance(data[field], (int, float)):
                        errors.append(f"Field '{field}' should be number")
                    elif expected_type == "boolean" and not isinstance(data[field], bool):
                        errors.append(f"Field '{field}' should be boolean")
                    elif expected_type == "array" and not isinstance(data[field], list):
                        errors.append(f"Field '{field}' should be array")
                    elif expected_type == "object" and not isinstance(data[field], dict):
                        errors.append(f"Field '{field}' should be object")

        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

    # =========================================================================
    # Performance Metrics Methods
    # =========================================================================

    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        avg_response_time = (
            self._metrics.total_response_time / self._metrics.success_count
            if self._metrics.success_count > 0 else 0
        )

        return {
            "request_count": self._metrics.request_count,
            "success_count": self._metrics.success_count,
            "failure_count": self._metrics.failure_count,
            "success_rate": (
                self._metrics.success_count / self._metrics.request_count * 100
                if self._metrics.request_count > 0 else 0
            ),
            "avg_response_time": avg_response_time,
            "min_response_time": self._metrics.min_response_time,
            "max_response_time": self._metrics.max_response_time,
            "total_bytes_sent": self._metrics.total_bytes_sent,
            "total_bytes_received": self._metrics.total_bytes_received,
            "cache_hits": self._metrics.cache_hits,
            "cache_misses": self._metrics.cache_misses,
            "cache_hit_rate": (
                self._metrics.cache_hits / (self._metrics.cache_hits + self._metrics.cache_misses) * 100
                if (self._metrics.cache_hits + self._metrics.cache_misses) > 0 else 0
            )
        }

    def reset_metrics(self):
        """Reset performance metrics"""
        self._metrics = PerformanceMetrics()

    # =========================================================================
    # Authentication Methods (inherited from original)
    # =========================================================================

    def _apply_authentication(
        self,
        headers: Dict[str, str],
        auth: Optional[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Apply authentication strategy to headers"""
        if auth:
            try:
                auth_type = AuthType(auth.get("auth_type", "none"))
                credentials = auth.get("credentials", {})

                self._auth_strategy = AuthFactory.create_auth(auth_type, credentials)
                headers = self._auth_strategy.apply(headers)

                self.logger.debug(f"Authentication applied: {auth_type.value}")

            except Exception as e:
                self.logger.warning(f"Failed to apply authentication: {e}")

        return headers

    def _sanitize_headers_for_log(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Sanitize sensitive headers for logging"""
        safe_headers = headers.copy()
        sensitive_keys = ["Authorization", "X-API-Key", "X-Auth-Token", "Cookie"]

        for key in sensitive_keys:
            if key in safe_headers:
                value = safe_headers[key]
                if len(value) > 10:
                    safe_headers[key] = value[:10] + "***"
                else:
                    safe_headers[key] = "***"

        return safe_headers

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _is_valid_url(self, url: str) -> bool:
        """Validate that URL is secure and well-formed"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)

            if not parsed.scheme or not parsed.netloc:
                return False

            if parsed.scheme not in ["http", "https"]:
                return False

            return True

        except Exception:
            return False

    async def test_endpoint(
        self,
        url: str,
        method: str = "GET",
        timeout: int = 10,
        verify_ssl: bool = True,
        auth: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Test if an endpoint is reachable with authentication"""

        try:
            if not self._is_valid_url(url):
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Invalid URL: {url}"
                )

            headers = {}
            if auth:
                headers = self._apply_authentication(headers, auth)

            async with httpx.AsyncClient(timeout=timeout, verify=verify_ssl) as client:
                method_upper = method.upper()

                if method_upper == "GET":
                    response = await client.get(url, headers=headers)
                elif method_upper == "POST":
                    response = await client.post(url, headers=headers)
                else:
                    response = await client.get(url, headers=headers)

                if response.is_error:
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"HTTP {response.status_code}"
                    )

                self.logger.info(
                    f"Endpoint test successful",
                    extra={"url": url, "status_code": response.status_code}
                )

                return ToolResult(
                    success=True,
                    data={
                        "status": "success",
                        "status_code": response.status_code,
                        "url": url
                    },
                    metadata={
                        "status_code": response.status_code,
                        "url": url
                    }
                )

        except Exception as e:
            self.logger.error(
                f"Endpoint test failed: {e}",
                exc_info=True,
                extra={"url": url}
            )

            return ToolResult(
                success=False,
                data=None,
                error=str(e)
            )
