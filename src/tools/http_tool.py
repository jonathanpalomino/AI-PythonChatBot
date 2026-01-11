# =============================================================================
# src/tools/http_tool.py
# HTTP Request Tool
# =============================================================================
"""
Tool for making HTTP requests (GET, POST, etc.) to external APIs
"""

from typing import List, Dict, Any, Optional
import json

import httpx

from src.tools.base_tool import BaseTool, ToolCategory, ToolParameter, ToolResult
from src.utils.logger import get_logger


class HTTPTool(BaseTool):
    """Tool for making HTTP requests to external APIs"""

    def __init__(self):
        self.logger = get_logger(__name__)
        super().__init__()

    # =============================================================================
    # Tool Definition
    # =============================================================================

    @property
    def name(self) -> str:
        return "http_request"

    @property
    def description(self) -> str:
        return "Make HTTP requests (GET, POST, PUT, DELETE) to external APIs with custom headers and body"

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
                description="Full URL for the HTTP request (must include http:// or https://)",
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
                name="auth_token",
                type="string",
                description="Optional authentication token (will be added to Authorization header)",
                required=False,
                default=None,
                example="your-api-token-here"
            ),
            ToolParameter(
                name="verify_ssl",
                type="boolean",
                description="Whether to verify SSL certificates (default: true). Set to false to bypass SSL verification for self-signed certificates.",
                required=False,
                default=True,
                example=True
            )
        ]

    # =============================================================================
    # Execution
    # =============================================================================

    async def execute(
        self,
        url: str,
        method: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
        auth_token: Optional[str] = None,
        verify_ssl: bool = True
    ) -> ToolResult:
        """Execute HTTP request"""
        try:
            # Validate inputs
            await self.validate_input(
                url=url,
                method=method,
                headers=headers or {},
                params=params or {},
                body=body,
                timeout=timeout,
                auth_token=auth_token,
                verify_ssl=verify_ssl
            )

            # Validate URL
            if not self._is_valid_url(url):
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Invalid URL: {url}"
                )

            # Prepare headers
            request_headers = headers or {}
            
            # Add auth token if provided
            if auth_token:
                request_headers["Authorization"] = f"Bearer {auth_token}"
            
            # Set content type for JSON body
            if body is not None and "Content-Type" not in request_headers:
                request_headers["Content-Type"] = "application/json"

            # Make HTTP request
            async with httpx.AsyncClient(timeout=timeout, verify=verify_ssl) as client:
                method_upper = method.upper()
                # Log full request details for debugging
                self.logger.info(
                    f"📤 HTTP REQUEST: {method_upper} {url}\n"
                    f"Headers: {request_headers}\n"
                    f"Params: {params}\n"
                    f"Body: {body}"
                )

                if method_upper == "GET":
                    response = await client.get(
                        url,
                        params=params,
                        headers=request_headers
                    )
                elif method_upper == "POST":
                    response = await client.post(
                        url,
                        json=body,
                        headers=request_headers
                    )
                elif method_upper == "PUT":
                    response = await client.put(
                        url,
                        json=body,
                        headers=request_headers
                    )
                elif method_upper == "DELETE":
                    response = await client.delete(
                        url,
                        json=body,
                        headers=request_headers
                    )
                elif method_upper == "PATCH":
                    response = await client.patch(
                        url,
                        json=body,
                        headers=request_headers
                    )
                else:
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"Unsupported HTTP method: {method}"
                    )

                # Process response
                if response.is_error:
                    error_detail = f"HTTP {response.status_code}: {response.text}"
                    self.logger.error(
                        f"HTTP request failed: {error_detail}",
                        extra={"url": url, "method": method, "status_code": response.status_code}
                    )
                    return ToolResult(
                        success=False,
                        data=None,
                        error=error_detail
                    )

                # Parse response body
                try:
                    response_data = response.json()
                except Exception:
                    response_data = response.text

                # Log full response details for debugging
                self.logger.info(
                    f"📥 HTTP RESPONSE: {response.status_code} {url}\n"
                    f"Data: {response_data}"
                )

                return ToolResult(
                    success=True,
                    data=response_data,
                    metadata={
                        "http_status": response.status_code,
                        "http_method": method,
                        "url": url,
                        "headers": dict(response.headers),
                        "response_size": len(str(response_data))
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

    # =============================================================================
    # Helper Methods
    # =============================================================================

    def _is_valid_url(self, url: str) -> bool:
        """Validar que la URL sea segura y bien formada"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            
            # Validar que tenga esquema y dominio
            if not parsed.scheme or not parsed.netloc:
                return False
            
            # Validar que el esquema sea HTTP o HTTPS
            if parsed.scheme not in ["http", "https"]:
                return False
            
            # Validar que no sea una URL local o privada
            if parsed.netloc.startswith("localhost") or parsed.netloc.startswith("127.0.0.1"):
                return False
            
            return True
        except Exception:
            return False

    async def test_endpoint(
        self,
        url: str,
        method: str = "GET",
        timeout: int = 10,
        verify_ssl: bool = True
    ) -> ToolResult:
        """Test if an endpoint is reachable"""
        try:
            if not self._is_valid_url(url):
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Invalid URL: {url}"
                )

            async with httpx.AsyncClient(timeout=timeout, verify=verify_ssl) as client:
                method_upper = method.upper()
                
                if method_upper == "GET":
                    response = await client.get(url)
                elif method_upper == "POST":
                    response = await client.post(url)
                else:
                    response = await client.get(url)

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
