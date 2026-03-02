# =============================================================================
# src/utils/health_checker.py
# Centralized Health Check Utilities
# =============================================================================
"""
Centralized health check utilities for external services.
Provides unified methods for checking service availability.

Includes both:
- System-level health checks (Redis, Celery, Qdrant) - static methods
- RAG-specific health checks (Qdrant, Embeddings, BM25) - async instance methods
"""
import socket
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict

import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class HealthCheckResult:
    """Result of health check"""
    is_healthy: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# Health Checker
# =============================================================================

class HealthChecker:
    """
    Centralized health check utilities.
    
    Provides both:
    - Static methods for system-level checks (socket, HTTP, processes)
    - Async instance methods for RAG-specific checks (Qdrant, embeddings, BM25)
    
    Usage:
        # System checks (static methods)
        HealthChecker.check_socket('localhost', 6379)
        HealthChecker.check_celery_worker()
        
        # RAG checks (instance methods)
        checker = HealthChecker()
        result = await checker.perform_health_check(qdrant_client, embedding_service, bm25_indexes)
    """

    # =========================================================================
    # System-Level Health Checks (Static Methods)
    # =========================================================================

    @staticmethod
    def check_socket(host: str, port: int, timeout: int = 1) -> bool:
        """
        Check if a socket connection can be established.

        Args:
            host: Host address to check
            port: Port number to check
            timeout: Connection timeout in seconds

        Returns:
            True if connection successful, False otherwise
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                result = s.connect_ex((host, port))
                return result == 0
        except Exception as e:
            logger.debug(f"Socket check failed for {host}:{port}: {e}")
            return False

    @staticmethod
    def check_http(url: str, timeout: int = 2) -> bool:
        """
        Check if an HTTP endpoint is accessible.

        Args:
            url: URL to check
            timeout: Request timeout in seconds

        Returns:
            True if endpoint returns 200, False otherwise
        """
        try:
            response = requests.get(url, timeout=timeout)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    @staticmethod
    def check_process(process_name: str) -> bool:
        """
        Check if a process is running (Windows).

        Args:
            process_name: Name of the process to check

        Returns:
            True if process is running, False otherwise
        """
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"IMAGENAME eq {process_name}", "/FO", "CSV"],
                capture_output=True,
                text=True
            )
            return process_name.lower() in result.stdout.lower()
        except Exception as e:
            logger.debug(f"Process check failed for {process_name}: {e}")
            return False

    @staticmethod
    def check_celery_worker() -> bool:
        """
        Check if a Celery worker process is running.

        Uses wmic to check for Python processes with 'celery.*worker' in command line.

        Returns:
            True if Celery worker is running, False otherwise
        """
        try:
            result = subprocess.run(
                ["wmic", "process", "where", "name='python.exe' or name='pythonw.exe'", "get", "commandline"],
                capture_output=True,
                text=True,
                timeout=5
            )
            # Check if 'celery' and 'worker' appear in the command line output
            output = result.stdout.lower()
            return "celery" in output and "worker" in output
        except Exception as e:
            logger.debug(f"Celery worker health check failed: {e}")
            return False

    @staticmethod
    def check_celery_flower(port: int = 5555) -> bool:
        """
        Check if Celery Flower is running.

        Checks both the port and the process.

        Args:
            port: Flower port (default: 5555)

        Returns:
            True if Flower is running, False otherwise
        """
        # Check if port is in use
        if not HealthChecker.check_socket('127.0.0.1', port, timeout=1):
            return False

        # Additional check: verify it's actually a Flower process
        try:
            result = subprocess.run(
                ["wmic", "process", "where", "name='python.exe' or name='pythonw.exe'", "get", "commandline"],
                capture_output=True,
                text=True,
                timeout=5
            )
            output = result.stdout.lower()
            return "flower" in output
        except Exception as e:
            logger.debug(f"Celery Flower process check failed: {e}")
            # If process check fails but port is open, assume Flower is running
            return True

    # =========================================================================
    # RAG-Specific Health Checks (Async Instance Methods)
    # =========================================================================

    def __init__(self):
        """Initialize HealthChecker for RAG-specific checks."""
        self.logger = get_logger(__name__)

    async def check_qdrant_connection(self, qdrant_client) -> bool:
        """
        Check Qdrant connection.

        Args:
            qdrant_client: Qdrant client instance

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try to get collections list
            collections = await qdrant_client.get_collections()
            self.logger.debug(f"Qdrant health check: {len(collections.collections)} collections found")
            return True
        except Exception as e:
            self.logger.error(f"Qdrant health check failed: {e}", exc_info=True)
            return False

    async def check_embedding_service(self, embedding_service) -> bool:
        """
        Check embedding service.

        Args:
            embedding_service: Embedding service instance

        Returns:
            True if embedding service is working, False otherwise
        """
        try:
            # Try to generate a test embedding
            test_text = "health check"
            embedding = await embedding_service.generate_embedding(test_text)
            is_valid = len(embedding) > 0
            self.logger.debug(f"Embedding service health check: {'OK' if is_valid else 'FAILED'}")
            return is_valid
        except Exception as e:
            self.logger.error(f"Embedding service health check failed: {e}", exc_info=True)
            return False

    async def check_bm25_indexes(self, bm25_indexes: Dict[str, Any]) -> bool:
        """
        Check BM25 indexes.

        Args:
            bm25_indexes: Dictionary of BM25 indexes

        Returns:
            True if indexes are loaded, False otherwise
        """
        try:
            # Check if indexes are loaded
            is_valid = len(bm25_indexes) > 0
            self.logger.debug(f"BM25 indexes health check: {len(bm25_indexes)} indexes loaded")
            return is_valid
        except Exception as e:
            self.logger.error(f"BM25 indexes health check failed: {e}", exc_info=True)
            return False

    async def perform_health_check(
        self,
        qdrant_client,
        embedding_service,
        bm25_indexes: Dict[str, Any]
    ) -> HealthCheckResult:
        """
        Perform comprehensive health check for RAG components.

        Args:
            qdrant_client: Qdrant client instance
            embedding_service: Embedding service instance
            bm25_indexes: Dictionary of BM25 indexes

        Returns:
            HealthCheckResult with check results
        """
        checks = {}
        details = {}

        # Check Qdrant
        checks["qdrant"] = await self.check_qdrant_connection(qdrant_client)

        # Check embedding service
        checks["embedding_service"] = await self.check_embedding_service(embedding_service)

        # Check BM25 indexes
        checks["bm25_indexes"] = await self.check_bm25_indexes(bm25_indexes)

        # Overall health
        is_healthy = all(checks.values())

        # Add details
        details["checks"] = checks
        details["bm25_index_count"] = len(bm25_indexes)

        return HealthCheckResult(
            is_healthy=is_healthy,
            checks=checks,
            details=details,
            timestamp=datetime.utcnow()
        )
