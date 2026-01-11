# =============================================================================
# config/settings.py
# Configuration management
# =============================================================================
"""
Configuración centralizada usando Pydantic Settings
Soporta variables de entorno y archivos .env
"""
import os
from pathlib import Path
from typing import Optional, List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings"""

    # =============================================================================
    # General
    # =============================================================================
    APP_NAME: str = "RAG Chatbot"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field("local", description="local, development, production")
    DEBUG: bool = Field(False, description="Debug mode")

    # =============================================================================
    # Database (PostgreSQL)
    # =============================================================================
    DATABASE_URL: str = Field(
        "postgresql://chatbot_ia:chatbot_ia@localhost:5432/chatbot_ia_db",
        description="PostgreSQL connection string"
    )
    DATABASE_ECHO: bool = Field(False, description="SQLAlchemy echo SQL")
    DATABASE_POOL_SIZE: int = Field(5, ge=1)
    DATABASE_MAX_OVERFLOW: int = Field(10, ge=0)

    # =============================================================================
    # Qdrant (Vector Database)
    # =============================================================================
    QDRANT_URL: str = Field(
        "http://localhost:6333",
        description="Qdrant server URL"
    )
    QDRANT_API_KEY: Optional[str] = Field(None, description="Qdrant API key (for cloud)")
    QDRANT_TIMEOUT: int = Field(60, description="Request timeout in seconds")

    # Default collection settings
    COLLECTION_NAME: str = Field("documents", description="Default collection name")
    VECTOR_SIZE: int = Field(768, description="Vector dimension (auto-detected from embedding model)")

    # =============================================================================
    # Redis (Cache & Queue)
    # =============================================================================
    REDIS_URL: str = Field(
        "redis://localhost:6379/0",
        description="Redis connection string"
    )
    REDIS_MAX_CONNECTIONS: int = Field(10, ge=1)
    CACHE_TTL: int = Field(3600, description="Default cache TTL in seconds")

    # =============================================================================
    # LLM Providers
    # =============================================================================

    # Ollama (Local)
    OLLAMA_BASE_URL: str = Field("http://localhost:11434", description="Ollama server URL")
    EMBEDDING_MODEL: str = Field("mxbai-embed-large", description="Default embedding model")

    # Contextual Retrieval (improves RAG quality by +35%)
    ENABLE_CONTEXTUAL_RETRIEVAL: bool = Field(
        False,
        description="Enable LLM-generated context for chunks before indexing"
    )
    CONTEXT_GENERATION_MODEL: str = Field(
        "qwen2.5:3b",  # Fast, small model for context generation
        description="Model for generating chunk context descriptions"
    )
    CONTEXT_MAX_TOKENS: int = Field(
        50,
        description="Max tokens for generated context"
    )
    LLM_MODEL: str = Field("mistral", description="Default LLM model")

    # Obsidian Configuration
    ENABLE_OBSIDIAN_GRAPH: bool = True  # Enable graph-aware processing
    OBSIDIAN_MAX_EXPANSION_LINKS: int = 3  # Max links to follow when expanding
    OBSIDIAN_CACHE_TTL: int = 3600  # Cache TTL for vault graphs (1 hour)

    # OpenAI
    OPENAI_API_KEY: Optional[str] = Field(None, description="OpenAI API key")
    OPENAI_ORG_ID: Optional[str] = Field(None, description="OpenAI organization ID")

    # Anthropic (Claude)
    ANTHROPIC_API_KEY: Optional[str] = Field(None, description="Anthropic API key")

    # Google (Gemini)
    GOOGLE_API_KEY: Optional[str] = Field(None, description="Google AI API key")

    # OpenRouter
    OPENROUTER_API_KEY: Optional[str] = Field(None, description="OpenRouter API key")

    # Groq
    GROQ_API_KEY: Optional[str] = Field(None, description="Groq API key")


    # Obsidian Configuration
    ENABLE_OBSIDIAN_GRAPH: bool = True  # Enable graph-aware processing
    OBSIDIAN_MAX_EXPANSION_LINKS: int = 3  # Max links to follow when expanding
    OBSIDIAN_CACHE_TTL: int = 3600  # Cache TTL for vault graphs (1 hour)
    
    # =============================================================================
    # File Storage
    # =============================================================================
    FILE_STORAGE_TYPE: str = Field(
        "local",
        description="Storage type: local, s3, minio"
    )
    UPLOAD_DIR: Path = Field(
        Path("./data/uploads"),
        description="Local upload directory"
    )
    MAX_UPLOAD_SIZE: int = Field(
        100 * 1024 * 1024,  # 100MB
        description="Max upload size in bytes"
    )
    # File extensions are now dynamically determined from document loaders
    # See get_allowed_extensions() method below
    ALLOWED_EXTENSIONS: List[str] = Field(
        default_factory=lambda: [],
        description="Allowed file extensions (dynamically loaded from document loaders)"
    )

    # AWS S3 (if FILE_STORAGE_TYPE = "s3")
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_S3_BUCKET: Optional[str] = None
    AWS_REGION: str = "us-east-1"

    # =============================================================================
    # API Settings
    # =============================================================================
    API_PREFIX: str = "/api/v1"
    CORS_ORIGINS: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:5173",
                                 "http://localhost:4200"],
        description="Allowed CORS origins"
    )
    CORS_CREDENTIALS: bool = True
    CORS_METHODS: List[str] = Field(default_factory=lambda: ["*"])
    CORS_HEADERS: List[str] = Field(default_factory=lambda: ["*"])

    # Rate limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 60

    # =============================================================================
    # Background Jobs
    # =============================================================================
    CELERY_BROKER_URL: str = Field(
        "redis://localhost:6379/1",
        description="Celery broker URL"
    )
    CELERY_RESULT_BACKEND: str = Field(
        "redis://localhost:6379/2",
        description="Celery result backend"
    )

    # =============================================================================
    # Security
    # =============================================================================
    SECRET_KEY: str = Field(
        "your-secret-key-change-in-production",
        description="Secret key for encryption"
    )
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    # =============================================================================
    # Logging
    # =============================================================================
    LOG_LEVEL: str = Field("INFO", description="Logging level")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: Optional[Path] = Field(None, description="Log file path")

    # =============================================================================
    # RAG Settings
    # =============================================================================
    TRACKING_FILE: Path = Field(
        Path("./data/sync_tracking.json"),
        description="Sync tracking file"
    )
    AUTO_PROCESS_FILES: bool = Field(
        True,
        description="Automatically process uploaded files and index to Qdrant"
    )
    DEFAULT_CHUNK_SIZE: int = Field(1000, description="Default text chunk size")
    DEFAULT_CHUNK_OVERLAP: int = Field(200, description="Chunk overlap")
    DEFAULT_CHUNK_STRATEGY: str = Field("fixed", description="Default chunking strategy: fixed, sentence, paragraph, semantic")

    # =============================================================================
    # Re-ranking Settings
    # =============================================================================
    RERANK_MODEL: str = Field(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for re-ranking"
    )
    RERANK_BATCH_SIZE: int = Field(32, description="Batch size for re-ranking inference")
    RERANK_DEVICE: Optional[str] = Field(None,
                                         description="Device for re-ranking (cuda/cpu/None=auto)")

    # =============================================================================
    # Hybrid Search Settings
    # =============================================================================
    BM25_K1: float = Field(1.5, description="BM25 k1 parameter (term frequency saturation)")
    BM25_B: float = Field(0.75, description="BM25 b parameter (length normalization)")
    BM25_INDEX_DIR: Path = Field(
        Path("./data/bm25_indexes"),
        description="Directory for BM25 index storage"
    )
    HYBRID_FUSION_METHOD: str = Field("rrf", description="Fusion method: rrf or weighted")
    HYBRID_DEFAULT_ALPHA: float = Field(0.5,
                                        description="Default alpha for hybrid search (0.0=lexical, 1.0=semantic)")

    # =============================================================================
    # Parent Document Retrieval Settings
    # =============================================================================
    PARENT_RETRIEVAL_ENABLED: bool = Field(False,
                                           description="Enable parent document retrieval by default")
    PARENT_RETRIEVAL_MODE: str = Field("full_parent",
                                       description="Retrieval mode: full_parent or windowed")
    PARENT_WINDOW_SIZE: int = Field(1,
                                    description="Number of adjacent chunks to include in windowed mode")

    # =============================================================================
    # Model Config
    # =============================================================================
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    # =============================================================================
    # Validators
    # =============================================================================

    @field_validator("UPLOAD_DIR", "TRACKING_FILE", "BM25_INDEX_DIR")
    @classmethod
    def create_directories(cls, v: Path) -> Path:
        """Create directories if they don't exist"""
        if not v.exists():
            v.parent.mkdir(parents=True, exist_ok=True)
            if v.suffix == "":  # It's a directory
                v.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment value"""
        allowed = ["local", "development", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"ENVIRONMENT must be one of {allowed}")
        return v

    # =============================================================================
    # Utility Methods
    # =============================================================================

    def get_vault_path(self) -> Path:
        """Get vault path (for backward compatibility with existing code)"""
        vault_path_str = os.getenv("VAULT_PATH", "./data/vault")
        vault_path = Path(vault_path_str)
        
        # Crear directorio si no existe
        if not vault_path.exists():
            vault_path.mkdir(parents=True, exist_ok=True)
            
        return vault_path

    def is_production(self) -> bool:
        """Check if running in production"""
        return self.ENVIRONMENT == "production"

    def is_local(self) -> bool:
        """Check if running locally"""
        return self.ENVIRONMENT == "local"

    def get_allowed_extensions(self) -> List[str]:
        """
        Get allowed file extensions dynamically from document loaders

        Returns:
            List of allowed file extensions (e.g., ['.pdf', '.docx', ...])
        """
        # Import here to avoid circular dependency
        try:
            from src.document_loaders import DocumentLoaderFactory
            extensions = DocumentLoaderFactory.get_supported_extensions()
            return sorted(list(extensions))
        except ImportError:
            # Fallback to empty list if loaders not available
            return []

    def validate(self):
        """Validate configuration"""
        # Check required settings based on environment
        if self.is_production():
            if "change-in-production" in self.SECRET_KEY:
                raise ValueError("SECRET_KEY must be changed in production")

            if self.FILE_STORAGE_TYPE == "s3":
                if not self.AWS_ACCESS_KEY_ID or not self.AWS_SECRET_ACCESS_KEY:
                    raise ValueError("AWS credentials required for S3 storage")

        # Validate provider API keys if needed
        # (handled by optional fields)

        print("✅ Configuration validated")


# =============================================================================
# Create global settings instance
# =============================================================================

settings = Settings()


# =============================================================================
# Environment-specific configurations
# =============================================================================

def get_settings() -> Settings:
    """Get settings instance (for dependency injection)"""
    return settings


def get_database_url() -> str:
    """Get database URL"""
    return settings.DATABASE_URL


def get_qdrant_config() -> dict:
    """Get Qdrant configuration"""
    config = {
        "url": settings.QDRANT_URL,
        "timeout": settings.QDRANT_TIMEOUT,
    }
    if settings.QDRANT_API_KEY:
        config["api_key"] = settings.QDRANT_API_KEY
    return config

