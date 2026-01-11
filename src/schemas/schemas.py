# =============================================================================
# src/schemas/schemas.py
# Pydantic Schemas for Request/Response validation
# =============================================================================
"""
Schemas de validación y serialización usando Pydantic v2
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict

from src.models.models import (
    MessageRole, ProcessingStatus, VisibilityType,
    HallucinationMode, ToolMode, ToolType
)


# =============================================================================
# Base Schemas
# =============================================================================

class BaseSchema(BaseModel):
    """Base schema with common config"""
    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Prompt Template Schemas
# =============================================================================

class PromptTemplateVariable(BaseModel):
    """Variable definition for prompt templates"""
    name: str
    type: str = Field(..., description="text, select, number, boolean")
    options: Optional[List[str]] = None
    default: Optional[Any] = None
    required: bool = False
    description: Optional[str] = None


class PromptTemplateSettings(BaseModel):
    """Settings for prompt template"""
    recommended_provider: Optional[str] = None
    recommended_model: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = None
    default_tools: Optional[List[str]] = []
    hallucination_mode: Optional[HallucinationMode] = None


class PromptTemplateCreate(BaseModel):
    """Create prompt template"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    category: str
    visibility: VisibilityType = VisibilityType.PUBLIC
    system_prompt: str = Field(..., min_length=1)
    user_prompt_template: Optional[str] = None
    variables: List[PromptTemplateVariable] = []
    settings: PromptTemplateSettings = Field(default_factory=PromptTemplateSettings)
    created_by: Optional[str] = None


class PromptTemplateUpdate(BaseModel):
    """Update prompt template"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    category: Optional[str] = None
    visibility: Optional[VisibilityType] = None
    system_prompt: Optional[str] = None
    user_prompt_template: Optional[str] = None
    variables: Optional[List[PromptTemplateVariable]] = None
    settings: Optional[PromptTemplateSettings] = None
    is_active: Optional[bool] = None


class PromptTemplateResponse(BaseSchema):
    """Prompt template response"""
    id: UUID
    name: str
    description: Optional[str]
    category: str
    visibility: VisibilityType
    system_prompt: str
    user_prompt_template: Optional[str]
    variables: List[Dict[str, Any]]
    settings: Dict[str, Any]
    version: int
    is_active: bool
    created_by: Optional[str]
    created_at: datetime
    updated_at: datetime


# =============================================================================
# Qdrant Collection Schemas
# =============================================================================

class QdrantCollectionCreate(BaseModel):
    """Create Qdrant collection registry"""
    name: str = Field(..., min_length=1, max_length=255)
    display_name: str
    description: Optional[str] = None
    category: Optional[str] = None
    visibility: VisibilityType = VisibilityType.PUBLIC
    # metadata: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict, alias="extra_metadata")


class QdrantCollectionUpdate(BaseModel):
    """Update Qdrant collection"""
    display_name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    is_active: Optional[bool] = None
    visibility: Optional[VisibilityType] = None
    vector_count: Optional[int] = None
    last_synced: Optional[datetime] = None
    # metadata: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = Field(None, alias="extra_metadata")


class QdrantCollectionResponse(BaseSchema):
    """Qdrant collection response"""
    id: UUID
    name: str
    display_name: str
    description: Optional[str]
    category: Optional[str]
    vector_count: int
    last_synced: Optional[datetime]
    is_active: bool
    visibility: VisibilityType
    # metadata: Dict[str, Any]
    metadata: Dict[str, Any] = Field(alias="extra_metadata")
    created_at: datetime
    updated_at: datetime


# =============================================================================
# Project Schemas
# =============================================================================

class ProjectCreate(BaseModel):
    """Create project"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None


class ProjectUpdate(BaseModel):
    """Update project"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    is_active: Optional[bool] = None


class ProjectResponse(BaseSchema):
    """Project response"""
    id: UUID
    name: str
    description: Optional[str]
    is_active: bool
    created_at: datetime
    updated_at: datetime




# =============================================================================
# Conversation Schemas
# =============================================================================

class MemoryConfig(BaseModel):
    """Memory configuration settings"""
    semantic_enabled: bool = False
    search_k: int = Field(5, ge=1, le=20)
    score_threshold: float = Field(0.3, ge=0.0, le=1.0)
    auto_index: bool = True
    temporal_window_hours: Optional[int] = None


class HallucinationControlSettings(BaseModel):
    """Hallucination control settings"""
    mode: HallucinationMode = HallucinationMode.BALANCED
    require_sources: bool = False
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)


class ConversationSettings(BaseModel):
    """Conversation settings"""
    provider: str = "local"
    model: str = "mistral"
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(2000, ge=1)
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0)
    stream_chat: bool = False

    # Conversation history
    max_history_messages: int = Field(5, ge=1, le=50,
                                      description="Maximum number of conversation history messages to include in LLM context")

    # Tool configuration
    tool_mode: ToolMode = ToolMode.MANUAL
    enabled_tools: List[str] = Field(default_factory=list)
    available_tools: Optional[List[str]] = None  # For agent mode
    allow_tool_chaining: bool = False

    # Memory configuration (replaces deep_thinking tool)
    memory_config: MemoryConfig = Field(default_factory=MemoryConfig)

    # RAG
    rag_enabled: bool = False

    # Hallucination control
    hallucination_control: HallucinationControlSettings = Field(
        default_factory=HallucinationControlSettings
    )

    # Embedding configuration
    embedding_model: Optional[str] = None
    
    # Ollama-specific configuration
    num_ctx: Optional[int] = Field(None, ge=128, description="Context window size for Ollama models")
    num_gpu: Optional[int] = Field(None, ge=0, description="Number of GPU layers for Ollama models")
    num_thread: Optional[int] = Field(None, ge=1, description="Number of threads for Ollama models")
    num_batch: Optional[int] = Field(None, ge=1, description="Batch size for Ollama models")


class ConversationCreate(BaseModel):
    """Create conversation"""
    title: str = Field(..., min_length=1, max_length=500)
    project_id: Optional[UUID] = None  # NEW
    prompt_template_id: Optional[UUID] = None
    settings: ConversationSettings = Field(default_factory=ConversationSettings)
    # metadata: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict, alias="extra_metadata")


class ConversationUpdate(BaseModel):
    """Update conversation"""
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    settings: Optional[ConversationSettings] = None
    # metadata: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = Field(None, alias="extra_metadata")


class ConversationResponse(BaseSchema):
    """Conversation response (full details)"""
    id: UUID
    title: str
    project_id: Optional[UUID] = None  # NEW
    prompt_template_id: Optional[UUID]
    settings: Dict[str, Any]
    # metadata: Dict[str, Any]
    metadata: Dict[str, Any] = Field(alias="extra_metadata")  # <-- alias aquí
    created_at: datetime
    updated_at: datetime

    # Optional nested data
    message_count: Optional[int] = None
    last_message_at: Optional[datetime] = None


class ConversationListItem(BaseSchema):
    """Lightweight conversation item for list view (headers only)"""
    id: UUID
    title: str
    message_count: int = 0
    last_message_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    # Optional: include minimal settings preview if needed
    provider: Optional[str] = None  # From settings
    model: Optional[str] = None  # From settings
    embedding_model: Optional[str] = None  # From settings


# =============================================================================
# Message Schemas
# =============================================================================

class MessageMetadata(BaseModel):
    """Message metadata"""
    provider: Optional[str] = None
    model: Optional[str] = None
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    latency_ms: Optional[int] = None
    confidence_score: Optional[float] = None
    has_sources: bool = False
    tools_used: List[str] = Field(default_factory=list)
    rag_sources: List[Dict[str, Any]] = Field(default_factory=list)
    code_analysis: Optional[Dict[str, Any]] = None


class MessageAttachment(BaseModel):
    """Message attachment info"""
    file_id: UUID
    file_name: str
    file_type: str
    size: int


class MessageCreate(BaseModel):
    """Create message"""
    conversation_id: UUID
    role: MessageRole
    content: str = Field(..., min_length=1)
    # metadata: MessageMetadata = Field(default_factory=MessageMetadata)
    metadata: MessageMetadata = Field(default_factory=MessageMetadata, alias="extra_metadata")
    attachments: List[MessageAttachment] = Field(default_factory=list)


class MessageResponse(BaseSchema):
    """Message response"""
    id: UUID
    conversation_id: UUID
    role: MessageRole
    content: str
    thinking_content: Optional[str] = None
    is_active: bool = True
    # metadata: Dict[str, Any]
    metadata: Dict[str, Any] = Field(alias="extra_metadata")
    attachments: List[Dict[str, Any]]
    created_at: datetime


# =============================================================================
# File Schemas
# =============================================================================

class FileMetadata(BaseModel):
    """File metadata"""
    original_name: Optional[str] = None
    hash: Optional[str] = None
    extracted_text: Optional[str] = None
    language: Optional[str] = None  # For code files
    embedding_ids: List[str] = Field(default_factory=list)
    page_count: Optional[int] = None  # For documents
    analysis_result: Optional[Dict[str, Any]] = None  # For code


class FileUpload(BaseModel):
    """File upload info (from form)"""
    conversation_id: Optional[UUID] = None
    project_id: Optional[UUID] = None  # NEW


class FileResponse(BaseSchema):
    """File response"""
    id: UUID
    conversation_id: Optional[UUID]
    project_id: Optional[UUID] = None
    file_name: str
    file_type: str
    file_size: int
    storage_path: str
    mime_type: Optional[str]
    processed: bool
    processing_status: ProcessingStatus
    # metadata: Dict[str, Any]
    metadata: Dict[str, Any] = Field(alias="extra_metadata")
    uploaded_at: datetime


# =============================================================================
# Tool Configuration Schemas
# =============================================================================

class RAGToolConfig(BaseModel):
    """RAG tool configuration"""
    collections: List[str] = Field(default_factory=list)
    k: int = Field(5, ge=1, le=20)
    score_threshold: float = Field(0.5, ge=0.0, le=1.0)
    filters: Dict[str, Any] = Field(default_factory=dict)
    always_execute: bool = False
    rerank: bool = False


class ToolConfigurationCreate(BaseModel):
    """Create tool configuration"""
    conversation_id: UUID
    tool_name: str
    config: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True


class ToolConfigurationUpdate(BaseModel):
    """Update tool configuration"""
    config: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class ToolConfigurationResponse(BaseSchema):
    """Tool configuration response"""
    id: UUID
    conversation_id: UUID
    tool_name: str
    config: Dict[str, Any]
    is_active: bool
    created_at: datetime
    updated_at: datetime


# =============================================================================
# Custom Tool Schemas
# =============================================================================

class CustomToolParameter(BaseModel):
    """Parameter definition for custom tools"""
    name: str
    type: str = Field(..., description="string, integer, boolean, array, object")
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None


class CustomToolCreate(BaseModel):
    """Create custom tool"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    tool_type: ToolType = Field(default=ToolType.http_request)
    configuration: Dict[str, Any] = Field(default_factory=dict)
    visibility: VisibilityType = VisibilityType.PUBLIC
    is_active: bool = True


class CustomToolUpdate(BaseModel):
    """Update custom tool"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    tool_type: Optional[ToolType] = None
    configuration: Optional[Dict[str, Any]] = None
    visibility: Optional[VisibilityType] = None
    is_active: Optional[bool] = None


class CustomToolResponse(BaseSchema):
    """Custom tool response"""
    id: UUID
    name: str
    description: Optional[str]
    tool_type: ToolType
    configuration: Dict[str, Any]
    visibility: VisibilityType
    is_active: bool
    created_at: datetime
    updated_at: datetime


# =============================================================================
# Chat Request/Response (Main API)
# =============================================================================

class ChatRequest(BaseModel):
    """Chat request"""
    message: str = Field(..., min_length=1)
    conversation_id: Optional[UUID] = None
    file_ids: List[UUID] = Field(default_factory=list)
    stream: bool = False

    # Optional overrides for this specific message
    temperature_override: Optional[float] = None
    max_tokens_override: Optional[int] = None


class ChatResponse(BaseModel):
    """Chat response"""
    conversation_id: UUID
    message: MessageResponse
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    tools_executed: List[str] = Field(default_factory=list)
    confidence_score: Optional[float] = None
    thinking_content: Optional[str] = None  # Reasoning/thinking from models like Qwen3, DeepSeek R1


# =============================================================================
# Utility Schemas
# =============================================================================

class PaginationParams(BaseModel):
    """Pagination parameters"""
    skip: int = Field(0, ge=0)
    limit: int = Field(50, ge=1, le=100)


class ListResponse(BaseModel):
    """Generic list response with pagination"""
    items: List[Any]
    total: int
    skip: int
    limit: int


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
