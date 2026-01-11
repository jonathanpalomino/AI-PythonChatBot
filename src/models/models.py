# =============================================================================
# src/models/models.py
# SQLAlchemy Models for RAG Chatbot
# =============================================================================
"""
Modelos de base de datos usando SQLAlchemy 2.0+
"""
import enum
from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import uuid4

from sqlalchemy import (
    String, Text, Integer, BigInteger, Boolean, DateTime, Enum,
    ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


# =============================================================================
# Base Class
# =============================================================================

class Base(DeclarativeBase):
    """Base class for all models"""
    pass


# Import new models to ensure they are registered with Base metadata
# (This is important for Alembic autogenerate)
# note: we do this inside the file but after Base definition or at end?
# Circular import risk if LLMModel imports Base from here.
# Yes, LLMModel imports Base from src.models.models.
# So we can't import LLMModel at top level easily if it depends on Base defined here.
# Actually, Python resolves classes at runtime.
# But for typical Alembic setup, we just need to make sure LLMModel is imported *somewhere* that env.py imports.
# Let's check if we can add it to __init__.py instead or just rely on separate import.
# But user asked to "update src/models/models.py to include new model".
# The cleanest way given `LLMModel` inherits `Base` from `models.py` is to leave `models.py` alone
# and ensure `src/models/__init__.py` imports both.
# Let's check `src/models/__init__.py`.


# =============================================================================
# Enums
# =============================================================================

class MessageRole(str, enum.Enum):
    """Role of message sender"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ProcessingStatus(str, enum.Enum):
    """File processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


class VisibilityType(str, enum.Enum):
    """Visibility/sharing level"""
    PUBLIC = "public"
    PRIVATE = "private"
    SHARED = "shared"


class HallucinationMode(str, enum.Enum):
    """Control mode for hallucinations"""
    STRICT = "strict"
    BALANCED = "balanced"
    CREATIVE = "creative"


class ToolMode(str, enum.Enum):
    """Tool execution mode"""
    AGENT = "agent"  # AI decides which tools to use
    MANUAL = "manual"  # User/system decides


class ToolType(str, enum.Enum):
    """Custom tool types"""
    http_request = "http_request"
    sql_query = "sql_query"
    rag_search = "rag_search"
    custom = "custom"
    
    @classmethod
    def _missing_(cls, value):
        """Allow case-insensitive matching"""
        if isinstance(value, str):
            for member in cls:
                if member.value.lower() == value.lower():
                    return member
        return None


# =============================================================================
# Models
# =============================================================================

class PromptTemplate(Base):
    """Prompt templates - predefined and custom"""
    __tablename__ = "prompt_templates"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    visibility: Mapped[VisibilityType] = mapped_column(
        Enum(VisibilityType,
             create_type=False,
             native_enum=True),
        default=VisibilityType.PUBLIC,
        index=True
    )

    system_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    user_prompt_template: Mapped[Optional[str]] = mapped_column(Text)

    # Dynamic variables for templates
    variables: Mapped[List[Dict]] = mapped_column(JSONB, default=list)

    # Settings: recommended_provider, model, temperature, default_tools, etc.
    settings: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)

    version: Mapped[int] = mapped_column(Integer, default=1)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_by: Mapped[Optional[str]] = mapped_column(String(255))

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now()
    )

    # Relationships
    conversations: Mapped[List["Conversation"]] = relationship(
        back_populates="prompt_template"
    )

    # Override table args to avoid metadata conflict
    __table_args__ = ()

    def __repr__(self):
        return f"<PromptTemplate(name={self.name}, category={self.category})>"


class QdrantCollection(Base):
    """Registry of Qdrant vector collections"""
    __tablename__ = "qdrant_collections"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    category: Mapped[Optional[str]] = mapped_column(String(100), index=True)

    vector_count: Mapped[int] = mapped_column(Integer, default=0)
    last_synced: Mapped[Optional[datetime]] = mapped_column(DateTime)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    visibility: Mapped[VisibilityType] = mapped_column(
        Enum(VisibilityType,
             create_type=False,
             native_enum=True),
        default=VisibilityType.PUBLIC
    )

    # Metadata: source_path, embedding_model, indexed_file_count, etc.
    extra_metadata: Mapped[Dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now()
    )

    def __repr__(self):
        return f"<QdrantCollection(name={self.name}, vectors={self.vector_count})>"


class Conversation(Base):
    """Chat conversations"""
    __tablename__ = "conversations"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    title: Mapped[str] = mapped_column(String(500), nullable=False)

    prompt_template_id: Mapped[Optional[UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("prompt_templates.id", ondelete="SET NULL"),
        index=True
    )

    # Settings: provider, model, temperature, tool_mode, enabled_tools, etc.
    settings: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)

    # Additional metadata: tags, folder, etc.
    extra_metadata: Mapped[Dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        index=True
    )

    # Relationships
    prompt_template: Mapped[Optional["PromptTemplate"]] = relationship(
        back_populates="conversations"
    )
    project_id: Mapped[Optional[UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="SET NULL"),
        index=True
    )
    project: Mapped[Optional["Project"]] = relationship(
        back_populates="conversations"
    )

    messages: Mapped[List["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan"
    )
    files: Mapped[List["File"]] = relationship(
        back_populates="conversation"
    )
    tool_configurations: Mapped[List["ToolConfiguration"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan"
    )
    memory: Mapped[List["ConversationMemory"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index('idx_conversations_settings', 'settings', postgresql_using='gin'),
    )

    def __repr__(self):
        return f"<Conversation(id={self.id}, title={self.title[:30]})>"


class Message(Base):
    """Individual messages in conversations"""
    __tablename__ = "messages"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    conversation_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False
    )

    role: Mapped[MessageRole] = mapped_column(
        Enum(MessageRole,
             create_type=False,
             native_enum=True,
             values_callable=lambda obj: [e.value for e in obj]),
        nullable=False)

    content: Mapped[str] = mapped_column(Text, nullable=False)
    thinking_content: Mapped[Optional[str]] = mapped_column(Text, default=None)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)

    # Metadata: provider, model, tokens_used, cost, confidence_score, tools_used, etc.
    extra_metadata: Mapped[Dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)

    # File attachments
    attachments: Mapped[List[Dict]] = mapped_column(JSONB, default=list)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        index=True
    )

    # Relationships
    conversation: Mapped["Conversation"] = relationship(back_populates="messages")

    # Indexes
    __table_args__ = (
        Index('idx_messages_conversation_created', 'conversation_id', 'created_at'),
        Index('idx_messages_metadata', 'metadata', postgresql_using='gin'),
    )

    def __repr__(self):
        return f"<Message(role={self.role}, content={self.content[:30]})>"


class File(Base):
    """Uploaded files attached to conversations"""
    __tablename__ = "files"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    conversation_id: Mapped[Optional[UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="SET NULL"),
        index=True
    )

    file_name: Mapped[str] = mapped_column(String(500), nullable=False)
    file_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    mime_type: Mapped[Optional[str]] = mapped_column(String(255))

    processed: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    processing_status: Mapped[ProcessingStatus] = mapped_column(
        Enum(ProcessingStatus,
             create_type=False,
             native_enum=True),
        default=ProcessingStatus.PENDING
    )

    # Metadata: hash, extracted_text, language, embedding_ids, analysis_result, etc.
    extra_metadata: Mapped[Dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)

    uploaded_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    conversation: Mapped[Optional["Conversation"]] = relationship(back_populates="files")
    
    project_id: Mapped[Optional[UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        index=True
    )
    project: Mapped[Optional["Project"]] = relationship(
        back_populates="files"
    )

    def __repr__(self):
        return f"<File(name={self.file_name}, type={self.file_type})>"


class ToolConfiguration(Base):
    """Tool configurations per conversation"""
    __tablename__ = "tool_configurations"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    conversation_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False
    )

    tool_name: Mapped[str] = mapped_column(String(100), nullable=False)

    # Tool-specific configuration
    config: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now()
    )

    # Relationships
    conversation: Mapped["Conversation"] = relationship(
        back_populates="tool_configurations"
    )

    # Constraints
    __table_args__ = (
        Index('idx_tool_configs_conversation', 'conversation_id'),
        UniqueConstraint('conversation_id', 'tool_name',
                         name='tool_configurations_conversation_id_tool_name_key'),
    )

    def __repr__(self):
        return f"<ToolConfiguration(tool={self.tool_name}, active={self.is_active})>"


class ConversationMemory(Base):
    """Memory/summary storage for conversations"""
    __tablename__ = "conversation_memory"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    conversation_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    summary: Mapped[str] = mapped_column(Text, nullable=False)
    key_points: Mapped[List[str]] = mapped_column(JSONB, default=list)
    token_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    conversation: Mapped["Conversation"] = relationship(back_populates="memory")

    def __repr__(self):
        return f"<ConversationMemory(conv_id={self.conversation_id}, tokens={self.token_count})>"


class Project(Base):
    """
    Project entity to group files and conversations.
    Files uploaded to a project are indexed in a shared collection.
    """
    __tablename__ = "projects"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
     
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
     
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now()
    )

    # Relationships
    conversations: Mapped[List["Conversation"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan"
    )
    files: Mapped[List["File"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Project(name={self.name})>"


class CustomTool(Base):
    """Custom tools created by users"""
    __tablename__ = "custom_tools"
    
    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text)
    
    # Tool type (http, sql, etc.)
    tool_type: Mapped[ToolType] = mapped_column(
        Enum(ToolType,
             create_type=False,
             native_enum=True,
             name='tooltype'),
        default=ToolType.http_request,
        index=True
    )
    
    # Is this a template tool type? (True for base types like http, sql, custom)
    is_template: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    
    # Flexible configuration stored as JSON
    # Structure depends on tool_type:
    # - http: {url, method, headers, parameters}
    # - sql: {database_type, host, port, database, username, password, query_template}
    configuration: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    
    # Configuration schema for the tool type (used by API to describe available fields)
    # This stores the metadata like required, default, enum, etc.
    # For template tools, this defines the schema. For custom tools, this can override the template.
    config_schema: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, default=None)
    
    # Example configuration
    example: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, default=None)
    
    # Visibility control (public, private, shared)
    visibility: Mapped[VisibilityType] = mapped_column(
        Enum(VisibilityType,
             create_type=False,
             native_enum=True),
        default=VisibilityType.PUBLIC,
        index=True
    )
    
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now()
    )
    
    def __repr__(self):
        return f"<CustomTool(name={self.name}, type={self.tool_type}, visibility={self.visibility})>"
