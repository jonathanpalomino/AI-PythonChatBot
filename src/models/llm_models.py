# =============================================================================
# src/models/llm_models.py
# Reference Model for LLM Capabilities
# =============================================================================

from datetime import datetime
from uuid import uuid4

from sqlalchemy import String, Boolean, DateTime, Integer, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.models.models import Base


class LLMModel(Base):
    """
    Registry of available LLM models and their capabilities.
    Allows prioritizing specific models and overriding inferred types (e.g. fixing 'chat' to 'reasoning').
    """
    __tablename__ = "llm_models"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Composite unique key conceptual: (provider, model_name)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)  # e.g., 'local', 'openai'
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)  # e.g., 'deepseek-r1'

    # Capabilities
    model_type: Mapped[str] = mapped_column(String(50), default="chat",
                                            nullable=False)  # chat, reasoning, vision, embedding
    context_window: Mapped[int] = mapped_column(Integer, default=4096)
    supports_streaming: Mapped[bool] = mapped_column(Boolean, default=True)
    supports_function_calling: Mapped[bool] = mapped_column(Boolean, default=False)

    # Operational flags
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_custom: Mapped[bool] = mapped_column(Boolean, default=False)  # If manually added by user
    supports_thinking: Mapped[bool] = mapped_column(Boolean,
                                                    default=False)  # Can emit <think> tags (gemma3, qwen3, etc.)

    # Hardware requirements and capabilities
    cpu_supported: Mapped[bool] = mapped_column(Boolean, default=True)  # Can run on CPU
    gpu_required: Mapped[bool] = mapped_column(Boolean, default=False)  # Requires GPU
    parent_retrieval_supported: Mapped[bool] = mapped_column(Boolean, default=True)  # Supports parent document retrieval
    
    # Pricing info
    is_free: Mapped[bool] = mapped_column(Boolean, default=False)
    cost_per_1k_input: Mapped[float] = mapped_column(Float, default=0.0)
    cost_per_1k_output: Mapped[float] = mapped_column(Float, default=0.0)
    
    last_seen: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow,
                                                 onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<LLMModel(name={self.model_name}, type={self.model_type})>"
