# =============================================================================
# src/models/refresh_token.py
# Modelo ORM: RefreshToken
# =============================================================================
"""
Almacena refresh tokens persistidos por usuario.

Estrategia de invalidación:
  · Individual  : is_revoked=True  (logout de un dispositivo)
  · Masiva      : User.token_version += 1  (logout everywhere / cambio de password)
    → token.token_version < user.token_version  ⟹  token inválido

El campo token_hash almacena SHA-256 del valor raw.
Nunca se persiste el token en texto plano.
"""

from datetime import datetime
from typing import Optional, TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import String, DateTime, Integer, Boolean, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from src.models.models import Base
from src.utils.date_utils import get_current_utc

if TYPE_CHECKING:
    from src.models.user import User


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    user_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # SHA-256 del token raw (hex, 64 chars)
    token_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )
    # Debe coincidir con User.token_version al momento de validar
    token_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    is_revoked: Mapped[bool]     = mapped_column(Boolean, nullable=False, default=False, index=True)

    # Auditoría opcional
    user_agent: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45),  nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=get_current_utc, nullable=False
    )

    user: Mapped["User"] = relationship("User", back_populates="refresh_tokens")

    __table_args__ = (
        Index("idx_refresh_tokens_user_valid", "user_id", "is_revoked", "expires_at"),
    )

    def __repr__(self) -> str:
        return f"<RefreshToken user={self.user_id} revoked={self.is_revoked}>"
