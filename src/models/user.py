# =============================================================================
# src/models/user.py
# Modelo User — entidad central de autenticación
# =============================================================================
"""
Usuarios del sistema con soporte para:
  · Autenticación local (email + bcrypt hash).
  · OAuth social (Google, GitHub, Microsoft) — sin password.
  · Control de acceso basado en roles (RBAC) via user_roles → Role.
  · Invalidación masiva de tokens via token_version (incrementar = logout everywhere).
  · Soft-delete via is_active=False (nunca borrar filas de User).

Relaciones:
  User ──< UserRole >── Role          (M:N via tabla pivot)
  User ──< RefreshToken               (1:N, cascade delete-orphan)
  User ──< ExternalOAuthToken         (1:N, cascade delete-orphan)
"""

import enum
from datetime import datetime
from typing import Optional, List, TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import (
    String, Text, Boolean, DateTime, Integer,
    Enum, Index, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from src.models.models import Base
from src.utils.date_utils import get_current_utc

if TYPE_CHECKING:
    from src.models.refresh_token import RefreshToken
    from src.models.external_oauth_token import ExternalOAuthToken
    from src.models.role import Role, UserRole


# ---------------------------------------------------------------------------
# Enum: estado de cuenta
# ---------------------------------------------------------------------------

class UserStatus(str, enum.Enum):
    """Estado del ciclo de vida de la cuenta."""
    ACTIVE = "active"           # cuenta operativa
    INACTIVE = "inactive"       # deshabilitada manualmente
    SUSPENDED = "suspended"     # bloqueada por política (abuso, deuda, etc.)
    PENDING_VERIFY = "pending_verify"  # email aún no verificado


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

class User(Base):
    """
    Entidad de usuario.

    Notas de diseño:
      · password_hash es Optional: los usuarios OAuth puro no tienen contraseña local.
      · token_version se incrementa para invalidar todos los refresh tokens activos
        sin necesitar una blacklist de JWTs.
      · login_count y last_login_at son auditables en capa de servicio, no triggers.
      · avatar_url puede ser una URL externa (OAuth) o una ruta interna (upload).
    """
    __tablename__ = "users"

    # ------------------------------------------------------------------
    # PK
    # ------------------------------------------------------------------
    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )

    # ------------------------------------------------------------------
    # Identidad
    # ------------------------------------------------------------------
    email: Mapped[str] = mapped_column(
        String(320),   # RFC 5321 max
        nullable=False,
        unique=True,
        index=True,
    )
    username: Mapped[Optional[str]] = mapped_column(
        String(64),
        unique=True,
        index=True,
        nullable=True,
        comment="Alias público opcional. NULL hasta que el usuario lo configure.",
    )
    full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    avatar_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # ------------------------------------------------------------------
    # Autenticación local
    # ------------------------------------------------------------------
    password_hash: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="bcrypt hash. NULL para cuentas puramente OAuth.",
    )

    # ------------------------------------------------------------------
    # Estado y control de acceso
    # ------------------------------------------------------------------
    status: Mapped[UserStatus] = mapped_column(
        Enum(UserStatus, create_type=False, native_enum=True, name="userstatus"),
        nullable=False,
        default=UserStatus.ACTIVE,
        index=True,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
        comment="Shortcut de is_active para queries rápidas. Refleja status==ACTIVE.",
    )
    is_superuser: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Bypass total de permisos. Solo para administradores del sistema.",
    )

    # ------------------------------------------------------------------
    # Seguridad de tokens
    # ------------------------------------------------------------------
    token_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        comment=(
            "Versión del token. Incrementar invalida TODOS los refresh tokens "
            "activos del usuario (logout everywhere / cambio de contraseña)."
        ),
    )

    # ------------------------------------------------------------------
    # Verificación de email
    # ------------------------------------------------------------------
    email_verified: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    email_verified_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # ------------------------------------------------------------------
    # Auditoría de uso
    # ------------------------------------------------------------------
    last_login_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    login_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # ------------------------------------------------------------------
    # Timestamps
    # ------------------------------------------------------------------
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=get_current_utc,
        nullable=False,
    )

    # ------------------------------------------------------------------
    # Relaciones
    # ------------------------------------------------------------------
    refresh_tokens: Mapped[List["RefreshToken"]] = relationship(
        "RefreshToken",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="select",
        passive_deletes=True,
    )
    external_oauth_tokens: Mapped[List["ExternalOAuthToken"]] = relationship(
        "ExternalOAuthToken",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="select",
        passive_deletes=True,
    )
    user_roles: Mapped[List["UserRole"]] = relationship(
        "UserRole",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="select",
        passive_deletes=True,
    )

    # ------------------------------------------------------------------
    # Índices y constraints
    # ------------------------------------------------------------------
    __table_args__ = (
        Index("idx_users_email_active", "email", "is_active"),
        Index("idx_users_status", "status"),
        # email ya tiene unique=True en la columna — no duplicar UniqueConstraint
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def has_local_auth(self) -> bool:
        """True si el usuario puede autenticarse con email+contraseña."""
        return self.password_hash is not None

    @property
    def role_names(self) -> list[str]:
        """Lista de nombres de roles asignados (requiere eager load de user_roles)."""
        return [ur.role.name for ur in self.user_roles if ur.role is not None]

    def __repr__(self) -> str:
        return f"<User id={self.id!s:.8} email={self.email!r} status={self.status.value!r}>"
