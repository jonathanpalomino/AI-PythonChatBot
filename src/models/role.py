# =============================================================================
# src/models/role.py
# Modelos ORM: Permission, Role, UserRole  +  association table role_permissions
# =============================================================================
"""
Separado de auth_models.py para evitar imports circulares.

user.py importa desde aquí:
    from src.models.role import Role, UserRole

auth_models.py re-exporta ResourceType / ActionType para back-compat.
"""

import enum
from datetime import datetime
from typing import Optional, List, TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import (
    String, Text, Boolean, DateTime, ForeignKey,
    Table, Column, Index, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from src.models.models import Base
from src.utils.date_utils import get_current_utc

if TYPE_CHECKING:
    from src.models.user import User


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ResourceType(str, enum.Enum):
    CONVERSATION    = "conversation"
    MESSAGE         = "message"
    FILE            = "file"
    PROJECT         = "project"
    TOOL            = "tool"
    COLLECTION      = "collection"
    PROMPT_TEMPLATE = "prompt_template"
    USER            = "user"
    ROLE            = "role"
    ANY             = "*"


class ActionType(str, enum.Enum):
    CREATE  = "create"
    READ    = "read"
    UPDATE  = "update"
    DELETE  = "delete"
    EXECUTE = "execute"
    ANY     = "*"


# ---------------------------------------------------------------------------
# Association table: role_permissions  (M:N sin atributos extra → Table pura)
# ---------------------------------------------------------------------------

role_permissions = Table(
    "role_permissions",
    Base.metadata,
    Column(
        "role_id",
        UUID(as_uuid=True),
        ForeignKey("roles.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "permission_id",
        UUID(as_uuid=True),
        ForeignKey("permissions.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


# ---------------------------------------------------------------------------
# Permission
# ---------------------------------------------------------------------------

class Permission(Base):
    """Permiso granular: par resource + action."""
    __tablename__ = "permissions"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Almacenado como varchar para evitar conflicto con Enum nativo de
    # versiones anteriores de auth_models.py que ya estén en la BD.
    resource: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    action:   Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    roles: Mapped[List["Role"]] = relationship(
        secondary=role_permissions,
        back_populates="permissions",
    )

    __table_args__ = (
        UniqueConstraint("resource", "action", name="uq_permission_resource_action"),
        Index("idx_permissions_resource_action", "resource", "action"),
    )

    def __repr__(self) -> str:
        return f"<Permission {self.resource}:{self.action}>"


# ---------------------------------------------------------------------------
# Role
# ---------------------------------------------------------------------------

class Role(Base):
    """Rol que agrupa permisos y se asigna a usuarios."""
    __tablename__ = "roles"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    name:        Mapped[str]           = mapped_column(String(100), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_system:   Mapped[bool]          = mapped_column(Boolean, default=False, index=True)
    is_active:   Mapped[bool]          = mapped_column(Boolean, default=True,  index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=get_current_utc,
        nullable=False,
    )

    permissions: Mapped[List["Permission"]] = relationship(
        secondary=role_permissions,
        back_populates="roles",
    )
    user_roles: Mapped[List["UserRole"]] = relationship(
        "UserRole",
        back_populates="role",
        cascade="all, delete-orphan",
        lazy="select",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"<Role {self.name}>"


# ---------------------------------------------------------------------------
# UserRole  (clase ORM — soporta cascade delete-orphan y auditoría)
# ---------------------------------------------------------------------------

class UserRole(Base):
    """
    Tabla pivot User ↔ Role como clase ORM con timestamps de auditoría.
    Permite cascade='all, delete-orphan' desde User.user_roles.
    """
    __tablename__ = "user_roles"

    user_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    role_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("roles.id", ondelete="CASCADE"),
        primary_key=True,
    )
    assigned_at: Mapped[datetime] = mapped_column(
        DateTime, default=get_current_utc, nullable=False
    )

    user: Mapped["User"] = relationship("User", back_populates="user_roles")
    role: Mapped["Role"] = relationship("Role", back_populates="user_roles")

    def __repr__(self) -> str:
        return f"<UserRole user={self.user_id} role={self.role_id}>"
