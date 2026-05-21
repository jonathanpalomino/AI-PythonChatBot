# src/models/__init__.py
# Registro central de modelos — importar aquí garantiza que Alembic
# detecte todas las tablas al hacer autogenerate.

from src.models.models import (
    Base,
    # Mixins
    CreatedAtMixin,
    UpdatedAtMixin,
    TimestampMixin,
    # Enums existentes
    MessageRole,
    ProcessingStatus,
    VisibilityType,
    HallucinationMode,
    ToolMode,
    ToolType,
    # Modelos existentes
    PromptTemplate,
    QdrantCollection,
    Conversation,
    Message,
    File,
    ToolConfiguration,
    ConversationMemory,
    Project,
    CustomTool,
)

# Modelos de autenticación (Módulo A)
from src.models.user import User, UserStatus
from src.models.refresh_token import RefreshToken
from src.models.external_oauth_token import ExternalOAuthToken, OAuthProvider

# Modelos RBAC (Módulo B)
from src.models.role import Role, Permission, UserRole, role_permissions


__all__ = [
    # Base
    "Base",
    # Mixins
    "CreatedAtMixin",
    "UpdatedAtMixin",
    "TimestampMixin",
    # Enums existentes
    "MessageRole",
    "ProcessingStatus",
    "VisibilityType",
    "HallucinationMode",
    "ToolMode",
    "ToolType",
    # Modelos existentes
    "PromptTemplate",
    "QdrantCollection",
    "Conversation",
    "Message",
    "File",
    "ToolConfiguration",
    "ConversationMemory",
    "Project",
    "CustomTool",
    # Auth
    "User",
    "UserStatus",
    "RefreshToken",
    "ExternalOAuthToken",
    "OAuthProvider",
    # RBAC
    "Role",
    "Permission",
    "UserRole",
]
