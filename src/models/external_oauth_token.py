# =============================================================================
# src/models/external_oauth_token.py
# Modelo ExternalOAuthToken — tokens OAuth de proveedores externos
# =============================================================================
"""
Almacena los tokens de acceso y refresco emitidos por proveedores OAuth externos
(Google, GitHub, Microsoft, etc.) para un usuario del sistema.

Diferencia con RefreshToken:
  · RefreshToken    → tokens INTERNOS del propio sistema de auth.
  · ExternalOAuthToken → tokens de TERCEROS para llamar APIs externas
                         (Google Drive, GitHub API, MS Graph, etc.).

Seguridad de almacenamiento:
  · access_token y refresh_token se almacenan CIFRADOS con AES-256-GCM
    usando CryptoService. Las columnas son VARCHAR con el ciphertext base64.
  · El descifrado ocurre en capa de servicio (ExternalOAuthService), nunca
    en el modelo ni en el repositorio.

Constraint de unicidad:
  · (user_id, provider) → un usuario tiene como máximo un token activo
    por proveedor. Si se re-autentica, el servicio hace upsert.

Campos opcionales de auditoría:
  · scopes: permisos otorgados por el usuario en el proveedor.
  · provider_user_id: ID del usuario en el sistema del proveedor (útil para
    vincular cuentas y detectar conflictos si dos usuarios locales intentan
    vincular la misma cuenta de Google).
"""

import enum
from datetime import datetime
from typing import Optional, TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import (
    String, Text, Boolean, DateTime,
    Enum, ForeignKey, Index, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from src.models.models import Base
from src.utils.date_utils import get_current_utc

if TYPE_CHECKING:
    from src.models.user import User


# ---------------------------------------------------------------------------
# Enum: proveedores OAuth soportados
# ---------------------------------------------------------------------------

class OAuthProvider(str, enum.Enum):
    """Proveedores OAuth externos soportados."""
    GOOGLE = "google"
    GITHUB = "github"
    MICROSOFT = "microsoft"
    GITLAB = "gitlab"
    BITBUCKET = "bitbucket"

    @classmethod
    def _missing_(cls, value: object) -> Optional["OAuthProvider"]:
        """Case-insensitive lookup — consistente con el resto de enums del proyecto."""
        if isinstance(value, str):
            for member in cls:
                if member.value.lower() == value.lower():
                    return member
        return None


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

class ExternalOAuthToken(Base):
    """
    Token OAuth de un proveedor externo vinculado a un usuario del sistema.

    Uso típico en servicio:
        crypto = get_crypto_service()
        # Al guardar:
        token.access_token_enc  = crypto.encrypt(raw_access_token)
        token.refresh_token_enc = crypto.encrypt(raw_refresh_token) if raw else None
        # Al leer:
        raw_access  = crypto.decrypt(token.access_token_enc)
        raw_refresh = crypto.decrypt(token.refresh_token_enc) if token.refresh_token_enc else None
    """
    __tablename__ = "external_oauth_tokens"

    # ------------------------------------------------------------------
    # PK
    # ------------------------------------------------------------------
    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid4
    )

    # ------------------------------------------------------------------
    # FK al usuario del sistema
    # ------------------------------------------------------------------
    user_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # ------------------------------------------------------------------
    # Proveedor
    # ------------------------------------------------------------------
    provider: Mapped[OAuthProvider] = mapped_column(
        Enum(
            OAuthProvider,
            create_type=False,
            native_enum=True,
            name="oauthprovider",
        ),
        nullable=False,
        index=True,
    )

    # ------------------------------------------------------------------
    # ID del usuario en el sistema del proveedor (para deduplicación)
    # ------------------------------------------------------------------
    provider_user_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="ID del usuario en el sistema del proveedor (ej: Google sub, GitHub id).",
    )

    # ------------------------------------------------------------------
    # Tokens cifrados — NUNCA almacenar en texto plano
    # ------------------------------------------------------------------
    access_token_enc: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="AES-256-GCM ciphertext del access token. Descifrar con CryptoService.",
    )
    refresh_token_enc: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment=(
            "AES-256-GCM ciphertext del refresh token. "
            "NULL si el proveedor no emite refresh token (ej: GitHub sin offline_access)."
        ),
    )

    # ------------------------------------------------------------------
    # Metadatos del token
    # ------------------------------------------------------------------
    token_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="Bearer",
        comment="Tipo de token según el proveedor (casi siempre 'Bearer').",
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="Expiración del access_token. NULL si el proveedor no informa TTL.",
    )
    scopes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Scopes otorgados, separados por espacio (RFC 6749).",
    )

    # ------------------------------------------------------------------
    # Estado
    # ------------------------------------------------------------------
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
        comment="False si el token fue revocado o el usuario desvinculó la cuenta.",
    )

    # ------------------------------------------------------------------
    # Metadata adicional del proveedor (JSONB para extensibilidad)
    # Útil para almacenar campos del proveedor sin alterar el esquema:
    # email del proveedor, avatar_url, nombre, etc.
    # ------------------------------------------------------------------
    provider_metadata: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        default=None,
        comment="Datos adicionales del proveedor (profile info, etc.). Nunca tokens aquí.",
    )

    # ------------------------------------------------------------------
    # Timestamps
    # ------------------------------------------------------------------
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
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
    user: Mapped["User"] = relationship(
        "User",
        back_populates="external_oauth_tokens",
        lazy="select",
    )

    # ------------------------------------------------------------------
    # Constraints e índices
    # ------------------------------------------------------------------
    __table_args__ = (
        # Un usuario tiene exactamente un token activo por proveedor.
        # El servicio hace upsert cuando el usuario re-autentica.
        UniqueConstraint(
            "user_id",
            "provider",
            name="uq_external_oauth_tokens_user_provider",
        ),
        # Búsqueda rápida por proveedor + provider_user_id (deduplicación de cuentas)
        Index(
            "idx_ext_oauth_provider_user",
            "provider",
            "provider_user_id",
        ),
        # Tokens próximos a expirar (job de renovación proactiva)
        Index(
            "idx_ext_oauth_expires_active",
            "expires_at",
            "is_active",
        ),
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def is_expired(self) -> bool:
        """
        True si el access_token ha expirado.

        Si expires_at es NULL (proveedor no informó TTL), retorna False
        y el caller debe intentar usar el token y manejar el 401.
        """
        if self.expires_at is None:
            return False
        from datetime import timezone
        return datetime.now(tz=timezone.utc) >= self.expires_at

    @property
    def can_refresh(self) -> bool:
        """True si existe un refresh_token cifrado disponible."""
        return self.refresh_token_enc is not None

    def __repr__(self) -> str:
        return (
            f"<ExternalOAuthToken id={self.id!s:.8} "
            f"provider={self.provider.value!r} "
            f"user_id={self.user_id!s:.8} "
            f"active={self.is_active}>"
        )
