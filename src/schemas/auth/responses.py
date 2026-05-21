# =============================================================================
# src/schemas/auth/responses.py
# Pydantic response schemas para el módulo de autenticación
# =============================================================================
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # segundos hasta expiración del access token


class UserResponse(BaseModel):
    id: UUID
    email: str
    full_name: str | None
    is_active: bool
    is_superuser: bool
    email_verified: bool
    roles: list[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class MessageResponse(BaseModel):
    message: str
