# =============================================================================
# src/schemas/auth_schemas.py
# Pydantic v2 schemas for auth endpoints
# =============================================================================
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, field_validator


class UserRegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_-]+$")
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    full_name: Optional[str] = Field(None, max_length=255)

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str


class AssignRoleRequest(BaseModel):
    role_name: str


class CreateRoleRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    description: str = Field("", max_length=500)


class AddPermissionRequest(BaseModel):
    resource: str
    action: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class PermissionResponse(BaseModel):
    id: UUID
    name: str
    resource: str
    action: str
    description: Optional[str] = None
    model_config = {"from_attributes": True}


class RoleResponse(BaseModel):
    id: UUID
    name: str
    description: Optional[str] = None
    is_system: bool
    is_active: bool
    permissions: List[PermissionResponse] = []
    model_config = {"from_attributes": True}


class UserResponse(BaseModel):
    id: UUID
    username: str
    email: str
    full_name: Optional[str] = None
    status: str
    is_superuser: bool
    email_verified: bool
    roles: List[RoleResponse] = []
    created_at: datetime
    last_login: Optional[datetime] = None
    model_config = {"from_attributes": True}


class UserPublicResponse(BaseModel):
    id: UUID
    username: str
    full_name: Optional[str] = None
    roles: List[str] = []
    model_config = {"from_attributes": True}
