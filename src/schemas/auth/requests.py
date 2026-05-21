# =============================================================================
# src/schemas/auth/requests.py
# Pydantic request schemas para el módulo de autenticación
# =============================================================================
from pydantic import BaseModel, EmailStr, Field, field_validator


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: str | None = Field(default=None, max_length=255)

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("La contraseña debe tener al menos una mayúscula.")
        if not any(c.isdigit() for c in v):
            raise ValueError("La contraseña debe tener al menos un dígito.")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str
