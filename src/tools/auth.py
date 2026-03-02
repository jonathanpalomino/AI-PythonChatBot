# =============================================================================
# src/tools/auth.py
# Authentication Strategies for HTTP Tool
# =============================================================================
"""
Sistema de autenticación extensible para HTTP requests
Implementa diferentes estrategias de autenticación siguiendo el patrón Strategy
"""

import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Any


class AuthType(str, Enum):
    """Tipos de autenticación soportados"""
    NONE = "none"
    BASIC = "basic"
    BEARER = "bearer"
    API_KEY = "api_key"
    DIGEST = "digest"
    OAUTH2 = "oauth2"
    CUSTOM = "custom"


@dataclass
class AuthCredentials:
    """Credenciales de autenticación"""
    auth_type: AuthType
    credentials: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para serialización"""
        return {
            "auth_type": self.auth_type.value,
            "credentials": self.credentials
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuthCredentials":
        """Crear desde diccionario"""
        return cls(
            auth_type=AuthType(data.get("auth_type", "none")),
            credentials=data.get("credentials", {})
        )


class AuthStrategy(ABC):
    """Clase base abstracta para estrategias de autenticación"""

    @abstractmethod
    def apply(self, headers: Dict[str, str], **kwargs) -> Dict[str, str]:
        """
        Aplicar autenticación a los headers de la petición

        Args:
            headers: Headers existentes de la petición
            **kwargs: Parámetros adicionales según el tipo de auth

        Returns:
            Headers actualizados con información de autenticación
        """
        pass

    @abstractmethod
    def get_auth_type(self) -> AuthType:
        """Retornar el tipo de autenticación"""
        pass

    def validate_credentials(self, credentials: Dict[str, Any]) -> bool:
        """Validar que las credenciales sean válidas para esta estrategia"""
        return True


class NoAuth(AuthStrategy):
    """Sin autenticación"""

    def apply(self, headers: Dict[str, str], **kwargs) -> Dict[str, str]:
        return headers

    def get_auth_type(self) -> AuthType:
        return AuthType.NONE


class BasicAuth(AuthStrategy):
    """HTTP Basic Authentication (RFC 7617)"""

    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password

    def apply(self, headers: Dict[str, str], **kwargs) -> Dict[str, str]:
        credentials = f"{self.username}:{self.password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        headers["Authorization"] = f"Basic {encoded}"
        return headers

    def get_auth_type(self) -> AuthType:
        return AuthType.BASIC

    def validate_credentials(self, credentials: Dict[str, Any]) -> bool:
        return "username" in credentials and "password" in credentials


class BearerAuth(AuthStrategy):
    """Bearer Token Authentication (RFC 6750)"""

    def __init__(self, token: str):
        self.token = token

    def apply(self, headers: Dict[str, str], **kwargs) -> Dict[str, str]:
        headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def get_auth_type(self) -> AuthType:
        return AuthType.BEARER

    def validate_credentials(self, credentials: Dict[str, Any]) -> bool:
        return "token" in credentials


class ApiKeyAuth(AuthStrategy):
    """API Key Authentication (header, query param, or custom)"""

    def __init__(
        self,
        api_key: str,
        key_name: str = "X-API-Key",
        location: str = "header"  # "header", "query", "custom"
    ):
        self.api_key = api_key
        self.key_name = key_name
        self.location = location

    def apply(self, headers: Dict[str, str], **kwargs) -> Dict[str, str]:
        if self.location == "header":
            headers[self.key_name] = self.api_key
        elif self.location == "custom":
            # Para casos custom, el key_name puede incluir un prefijo
            headers[self.key_name] = self.api_key
        # Si es "query", se maneja en el método execute del HTTPTool
        return headers

    def get_auth_type(self) -> AuthType:
        return AuthType.API_KEY

    def validate_credentials(self, credentials: Dict[str, Any]) -> bool:
        return "api_key" in credentials


class DigestAuth(AuthStrategy):
    """HTTP Digest Authentication (RFC 7616) - Simplified version"""

    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password

    def apply(self, headers: Dict[str, str], **kwargs) -> Dict[str, str]:
        # Nota: Digest Auth requiere un challenge-response del servidor
        # Esta es una implementación simplificada que asume que httpx lo maneja
        # En producción, httpx.DigestAuth puede ser usado directamente
        headers["X-Digest-Username"] = self.username
        return headers

    def get_auth_type(self) -> AuthType:
        return AuthType.DIGEST

    def validate_credentials(self, credentials: Dict[str, Any]) -> bool:
        return "username" in credentials and "password" in credentials


class OAuth2Auth(AuthStrategy):
    """OAuth 2.0 Authentication"""

    def __init__(
        self,
        access_token: str,
        token_type: str = "Bearer",
        refresh_token: Optional[str] = None
    ):
        self.access_token = access_token
        self.token_type = token_type
        self.refresh_token = refresh_token

    def apply(self, headers: Dict[str, str], **kwargs) -> Dict[str, str]:
        headers["Authorization"] = f"{self.token_type} {self.access_token}"
        return headers

    def get_auth_type(self) -> AuthType:
        return AuthType.OAUTH2

    def validate_credentials(self, credentials: Dict[str, Any]) -> bool:
        return "access_token" in credentials


class CustomAuth(AuthStrategy):
    """
    Autenticación personalizada con headers custom
    Útil para APIs con esquemas propietarios
    """

    def __init__(self, custom_headers: Dict[str, str]):
        self.custom_headers = custom_headers

    def apply(self, headers: Dict[str, str], **kwargs) -> Dict[str, str]:
        headers.update(self.custom_headers)
        return headers

    def get_auth_type(self) -> AuthType:
        return AuthType.CUSTOM

    def validate_credentials(self, credentials: Dict[str, Any]) -> bool:
        return "headers" in credentials


class AuthFactory:
    """Factory para crear estrategias de autenticación"""

    @staticmethod
    def create_auth(
        auth_type: AuthType,
        credentials: Dict[str, Any]
    ) -> AuthStrategy:
        """
        Crear una estrategia de autenticación

        Args:
            auth_type: Tipo de autenticación
            credentials: Credenciales necesarias según el tipo

        Returns:
            Instancia de AuthStrategy correspondiente

        Raises:
            ValueError: Si el tipo de auth no es soportado o faltan credenciales
        """
        if auth_type == AuthType.NONE:
            return NoAuth()

        elif auth_type == AuthType.BASIC:
            if "username" not in credentials or "password" not in credentials:
                raise ValueError("Basic auth requires 'username' and 'password'")
            return BasicAuth(
                username=credentials["username"],
                password=credentials["password"]
            )

        elif auth_type == AuthType.BEARER:
            if "token" not in credentials:
                raise ValueError("Bearer auth requires 'token'")
            return BearerAuth(token=credentials["token"])

        elif auth_type == AuthType.API_KEY:
            if "api_key" not in credentials:
                raise ValueError("API Key auth requires 'api_key'")
            return ApiKeyAuth(
                api_key=credentials["api_key"],
                key_name=credentials.get("key_name", "X-API-Key"),
                location=credentials.get("location", "header")
            )

        elif auth_type == AuthType.DIGEST:
            if "username" not in credentials or "password" not in credentials:
                raise ValueError("Digest auth requires 'username' and 'password'")
            return DigestAuth(
                username=credentials["username"],
                password=credentials["password"]
            )

        elif auth_type == AuthType.OAUTH2:
            if "access_token" not in credentials:
                raise ValueError("OAuth2 requires 'access_token'")
            return OAuth2Auth(
                access_token=credentials["access_token"],
                token_type=credentials.get("token_type", "Bearer"),
                refresh_token=credentials.get("refresh_token")
            )

        elif auth_type == AuthType.CUSTOM:
            if "headers" not in credentials:
                raise ValueError("Custom auth requires 'headers' dict")
            return CustomAuth(custom_headers=credentials["headers"])

        else:
            raise ValueError(f"Unsupported auth type: {auth_type}")

    @staticmethod
    def create_from_dict(auth_data: Dict[str, Any]) -> AuthStrategy:
        """Crear desde diccionario de configuración"""
        auth_type = AuthType(auth_data.get("auth_type", "none"))
        credentials = auth_data.get("credentials", {})
        return AuthFactory.create_auth(auth_type, credentials)
