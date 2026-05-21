# =============================================================================
# src/services/auth/token_service.py
# Servicio de generación y verificación de JWT y refresh tokens
# =============================================================================
"""
Gestión de tokens de autenticación:

  Access Token  — JWT firmado HS256, vida corta (15 min por defecto).
                  Claims: sub (user_id), ver (token_version), roles, iat, exp.

  Refresh Token — String aleatorio de 32 bytes (256 bits de entropía).
                  Se almacena en DB como SHA-256(token_raw).
                  El caller recibe (token_raw, token_hash):
                    · token_raw  → se entrega al cliente.
                    · token_hash → se persiste en base de datos.

Separación de claves:
  · SECRET_KEY         → firma de access tokens.
  · REFRESH_SECRET_KEY → firma interna de refresh tokens si se necesita
                         verificación sin DB (actualmente no usado, reservado).

Errores propios:
  · TokenExpiredError   — el token expiró (exp < now).
  · TokenInvalidError   — firma inválida, claims faltantes o formato incorrecto.
"""

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID
from jose import JWTError, jwt, ExpiredSignatureError
from passlib.exc import InvalidTokenError

from src.services.auth.crypto_service import CryptoService

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Excepciones propias
# ---------------------------------------------------------------------------

class TokenError(Exception):
    """Base para errores de token."""


class TokenExpiredError(TokenError):
    """El access token ha expirado."""


class TokenInvalidError(TokenError):
    """El token es inválido (firma, formato o claims)."""


# ---------------------------------------------------------------------------
# TokenService
# ---------------------------------------------------------------------------

class TokenService:
    """
    Servicio de creación y verificación de tokens de autenticación.

    Uso:
        svc = TokenService.from_settings()
        access  = svc.create_access_token(user_id=uid, token_version=2, roles=["developer"])
        raw, h  = svc.create_refresh_token()
        payload = svc.verify_access_token(access)
    """

    __slots__ = (
        "_secret_key",
        "_algorithm",
        "_access_ttl",
        "_refresh_ttl",
    )

    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        access_ttl_minutes: int = 30,
        refresh_ttl_days: int = 7,
    ) -> None:
        self._secret_key     = secret_key
        self._algorithm  = algorithm
        self._access_ttl = timedelta(minutes=access_ttl_minutes)
        self._refresh_ttl = timedelta(days=refresh_ttl_days)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls) -> "TokenService":
        """
        Crea la instancia usando los valores de settings.

        Raises:
            ValueError: Si SECRET_KEY no está configurado.
        """
        from src.config.settings import settings  # import diferido

        if not settings.SECRET_KEY:
            raise ValueError(
                "settings.SECRET_KEY no está configurado. "
                "Genera una clave con: python -c 'import secrets; print(secrets.token_hex(32))'"
            )
        return cls(
            secret_key=settings.SECRET_KEY,
            algorithm=getattr(settings, "JWT_ALGORITHM", "HS256"),
            access_ttl_minutes=getattr(settings, "ACCESS_TOKEN_EXPIRE_MINUTES", 30),
            refresh_ttl_days=getattr(settings, "REFRESH_TOKEN_EXPIRE_DAYS", 7),
        )

    # ------------------------------------------------------------------
    # Access Token
    # ------------------------------------------------------------------

    def create_access_token(
        self,
        user_id: UUID,
        email: str,
        roles: list[str],
        token_version: int,
        extra: Optional[dict] = None,
    ) -> str:
        """
        Genera un JWT de acceso firmado.

        Claims incluidos:
          sub   → str(user_id)
          ver   → token_version (para invalidación sin blacklist)
          roles → lista de nombres de roles
          iat   → issued-at (UTC)
          exp   → expiration (UTC, now + expire_minutes)

        Args:
            user_id: UUID del usuario autenticado.
            token_version: Versión actual del token del usuario (campo en DB).
            roles: Lista de nombres de roles asignados al usuario.

        Returns:
            JWT firmado como string.
        """
        now = datetime.now(tz=timezone.utc)

        payload = {
            "sub": str(user_id),
            "email": email,
            "roles": roles,
            "ver": token_version,
            "iat": now,
            "exp": now + self._access_ttl,
            "type": "access",
        }
        if extra:
            payload.update(extra)

        return jwt.encode(payload, self._secret_key, algorithm=self._algorithm)

    def decode_access_token(self, token: str) -> dict:
        """
        Decodifica y valida firma + expiración.
        Lanza JWTError en cualquier fallo.
        """
        return jwt.decode(token, self._secret_key, algorithms=[self._algorithm])

    def verify_access_token(self, token: str) -> dict:
        """
        Verifica y decodifica un JWT de acceso.

        Valida:
          · Firma con SECRET_KEY.
          · Expiración (exp).
          · Presencia de claims obligatorios: sub, ver, roles.

        Args:
            token: JWT en formato string.

        Returns:
            Dict con el payload decodificado.

        Raises:
            TokenExpiredError: Si exp < now.
            TokenInvalidError: Si la firma es inválida o faltan claims.
        """
        try:
            payload = jwt.decode(
                token,
                self._secret_key,
                algorithms=[self._algorithm],
                options={"require": ["sub", "ver", "roles", "exp", "iat"]},
            )
        except ExpiredSignatureError as exc:
            raise TokenExpiredError("El access token ha expirado.") from exc
        except InvalidTokenError as exc:
            raise TokenInvalidError(f"Access token inválido: {exc}") from exc

        # Validación adicional de tipos de claims
        if not isinstance(payload.get("sub"), str):
            raise TokenInvalidError("Claim 'sub' inválido: debe ser string.")
        if not isinstance(payload.get("ver"), int):
            raise TokenInvalidError("Claim 'ver' inválido: debe ser entero.")
        if not isinstance(payload.get("roles"), list):
            raise TokenInvalidError("Claim 'roles' inválido: debe ser lista.")

        return payload

    def get_user_id_from_token(self, token: str) -> UUID:
        """
        Extrae el user_id de un access token válido.

        Shortcut conveniente para dependencias que solo necesitan el ID.

        Args:
            token: JWT en formato string.

        Returns:
            UUID del usuario.

        Raises:
            TokenExpiredError: Si el token ha expirado.
            TokenInvalidError: Si el token es inválido o sub no es UUID.
        """
        payload = self.verify_access_token(token)
        try:
            return UUID(payload["sub"])
        except (ValueError, KeyError) as exc:
            raise TokenInvalidError(
                f"Claim 'sub' no es un UUID válido: {payload.get('sub')!r}"
            ) from exc

    # ------------------------------------------------------------------
    # Refresh Token
    # ------------------------------------------------------------------

    def create_refresh_token(self) -> tuple[str, str, datetime]:
        """
        Genera un trío (token_raw, token_hash, expires_at) para un refresh token.

        · token_raw  — string de 64 bytes aleatorios.
                       Se entrega al cliente.
        · token_hash — Hash SHA-256 del token_raw.
                       Es lo único que se persiste en base de datos.
        · expires_at — Fecha de expiración (UTC).

        Returns:
            Tupla (token_raw, token_hash, expires_at).
        """
        from src.utils.date_utils import get_current_utc
        raw = CryptoService.generate_opaque_token(64)
        h = CryptoService.hash_token(raw)
        exp = get_current_utc() + self._refresh_ttl
        return raw, h, exp

    @staticmethod
    def create_state_token() -> str:
        return CryptoService.generate_opaque_token(32)

    @staticmethod
    def hash_refresh_token(token_raw: str) -> str:
        """
        Calcula el hash SHA-256 de un refresh token recibido del cliente.

        Usado en el endpoint /auth/refresh para buscar el token en DB.

        Args:
            token_raw: Token recibido del cliente.

        Returns:
            SHA-256 hex del token.
        """
        return hashlib.sha256(token_raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Instancia singleton lazy
# ---------------------------------------------------------------------------

_instance: TokenService | None = None


def get_token_service() -> TokenService:
    """
    Retorna la instancia singleton de TokenService.

    Se inicializa en el primer llamado. Thread-safe (GIL en asignación simple).

    Raises:
        ValueError: Si SECRET_KEY no está configurado en settings.
    """
    global _instance
    if _instance is None:
        _instance = TokenService.from_settings()
    return _instance
