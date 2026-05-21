# =============================================================================
# src/services/auth/crypto_service.py
# Servicios criptográficos: hash de contraseñas, generación de tokens opacos
# Cero dependencias del proyecto — solo stdlib + passlib
# =============================================================================
import hashlib
import secrets

from passlib.context import CryptContext

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class CryptoService:
    """Utilidades criptográficas sin dependencias del proyecto."""

    # ── Passwords ──────────────────────────────────────────────────────────
    @staticmethod
    def hash_password(plain: str) -> str:
        return _pwd_context.hash(plain)

    @staticmethod
    def verify_password(plain: str, hashed: str) -> bool:
        return _pwd_context.verify(plain, hashed)

    # ── Tokens opacos ──────────────────────────────────────────────────────
    @staticmethod
    def generate_opaque_token(nbytes: int = 64) -> str:
        """Genera token URL-safe de `nbytes` bytes de entropía."""
        return secrets.token_urlsafe(nbytes)

    @staticmethod
    def hash_token(raw_token: str) -> str:
        """SHA-256 hex del token raw. Se usa como índice en BD."""
        return hashlib.sha256(raw_token.encode()).hexdigest()

    @staticmethod
    def secure_compare(a: str, b: str) -> bool:
        """Comparación en tiempo constante para evitar timing attacks."""
        return secrets.compare_digest(a.encode(), b.encode())
