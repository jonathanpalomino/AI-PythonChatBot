# =============================================================================
# src/core/crypto_service.py
# Servicio de cifrado simétrico AES-256-GCM
# =============================================================================
"""
Cifrado y descifrado de datos sensibles (tokens OAuth externos).

Algoritmo : AES-256-GCM
  - Autenticado: detecta tampering sin capa adicional (MAC integrado).
  - IV aleatorio de 12 bytes por operación: nunca se reutiliza el mismo IV.
  - Output: base64url-safe (sin padding "=") → seguro para columnas VARCHAR.

Formato del ciphertext almacenado (todo concatenado, luego base64):
  [IV 12 bytes][TAG 16 bytes][CIPHERTEXT variable]

La clave proviene de settings.ENCRYPTION_KEY (hex de 32 bytes = 64 caracteres).
"""

import base64
import os
from typing import Final

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Tamaños fijos del protocolo — NO modificar sin migración de datos
_IV_SIZE: Final[int] = 12   # bytes — recomendación NIST para GCM
_TAG_SIZE: Final[int] = 16  # bytes — tag de autenticación GCM (128 bits)


class CryptoError(Exception):
    """Error base de operaciones criptográficas."""


class EncryptionError(CryptoError):
    """Fallo durante el cifrado."""


class DecryptionError(CryptoError):
    """Fallo durante el descifrado (dato corrupto, clave incorrecta o tampering)."""


class CryptoService:
    """
    Servicio de cifrado AES-256-GCM para datos sensibles en base de datos.

    Uso:
        svc = CryptoService.from_settings()
        ciphertext = svc.encrypt("mi_token_secreto")
        plaintext  = svc.decrypt(ciphertext)
    """

    __slots__ = ("_aesgcm",)

    def __init__(self, key_hex: str) -> None:
        """
        Inicializa el servicio con la clave en formato hexadecimal.

        Args:
            key_hex: Clave AES-256 como string hexadecimal (exactamente 64 caracteres).

        Raises:
            ValueError: Si la clave no tiene el tamaño correcto.
        """
        try:
            raw_key = bytes.fromhex(key_hex)
        except ValueError as exc:
            raise ValueError(
                "ENCRYPTION_KEY debe ser un string hexadecimal válido."
            ) from exc

        if len(raw_key) != 32:
            raise ValueError(
                f"ENCRYPTION_KEY debe representar exactamente 32 bytes (64 hex chars). "
                f"Recibidos: {len(raw_key)} bytes."
            )

        self._aesgcm = AESGCM(raw_key)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls) -> "CryptoService":
        """
        Crea una instancia usando la clave definida en settings.

        Raises:
            ValueError: Si ENCRYPTION_KEY no está configurado o es inválido.
        """
        from src.config.settings import settings  # import diferido — evita circular

        if not settings.ENCRYPTION_KEY:
            raise ValueError(
                "settings.ENCRYPTION_KEY no está configurado. "
                "Genera una clave con: python -c 'import secrets; print(secrets.token_hex(32))'"
            )
        return cls(settings.ENCRYPTION_KEY)

    # ------------------------------------------------------------------
    # Operaciones públicas
    # ------------------------------------------------------------------

    def encrypt(self, plaintext: str) -> str:
        """
        Cifra un string y retorna el resultado como base64url sin padding.

        Args:
            plaintext: Texto plano a cifrar (UTF-8).

        Returns:
            String base64url con formato [IV][TAG][CIPHERTEXT].

        Raises:
            EncryptionError: Si el cifrado falla por cualquier motivo.
        """
        if not isinstance(plaintext, str):
            raise TypeError(f"plaintext debe ser str, recibido: {type(plaintext).__name__}")

        try:
            iv = os.urandom(_IV_SIZE)
            # AESGCM.encrypt retorna ciphertext + tag (tag al final)
            ciphertext_with_tag = self._aesgcm.encrypt(iv, plaintext.encode("utf-8"), None)

            # Separar tag (últimos 16 bytes) del ciphertext
            tag = ciphertext_with_tag[-_TAG_SIZE:]
            ciphertext = ciphertext_with_tag[:-_TAG_SIZE]

            # Concatenar: IV | TAG | CIPHERTEXT
            payload = iv + tag + ciphertext
            return base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")

        except Exception as exc:
            logger.error("Error durante el cifrado AES-256-GCM", exc_info=True)
            raise EncryptionError("Fallo en operación de cifrado.") from exc

    def decrypt(self, ciphertext_b64: str) -> str:
        """
        Descifra un string producido por encrypt().

        Args:
            ciphertext_b64: String base64url sin padding producido por encrypt().

        Returns:
            Texto plano original (UTF-8).

        Raises:
            DecryptionError: Si el dato está corrupto, fue manipulado,
                             o la clave es incorrecta.
        """
        if not isinstance(ciphertext_b64, str):
            raise TypeError(f"ciphertext_b64 debe ser str, recibido: {type(ciphertext_b64).__name__}")

        try:
            # Restaurar padding base64 si fue eliminado
            padding = 4 - len(ciphertext_b64) % 4
            if padding != 4:
                ciphertext_b64 += "=" * padding

            payload = base64.urlsafe_b64decode(ciphertext_b64)
        except Exception as exc:
            raise DecryptionError("El dato no es base64url válido.") from exc

        min_size = _IV_SIZE + _TAG_SIZE + 1
        if len(payload) < min_size:
            raise DecryptionError(
                f"Payload demasiado corto ({len(payload)} bytes). "
                f"Mínimo esperado: {min_size} bytes."
            )

        try:
            iv = payload[:_IV_SIZE]
            tag = payload[_IV_SIZE: _IV_SIZE + _TAG_SIZE]
            ciphertext = payload[_IV_SIZE + _TAG_SIZE:]

            # AESGCM.decrypt espera ciphertext + tag concatenados
            plaintext_bytes = self._aesgcm.decrypt(iv, ciphertext + tag, None)
            return plaintext_bytes.decode("utf-8")

        except Exception as exc:
            # No exponer detalles internos del fallo criptográfico
            logger.warning("Intento de descifrado fallido — posible tampering o clave incorrecta.")
            raise DecryptionError(
                "Descifrado fallido: dato corrupto, manipulado o clave incorrecta."
            ) from exc


# ------------------------------------------------------------------
# Instancia singleton lazy — se inicializa la primera vez que se usa
# ------------------------------------------------------------------

_instance: CryptoService | None = None


def get_crypto_service() -> CryptoService:
    """
    Retorna la instancia singleton de CryptoService.

    La instancia se crea en el primer llamado y se reutiliza.
    Thread-safe para lecturas concurrentes (Python GIL protege la asignación).

    Raises:
        ValueError: Si ENCRYPTION_KEY no está configurado.
    """
    global _instance
    if _instance is None:
        _instance = CryptoService.from_settings()
    return _instance
