# =============================================================================
# src/services/auth/password_service.py
# Servicio de hashing y verificación de contraseñas
# =============================================================================
"""
Abstracción sobre bcrypt para hashing seguro de contraseñas.

Decisiones de diseño:
  - bcrypt con cost factor configurable (default 12).
    Cost 12 ≈ 250ms en hardware moderno — balance seguridad/UX.
    Cost 14+ para datos altamente sensibles a expensas de latencia.
  - Módulo sin estado (funciones puras) — no requiere instanciación.
  - check_needs_rehash() permite migración transparente si se sube el cost.

Dependencia: passlib[bcrypt] (ya incluido en requirements por cryptography).
"""

from passlib.context import CryptContext

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Contexto de passlib
# ---------------------------------------------------------------------------
# deprecated="auto" → rehash transparente si el algoritmo cambia en el futuro.
# schemes puede extenderse con ["bcrypt", "argon2"] si se migra a argon2.
_pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12,
)


# ---------------------------------------------------------------------------
# API pública — funciones puras, sin estado
# ---------------------------------------------------------------------------

def hash_password(plain_password: str) -> str:
    """
    Genera el hash bcrypt de una contraseña en texto plano.

    Args:
        plain_password: Contraseña en texto plano. No debe estar vacía.

    Returns:
        Hash bcrypt listo para almacenar en base de datos.

    Raises:
        ValueError: Si plain_password está vacío.
    """
    if not plain_password:
        raise ValueError("La contraseña no puede estar vacía.")
    return _pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifica si una contraseña en texto plano coincide con su hash.

    Timing-safe: passlib/bcrypt usa comparación de tiempo constante
    internamente para evitar timing attacks.

    Args:
        plain_password: Contraseña candidata en texto plano.
        hashed_password: Hash almacenado en base de datos.

    Returns:
        True si coinciden, False en caso contrario.
        Nunca lanza excepción por contraseña incorrecta.
    """
    if not plain_password or not hashed_password:
        return False
    try:
        return _pwd_context.verify(plain_password, hashed_password)
    except Exception:
        # Atrapar hashes malformados sin exponer detalles
        logger.warning("verify_password: hash malformado o algoritmo desconocido.")
        return False


def check_needs_rehash(hashed_password: str) -> bool:
    """
    Determina si un hash existente debe ser re-generado.

    Útil cuando se sube el cost factor de bcrypt: al siguiente login
    exitoso el caller puede regenerar el hash con el nuevo cost.

    Args:
        hashed_password: Hash almacenado en base de datos.

    Returns:
        True si el hash fue generado con parámetros obsoletos.
    """
    try:
        return _pwd_context.needs_update(hashed_password)
    except Exception:
        logger.warning("check_needs_rehash: no se pudo evaluar el hash.")
        return False
