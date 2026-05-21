# =============================================================================
# src/models/auth_models.py
# Re-exporta ResourceType y ActionType para compatibilidad con imports
# existentes en el proyecto.
#
# NOTA: Role, Permission, UserRole y las association tables se definen
# en src/models/role.py  para evitar imports circulares y duplicación
# de tablas en Base.metadata.
# =============================================================================
from src.models.role import ResourceType, ActionType  # noqa: F401

__all__ = ["ResourceType", "ActionType"]
