# =============================================================================
# src/services/auth/permission_service.py
# Bootstrap idempotente de roles y permisos del sistema
# =============================================================================
"""
Se llama desde el lifespan de main.py a través de get_permission_service().
Define los roles base que deben existir en la BD al arrancar la aplicación.
"""

from src.models.role import ResourceType, ActionType
from src.repositories.role_repository import RoleRepository, PermissionRepository
from src.repositories.user_repository import UserRepository
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Definición declarativa de roles del sistema
# ---------------------------------------------------------------------------

SYSTEM_ROLES: dict[str, dict] = {
    "superadmin": {
        "description": "Acceso total al sistema.",
        "is_system": True,
        "permissions": [
            {"resource": ResourceType.ANY, "action": ActionType.ANY},
        ],
    },
    "admin": {
        "description": "Administrador de usuarios y configuración.",
        "is_system": True,
        "permissions": [
            {"resource": ResourceType.USER,  "action": ActionType.CREATE},
            {"resource": ResourceType.USER,  "action": ActionType.READ},
            {"resource": ResourceType.USER,  "action": ActionType.UPDATE},
            {"resource": ResourceType.USER,  "action": ActionType.DELETE},
            {"resource": ResourceType.ROLE,  "action": ActionType.READ},
            {"resource": ResourceType.ROLE,  "action": ActionType.UPDATE},
            {"resource": ResourceType.TOOL,  "action": ActionType.ANY},
        ],
    },
    "user": {
        "description": "Usuario estándar del sistema.",
        "is_system": True,
        "permissions": [
            {"resource": ResourceType.CONVERSATION,    "action": ActionType.CREATE},
            {"resource": ResourceType.CONVERSATION,    "action": ActionType.READ},
            {"resource": ResourceType.CONVERSATION,    "action": ActionType.UPDATE},
            {"resource": ResourceType.CONVERSATION,    "action": ActionType.DELETE},
            {"resource": ResourceType.MESSAGE,         "action": ActionType.CREATE},
            {"resource": ResourceType.MESSAGE,         "action": ActionType.READ},
            {"resource": ResourceType.FILE,            "action": ActionType.CREATE},
            {"resource": ResourceType.FILE,            "action": ActionType.READ},
            {"resource": ResourceType.FILE,            "action": ActionType.DELETE},
            {"resource": ResourceType.PROMPT_TEMPLATE, "action": ActionType.READ},
            {"resource": ResourceType.COLLECTION,      "action": ActionType.READ},
        ],
    },
    "readonly": {
        "description": "Solo lectura sobre recursos públicos.",
        "is_system": True,
        "permissions": [
            {"resource": ResourceType.CONVERSATION,    "action": ActionType.READ},
            {"resource": ResourceType.MESSAGE,         "action": ActionType.READ},
            {"resource": ResourceType.PROMPT_TEMPLATE, "action": ActionType.READ},
            {"resource": ResourceType.COLLECTION,      "action": ActionType.READ},
        ],
    },
}


class PermissionService:

    def __init__(
        self,
        role_repo: RoleRepository,
        permission_repo: PermissionRepository,
        user_repo: UserRepository,
    ):
        self._roles       = role_repo
        self._permissions = permission_repo
        self._users       = user_repo

    async def bootstrap_system_roles(self) -> dict:
        """
        Idempotente: crea roles y permisos del sistema si no existen.
        Llamar en el lifespan de la aplicación.
        """
        logger.info("Bootstrapping system roles and permissions...")
        
        roles_created = 0
        permissions_created = 0

        for role_name, role_def in SYSTEM_ROLES.items():
            role = await self._roles.get_by_name_with_permissions(role_name)
            if not role:
                role = await self._roles.create(
                    name=role_name,
                    description=role_def["description"],
                    is_system=role_def["is_system"],
                    is_active=True,
                )
                logger.info(f"  Role created: {role_name}")
                roles_created += 1
                # Re-fetch with permissions to avoid lazy-loading error on append
                role = await self._roles.get_by_name_with_permissions(role_name)
            else:
                logger.debug(f"  Role already exists: {role_name}")

            for perm_def in role_def["permissions"]:
                resource  = perm_def["resource"].value
                action    = perm_def["action"].value
                perm_name = f"{resource}:{action}"

                perm = await self._permissions.get_by_resource_action(resource, action)
                if not perm:
                    perm = await self._permissions.create(
                        name=perm_name,
                        resource=resource,
                        action=action,
                    )
                    logger.info(f"    Permission created: {perm_name}")
                    permissions_created += 1

                # Asociar permiso al rol si aún no está vinculado
                if perm.id not in {p.id for p in role.permissions}:
                    role.permissions.append(perm)

        await self._roles.flush()
        logger.info("Bootstrap complete.")
        return {"roles_created": roles_created, "permissions_created": permissions_created}
