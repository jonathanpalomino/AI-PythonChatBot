# =============================================================================
# src/tools/git_tool.py
# Git Branch Comparison Tool
# =============================================================================
"""
Herramienta para comparar ramas en repositorios Git y obtener la lista de
archivos modificados. Soporta:
  - Modo API : Bitbucket Cloud (via atlassian-python-api + httpx para diffstat).
  - GitHub, GitLab via httpx directo.
  - Modo Binary: git diff local via subprocess.

Acciones disponibles:
  - diff             : Compara dos branches (comportamiento original).
  - scan_conflicts   : Escanea todos los branches y detecta conflictos vs master.
  - get_branch_changes: Obtiene archivos modificados en un branch específico.
"""

import asyncio
import subprocess
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import httpx

from src.tools.base_tool import BaseTool, ToolCategory, ToolParameter, ToolResult
from src.utils.logger import get_logger
from typing import Optional
from uuid import UUID

# =============================================================================
# Enums
# =============================================================================

class GitProvider(str, Enum):
    BITBUCKET = "bitbucket"
    GITHUB    = "github"
    GITLAB    = "gitlab"


class GitMode(str, Enum):
    API    = "api"
    BINARY = "binary"


class FileStatus(str, Enum):
    ADDED          = "added"
    MODIFIED       = "modified"
    REMOVED        = "removed"
    RENAMED        = "renamed"
    MERGE_CONFLICT = "merge_conflict"
    UNKNOWN        = "unknown"


class GitAction(str, Enum):
    DIFF               = "diff"
    SCAN_CONFLICTS     = "scan_conflicts"
    GET_BRANCH_CHANGES = "get_branch_changes"
    LIST_REPOSITORIES  = "list_repositories"
    LIST_BRANCHES      = "list_branches"


# =============================================================================
# GitTool
# =============================================================================

class GitTool(BaseTool):
    """
    Herramienta Git con múltiples acciones:

    - diff             : Compara dos branches y lista archivos cambiados.
    - scan_conflicts   : Lista todos los branches de un repo Bitbucket y detecta
                         cuáles tienen conflictos de merge contra master/main.
    - get_branch_changes: Muestra archivos modificados en un branch específico.

    Usa atlassian-python-api para operaciones de listado de branches en
    Bitbucket, y httpx para diffstat (no expuesto por la librería en Cloud).
    """

    _DEFAULT_URLS: Dict[str, str] = {
        GitProvider.BITBUCKET: "https://api.bitbucket.org",
        GitProvider.GITHUB:    "https://api.github.com",
        GitProvider.GITLAB:    "https://gitlab.com/api/v4",
    }

    def __init__(self):
        self.logger = get_logger(__name__)
        super().__init__()

    # -------------------------------------------------------------------------
    # BaseTool contract
    # -------------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "git_tool"

    @property
    def description(self) -> str:
        return (
            "Herramienta Git con cuatro modos: (1) 'diff': compara dos branches; "
            "(2) 'scan_conflicts': escanea todos los branches de un repositorio "
            "Bitbucket y detecta cuáles tienen conflicto con master; "
            "(3) 'get_branch_changes': muestra los archivos modificados en un "
            "branch específico; (4) 'list_repositories': lista los repositorios "
            "de un workspace o cuenta; (5) 'list_branches': lista las ramas de un repositorio."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.UTILITY

    @property
    def enabled_by_default(self) -> bool:
        return False

    def get_available_actions(self) -> List[Dict[str, Any]]:
        """Get list of available actions for this tool"""
        return [
            {
                "name": action.value,
                "description": self._get_action_description(action.value),
                "default_params": self._get_default_params(action.value)
            }
            for action in GitAction
        ]

    def _get_action_description(self, action_name: str) -> str:
        """Get human-readable description for an action"""
        descriptions = {
            "diff": "Compara dos branches y lista archivos cambiados",
            "scan_conflicts": "Escanea todos los branches de un repositorio Bitbucket y detecta cuáles tienen conflicto con master/main",
            "get_branch_changes": "Muestra los archivos modificados en un branch específico",
            "list_repositories": "Lista los repositorios de un workspace o cuenta",
            "list_branches": "Lista las ramas de un repositorio"
        }
        return descriptions.get(action_name, action_name)

    def _get_default_params(self, action_name: str) -> Dict[str, Any]:
        """Get default parameters for an action"""
        defaults = {
            "diff": {"action": "diff"},
            "scan_conflicts": {"action": "scan_conflicts"},
            "get_branch_changes": {"action": "get_branch_changes"},
            "list_repositories": {"action": "list_repositories"},
            "list_branches": {"action": "list_branches"}
        }
        return defaults.get(action_name, {"action": action_name})

    @property
    def llm_hint(self) -> Optional[str]:
        return (
            "Usa 'git_tool' para: comparar branches (action='diff'), "
            "detectar qué branches entran en conflicto con master (action='scan_conflicts'), "
            "revisar los cambios de un branch (action='get_branch_changes'), "
            "o listar repositorios de un workspace/cuenta (action='list_repositories'). "
            "Para 'list_repositories', 'repository' debe ser solo el workspace (ej: 'my-workspace'). "
            "Para las demás acciones, 'repository' debe ser 'workspace/repo_slug'."
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description=(
                    "Acción a ejecutar: "
                    "'diff' (comparar dos branches), "
                    "'scan_conflicts' (detectar branches con conflicto vs master), "
                    "'get_branch_changes' (ver archivos cambiados en un branch), "
                    "'list_repositories' (listar repositorios de un workspace o cuenta)."
                ),
                required=False,
                default="diff",
                enum=["diff", "scan_conflicts", "get_branch_changes", "list_repositories"],
            ),
            ToolParameter(
                name="repository",
                type="string",
                description=(
                    "Ruta del repositorio ('workspace/repo_slug') o nombre del workspace ('workspace'). "
                    "Para 'list_repositories': si quieres buscar en un espacio específico usa solo el 'workspace' (ej: 'map-pe-devops'). Si se omite, busca en todos. "
                    "Para otras acciones: usa el formato completo (ej: 'map-pe-devops/tron2000-scripts')."
                ),
                required=False,
                example="map-pe-devops/tron2000-scripts",
            ),
            ToolParameter(
                name="branch",
                type="string",
                description=(
                    "Rama de origen (la que contiene los cambios a comparar). "
                    "Requerido para 'diff' y 'get_branch_changes'. "
                    "No usado en 'scan_conflicts'."
                ),
                required=False,
                example="feature/my-branch",
            ),
            ToolParameter(
                name="compare_branch",
                type="string",
                description="Rama base con la que se compara. Por defecto 'master'.",
                required=False,
                default="master",
                example="master",
            ),
            ToolParameter(
                name="provider",
                type="string",
                description="Proveedor Git: 'bitbucket', 'github' o 'gitlab'.",
                required=False,
                default="bitbucket",
                enum=["bitbucket", "github", "gitlab"],
            ),
            ToolParameter(
                name="mode",
                type="string",
                description="Modo de consulta: 'api' o 'binary' (git local).",
                required=False,
                default="api",
                enum=["api", "binary"],
            ),
            ToolParameter(
                name="api_token",
                type="string",
                description=(
                    "Token de autenticación. "
                    "Bitbucket Cloud: 'username:app_password'. "
                    "GitHub/GitLab: personal access token."
                ),
                required=False,
                default=None,
            ),
            ToolParameter(
                name="api_base_url",
                type="string",
                description="URL base de la API. Por defecto usa la URL pública del proveedor.",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="local_repo_path",
                type="string",
                description="Ruta al repositorio local (solo para mode='binary').",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="concurrency",
                type="integer",
                description=(
                    "Número de requests paralelas al escanear branches "
                    "(solo para action='scan_conflicts'). Por defecto 5."
                ),
                required=False,
                default=5,
            ),
            ToolParameter(
                name="exclude_branches",
                type="string",
                description=(
                    "Branches a excluir del scan (CSV). "
                    "Ej: 'master,main,develop'. "
                    "Solo para action='scan_conflicts'."
                ),
                required=False,
                default=None,
            ),
            ToolParameter(
                name="include_pattern",
                type="string",
                description=(
                    "Filtrar branches por prefijo (ej: 'feature/'). "
                    "Solo para action='scan_conflicts'."
                ),
                required=False,
                default=None,
            ),
            ToolParameter(
                name="timeout",
                type="integer",
                description="Timeout en segundos para llamadas API o subprocesos. Por defecto 30.",
                required=False,
                default=30,
            ),
            ToolParameter(
                name="user_id",
                type="string",
                description="UUID del usuario autenticado. Si se provee, se usa su OAuth token de Bitbucket.",
                required=False,
                default=None,
            ),
        ]

    # -------------------------------------------------------------------------
    # Intent definitions
    # -------------------------------------------------------------------------

    def get_intent_definitions(self) -> Dict[str, Any]:
        return {
            "list_git_repositories": {
                "description": "Listar los repositorios existentes en un workspace o cuenta de Bitbucket/GitHub/GitLab",
                "action_name": "list_repositories",
                "requires_target": False,
                "target_patterns": [r"(?:workspace|cuenta|organización|org|de)\s+([\w\-]+)"],
                "examples": [
                    "lista los repositorios de bitbucket",
                    "qué repos existen en el workspace",
                    "muestra todos los repositorios disponibles",
                    "listar repositorios existentes",
                    "cuáles son los repos del proyecto",
                ],
                "default_params": {"action": "list_repositories", "provider": "bitbucket"},
            },
            "scan_git_conflicts": {
                "description": "Detectar qué branches tienen conflicto de merge contra master en un repositorio Bitbucket",
                "action_name": "scan_conflicts",
                "requires_target": True,
                "target_patterns": [r"(?:proyecto|proyecto|repo|repositorio|en)\s+([\w\-/]+)"],
                "examples": [
                    "qué branches tienen conflicto con master en tron2000-scripts",
                    "analiza los conflictos del proyecto xyz en bitbucket",
                    "revisa qué ramas entran en conflicto con master",
                    "escanea conflictos en bitbucket",
                    "evalúa los conflictos del repo tron2000",
                    "cuáles branches están en conflicto",
                ],
                "default_params": {"action": "scan_conflicts", "provider": "bitbucket"},
            },
            "get_git_branch_changes": {
                "description": "Ver los archivos modificados en un branch específico de un repositorio",
                "action_name": "get_branch_changes",
                "requires_target": True,
                "target_patterns": [r"(?:branch|rama)\s+([\w\-/]+)"],
                "examples": [
                    "revisa los cambios del branch feature/hhh en tron2000-scripts",
                    "qué archivos cambió feature/my-branch",
                    "muestra el diff de feature/login vs master",
                    "qué modificó el branch release/v2",
                    "cambios en la rama feature/nueva-api",
                ],
                "default_params": {"action": "get_branch_changes", "provider": "bitbucket"},
            },
            "git_diff_branches": {
                "description": "Comparar dos branches de Git y listar archivos cambiados",
                "action_name": "diff",
                "requires_target": False,
                "target_patterns": [],
                "examples": [
                    "compara feature/x con master",
                    "diferencias entre develop y main",
                    "qué cambió entre release/v1 y main",
                ],
                "default_params": {"action": "diff", "provider": "bitbucket"},
            },
            "list_git_branches": {
                "description": "Listar las ramas (branches) de un repositorio",
                "action_name": "list_branches",
                "requires_target": True,
                "target_patterns": [r"(?:proyecto|repo|repositorio|en|del)\\s+([\\w\\-/]+)"],
                "examples": [
                    "lista las ramas del repo tron2000-scripts",
                    "qué branches existen en bitbucket",
                    "muestra todas las ramas del proyecto",
                    "listar branches del repositorio",
                ],
                "default_params": {"action": "list_branches", "provider": "bitbucket"},
            },
        }

    def params_from_intent(self, intent_result: Any) -> Dict[str, Any]:
        params = {"action": intent_result.intent_def.action_name}
        if intent_result.target:
            action = intent_result.intent_def.action_name
            if action == "get_branch_changes":
                params["branch"] = intent_result.target
            elif action in ("scan_conflicts", "list_branches"):
                params["repository"] = intent_result.target
        if intent_result.intent_def.default_params:
            params.update(intent_result.intent_def.default_params)
        return params

    # -------------------------------------------------------------------------
    # execute() — router principal
    # -------------------------------------------------------------------------

    async def execute(self, **kwargs) -> ToolResult:
        """Enruta a la acción correspondiente según el parámetro 'action'."""
        action:         str           = kwargs.get("action", "diff")
        repository:     Optional[str] = kwargs.get("repository")
        branch:         str           = kwargs.get("branch", "")
        compare_branch: str           = kwargs.get("compare_branch", "master")
        provider:       str           = kwargs.get("provider", "bitbucket")
        mode:           str           = kwargs.get("mode", "api")
        api_token:      Optional[str] = kwargs.get("api_token")
        api_base_url:   Optional[str] = kwargs.get("api_base_url")
        local_repo_path:Optional[str] = kwargs.get("local_repo_path")
        concurrency:    int           = kwargs.get("concurrency", 5)
        exclude_branches:Optional[str]= kwargs.get("exclude_branches")
        include_pattern:Optional[str]= kwargs.get("include_pattern")
        timeout:        int           = kwargs.get("timeout", 30)
        user_id                       = kwargs.get("user_id")

        # Validate repository is provided for API actions that require it
        if mode == "api" and action != "list_repositories" and not repository:
            return ToolResult(
                success=False, data=None,
                error=f"El parámetro 'repository' es requerido para action='{action}' en modo api."
            )

        # Resolver token: OAuth primero, fallback a settings
        resolved_token = api_token or await self._resolve_token(provider, user_id=user_id)

        self.logger.info(
            f"[git_tool] action={action} mode={mode} provider={provider} "
            f"repo={repository} branch={branch} compare={compare_branch}"
        )

        try:
            if action == GitAction.LIST_REPOSITORIES:
                return await self._action_list_repositories(
                    repository=repository,
                    provider=GitProvider(provider),
                    api_token=resolved_token,
                    api_base_url=api_base_url,
                    include_pattern=include_pattern,
                    timeout=timeout,
                )
            elif action == GitAction.SCAN_CONFLICTS:
                return await self._action_scan_conflicts(
                    repository=repository,
                    compare_branch=compare_branch,
                    provider=GitProvider(provider),
                    api_token=resolved_token,
                    api_base_url=api_base_url,
                    concurrency=concurrency,
                    exclude_branches=exclude_branches,
                    include_pattern=include_pattern,
                    timeout=timeout,
                )
            elif action == GitAction.LIST_BRANCHES:
                return await self._action_list_branches(
                    repository=repository,
                    provider=GitProvider(provider),
                    api_token=resolved_token,
                    api_base_url=api_base_url,
                    timeout=timeout,
                )
            elif action == GitAction.GET_BRANCH_CHANGES:
                if not branch:
                    return ToolResult(
                        success=False, data=None,
                        error="El parámetro 'branch' es requerido para action='get_branch_changes'."
                    )
                return await self._action_get_branch_changes(
                    repository=repository,
                    branch=branch,
                    compare_branch=compare_branch,
                    provider=GitProvider(provider),
                    api_token=resolved_token,
                    api_base_url=api_base_url,
                    timeout=timeout,
                )
            else:  # "diff" — comportamiento original
                if mode == GitMode.BINARY:
                    return await self._execute_binary(
                        branch=branch,
                        compare_branch=compare_branch,
                        local_repo_path=local_repo_path,
                        timeout=timeout,
                    )
                return await self._execute_api(
                    repository=repository,
                    branch=branch,
                    compare_branch=compare_branch,
                    provider=GitProvider(provider),
                    api_token=resolved_token,
                    api_base_url=api_base_url,
                    timeout=timeout,
                )
        except ValueError as e:
            return ToolResult(success=False, data=None, error=str(e))
        except Exception as e:
            self.logger.error(f"[git_tool] Unexpected error: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))

    # -------------------------------------------------------------------------
    # Auth helpers
    # -------------------------------------------------------------------------
    async def _resolve_token(
        self,
        provider: str,
        user_id: Optional[UUID] = None,
    ) -> Optional[str]:
        """
        Estrategia de resolución de token (en orden de prioridad):
          1. OAuth token del usuario (si user_id está disponible)  → Bearer
          2. Fallback al usuario 'default@example.com' (para desarrollo local stateless)
          3. settings.BITBUCKET_API_TOKEN (email:token Basic Auth) → fallback global
          4. None → la llamada fallará con 401 (esperado)
        """
        resolved_uid = user_id
        if not resolved_uid and provider == "bitbucket":
            try:
                from src.database.connection import get_async_db_session
                from src.repositories.user_repository import UserRepository
                async with get_async_db_session() as session:
                    user_repo = UserRepository(session)
                    user = await user_repo.get_by_email("default@example.com")
                    if user:
                        resolved_uid = user.id
            except Exception as e:
                self.logger.warning(f"[git_tool] No se pudo obtener el usuario por defecto: {e}")

        if resolved_uid and provider == "bitbucket":
            try:
                from uuid import UUID
                uid = UUID(str(resolved_uid)) if not isinstance(resolved_uid, UUID) else resolved_uid
                from src.utils.service_factory import get_bb_oauth_service

                async with get_bb_oauth_service() as svc:
                    oauth_token = await svc.get_valid_access_token(uid)

                if oauth_token:
                    self.logger.debug(f"[git_tool] Usando OAuth token para user={uid}")
                    return f"__oauth__:{oauth_token}"  # prefijo para _build_headers
            except Exception as e:
                self.logger.warning(f"[git_tool] No se pudo obtener OAuth token: {e}")

        # Fallback a token global de settings
        try:
            from src.config.settings import settings
            if provider == "bitbucket":
                return getattr(settings, "BITBUCKET_API_TOKEN", None)
            elif provider == "github":
                return getattr(settings, "GITHUB_API_TOKEN", None)
            elif provider == "gitlab":
                return getattr(settings, "GITLAB_API_TOKEN", None)
        except Exception:
            pass
        return None

    def _build_headers(
        self, provider: "GitProvider", api_token: Optional[str]
    ) -> dict:
        headers = {"Accept": "application/json"}
        if not api_token:
            return headers

        if provider == GitProvider.BITBUCKET:
            if api_token.startswith("__oauth__:"):
                # Token OAuth — usar como Bearer directamente
                raw = api_token.split(":", 1)[1]
                headers["Authorization"] = f"Bearer {raw}"
            elif ":" in api_token:
                import base64
                encoded = base64.b64encode(api_token.encode()).decode()
                headers["Authorization"] = f"Basic {encoded}"
            else:
                headers["Authorization"] = f"Bearer {api_token}"
        elif provider == GitProvider.GITHUB:
            headers["Authorization"] = f"Bearer {api_token}"
            headers["X-GitHub-Api-Version"] = "2022-11-28"
        elif provider == GitProvider.GITLAB:
            headers["PRIVATE-TOKEN"] = api_token
        return headers

    # -------------------------------------------------------------------------
    # Action: list_repositories
    # -------------------------------------------------------------------------

    async def _action_list_repositories(
        self,
        repository: Optional[str],
        provider: GitProvider,
        api_token: Optional[str],
        api_base_url: Optional[str],
        include_pattern: Optional[str],
        timeout: int,
    ) -> ToolResult:
        """
        Lista los repositorios. Si se indica repository/workspace, se lista de ahí.
        Si no se indica, se listan todos a los que el usuario tiene acceso a través de su token.
        """
        workspace = None
        if repository:
            # Extraer solo el workspace si viene en formato 'workspace/repo'
            workspace = repository.split("/")[0] if "/" in repository else repository

        base_url = api_base_url or self._DEFAULT_URLS[provider]
        headers  = self._build_headers(provider, api_token)

        try:
            repos = await self._fetch_repositories(
                workspace=workspace,
                provider=provider,
                base_url=base_url,
                headers=headers,
                timeout=timeout,
            )
        except httpx.HTTPStatusError as e:
            return ToolResult(
                success=False, data=None,
                error=f"HTTP {e.response.status_code}: {e.response.text[:300]}"
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=f"Error listando repositorios: {e}")

        # Filtrar por patrón si se indicó
        if include_pattern:
            repos = [r for r in repos if include_pattern.lower() in r.get("slug", "").lower()
                     or include_pattern.lower() in r.get("name", "").lower()]

        return ToolResult(
            success=True,
            data={"repositories": repos, "total_count": len(repos)},
            metadata={
                "action":    "list_repositories",
                "provider":  provider.value,
                "workspace": workspace or "all",
            },
        )

    async def _fetch_repositories(
        self,
        workspace: Optional[str],
        provider: GitProvider,
        base_url: str,
        headers: Dict[str, str],
        timeout: int,
    ) -> List[Dict[str, Any]]:
        """Llama a la API correspondiente y devuelve lista de repos."""
        repos: List[Dict[str, Any]] = []

        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            if provider == GitProvider.BITBUCKET:
                workspaces_to_fetch = []
                if workspace:
                    workspaces_to_fetch.append(workspace)
                else:
                    # Fetch all workspaces the user has access to
                    ws_url: Optional[str] = f"{base_url}/2.0/workspaces"
                    while ws_url:
                        ws_resp = await client.get(ws_url)
                        ws_resp.raise_for_status()
                        ws_data = ws_resp.json()
                        for ws in ws_data.get("values", []):
                            slug = ws.get("slug")
                            if slug:
                                workspaces_to_fetch.append(slug)
                        ws_url = ws_data.get("next")
                        
                for ws_slug in workspaces_to_fetch:
                    url: Optional[str] = f"{base_url}/2.0/repositories/{ws_slug}?pagelen=100&role=member"
                    while url:
                        resp = await client.get(url)
                        if resp.status_code in (403, 404):
                            break
                        resp.raise_for_status()
                        data = resp.json()
                        for r in data.get("values", []):
                            repos.append({
                                "slug":        r.get("slug", ""),
                                "name":        r.get("name", ""),
                                "description": r.get("description", ""),
                                "is_private":  r.get("is_private", True),
                                "language":    r.get("language", ""),
                                "full_name":   r.get("full_name", ""),
                                "updated_on":  r.get("updated_on", ""),
                            })
                        url = data.get("next")

            elif provider == GitProvider.GITHUB:
                page, per_page = 1, 100
                while True:
                    if workspace:
                        url = (
                            f"{base_url}/orgs/{workspace}/repos"
                            f"?per_page={per_page}&page={page}&type=all"
                        )
                    else:
                        url = (
                            f"{base_url}/user/repos"
                            f"?per_page={per_page}&page={page}&type=all"
                        )
                    resp = await client.get(url)
                    # Si es cuenta personal (no org), fallback a /users/{workspace}/repos
                    if workspace and resp.status_code == 404:
                        url = (
                            f"{base_url}/users/{workspace}/repos"
                            f"?per_page={per_page}&page={page}&type=all"
                        )
                        resp = await client.get(url)
                    resp.raise_for_status()
                    data = resp.json()
                    for r in data:
                        repos.append({
                            "slug":        r.get("name", ""),
                            "name":        r.get("full_name", ""),
                            "description": r.get("description") or "",
                            "is_private":  r.get("private", True),
                            "language":    r.get("language") or "",
                            "full_name":   r.get("full_name", ""),
                            "updated_on":  r.get("updated_at", ""),
                        })
                    if len(data) < per_page:
                        break
                    page += 1

            elif provider == GitProvider.GITLAB:
                import urllib.parse
                page = 1
                while True:
                    if workspace:
                        enc = urllib.parse.quote(workspace, safe="")
                        url = (
                            f"{base_url}/groups/{enc}/projects"
                            f"?per_page=100&page={page}&include_subgroups=true"
                        )
                    else:
                        url = (
                            f"{base_url}/projects"
                            f"?membership=true&per_page=100&page={page}"
                        )
                    resp = await client.get(url)
                    resp.raise_for_status()
                    data = resp.json()
                    for r in data:
                        repos.append({
                            "slug":        r.get("path", ""),
                            "name":        r.get("name", ""),
                            "description": r.get("description") or "",
                            "is_private":  r.get("visibility", "private") != "public",
                            "language":    "",
                            "full_name":   r.get("path_with_namespace", ""),
                            "updated_on":  r.get("last_activity_at", ""),
                        })
                    if not resp.headers.get("X-Next-Page"):
                        break
                    page = int(resp.headers["X-Next-Page"])

        return repos

    # -------------------------------------------------------------------------
    # Action: list_branches
    # -------------------------------------------------------------------------

    async def _action_list_branches(
        self,
        repository: Optional[str],
        provider: GitProvider,
        api_token: Optional[str],
        api_base_url: Optional[str],
        timeout: int,
    ) -> ToolResult:
        """Lista las ramas de un repositorio usando el método _list_branches que ya existía."""
        if not repository:
            return ToolResult(
                success=False, data=None,
                error="El parámetro 'repository' es requerido para action='list_branches'."
            )
            
        base_url = api_base_url or self._DEFAULT_URLS[provider]
        headers  = self._build_headers(provider, api_token)

        try:
            branches = await self._list_branches(
                repository=repository,
                provider=provider,
                api_token=api_token,
                base_url=base_url,
                headers=headers,
                timeout=timeout,
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=f"Error listando ramas: {e}")

        return ToolResult(
            success=True,
            data={"branches": branches, "total_count": len(branches)},
            metadata={
                "action": "list_branches",
                "provider": provider.value,
                "repository": repository,
            },
        )

    # -------------------------------------------------------------------------
    # Action: scan_conflicts
    # -------------------------------------------------------------------------

    async def _action_scan_conflicts(
        self,
        repository: Optional[str],
        compare_branch: str,
        provider: GitProvider,
        api_token: Optional[str],
        api_base_url: Optional[str],
        concurrency: int,
        exclude_branches: Optional[str],
        include_pattern: Optional[str],
        timeout: int,
    ) -> ToolResult:
        """
        Lista todos los branches del repo y comprueba cuáles tienen conflicto
        de merge contra compare_branch (normalmente 'master').

        Usa atlassian-python-api en executor para listar branches (sync)
        y httpx para el diffstat (async, en paralelo con semáforo).
        """
        base_url = api_base_url or self._DEFAULT_URLS[provider]
        headers  = self._build_headers(provider, api_token)

        # Construir set de exclusiones
        excluded: set = set()
        if exclude_branches:
            excluded = {b.strip() for b in exclude_branches.split(",") if b.strip()}
        excluded.add(compare_branch)  # Siempre excluir la rama base

        # 1. Listar todos los branches
        try:
            all_branches = await self._list_branches(
                repository=repository,
                provider=provider,
                api_token=api_token,
                base_url=base_url,
                headers=headers,
                timeout=timeout,
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=f"Error listando branches: {e}")

        # Filtrar exclusiones y patrón de inclusión
        branches_to_scan = [
            b for b in all_branches
            if b not in excluded
            and (not include_pattern or b.startswith(include_pattern))
        ]

        self.logger.info(
            f"[git_tool] scan_conflicts: {len(all_branches)} total, "
            f"{len(branches_to_scan)} a analizar contra '{compare_branch}'"
        )

        # 2. Escanear conflictos en paralelo con semáforo
        semaphore = asyncio.Semaphore(concurrency)
        results: List[Dict[str, Any]] = []

        async def check_one(branch_name: str):
            async with semaphore:
                return await self._check_branch_conflict(
                    branch=branch_name,
                    compare_branch=compare_branch,
                    repository=repository,
                    base_url=base_url,
                    headers=headers,
                    timeout=timeout,
                )

        tasks = [check_one(b) for b in branches_to_scan]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 3. Clasificar resultados
        conflicted: List[Dict] = []
        clean_with_changes: List[Dict] = []
        no_changes: List[Dict] = []
        errors: List[Dict] = []

        for branch_name, result in zip(branches_to_scan, results):
            if isinstance(result, Exception):
                errors.append({"branch": branch_name, "error": str(result)})
                continue
            status = result.get("conflict_status")
            entry  = {
                "branch":           branch_name,
                "conflict_files":   result.get("conflict_files", []),
                "conflict_count":   result.get("conflict_count", 0),
                "total_changes":    result.get("total_changes", 0),
                "changed_files":    result.get("changed_files", []),
            }
            if status == "conflict":
                conflicted.append(entry)
            elif status == "clean":
                clean_with_changes.append(entry)
            else:
                no_changes.append(entry)

        return ToolResult(
            success=True,
            data={
                "conflicted":         conflicted,
                "clean_with_changes": clean_with_changes,
                "no_changes":         no_changes,
                "errors":             errors,
            },
            metadata={
                "action":           "scan_conflicts",
                "provider":         provider.value,
                "repository":       repository,
                "compare_branch":   compare_branch,
                "total_scanned":    len(branches_to_scan),
                "total_branches":   len(all_branches),
                "conflict_count":   len(conflicted),
            },
        )

    async def _list_branches(
        self,
        repository: str,
        provider: GitProvider,
        api_token: Optional[str],
        base_url: str,
        headers: Dict[str, str],
        timeout: int,
    ) -> List[str]:
        """
        Lista todos los branches del repositorio.
        Para Bitbucket Cloud usa atlassian-python-api (sync) en executor.
        Para GitHub/GitLab usa httpx directamente.
        """
        if provider == GitProvider.BITBUCKET:
            return await self._list_bitbucket_branches_sdk(
                repository=repository,
                api_token=api_token,
                base_url=base_url,
                timeout=timeout,
            )
        else:
            return await self._list_branches_httpx(
                repository=repository,
                provider=provider,
                base_url=base_url,
                headers=headers,
                timeout=timeout,
            )

    async def _list_bitbucket_branches_sdk(
        self,
        repository: str,
        api_token: Optional[str],
        base_url: str,
        timeout: int,
    ) -> List[str]:
        """
        Usa atlassian-python-api para listar branches en Bitbucket Cloud.
        La librería es síncrona, la ejecutamos en un executor.
        """
        def _sync_list():
            try:
                from atlassian.bitbucket import Cloud

                # api_token esperado como "username:app_password" o token Bearer
                username, password, token = None, None, None
                if api_token:
                    if api_token.startswith("__oauth__:"):
                        token = api_token.split(":", 1)[1]
                    elif ":" in api_token:
                        username, password = api_token.split(":", 1)
                    else:
                        token = api_token

                cloud = Cloud(username=username, password=password, token=token, cloud=True)

                # repository = "workspace/repo_slug"
                workspace_slug, repo_slug = repository.split("/", 1)
                repo = cloud.workspaces.get(workspace_slug).repositories.get(repo_slug)

                branches = []
                for branch in repo.branches.each():
                    branches.append(branch.name)
                return branches
            except Exception as e:
                raise RuntimeError(f"atlassian-python-api error: {e}") from e

        loop = asyncio.get_event_loop()
        branches = await loop.run_in_executor(None, _sync_list)
        return branches

    async def _list_branches_httpx(
        self,
        repository: str,
        provider: GitProvider,
        base_url: str,
        headers: Dict[str, str],
        timeout: int,
    ) -> List[str]:
        """Fallback httpx para listar branches en GitHub/GitLab."""
        branches: List[str] = []

        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            if provider == GitProvider.GITHUB:
                page, per_page = 1, 100
                while True:
                    url = f"{base_url}/repos/{repository}/branches?per_page={per_page}&page={page}"
                    resp = await client.get(url)
                    resp.raise_for_status()
                    data = resp.json()
                    branches.extend(b["name"] for b in data)
                    if len(data) < per_page:
                        break
                    page += 1
            elif provider == GitProvider.GITLAB:
                import urllib.parse
                enc = urllib.parse.quote(repository, safe="")
                page = 1
                while True:
                    url = f"{base_url}/projects/{enc}/repository/branches?per_page=100&page={page}"
                    resp = await client.get(url)
                    resp.raise_for_status()
                    data = resp.json()
                    branches.extend(b["name"] for b in data)
                    if not resp.headers.get("X-Next-Page"):
                        break
                    page = int(resp.headers["X-Next-Page"])
            else:
                # Bitbucket httpx fallback
                url = f"{base_url}/2.0/repositories/{repository}/refs/branches?pagelen=100"
                while url:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    data = resp.json()
                    branches.extend(v["name"] for v in data.get("values", []))
                    url = data.get("next")

        return branches

    async def _check_branch_conflict(
        self,
        branch: str,
        compare_branch: str,
        repository: Optional[str],
        base_url: str,
        headers: Dict[str, str],
        timeout: int,
    ) -> Dict[str, Any]:
        """
        Llama al endpoint diffstat de Bitbucket Cloud para un branch.
        Detecta si hay archivos con status='merge conflict'.
        Si repository es None, devuelve resultado vacío (sin error).
        """
        if not repository:
            return {
                "conflict_status": "no_changes",
                "conflict_files":  [],
                "changed_files":   [],
                "total_changes":   0,
            }
        spec = f"{compare_branch}..{branch}"
        url  = f"{base_url}/2.0/repositories/{repository}/diffstat/{spec}"

        changed_files: List[Dict] = []
        conflict_files: List[Dict] = []

        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            while url:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()

                for entry in data.get("values", []):
                    status   = entry.get("status", "unknown")
                    new_file = entry.get("new") or {}
                    old_file = entry.get("old") or {}
                    path     = new_file.get("path") or old_file.get("path") or ""

                    file_info = {
                        "path":          path,
                        "status":        status,
                        "lines_added":   entry.get("lines_added", 0),
                        "lines_removed": entry.get("lines_removed", 0),
                    }
                    changed_files.append(file_info)
                    if status == "merge conflict":
                        conflict_files.append(file_info)

                url = data.get("next")

        has_conflict = len(conflict_files) > 0
        has_changes  = len(changed_files) > 0

        return {
            "conflict_status": "conflict" if has_conflict else ("clean" if has_changes else "no_changes"),
            "conflict_files":  conflict_files,
            "conflict_count":  len(conflict_files),
            "changed_files":   changed_files,
            "total_changes":   len(changed_files),
        }

    # -------------------------------------------------------------------------
    # Action: get_branch_changes
    # -------------------------------------------------------------------------

    async def _action_get_branch_changes(
        self,
        repository: str,
        branch: str,
        compare_branch: str,
        provider: GitProvider,
        api_token: Optional[str],
        api_base_url: Optional[str],
        timeout: int,
    ) -> ToolResult:
        """
        Wrapper específico para obtener los cambios de un branch, devolviendo
        un objeto más estructurado. En Bitbucket, usa el mismo mecanismo de
        diffstat que action='diff', pero su output será más narrativo.
        """
        # Reutilizamos el comportamiento core del diff
        result = await self._execute_api(
            repository=repository,
            branch=branch,
            compare_branch=compare_branch,
            provider=provider,
            api_token=api_token,
            api_base_url=api_base_url,
            timeout=timeout,
        )
        if result.success:
            # Etiquetamos explícitamente para que format_output sepa cómo renderizar
            result.metadata["action"] = GitAction.GET_BRANCH_CHANGES.value
        return result

    # -------------------------------------------------------------------------
    # Action: diff (y fallback general para fetch)
    # -------------------------------------------------------------------------

    async def _execute_api(
        self,
        repository: str,
        branch: str,
        compare_branch: str,
        provider: GitProvider,
        api_token: Optional[str],
        api_base_url: Optional[str],
        timeout: int,
    ) -> ToolResult:
        """Consulta la API REST del proveedor para obtener diff completo."""
        base_url = api_base_url or self._DEFAULT_URLS[provider]
        headers  = self._build_headers(provider, api_token)

        changed_files: List[Dict[str, Any]] = []
        api_calls = 0

        try:
            async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
                if provider == GitProvider.BITBUCKET:
                    changed_files, api_calls = await self._fetch_bitbucket(
                        client, base_url, repository, branch, compare_branch
                    )
                elif provider == GitProvider.GITHUB:
                    changed_files, api_calls = await self._fetch_github(
                        client, base_url, repository, branch, compare_branch
                    )
                elif provider == GitProvider.GITLAB:
                    changed_files, api_calls = await self._fetch_gitlab(
                        client, base_url, repository, branch, compare_branch
                    )
        except httpx.TimeoutException:
            return ToolResult(
                success=False, data=None,
                error=f"API request timed out after {timeout}s"
            )
        except httpx.HTTPStatusError as e:
            return ToolResult(
                success=False, data=None,
                error=f"HTTP {e.response.status_code}: {e.response.text[:300]}"
            )

        return ToolResult(
            success=True,
            data={
                "changed_files": changed_files,
                "total_count": len(changed_files),
            },
            metadata={
                "action": "diff",
                "provider": provider.value,
                "mode": "api",
                "repository": repository,
                "branch": branch,
                "compare_branch": compare_branch,
                "api_calls": api_calls,
            },
        )

    # -- Fetch Helpers por proveedor (Diff original) ------------------------- #

    async def _fetch_bitbucket(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        repository: Optional[str],
        branch: str,
        compare_branch: str,
    ) -> tuple[List[Dict[str, Any]], int]:
        if not repository:
            return [], 0
        spec = f"{compare_branch}..{branch}"
        url  = f"{base_url}/2.0/repositories/{repository}/diffstat/{spec}"
        files: List[Dict[str, Any]] = []
        api_calls = 0

        while url:
            response = await client.get(url)
            response.raise_for_status()
            api_calls += 1
            data = response.json()

            for entry in data.get("values", []):
                status = entry.get("status", "unknown")
                new_file = entry.get("new") or {}
                old_file = entry.get("old") or {}
                path = (new_file.get("path") or old_file.get("path") or "")
                old_path = old_file.get("path") if status == "renamed" else None

                files.append({
                    "path": path,
                    "old_path": old_path,
                    "status": self._map_bitbucket_status(status),
                    "lines_added":   entry.get("lines_added", 0),
                    "lines_removed": entry.get("lines_removed", 0),
                })

            url = data.get("next")
        return files, api_calls

    def _map_bitbucket_status(self, status: str) -> str:
        mapping = {
            "added":          FileStatus.ADDED,
            "modified":       FileStatus.MODIFIED,
            "removed":        FileStatus.REMOVED,
            "renamed":        FileStatus.RENAMED,
            "merge conflict": FileStatus.MERGE_CONFLICT,
        }
        return mapping.get(status, FileStatus.UNKNOWN).value

    async def _fetch_github(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        repository: str,
        branch: str,
        compare_branch: str,
    ) -> tuple[List[Dict[str, Any]], int]:
        files: List[Dict[str, Any]] = []
        api_calls = 0
        page = 1
        per_page = 100

        while True:
            url = (
                f"{base_url}/repos/{repository}/compare"
                f"/{compare_branch}...{branch}"
                f"?per_page={per_page}&page={page}"
            )
            response = await client.get(url)
            response.raise_for_status()
            api_calls += 1
            data = response.json()

            page_files = data.get("files", [])
            for f in page_files:
                files.append({
                    "path":          f.get("filename", ""),
                    "old_path":      f.get("previous_filename"),
                    "status":        self._map_github_status(f.get("status", "")),
                    "lines_added":   f.get("additions", 0),
                    "lines_removed": f.get("deletions", 0),
                    "changes":       f.get("changes", 0),
                })
            if len(page_files) < per_page:
                break
            page += 1
        return files, api_calls

    def _map_github_status(self, status: str) -> str:
        mapping = {
            "added":    FileStatus.ADDED,
            "modified": FileStatus.MODIFIED,
            "removed":  FileStatus.REMOVED,
            "renamed":  FileStatus.RENAMED,
        }
        return mapping.get(status, FileStatus.UNKNOWN).value

    async def _fetch_gitlab(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        repository: str,
        branch: str,
        compare_branch: str,
    ) -> tuple[List[Dict[str, Any]], int]:
        import urllib.parse
        encoded_repo = urllib.parse.quote(repository, safe="")
        files: List[Dict[str, Any]] = []
        api_calls = 0
        page = 1

        while True:
            url = (
                f"{base_url}/projects/{encoded_repo}/repository/compare"
                f"?from={compare_branch}&to={branch}&page={page}&per_page=100"
            )
            response = await client.get(url)
            response.raise_for_status()
            api_calls += 1
            data = response.json()

            for diff in data.get("diffs", []):
                status = self._map_gitlab_status(diff)
                files.append({
                    "path":          diff.get("new_path", ""),
                    "old_path":      diff.get("old_path") if diff.get("renamed_file") else None,
                    "status":        status,
                    "lines_added":   None,
                    "lines_removed": None,
                })
            next_page = response.headers.get("X-Next-Page", "")
            if not next_page:
                break
            page = int(next_page)
        return files, api_calls

    def _map_gitlab_status(self, diff: Dict[str, Any]) -> str:
        if diff.get("new_file"):
            return FileStatus.ADDED.value
        if diff.get("deleted_file"):
            return FileStatus.REMOVED.value
        if diff.get("renamed_file"):
            return FileStatus.RENAMED.value
        return FileStatus.MODIFIED.value

    # -------------------------------------------------------------------------
    # Binary Mode (Fallback git local)
    # -------------------------------------------------------------------------

    async def _execute_binary(
        self,
        branch: str,
        compare_branch: str,
        local_repo_path: Optional[str],
        timeout: int,
    ) -> ToolResult:
        import shutil

        if not shutil.which("git"):
            return ToolResult(
                success=False, data=None,
                error="git binary not found in PATH. Install git or use mode='api'."
            )

        cmd = ["git", "diff", "--name-status", f"{compare_branch}..{branch}"]
        cwd = local_repo_path or None

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=cwd,
                    timeout=timeout,
                ),
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False, data=None,
                error=f"git diff timed out after {timeout}s"
            )
        except FileNotFoundError:
            return ToolResult(
                success=False, data=None,
                error=f"Repository path not found: {local_repo_path}"
            )

        if result.returncode != 0:
            return ToolResult(
                success=False, data=None,
                error=f"git diff failed (exit {result.returncode}): {result.stderr.strip()}"
            )

        changed_files = self._parse_git_diff_output(result.stdout)

        return ToolResult(
            success=True,
            data={
                "changed_files": changed_files,
                "total_count": len(changed_files),
            },
            metadata={
                "action": "diff",
                "mode": "binary",
                "branch": branch,
                "compare_branch": compare_branch,
                "local_repo_path": local_repo_path or "cwd",
                "command": " ".join(cmd),
            },
        )

    def _parse_git_diff_output(self, output: str) -> List[Dict[str, Any]]:
        files: List[Dict[str, Any]] = []
        _status_map = {
            "A": FileStatus.ADDED,
            "M": FileStatus.MODIFIED,
            "D": FileStatus.REMOVED,
            "R": FileStatus.RENAMED,
        }

        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            raw_status = parts[0]
            letter = raw_status[0].upper()
            status = _status_map.get(letter, FileStatus.UNKNOWN).value

            if letter == "R" and len(parts) >= 3:
                files.append({
                    "path":          parts[2],
                    "old_path":      parts[1],
                    "status":        status,
                    "lines_added":   None,
                    "lines_removed": None,
                })
            elif len(parts) >= 2:
                files.append({
                    "path":          parts[1],
                    "old_path":      None,
                    "status":        status,
                    "lines_added":   None,
                    "lines_removed": None,
                })

        return files

    # -------------------------------------------------------------------------
    # Output formatting
    # -------------------------------------------------------------------------

    def format_output(self, result: ToolResult) -> str:
        if not result.success:
            return f"❌ git_tool error: {result.error}"

        meta   = result.metadata or {}
        action = meta.get("action", "diff")

        if action == "list_repositories":
            return self._format_list_repositories(result)
        elif action == "scan_conflicts":
            return self._format_scan_conflicts(result)
        elif action == "list_branches":
            return self._format_list_branches(result)
        elif action == "get_branch_changes":
            return self._format_get_branch_changes(result)
        else:
            return self._format_diff(result)

    def _format_list_repositories(self, result: ToolResult) -> str:
        data  = result.data or {}
        meta  = result.metadata or {}
        repos = data.get("repositories", [])
        total = data.get("total_count", 0)

        lines = [
            f"📦 Repositorios en workspace: {meta.get('workspace', '')}",
            f"   Proveedor : {meta.get('provider', '')}",
            f"   Total     : {total} repositorio(s)",
            "",
        ]

        for r in repos:
            visibility = "🔒" if r.get("is_private") else "🌐"
            lang = f" [{r['language']}]" if r.get("language") else ""
            desc = f" — {r['description']}" if r.get("description") else ""
            updated = r.get("updated_on", "")[:10] if r.get("updated_on") else ""
            updated_str = f" (actualizado: {updated})" if updated else ""
            lines.append(f"  {visibility}  {r.get('full_name') or r.get('slug', '')}{lang}{desc}{updated_str}")

        return "\n".join(lines)

    def _format_scan_conflicts(self, result: ToolResult) -> str:
        data = result.data or {}
        meta = result.metadata or {}

        conflicted = data.get("conflicted", [])
        clean      = data.get("clean_with_changes", [])
        no_changes = data.get("no_changes", [])
        errors     = data.get("errors", [])

        repo = meta.get("repository", "")
        base = meta.get("compare_branch", "")

        lines = [
            f"🔍 Git Conflict Scan — {repo}",
            f"   Base branch: {base}",
            f"   Scanned: {meta.get('total_scanned', 0)} branches",
            "",
        ]

        if conflicted:
            lines.append(f"🔴 CONFLICTOS CON MASTER ({len(conflicted)}):")
            for b in conflicted:
                lines.append(f"  ⚠️  {b['branch']} — {b['conflict_count']} archivos en conflicto")
            lines.append("")

        if clean:
            lines.append(f"✅ BRANCHES LIMPIOS CON CAMBIOS ({len(clean)}):")
            for b in clean:
                lines.append(f"  ✔  {b['branch']} — {b['total_changes']} archivos modificados")
            lines.append("")

        if no_changes:
            lines.append(f"⬜ BRANCHES SIN CAMBIOS RESPECTO A {base.upper()} ({len(no_changes)}):")
            for b in no_changes[:10]:
                lines.append(f"  •  {b['branch']}")
            if len(no_changes) > 10:
                lines.append(f"  •  ... y {len(no_changes) - 10} más")
            lines.append("")

        if errors:
            lines.append(f"❌ ERRORES AL ESCANEAR ({len(errors)}):")
            for e in errors:
                lines.append(f"  •  {e.get('branch')}: {e.get('error')}")

        return "\n".join(lines)

    def _format_list_branches(self, result: ToolResult) -> str:
        data = result.data or {}
        meta = result.metadata or {}
        branches = data.get("branches", [])
        total = data.get("total_count", 0)

        lines = [
            f"🌿 Ramas en el repositorio: {meta.get('repository', '')}",
            f"   Proveedor : {meta.get('provider', '')}",
            f"   Total     : {total} rama(s)",
            "",
        ]

        for b in branches:
            lines.append(f"  • {b}")

        return "\n".join(lines)

    def _format_get_branch_changes(self, result: ToolResult) -> str:
        data  = result.data or {}
        files = data.get("changed_files", [])
        total = data.get("total_count", 0)
        meta  = result.metadata or {}

        lines = [
            f"📄 Cambios en rama: {meta.get('branch', '')}",
            f"   (Comparado con {meta.get('compare_branch', '')} en {meta.get('repository', '')})",
            f"   Total archivos: {total}",
            "",
        ]

        # Contar adiciones y eliminaciones si están disponibles
        total_added = sum((f.get("lines_added") or 0) for f in files)
        total_removed = sum((f.get("lines_removed") or 0) for f in files)
        if total_added > 0 or total_removed > 0:
            lines.append(f"   Líneas: +{total_added} -{total_removed}")
            lines.append("")

        status_icons = {
            "added":          "🟢 Added   ",
            "modified":       "🟡 Modified",
            "removed":        "🔴 Removed ",
            "renamed":        "🔵 Renamed ",
            "merge_conflict": "⚠️ CONFLICT",
            "unknown":        "⬜ Unknown ",
        }

        for f in files:
            icon = status_icons.get(f["status"], "⬜ Unknown ")
            path = f["path"]
            if f.get("old_path"):
                path = f"{f['old_path']} → {path}"

            lines_info = ""
            if f.get("lines_added") or f.get("lines_removed"):
                lines_info = f" (+{f.get('lines_added', 0)} -{f.get('lines_removed', 0)})"

            lines.append(f"  {icon} | {path}{lines_info}")

        return "\n".join(lines)

    def _format_diff(self, result: ToolResult) -> str:
        # Fallback al formato original de diff
        data  = result.data or {}
        files = data.get("changed_files", [])
        total = data.get("total_count", 0)
        meta  = result.metadata or {}

        lines = [
            f"🔀 Git diff — {meta.get('branch', '')} ← {meta.get('compare_branch', '')}",
            f"   Provider : {meta.get('provider', meta.get('mode', ''))}",
            f"   Total    : {total} file(s) changed",
            "",
        ]

        status_icons = {
            "added":          "🟢 A",
            "modified":       "🟡 M",
            "removed":        "🔴 D",
            "renamed":        "🔵 R",
            "merge_conflict": "⚠️ C",
            "unknown":        "⬜ ?",
        }

        for f in files:
            icon = status_icons.get(f["status"], "⬜ ?")
            path = f["path"]
            if f.get("old_path"):
                path = f"{f['old_path']} → {path}"
            lines.append(f"  {icon}  {path}")

        return "\n".join(lines)

