# =============================================================================
# src/tools/sftp_tool.py
# Herramienta SFTP para operaciones de transferencia de archivos
# =============================================================================
"""
Herramienta física para operaciones SFTP: listar, descargar, subir, eliminar,
renombrar, mover y cambiar permisos de archivos en servidores remotos.

Autenticación soportada:
- Usuario + contraseña (via variable de entorno o parámetro explícito)
- Clave privada SSH (ruta al archivo)

Características:
- Reintentos automáticos para conexiones fallidas (configurable)
- Timeouts por operación
- Streaming para archivos grandes
- Logging mediante el logger del proyecto
"""

import asyncio
import os
import stat
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.tools.base_tool import BaseTool, ToolCategory, ToolParameter, ToolResult
from src.utils.logger import get_logger


# =============================================================================
# Constantes
# =============================================================================

OPERATIONS = ["list", "download", "upload", "delete", "rename", "move", "permissions"]

# Tamaño de chunk para streaming (1 MB)
STREAM_CHUNK_SIZE = 1024 * 1024

# Máximo de reintentos de conexión
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY_S = 2.0
DEFAULT_TIMEOUT_S = 30


# =============================================================================
# Data classes auxiliares
# =============================================================================

@dataclass
class SFTPFileEntry:
    """Entrada de archivo/directorio en un listado SFTP."""
    name: str
    path: str
    is_directory: bool
    size: int
    permissions: str
    modified_time: Optional[str]


@dataclass
class SFTPListResult:
    """Resultado de la operación 'list'."""
    path: str
    entries: List[SFTPFileEntry] = field(default_factory=list)
    total_count: int = 0
    directories: int = 0
    files_count: int = 0


# =============================================================================
# SFTPTool
# =============================================================================

class SFTPTool(BaseTool):
    """
    Herramienta para operaciones SFTP sobre servidores remotos.

    Soporta: listar, descargar, subir, eliminar, renombrar, mover y
    cambiar permisos de archivos.

    La autenticación se puede realizar con contraseña o clave privada SSH.
    Las contraseñas nunca se loguean; se recomienda usar variables de entorno
    (SFTP_PASSWORD) en lugar de pasarlas como parámetro explícito.
    """

    auto_discover: bool = True

    def __init__(self):
        self.logger = get_logger(__name__)
        super().__init__()

    # =========================================================================
    # Metadatos
    # =========================================================================

    @property
    def name(self) -> str:
        return "sftp_tool"

    @property
    def description(self) -> str:
        return (
            "Herramienta para operaciones SFTP sobre servidores remotos. "
            "Soporta: listar directorio, descargar archivo, subir archivo, "
            "eliminar archivo, renombrar, mover y cambiar permisos."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.UTILITY

    @property
    def enabled_by_default(self) -> bool:
        return False

    @property
    def llm_hint(self) -> Optional[str]:
        return (
            "Un servidor SFTP está disponible. Usa 'sftp_tool' para listar archivos "
            "remotos, descargar, subir, eliminar, renombrar, mover o cambiar permisos. "
            "Siempre especifica host, username, operation y remote_path."
        )

    # content_prompt define cómo el LLM debe interpretar los resultados SFTP
    @property
    def content_prompt(self) -> str:
        return (
            "El resultado de la herramienta SFTP contiene información sobre archivos "
            "y directorios remotos. Interpreta los campos: 'entries' como listado de "
            "archivos (name, size, permissions, modified_time), 'message' como "
            "confirmación de operación completada, y 'error' como descripción del "
            "problema si la operación falló. Presenta los resultados de forma clara "
            "y concisa al usuario."
        )

    # =========================================================================
    # Parámetros
    # =========================================================================

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="host",
                type="string",
                description="Host o IP del servidor SFTP.",
                required=True,
                example="sftp.example.com"
            ),
            ToolParameter(
                name="username",
                type="string",
                description="Usuario de autenticación SFTP.",
                required=True,
                example="deploy"
            ),
            ToolParameter(
                name="operation",
                type="string",
                description=(
                    "Operación a realizar: "
                    "list=listar directorio, download=descargar archivo, "
                    "upload=subir archivo, delete=eliminar archivo, "
                    "rename=renombrar, move=mover, permissions=cambiar permisos."
                ),
                required=True,
                enum=OPERATIONS,
                example="list"
            ),
            ToolParameter(
                name="remote_path",
                type="string",
                description="Ruta remota del archivo o directorio.",
                required=True,
                example="/home/deploy/data"
            ),
            ToolParameter(
                name="port",
                type="integer",
                description="Puerto del servidor SFTP (default: 22).",
                required=False,
                default=22,
                example=22
            ),
            ToolParameter(
                name="password",
                type="string",
                description=(
                    "Contraseña de autenticación. Preferir variable de entorno "
                    "SFTP_PASSWORD en lugar de pasar este parámetro explícito."
                ),
                required=False,
                default=None
            ),
            ToolParameter(
                name="private_key_path",
                type="string",
                description="Ruta absoluta a la clave privada SSH (alternativa a password).",
                required=False,
                default=None,
                example="/home/user/.ssh/id_rsa"
            ),
            ToolParameter(
                name="local_path",
                type="string",
                description="Ruta local para download (destino) o upload (origen).",
                required=False,
                default=None,
                example="/tmp/downloaded_file.txt"
            ),
            ToolParameter(
                name="new_path",
                type="string",
                description="Nueva ruta remota para las operaciones rename y move.",
                required=False,
                default=None,
                example="/home/deploy/backup/file.txt"
            ),
            ToolParameter(
                name="permissions",
                type="string",
                description="Permisos en formato octal (ej: '755', '644') para la operación permissions.",
                required=False,
                default=None,
                example="755"
            ),
            ToolParameter(
                name="connection_timeout",
                type="integer",
                description=f"Timeout de conexión en segundos (default: {DEFAULT_TIMEOUT_S}).",
                required=False,
                default=DEFAULT_TIMEOUT_S,
                example=30
            ),
            ToolParameter(
                name="max_retries",
                type="integer",
                description=f"Número máximo de reintentos de conexión (default: {DEFAULT_MAX_RETRIES}).",
                required=False,
                default=DEFAULT_MAX_RETRIES,
                example=3
            ),
        ]

    # =========================================================================
    # Ejecución principal
    # =========================================================================

    async def execute(
        self,
        host: str,
        username: str,
        operation: str,
        remote_path: str,
        port: int = 22,
        password: Optional[str] = None,
        private_key_path: Optional[str] = None,
        local_path: Optional[str] = None,
        new_path: Optional[str] = None,
        permissions: Optional[str] = None,
        connection_timeout: int = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        **kwargs
    ) -> ToolResult:
        """Ejecuta la operación SFTP solicitada con reintentos automáticos."""

        # Resolver contraseña desde entorno si no se pasó explícitamente
        resolved_password = password or os.environ.get("SFTP_PASSWORD")

        # Ejecutar en un thread para no bloquear el event loop (paramiko es síncrono)
        return await asyncio.get_event_loop().run_in_executor(
            None,
            self._execute_sync,
            host, port, username, resolved_password, private_key_path,
            operation, remote_path, local_path, new_path, permissions,
            connection_timeout, max_retries
        )

    # =========================================================================
    # Implementación síncrona (ejecutada en thread pool)
    # =========================================================================

    def _execute_sync(
        self,
        host: str,
        port: int,
        username: str,
        password: Optional[str],
        private_key_path: Optional[str],
        operation: str,
        remote_path: str,
        local_path: Optional[str],
        new_path: Optional[str],
        permissions: Optional[str],
        connection_timeout: int,
        max_retries: int
    ) -> ToolResult:
        """Versión síncrona ejecutada en executor para no bloquear asyncio."""
        try:
            import paramiko
        except ImportError:
            return ToolResult(
                success=False,
                data=None,
                error=(
                    "La librería paramiko no está instalada. "
                    "Ejecuta: pip install paramiko>=3.0.0"
                )
            )

        client = None
        sftp = None
        last_error = ""

        for attempt in range(1, max_retries + 1):
            try:
                self.logger.info(
                    f"[sftp_tool] Conectando a {host}:{port} (intento {attempt}/{max_retries})"
                )
                client = self._create_ssh_client(
                    paramiko, host, port, username, password,
                    private_key_path, connection_timeout
                )
                sftp = client.open_sftp()
                sftp.get_channel().settimeout(connection_timeout)
                self.logger.info(f"[sftp_tool] Conexión establecida con {host}:{port}")
                break

            except Exception as e:
                last_error = str(e)
                self.logger.warning(
                    f"[sftp_tool] Intento {attempt}/{max_retries} fallido: {last_error}"
                )
                if client:
                    try:
                        client.close()
                    except Exception:
                        pass
                    client = None
                sftp = None

                if attempt < max_retries:
                    time.sleep(DEFAULT_RETRY_DELAY_S * attempt)  # backoff incremental

        if sftp is None:
            return ToolResult(
                success=False,
                data=None,
                error=f"No se pudo conectar a {host}:{port} tras {max_retries} intentos. "
                      f"Último error: {last_error}"
            )

        try:
            return self._dispatch_operation(
                sftp, operation, remote_path,
                local_path, new_path, permissions
            )
        except Exception as e:
            self.logger.error(f"[sftp_tool] Error en operación '{operation}': {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))
        finally:
            try:
                sftp.close()
            except Exception:
                pass
            try:
                client.close()
            except Exception:
                pass

    def _create_ssh_client(
        self,
        paramiko,
        host: str,
        port: int,
        username: str,
        password: Optional[str],
        private_key_path: Optional[str],
        timeout: int
    ):
        """Crea y configura el cliente SSH con la estrategia de auth disponible."""
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        connect_kwargs: Dict[str, Any] = {
            "hostname": host,
            "port": port,
            "username": username,
            "timeout": timeout,
            "banner_timeout": timeout,
            "auth_timeout": timeout,
        }

        if private_key_path:
            self.logger.debug(f"[sftp_tool] Auth: clave privada ({private_key_path})")
            connect_kwargs["key_filename"] = private_key_path
        elif password:
            self.logger.debug("[sftp_tool] Auth: contraseña (no se registra el valor)")
            connect_kwargs["password"] = password
        else:
            # Intentar autenticación por agente SSH / claves en ~/.ssh
            self.logger.debug("[sftp_tool] Auth: agente SSH / claves locales")
            connect_kwargs["look_for_keys"] = True
            connect_kwargs["allow_agent"] = True

        client.connect(**connect_kwargs)
        return client

    # =========================================================================
    # Despachador de operaciones
    # =========================================================================

    def _dispatch_operation(
        self,
        sftp,
        operation: str,
        remote_path: str,
        local_path: Optional[str],
        new_path: Optional[str],
        permissions: Optional[str]
    ) -> ToolResult:
        """Despacha la operación al método correspondiente."""
        op = operation.lower()

        if op == "list":
            return self._list_directory(sftp, remote_path)
        elif op == "download":
            if not local_path:
                return ToolResult(
                    success=False, data=None,
                    error="El parámetro 'local_path' es requerido para la operación 'download'."
                )
            return self._download_file(sftp, remote_path, local_path)
        elif op == "upload":
            if not local_path:
                return ToolResult(
                    success=False, data=None,
                    error="El parámetro 'local_path' es requerido para la operación 'upload'."
                )
            return self._upload_file(sftp, local_path, remote_path)
        elif op == "delete":
            return self._delete_file(sftp, remote_path)
        elif op == "rename":
            if not new_path:
                return ToolResult(
                    success=False, data=None,
                    error="El parámetro 'new_path' es requerido para la operación 'rename'."
                )
            return self._rename_file(sftp, remote_path, new_path)
        elif op == "move":
            if not new_path:
                return ToolResult(
                    success=False, data=None,
                    error="El parámetro 'new_path' es requerido para la operación 'move'."
                )
            return self._move_file(sftp, remote_path, new_path)
        elif op == "permissions":
            if not permissions:
                return ToolResult(
                    success=False, data=None,
                    error="El parámetro 'permissions' (ej: '755') es requerido para esta operación."
                )
            return self._change_permissions(sftp, remote_path, permissions)
        else:
            return ToolResult(
                success=False, data=None,
                error=f"Operación desconocida: '{operation}'. Válidas: {OPERATIONS}"
            )

    # =========================================================================
    # Operaciones individuales
    # =========================================================================

    def _list_directory(self, sftp, path: str) -> ToolResult:
        """Lista el contenido de un directorio remoto."""
        self.logger.info(f"[sftp_tool] LIST {path}")
        try:
            attrs_list = sftp.listdir_attr(path)
        except Exception as e:
            return ToolResult(success=False, data=None, error=f"No se pudo listar '{path}': {e}")

        entries: List[Dict[str, Any]] = []
        dirs, files = 0, 0

        for attr in attrs_list:
            is_dir = stat.S_ISDIR(attr.st_mode) if attr.st_mode else False
            perms = oct(stat.S_IMODE(attr.st_mode)) if attr.st_mode else "?"
            mtime = (
                datetime.utcfromtimestamp(attr.st_mtime).isoformat()
                if attr.st_mtime else None
            )
            entry = {
                "name": attr.filename,
                "path": f"{path.rstrip('/')}/{attr.filename}",
                "is_directory": is_dir,
                "size": attr.st_size or 0,
                "permissions": perms,
                "modified_time": mtime,
            }
            entries.append(entry)
            if is_dir:
                dirs += 1
            else:
                files += 1

        result_data = {
            "path": path,
            "entries": entries,
            "total_count": len(entries),
            "directories": dirs,
            "files_count": files,
        }

        return ToolResult(
            success=True,
            data=result_data,
            metadata={"operation": "list", "path": path, "total": len(entries)}
        )

    def _download_file(self, sftp, remote_path: str, local_path: str) -> ToolResult:
        """
        Descarga un archivo remoto a una ruta local.
        Para archivos grandes usa callback de progreso en chunks.
        """
        self.logger.info(f"[sftp_tool] DOWNLOAD {remote_path} → {local_path}")

        # Asegurar que el directorio local existe
        local_dir = os.path.dirname(local_path)
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)

        transferred_bytes = 0

        def _progress(transferred: int, total: int) -> None:
            nonlocal transferred_bytes
            transferred_bytes = transferred
            if total and total > 0:
                pct = int(transferred / total * 100)
                self.logger.debug(f"[sftp_tool] Descarga: {pct}% ({transferred}/{total} bytes)")

        try:
            file_stat = sftp.stat(remote_path)
            file_size = file_stat.st_size if file_stat else 0
        except Exception:
            file_size = 0

        try:
            sftp.get(remote_path, local_path, callback=_progress)
        except Exception as e:
            return ToolResult(
                success=False, data=None,
                error=f"Error al descargar '{remote_path}': {e}"
            )

        return ToolResult(
            success=True,
            data={
                "message": f"Archivo descargado exitosamente: {remote_path} → {local_path}",
                "remote_path": remote_path,
                "local_path": local_path,
                "bytes_transferred": transferred_bytes or file_size,
            },
            metadata={"operation": "download", "remote_path": remote_path, "local_path": local_path}
        )

    def _upload_file(self, sftp, local_path: str, remote_path: str) -> ToolResult:
        """
        Sube un archivo local a una ruta remota.
        Para archivos grandes usa callback de progreso en chunks.
        """
        self.logger.info(f"[sftp_tool] UPLOAD {local_path} → {remote_path}")

        if not os.path.exists(local_path):
            return ToolResult(
                success=False, data=None,
                error=f"El archivo local no existe: {local_path}"
            )

        local_size = os.path.getsize(local_path)
        transferred_bytes = 0

        def _progress(transferred: int, total: int) -> None:
            nonlocal transferred_bytes
            transferred_bytes = transferred
            if total and total > 0:
                pct = int(transferred / total * 100)
                self.logger.debug(f"[sftp_tool] Subida: {pct}% ({transferred}/{total} bytes)")

        try:
            sftp.put(local_path, remote_path, callback=_progress)
        except Exception as e:
            return ToolResult(
                success=False, data=None,
                error=f"Error al subir '{local_path}' a '{remote_path}': {e}"
            )

        return ToolResult(
            success=True,
            data={
                "message": f"Archivo subido exitosamente: {local_path} → {remote_path}",
                "local_path": local_path,
                "remote_path": remote_path,
                "bytes_transferred": transferred_bytes or local_size,
            },
            metadata={"operation": "upload", "local_path": local_path, "remote_path": remote_path}
        )

    def _delete_file(self, sftp, path: str) -> ToolResult:
        """Elimina un archivo remoto."""
        self.logger.info(f"[sftp_tool] DELETE {path}")
        try:
            sftp.remove(path)
        except Exception as e:
            # Intentar eliminar como directorio vacío
            try:
                sftp.rmdir(path)
            except Exception:
                return ToolResult(
                    success=False, data=None,
                    error=f"No se pudo eliminar '{path}': {e}"
                )

        return ToolResult(
            success=True,
            data={"message": f"Eliminado exitosamente: {path}", "path": path},
            metadata={"operation": "delete", "path": path}
        )

    def _rename_file(self, sftp, old_path: str, new_path: str) -> ToolResult:
        """Renombra un archivo o directorio remoto."""
        self.logger.info(f"[sftp_tool] RENAME {old_path} → {new_path}")
        try:
            sftp.rename(old_path, new_path)
        except Exception as e:
            return ToolResult(
                success=False, data=None,
                error=f"No se pudo renombrar '{old_path}' a '{new_path}': {e}"
            )

        return ToolResult(
            success=True,
            data={
                "message": f"Renombrado exitosamente: {old_path} → {new_path}",
                "old_path": old_path,
                "new_path": new_path,
            },
            metadata={"operation": "rename", "old_path": old_path, "new_path": new_path}
        )

    def _move_file(self, sftp, source: str, destination: str) -> ToolResult:
        """
        Mueve un archivo o directorio a otra ruta remota.
        SFTP no tiene operación move nativa; se implementa como rename.
        """
        self.logger.info(f"[sftp_tool] MOVE {source} → {destination}")
        try:
            sftp.rename(source, destination)
        except Exception as e:
            return ToolResult(
                success=False, data=None,
                error=f"No se pudo mover '{source}' a '{destination}': {e}"
            )

        return ToolResult(
            success=True,
            data={
                "message": f"Movido exitosamente: {source} → {destination}",
                "source": source,
                "destination": destination,
            },
            metadata={"operation": "move", "source": source, "destination": destination}
        )

    def _change_permissions(self, sftp, path: str, permissions: str) -> ToolResult:
        """Cambia los permisos de un archivo o directorio remoto."""
        self.logger.info(f"[sftp_tool] CHMOD {permissions} {path}")

        # Validar y convertir permisos de octal a entero
        permissions_clean = permissions.lstrip("0o").lstrip("0") or "0"
        if not permissions_clean.isdigit() or len(permissions) not in (3, 4):
            return ToolResult(
                success=False, data=None,
                error=f"Formato de permisos inválido: '{permissions}'. Use formato octal como '755' o '644'."
            )

        try:
            mode = int(permissions, 8)
        except ValueError:
            return ToolResult(
                success=False, data=None,
                error=f"Permisos inválidos: '{permissions}'. Deben ser dígitos octales (0-7)."
            )

        try:
            sftp.chmod(path, mode)
        except Exception as e:
            return ToolResult(
                success=False, data=None,
                error=f"No se pudo cambiar permisos de '{path}' a '{permissions}': {e}"
            )

        return ToolResult(
            success=True,
            data={
                "message": f"Permisos cambiados exitosamente: {path} → {permissions}",
                "path": path,
                "permissions": permissions,
                "mode_int": mode,
            },
            metadata={"operation": "permissions", "path": path, "permissions": permissions}
        )

    # =========================================================================
    # Formato de salida para el LLM
    # =========================================================================

    def format_output(self, result: ToolResult) -> str:
        """Formatea el resultado SFTP para consumo del LLM."""
        if not result.success:
            return f"❌ Error SFTP: {result.error}"

        data = result.data
        if not isinstance(data, dict):
            return str(data)

        # Listado de directorio
        if "entries" in data:
            lines = [f"📁 Directorio: {data['path']}",
                     f"   Total: {data['total_count']} elementos "
                     f"({data['directories']} dirs, {data['files_count']} archivos)"]
            for e in data["entries"]:
                icon = "📁" if e["is_directory"] else "📄"
                size_str = f"{e['size']:,} bytes" if not e["is_directory"] else ""
                lines.append(f"  {icon} {e['name']}  {e['permissions']}  {size_str}  {e.get('modified_time', '')}")
            return "\n".join(lines)

        # Otras operaciones: mostrar mensaje y detalles
        lines = []
        if "message" in data:
            lines.append(f"✅ {data['message']}")
        if "bytes_transferred" in data:
            lines.append(f"   Bytes transferidos: {data['bytes_transferred']:,}")
        return "\n".join(lines) if lines else str(data)
