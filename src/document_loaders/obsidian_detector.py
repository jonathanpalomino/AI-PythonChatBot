# src/document_loaders/obsidian_detector.py
"""
Obsidian Vault Detection System
Detects whether a path is an Obsidian vault, single file, or subset
"""
from pathlib import Path
from typing import Optional, Dict, Literal, List
from dataclasses import dataclass
import json

from src.utils.logger import get_logger


@dataclass
class ObsidianContext:
    """Contexto de un archivo o vault de Obsidian"""
    type: Literal["vault", "single_file", "subset"]
    vault_root: Optional[Path]
    files: List[Path]
    is_obsidian: bool
    vault_config: Optional[Dict] = None

    def __repr__(self):
        return (f"ObsidianContext(type={self.type}, is_obsidian={self.is_obsidian}, "
                f"vault_root={self.vault_root}, files_count={len(self.files)})")


class ObsidianDetector:
    """Detecta si una ruta es un vault Obsidian o archivos individuales"""

    def __init__(self):
        self.logger = get_logger(__name__)

    def detect(self, path: Path) -> ObsidianContext:
        """
        Detecta el contexto de Obsidian desde una ruta

        Cases:
        1. Ruta es un vault completo (tiene .obsidian/)
        2. Ruta es un archivo dentro de un vault
        3. Ruta es carpeta con múltiples .md (sin .obsidian)
        4. Ruta es archivo .md suelto

        Args:
            path: Ruta a analizar (archivo o directorio)

        Returns:
            ObsidianContext con información del contexto detectado
        """
        path = Path(path).resolve()

        self.logger.debug(f"Detecting Obsidian context for: {path}")

        # CASO 1: La ruta ES un directorio con .obsidian
        if path.is_dir() and (path / ".obsidian").exists():
            self.logger.info(f"Detected complete Obsidian vault at: {path}")
            return self._detect_vault(path)

        # CASO 2: Archivo individual - buscar vault padre
        if path.is_file() and path.suffix == ".md":
            vault_root = self._find_parent_vault(path)

            if vault_root:
                self.logger.info(
                    f"File is part of Obsidian vault",
                    extra={
                        "file": path.name,
                        "vault_root": str(vault_root)
                    }
                )
                return ObsidianContext(
                    type="single_file",
                    vault_root=vault_root,
                    files=[path],
                    is_obsidian=True,
                    vault_config=self._read_vault_config(vault_root)
                )
            else:
                # Archivo markdown suelto (no es Obsidian)
                self.logger.debug(f"Standalone markdown file (not Obsidian): {path.name}")
                return ObsidianContext(
                    type="single_file",
                    vault_root=None,
                    files=[path],
                    is_obsidian=False
                )

        # CASO 3: Directorio con .md pero sin .obsidian
        if path.is_dir():
            md_files = list(path.rglob("*.md"))

            if md_files:
                # Buscar si algún padre es vault
                vault_root = self._find_parent_vault(path)

                is_obsidian = vault_root is not None
                context_type = "subset" if is_obsidian else "subset"

                self.logger.info(
                    f"Directory with markdown files",
                    extra={
                        "path": str(path),
                        "files_count": len(md_files),
                        "is_obsidian": is_obsidian,
                        "vault_root": str(vault_root) if vault_root else None
                    }
                )

                return ObsidianContext(
                    type=context_type,
                    vault_root=vault_root,
                    files=md_files,
                    is_obsidian=is_obsidian,
                    vault_config=self._read_vault_config(vault_root) if vault_root else None
                )

        # CASO 4: Ruta inválida o sin archivos
        self.logger.warning(f"Invalid path or no markdown files found: {path}")
        return ObsidianContext(
            type="single_file",
            vault_root=None,
            files=[],
            is_obsidian=False
        )

    def _detect_vault(self, vault_path: Path) -> ObsidianContext:
        """Detecta vault completo"""
        md_files = list(vault_path.rglob("*.md"))

        # Filtrar archivos en .obsidian/, .trash/ y otras carpetas ocultas
        md_files = [
            f for f in md_files
            if not any(p.startswith('.') for p in f.relative_to(vault_path).parts)
        ]

        self.logger.info(
            f"Detected Obsidian vault",
            extra={
                "path": str(vault_path),
                "notes_count": len(md_files),
                "total_md_files": len(list(vault_path.rglob("*.md")))
            }
        )

        return ObsidianContext(
            type="vault",
            vault_root=vault_path,
            files=md_files,
            is_obsidian=True,
            vault_config=self._read_vault_config(vault_path)
        )

    def _find_parent_vault(self, path: Path) -> Optional[Path]:
        """
        Busca recursivamente hacia arriba para encontrar .obsidian/

        Ejemplo:
        /home/user/vault/subfolder/nota.md
            -> busca en /home/user/vault/subfolder/.obsidian
            -> busca en /home/user/vault/.obsidian ✓ FOUND

        Args:
            path: Ruta inicial (archivo o directorio)

        Returns:
            Path del vault root o None si no se encuentra
        """
        current = path if path.is_dir() else path.parent

        # Limitar búsqueda a 5 niveles hacia arriba para evitar búsquedas infinitas
        max_depth = 5
        for level in range(max_depth):
            obsidian_dir = current / ".obsidian"

            if obsidian_dir.exists() and obsidian_dir.is_dir():
                self.logger.debug(
                    f"Found vault root at level {level}",
                    extra={"vault_root": str(current)}
                )
                return current

            # Subir un nivel
            parent = current.parent
            if parent == current:  # Llegamos a root del sistema
                break
            current = parent

        self.logger.debug(f"No vault root found after searching {max_depth} levels up")
        return None

    def _read_vault_config(self, vault_root: Path) -> Optional[Dict]:
        """
        Lee configuración del vault (.obsidian/app.json y workspace.json)

        Args:
            vault_root: Ruta raíz del vault

        Returns:
            Dict con configuración o None si no se puede leer
        """
        try:
            config = {}

            # Leer app.json (configuración general)
            app_config_file = vault_root / ".obsidian" / "app.json"
            if app_config_file.exists():
                with open(app_config_file, 'r', encoding='utf-8') as f:
                    config['app'] = json.load(f)

            # Leer workspace.json (estado actual del workspace)
            workspace_file = vault_root / ".obsidian" / "workspace.json"
            if workspace_file.exists():
                with open(workspace_file, 'r', encoding='utf-8') as f:
                    config['workspace'] = json.load(f)

            # Leer community-plugins.json
            plugins_file = vault_root / ".obsidian" / "community-plugins.json"
            if plugins_file.exists():
                with open(plugins_file, 'r', encoding='utf-8') as f:
                    config['plugins'] = json.load(f)

            if config:
                self.logger.debug(f"Read vault config with {len(config)} sections")
                return config

            return None

        except Exception as e:
            self.logger.warning(
                f"Could not read vault config: {e}",
                extra={"vault_root": str(vault_root)}
            )
            return None
