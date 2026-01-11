# =============================================================================
# src/sync/tracker.py
# =============================================================================
"""
Sistema de tracking para sincronizaciÃ³n incremental
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


class SyncTracker:
    """Rastrea cambios en documentos para sincronizaciÃ³n incremental"""

    def __init__(self, tracking_file: Path):
        self.tracking_file = tracking_file
        self.metadata: Dict[str, Dict] = {}

    def load(self):
        """Carga metadata existente"""
        try:
            with open(self.tracking_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            print(f"ðŸ“‹ Tracking cargado: {len(self.metadata)} archivos")
        except FileNotFoundError:
            print("ðŸ“‹ Iniciando tracking nuevo")
            self.metadata = {}

    def save(self):
        """Guarda metadata actualizada"""
        self.tracking_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.tracking_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        print(f"ðŸ’¾ Tracking guardado: {len(self.metadata)} archivos")

    @staticmethod
    def compute_hash(content: str) -> str:
        """Calcula hash SHA256 del contenido"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def has_changed(self, file_path: str, content: str) -> Tuple[bool, str]:
        """
        Verifica si un archivo ha cambiado

        Returns:
            (changed: bool, reason: str)
        """
        current_hash = self.compute_hash(content)
        tracked = self.metadata.get(file_path)

        if not tracked:
            return True, "nuevo"

        if tracked['hash'] != current_hash:
            return True, "modificado"

        return False, "sin_cambios"

    def update_file(self, file_path: str, content: str, point_ids: List[str]):
        """Actualiza informaciÃ³n de un archivo"""
        self.metadata[file_path] = {
            'hash': self.compute_hash(content),
            'lastModified': datetime.now().isoformat(),
            'pointIds': point_ids,
            'sections': len(point_ids)
        }

    def get_point_ids(self, file_path: str) -> List[str]:
        """Obtiene los IDs de vectores de un archivo"""
        return self.metadata.get(file_path, {}).get('pointIds', [])

    def remove_file(self, file_path: str):
        """Elimina un archivo del tracking"""
        self.metadata.pop(file_path, None)

    def get_tracked_files(self) -> List[str]:
        """Lista todos los archivos rastreados"""
        return list(self.metadata.keys())

    def get_stats(self) -> Dict:
        """Obtiene estadÃ­sticas del tracking"""
        if not self.metadata:
            return {
                'total_files': 0,
                'total_sections': 0,
                'total_vectors': 0
            }

        total_sections = sum(info['sections'] for info in self.metadata.values())
        total_vectors = sum(len(info['pointIds']) for info in self.metadata.values())

        return {
            'total_files': len(self.metadata),
            'total_sections': total_sections,
            'total_vectors': total_vectors
        }
