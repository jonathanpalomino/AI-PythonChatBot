import json
import sys
from pathlib import Path

# Agregar el directorio raíz del proyecto al PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from src.document_loaders.word_loader import WordLoaderUniversal

# Habilitar procesamiento paralelo
loader = WordLoaderUniversal(
    enable_parallel=True,
    max_workers=4
)

result = loader.load(Path("C:/Users/pvjonat/Downloads/IM05064909_V2 - Evidencia de pruebas.docx"))

# Ver estadísticas completas
print(json.dumps(loader.stats, indent=2))