# test_word_loader.py
import sys
from pathlib import Path

# Agregar el directorio raíz del proyecto al PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from src.document_loaders.word_loader import WordLoaderUniversal

# =============================================================================
# FUNCIÓN DE TEST RÁPIDO
# =============================================================================

def test_nested_tables(file_path: str):
    """
    Función de test rápido para verificar extracción de tablas anidadas.
    
    Usage:
        python -c "from src.document_loaders.word_loader import test_nested_tables; test_nested_tables('documento.docx')"
    """
    from pathlib import Path
    import json
    
    print("=" * 80)
    print("TEST: Extracción de Tablas Anidadas")
    print("=" * 80)
    
    loader = WordLoaderUniversal(enable_parallel=False)
    
    # 1. Analizar estructura
    print("\n1. ANÁLISIS DE ESTRUCTURA:")
    print("-" * 80)
    structure = loader.debug_table_structure(Path(file_path))
    print(json.dumps(structure, indent=2, ensure_ascii=False))
    
    # 2. Extracción completa
    print("\n2. EXTRACCIÓN COMPLETA:")
    print("-" * 80)
    result = loader.load(Path(file_path))
    
    # 3. Mostrar estadísticas
    print("\n3. ESTADÍSTICAS:")
    print("-" * 80)
    print(json.dumps(loader.stats, indent=2))
    
    # 4. Mostrar contenido de tablas
    print("\n4. CONTENIDO DE SECCIONES:")
    print("-" * 80)
    for idx, section in enumerate(result.sections):
        print(f"\nSección {idx}: {section.title}")
        print("-" * 40)
        
        # Mostrar solo primeros 500 caracteres
        content_preview = section.content[:500] if len(section.content) > 500 else section.content
        print(content_preview)
        
        if len(section.content) > 500:
            print(f"\n... ({len(section.content) - 500} caracteres más)")
        
        # Buscar branches en el contenido
        if 'feature/' in section.content or 'branch' in section.content.lower():
            print("\n🎯 BRANCH ENCONTRADO EN ESTA SECCIÓN!")
            
            # Extraer líneas con 'feature/' o 'branch'
            for line in section.content.split('\n'):
                if 'feature/' in line or 'branch' in line.lower():
                    print(f"  → {line.strip()}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETADO")
    print("=" * 80)

# Ruta a tu documento
file_path = "C:/Users/pvjonat/Downloads/IM05064909_V2 - Evidencia de pruebas.docx"

# Ejecutar test completo
test_nested_tables(file_path)