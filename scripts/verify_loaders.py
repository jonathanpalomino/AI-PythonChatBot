"""
Script simple para verificar las extensiones soportadas
"""
print("=" * 80)
print("VERIFICACIÓN DE DOCUMENT LOADERS")
print("=" * 80)

import os
# Importar solo lo necesario
import sys

# Agregar path
sys.path.insert(0, os.path.abspath('.'))

try:
    # Solo importar el factory
    from src.document_loaders import DocumentLoaderFactory

    print("\n✅ DocumentLoaderFactory importado correctamente")

    # Test 1: Get supported extensions
    print("\n1. Extensiones soportadas:")
    print("-" * 80)
    extensions = DocumentLoaderFactory.get_supported_extensions()
    print(f"Total extensiones: {len(extensions)}")
    print(f"Extensiones: {sorted(extensions)}")

    # Test 2: Get loader info
    print("\n2. Información de loaders:")
    print("-" * 80)
    loader_info = DocumentLoaderFactory.get_loader_info()
    print(f"\nLoaders disponibles: {len(loader_info['loaders'])}")
    for loader in loader_info['loaders']:
        extensions_str = ', '.join(loader['extensions'])
        print(f"  • {loader['name']}: {extensions_str}")

    print(f"\nTodas las extensiones: {loader_info['all_extensions']}")

    # Test 3: Verify new extensions
    print("\n3. Verificación de nuevas extensiones:")
    print("-" * 80)
    new_extensions = {'.csv', '.html', '.htm', '.xlsx', '.xls'}
    factory_extensions = DocumentLoaderFactory.get_supported_extensions()

    for ext in new_extensions:
        status = "✅" if ext in factory_extensions else "❌"
        print(f"{status} {ext}")

    # Summary
    print("\n" + "=" * 80)
    all_present = new_extensions.issubset(factory_extensions)
    if all_present:
        print("✅ ÉXITO: Todas las nuevas extensiones están disponibles")
    else:
        missing = new_extensions - factory_extensions
        print(f"❌ FALTAN: {missing}")
    print("=" * 80)

except ImportError as e:
    print(f"\n❌ Error al importar: {e}")
    import traceback

    traceback.print_exc()
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback

    traceback.print_exc()
