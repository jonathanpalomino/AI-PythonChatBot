"""
Script de verificación para probar las extensiones dinámicas
"""
import sys

sys.path.insert(0, 'e:/Repositorio/BITBUCKET-PERSONAL/PythonChatBot')

from src.document_loaders import DocumentLoaderFactory
from src.config.settings import settings

print("=" * 80)
print("VERIFICACIÓN DE EXTENSIONES DINÁMICAS")
print("=" * 80)

# Test 1: Get supported extensions from factory
print("\n1. Extensiones soportadas por DocumentLoaderFactory:")
print("-" * 80)
extensions = DocumentLoaderFactory.get_supported_extensions()
print(f"Total extensiones: {len(extensions)}")
print(f"Extensiones: {sorted(extensions)}")

# Test 2: Get detailed loader info
print("\n2. Información detallada de loaders:")
print("-" * 80)
loader_info = DocumentLoaderFactory.get_loader_info()
print(f"Total loaders: {len(loader_info['loaders'])}")
for loader in loader_info['loaders']:
    print(f"  - {loader['name']}: {loader['extensions']}")

# Test 3: Get allowed extensions from settings
print("\n3. Extensiones permitidas desde Settings:")
print("-" * 80)
allowed = settings.get_allowed_extensions()
print(f"Total: {len(allowed)}")
print(f"Extensiones: {allowed}")

# Test 4: Compare
print("\n4. Verificación de sincronización:")
print("-" * 80)
factory_extensions = sorted(DocumentLoaderFactory.get_supported_extensions())
settings_extensions = settings.get_allowed_extensions()

if factory_extensions == settings_extensions:
    print("✅ Las extensiones están sincronizadas correctamente")
else:
    print("❌ Las extensiones NO están sincronizadas")
    print(f"Factory: {factory_extensions}")
    print(f"Settings: {settings_extensions}")

# Test 5: Verify new extensions are included
print("\n5. Verificación de nuevas extensiones incluidas:")
print("-" * 80)
new_extensions = ['.csv', '.html', '.htm', '.xlsx', '.xls']
for ext in new_extensions:
    if ext in factory_extensions:
        print(f"✅ {ext} está incluido")
    else:
        print(f"❌ {ext} NO está incluido")

print("\n" + "=" * 80)
print("VERIFICACIÓN COMPLETADA")
print("=" * 80)
