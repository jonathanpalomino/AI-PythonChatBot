#!/usr/bin/env python3
"""
Script Final: Configuración Completa de Modelos con Catálogo (Versión Síncrona)
Esta versión usa la sincronización síncrona para evitar problemas de concurrencia.
"""

import sys
import os

# Añadir el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config.settings import settings


def main():
    """Script principal para configuración completa con catálogo (versión síncrona)."""

    print("🚀 Script Final: Configuración Completa de Modelos con Catálogo (Versión Síncrona)")
    print("=" * 80)
    print("Esta versión utiliza el catálogo predefinido para poblar")
    print("los campos de hardware y capacidades que no están")
    print("disponibles en la API de Ollama.")
    print()

    # Mostrar estadísticas del catálogo
    from src.models.model_catalog import model_catalog

    total_models = len(model_catalog.get_all_model_names())
    excellent_parent = len(model_catalog.get_excellent_parent_retrieval_models())
    not_recommended_parent = len(model_catalog.get_not_recommended_parent_retrieval_models())
    cpu_models = len(model_catalog.get_cpu_supported_models())
    gpu_models = len(model_catalog.get_gpu_required_models())

    print(f"📊 Estadísticas del Catálogo:")
    print(f"   Total de modelos: {total_models}")
    print(f"   Excelentes para parent retrieval: {excellent_parent}")
    print(f"   No recomendados para parent retrieval: {not_recommended_parent}")
    print(f"   Compatibles con CPU: {cpu_models}")
    print(f"   Que requieren GPU: {gpu_models}")
    print()

    print("🔧 Base de datos: " + settings.DATABASE_URL)
    print()

    # Paso 1: Sincronización con catálogo (versión síncrona)
    print("📋 Paso 1: Sincronización con catálogo de modelos...")

    try:
        from enhanced_sync_with_catalog_sync import enhanced_sync_available_models_with_catalog_sync
        enhanced_sync_available_models_with_catalog_sync()
        print("✅ Sincronización completada exitosamente!")
    except Exception as e:
        print(f"❌ Error en sincronización: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    print()
    print("✅ Configuración completa con catálogo finalizada!")
    print()
    print("💡 Próximos pasos:")
    print("   - Implementa el API para mostrar estas capacidades en el frontend")
    print("   - Añade validación para prevenir modelos incompatibles")
    print("   - Documenta las capacidades de hardware de cada modelo en la UI")
    print()
    print("🎯 Beneficios del Catálogo:")
    print("   - Información predefinida y confiable")
    print("   - No depende de la API de Ollama")
    print("   - Fácil de mantener y actualizar")
    print("   - Permite agregar más campos en el futuro")
    print("   - No se pierde información al recrear la tabla")
    print("   - Versión síncrona: evita problemas de concurrencia")

    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Todo está funcionando correctamente!")
        print("Puedes usar este script para futuras configuraciones.")
    else:
        print("\n❌ Hay errores que deben corregirse.")
        sys.exit(1)
