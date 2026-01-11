#!/usr/bin/env python3
"""
Script para consultar el catálogo de modelos y obtener recomendaciones.
"""

import sys
import os

# Añadir el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models.model_catalog import model_catalog


def show_model_info(model_name: str):
    """Muestra información detallada de un modelo."""
    
    spec = model_catalog.get_model_spec(model_name)
    
    if not spec:
        print(f"❌ Modelo '{model_name}' no encontrado en el catálogo.")
        return
    
    print(f"\n🔍 Información del Modelo: {spec.name}")
    print("=" * 50)
    print(f"Proveedor: {spec.provider}")
    print(f"Memoria RAM: {spec.memory_gb} GB" if spec.memory_gb else "Memoria RAM: Desconocida")
    print(f"CPU compatible: {'✅' if spec.cpu_supported else '❌'}")
    print(f"GPU requerida: {'✅' if spec.gpu_required else '❌'}")
    print(f"Parent Retrieval: {'✅' if spec.parent_retrieval_supported else '❌'}")
    print(f"Tipo de modelo: {spec.model_type}")
    print(f"Pensamiento soportado: {'✅' if spec.supports_thinking else '❌'}")
    print(f"Contexto: {spec.context_window} tokens" if spec.context_window else "Contexto: Desconocido")
    print(f"Multimodal: {'✅' if spec.is_multimodal else '❌'}")
    print(f"Funciones: {'✅' if spec.supports_function_calling else '❌'}")
    
    if spec.notes:
        print(f"\n📝 Notas:")
        print(f"   {spec.notes}")
    
    if spec.recommendations:
        print(f"\n💡 Recomendaciones:")
        for i, rec in enumerate(spec.recommendations, 1):
            print(f"   {i}. {rec}")
    
    # Configuración recomendada
    config = model_catalog.get_recommended_config(model_name)
    print(f"\n⚙️ Configuración Recomendada:")
    print(f"   k: {config['k']}")
    print(f"   score_threshold: {config['score_threshold']}")
    print(f"   max_context_chunks: {config['max_context_chunks']}")
    print(f"   parent_retrieval_supported: {config['parent_retrieval_supported']}")


def show_models_by_category():
    """Muestra modelos por categorías específicas."""
    
    print("\n📋 Modelos por Categoría")
    print("=" * 50)
    
    # Modelos excelentes para parent retrieval
    excellent = model_catalog.get_excellent_parent_retrieval_models()
    print(f"\n🟢 EXCELENTES para Parent Retrieval ({len(excellent)}):")
    for model in excellent:
        print(f"   • {model.name} ({model.memory_gb} GB) - {model.notes}")
    
    # Modelos no recomendados para parent retrieval
    not_recommended = model_catalog.get_not_recommended_parent_retrieval_models()
    print(f"\n🔴 NO RECOMENDADOS para Parent Retrieval ({len(not_recommended)}):")
    for model in not_recommended:
        print(f"   • {model.name} ({model.memory_gb} GB) - {model.notes}")
    
    # Modelos compatibles con CPU
    cpu_models = model_catalog.get_cpu_supported_models()
    print(f"\n💻 Modelos compatibles con CPU ({len(cpu_models)}):")
    for model in cpu_models:
        print(f"   • {model.name} ({model.memory_gb} GB)")
    
    # Modelos que requieren GPU
    gpu_models = model_catalog.get_gpu_required_models()
    print(f"\n🎮 Modelos que requieren GPU ({len(gpu_models)}):")
    for model in gpu_models:
        print(f"   • {model.name} ({model.memory_gb} GB)")


def interactive_query():
    """Consulta interactiva del catálogo."""
    
    print("\n🎯 Consulta Interactiva del Catálogo")
    print("=" * 50)
    
    while True:
        print("\nOpciones:")
        print("1. Buscar modelo por nombre")
        print("2. Ver modelos por categoría")
        print("3. Ver todos los nombres de modelos")
        print("4. Salir")
        
        choice = input("\nSelecciona una opción (1-4): ").strip()
        
        if choice == '1':
            model_name = input("Ingresa el nombre del modelo: ").strip()
            show_model_info(model_name)
        
        elif choice == '2':
            show_models_by_category()
        
        elif choice == '3':
            all_names = model_catalog.get_all_model_names()
            print(f"\n📋 Todos los modelos en el catálogo ({len(all_names)}):")
            for name in sorted(all_names):
                spec = model_catalog.get_model_spec(name)
                memory = f" ({spec.memory_gb} GB)" if spec.memory_gb else ""
                print(f"   • {name}{memory}")
        
        elif choice == '4':
            print("👋 ¡Hasta luego!")
            break
        
        else:
            print("❌ Opción no válida. Por favor, selecciona 1-4.")


def main():
    """Script principal para consultar el catálogo."""
    
    print("📚 Consulta del Catálogo de Modelos LLM")
    print("=" * 50)
    print("Este catálogo contiene información predefinida sobre:")
    print("- Compatibilidad con CPU/GPU")
    print("- Soporte para Parent Retrieval")
    print("- Configuración recomendada")
    print("- Recomendaciones específicas")
    print()
    
    # Mostrar estadísticas generales
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
    
    # Preguntar si desea consultar
    response = input("\n¿Deseas consultar el catálogo? (s/n): ").lower().strip()
    
    if response in ['s', 'si', 'y', 'yes']:
        interactive_query()
    else:
        print("✅ Consulta omitida.")


if __name__ == "__main__":
    main()