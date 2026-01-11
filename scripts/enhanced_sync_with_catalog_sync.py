#!/usr/bin/env python3
"""
Versión optimizada de sincronización con modelos locales Ollama.
Optimizada para eficiencia: batch processing, caché, y consultas optimizadas.
"""

import sys
import os
from datetime import datetime
from typing import Dict, List, Set, Tuple
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.providers.manager import provider_manager
from src.models.llm_models import LLMModel
from src.models.model_catalog import model_catalog
from src.config.settings import settings


# Catálogo optimizado con modelos Ollama locales
OLLAMA_LOCAL_MODELS = {
    # Modelos de Propósito General (ordenados por eficiencia memoria/capacidad)
    "qwen2.5:3b": {
        "type": "general", "size_gb": 1.9, "cpu": True, "gpu": False,
        "parent_retrieval": True, "context": 32768, "thinking": False,
        "function_calling": True, "multimodal": False
    },
    "qwen3:1.7b": {
        "type": "general", "size_gb": 1.4, "cpu": True, "gpu": False,
        "parent_retrieval": True, "context": 32768, "thinking": False,
        "function_calling": True, "multimodal": False
    },
    "qwen3:4b": {
        "type": "general", "size_gb": 2.5, "cpu": True, "gpu": False,
        "parent_retrieval": True, "context": 32768, "thinking": False,
        "function_calling": True, "multimodal": False
    },
    "qwen3:8b": {
        "type": "general", "size_gb": 5.2, "cpu": True, "gpu": False,
        "parent_retrieval": True, "context": 32768, "thinking": False,
        "function_calling": True, "multimodal": False
    },
    "gemma3:1b": {
        "type": "general", "size_gb": 0.815, "cpu": True, "gpu": False,
        "parent_retrieval": False, "context": 8192, "thinking": False,
        "function_calling": False, "multimodal": False,
        "notes": "Muy ligero, ideal para tareas simples"
    },
    "gemma2:2b": {
        "type": "general", "size_gb": 1.6, "cpu": True, "gpu": False,
        "parent_retrieval": False, "context": 8192, "thinking": False,
        "function_calling": False, "multimodal": False
    },
    "gemma3:4b": {
        "type": "general", "size_gb": 3.3, "cpu": True, "gpu": False,
        "parent_retrieval": True, "context": 8192, "thinking": False,
        "function_calling": True, "multimodal": False
    },
    "gemma3:latest": {
        "type": "general", "size_gb": 3.3, "cpu": True, "gpu": False,
        "parent_retrieval": True, "context": 8192, "thinking": False,
        "function_calling": True, "multimodal": False
    },
    "gemma3n:latest": {
        "type": "general", "size_gb": 7.5, "cpu": False, "gpu": True,
        "parent_retrieval": True, "context": 8192, "thinking": False,
        "function_calling": True, "multimodal": False
    },
    "phi:latest": {
        "type": "general", "size_gb": 1.6, "cpu": True, "gpu": False,
        "parent_retrieval": True, "context": 4096, "thinking": False,
        "function_calling": False, "multimodal": False
    },
    "mistral:latest": {
        "type": "general", "size_gb": 4.4, "cpu": True, "gpu": False,
        "parent_retrieval": True, "context": 8192, "thinking": False,
        "function_calling": True, "multimodal": False
    },
    "llama3:latest": {
        "type": "general", "size_gb": 4.7, "cpu": True, "gpu": False,
        "parent_retrieval": True, "context": 8192, "thinking": False,
        "function_calling": True, "multimodal": False
    },

    # Modelos de Código
    "deepseek-coder:latest": {
        "type": "code", "size_gb": 0.776, "cpu": True, "gpu": False,
        "parent_retrieval": False, "context": 16384, "thinking": False,
        "function_calling": True, "multimodal": False,
        "notes": "Versión ligera para código"
    },
    "deepseek-coder:6.7b": {
        "type": "code", "size_gb": 3.8, "cpu": True, "gpu": False,
        "parent_retrieval": True, "context": 16384, "thinking": False,
        "function_calling": True, "multimodal": False
    },
    "deepseek-coder-v2:16b": {
        "type": "code", "size_gb": 8.9, "cpu": False, "gpu": True,
        "parent_retrieval": True, "context": 16384, "thinking": False,
        "function_calling": True, "multimodal": False,
        "notes": "Requiere GPU para óptimo rendimiento"
    },

    # Modelos de Razonamiento
    "deepseek-r1:1.5b": {
        "type": "reasoning", "size_gb": 1.1, "cpu": True, "gpu": False,
        "parent_retrieval": False, "context": 32768, "thinking": True,
        "function_calling": True, "multimodal": False,
        "notes": "Razonamiento ligero"
    },
    "deepseek-r1:8b": {
        "type": "reasoning", "size_gb": 5.2, "cpu": True, "gpu": False,
        "parent_retrieval": True, "context": 32768, "thinking": True,
        "function_calling": True, "multimodal": False,
        "notes": "Razonamiento avanzado, ideal para parent retrieval"
    },

    # Modelos de Embeddings (optimizados para parent retrieval)
    "nomic-embed-text:latest": {
        "type": "embedding", "size_gb": 0.274, "cpu": True, "gpu": False,
        "parent_retrieval": True, "context": 8192, "thinking": False,
        "function_calling": False, "multimodal": False,
        "notes": "Excelente para embeddings de texto"
    },
    "mxbai-embed-large:latest": {
        "type": "embedding", "size_gb": 0.669, "cpu": True, "gpu": False,
        "parent_retrieval": True, "context": 512, "thinking": False,
        "function_calling": False, "multimodal": False,
        "notes": "Embeddings de alta calidad"
    },
    "bge-m3:latest": {
        "type": "embedding", "size_gb": 1.2, "cpu": True, "gpu": False,
        "parent_retrieval": True, "context": 8192, "thinking": False,
        "function_calling": False, "multimodal": False,
        "notes": "Multilingüe, excelente para parent retrieval"
    },
    "embeddinggemma:latest": {
        "type": "embedding", "size_gb": 0.621, "cpu": True, "gpu": False,
        "parent_retrieval": True, "context": 8192, "thinking": False,
        "function_calling": False, "multimodal": False
    },
}


def get_model_base_name(full_name: str) -> str:
    """Extrae el nombre base del modelo (sin versión/tag)."""
    return full_name.split(':')[0] if ':' in full_name else full_name


def enhanced_sync_available_models_with_catalog_sync():
    """
    Sincronización optimizada con batch processing y caché.
    """
    print("🚀 Sincronización optimizada con modelos Ollama locales...")

    engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
    Session = sessionmaker(bind=engine)

    with Session() as session:
        try:
            # 1. Obtener modelos de proveedores (con manejo de errores)
            all_provider_models = []
            for provider_name, provider in provider_manager.providers.items():
                try:
                    models = provider.get_available_models()
                    all_provider_models.extend(models)
                    print(f" ✅ {provider_name.value}: {len(models)} modelos")
                except Exception as e:
                    print(f" ⚠️  {provider_name.value}: {e}")

            print(f"📊 Total obtenidos: {len(all_provider_models)}")

            # 2. Cargar modelos existentes en batch (optimización)
            existing_models_query = session.execute(
                select(LLMModel.provider, LLMModel.model_name, LLMModel.id)
            ).all()

            existing_lookup: Dict[Tuple[str, str], int] = {
                (row[0], row[1]): row[2] for row in existing_models_query
            }
            print(f"💾 Modelos en BD: {len(existing_lookup)}")

            # 3. Preparar operaciones en batch
            to_update: List[Tuple[int, Dict]] = []
            to_create: List[LLMModel] = []
            matched_catalog = 0
            unmatched_catalog = 0

            for model_info in all_provider_models:
                key = (model_info.provider.value, model_info.name)
                model_base = get_model_base_name(model_info.name)

                # Buscar en catálogo local primero (más rápido)
                catalog_spec = OLLAMA_LOCAL_MODELS.get(model_info.name)

                # Si no se encuentra, buscar en catálogo general
                if not catalog_spec:
                    try:
                        general_spec = model_catalog.get_model_spec(model_info.name)
                        if general_spec:
                            catalog_spec = {
                                "cpu": general_spec.cpu_supported,
                                "gpu": general_spec.gpu_required,
                                "parent_retrieval": general_spec.parent_retrieval_supported,
                                "type": general_spec.model_type,
                                "thinking": general_spec.supports_thinking,
                                "context": general_spec.context_window,
                                "function_calling": general_spec.supports_function_calling,
                                "multimodal": general_spec.is_multimodal,
                            }
                    except:
                        pass

                # Preparar datos
                if catalog_spec:
                    matched_catalog += 1
                    data = {
                        "cpu_supported": catalog_spec.get("cpu", True),
                        "gpu_required": catalog_spec.get("gpu", False),
                        "parent_retrieval_supported": catalog_spec.get("parent_retrieval", False),
                        "model_type": catalog_spec.get("type", "general"),
                        "supports_thinking": catalog_spec.get("thinking", False),
                        "context_window": catalog_spec.get("context") or model_info.context_window,
                        "supports_function_calling": catalog_spec.get("function_calling", False),
                    }
                else:
                    unmatched_catalog += 1
                    data = {
                        "cpu_supported": model_info.cpu_supported,
                        "gpu_required": model_info.gpu_required,
                        "parent_retrieval_supported": model_info.parent_retrieval_supported,
                        "model_type": model_info.model_type.value,
                        "supports_thinking": model_info.supports_thinking,
                        "context_window": model_info.context_window,
                        "supports_function_calling": model_info.supports_function_calling,
                    }

                # Agregar a lista de actualización o creación
                if key in existing_lookup:
                    to_update.append((existing_lookup[key], {**data, "last_seen": datetime.utcnow()}))
                else:
                    to_create.append(LLMModel(
                        provider=model_info.provider.value,
                        model_name=model_info.name,
                        supports_streaming=model_info.supports_streaming,
                        is_active=True,
                        **data
                    ))

            # 4. Ejecutar operaciones en batch (más eficiente)
            if to_update:
                for model_id, updates in to_update:
                    session.query(LLMModel).filter(LLMModel.id == model_id).update(updates)

            if to_create:
                session.add_all(to_create)

            session.commit()

            # 5. Estadísticas finales
            print(f"\n✅ Sincronización completada:")
            print(f"   📝 Actualizados: {len(to_update)}")
            print(f"   ➕ Creados: {len(to_create)}")
            print(f"   📚 Encontrados en catálogo: {matched_catalog}")
            print(f"   ⚠️  No en catálogo: {unmatched_catalog}")

            # 6. Resumen por categorías (una sola consulta)
            all_models = session.execute(select(LLMModel)).scalars().all()

            stats = {
                "total": len(all_models),
                "parent_retrieval": sum(1 for m in all_models if m.parent_retrieval_supported),
                "cpu_only": sum(1 for m in all_models if m.cpu_supported and not m.gpu_required),
                "gpu_required": sum(1 for m in all_models if m.gpu_required),
                "thinking": sum(1 for m in all_models if m.supports_thinking),
                "embeddings": sum(1 for m in all_models if m.model_type == "embedding"),
            }

            print(f"\n📊 Resumen de Base de Datos:")
            print(f"   Total: {stats['total']}")
            print(f"   Parent Retrieval: {stats['parent_retrieval']}")
            print(f"   Solo CPU: {stats['cpu_only']}")
            print(f"   Requiere GPU: {stats['gpu_required']}")
            print(f"   Con razonamiento: {stats['thinking']}")
            print(f"   Embeddings: {stats['embeddings']}")

            # 7. Top modelos recomendados por categoría
            print(f"\n🎯 Recomendaciones por categoría:")

            # Parent Retrieval (embeddings primero)
            pr_models = [m for m in all_models if m.parent_retrieval_supported]
            pr_embeddings = [m for m in pr_models if m.model_type == "embedding"]
            pr_reasoning = [m for m in pr_models if m.model_type == "reasoning"]

            if pr_embeddings:
                print(f"   🔍 Embeddings (parent retrieval): {', '.join(m.model_name for m in pr_embeddings[:3])}")
            if pr_reasoning:
                print(f"   🧠 Razonamiento: {', '.join(m.model_name for m in pr_reasoning[:2])}")

            # CPU eficientes
            cpu_models = sorted(
                [m for m in all_models if m.cpu_supported and not m.gpu_required and m.model_type == "general"],
                key=lambda x: x.model_name
            )
            if cpu_models:
                print(f"   💻 CPU ligeros: {', '.join(m.model_name for m in cpu_models[:3])}")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            session.rollback()
            raise


if __name__ == "__main__":
    print("🔧 Sincronización Optimizada de Modelos Ollama Locales")
    print("=" * 70)
    print("Optimizaciones:")
    print("  • Batch processing para DB operations")
    print("  • Caché de modelos existentes")
    print("  • Lookup directo en diccionario (O(1))")
    print("  • Catálogo local con 21 modelos")
    print()

    enhanced_sync_available_models_with_catalog_sync()
