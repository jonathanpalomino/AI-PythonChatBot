#!/usr/bin/env python3
"""
Catálogo de Modelos LLM con Información Predefinida
Esta clase contiene información auxiliar predefinida para modelos conocidos
que no está disponible en la API de Ollama.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelSpec:
    """Especificaciones de un modelo LLM."""
    
    # Identificación
    name: str
    provider: str = "local"
    
    # Capacidad de hardware
    cpu_supported: bool = True
    gpu_required: bool = False
    memory_gb: Optional[float] = None  # Memoria RAM requerida en GB
    
    # Capacidad de RAG/Parent Retrieval
    parent_retrieval_supported: bool = True
    max_context_chunks: int = 5  # Máximo chunks recomendado para parent retrieval
    recommended_k: int = 3  # Valor recomendado de k para parent retrieval
    score_threshold: float = 0.6  # Threshold recomendado
    
    # Tipo de modelo
    model_type: str = "chat"  # chat, reasoning, vision, embedding
    supports_thinking: bool = False
    context_window: Optional[int] = None
    
    # Compatibilidad
    is_multimodal: bool = False
    supports_function_calling: bool = False
    
    # Notas y recomendaciones
    notes: str = ""
    recommendations: List[str] = field(default_factory=list)


class ModelCatalog:
    """Catálogo de especificaciones de modelos LLM."""
    
    def __init__(self):
        self.models: Dict[str, ModelSpec] = {}
        self._initialize_catalog()
    
    def _initialize_catalog(self):
        """Inicializa el catálogo con especificaciones predefinidas."""
        
        # Modelos EXCELENTES para Parent Retrieval
        self.models.update({
            # Modelos Grandes (7B+ parámetros) - Ideales para contextos largos
            "deepseek-r1:7b": ModelSpec(
                name="deepseek-r1:7b",
                memory_gb=4.7,
                parent_retrieval_supported=True,
                max_context_chunks=5,
                recommended_k=3,
                score_threshold=0.6,
                model_type="reasoning",
                supports_thinking=True,
                context_window=128000,
                supports_function_calling=True,
                notes="Arquitectura optimizada para RAG, excelente con contextos largos",
                recommendations=[
                    "Ideal para documentos extensos",
                    "Buen balance entre tamaño y performance",
                    "Recomendado para RAG avanzado"
                ]
            ),
            
            "deepseek-r1:8b": ModelSpec(
                name="deepseek-r1:8b",
                memory_gb=5.2,
                parent_retrieval_supported=True,
                max_context_chunks=5,
                recommended_k=3,
                score_threshold=0.6,
                model_type="reasoning",
                supports_thinking=True,
                context_window=128000,
                supports_function_calling=True,
                notes="Versión más grande, mejor manejo de contextos extensos",
                recommendations=[
                    "Mejor performance que 7b",
                    "Ideal para documentos muy extensos",
                    "Requiere más recursos"
                ]
            ),
            
            "qwen3:8b": ModelSpec(
                name="qwen3:8b",
                memory_gb=5.2,
                parent_retrieval_supported=True,
                max_context_chunks=5,
                recommended_k=3,
                score_threshold=0.6,
                model_type="reasoning",
                supports_thinking=True,
                context_window=128000,
                supports_function_calling=True,
                notes="Arquitectura moderna, buen soporte para contextos largos",
                recommendations=[
                    "Buen balance tamaño/rendimiento",
                    "Arquitectura Qwen3 optimizada",
                    "Recomendado para RAG general"
                ]
            ),
            
            "deepseek-coder-v2:16b": ModelSpec(
                name="deepseek-coder-v2:16b",
                memory_gb=8.9,
                cpu_supported=False,  # Demasiado grande para CPU
                gpu_required=True,
                parent_retrieval_supported=True,
                max_context_chunks=5,
                recommended_k=3,
                score_threshold=0.6,
                model_type="reasoning",
                supports_thinking=True,
                context_window=128000,
                supports_function_calling=True,
                notes="Excelente para documentos técnicos, gran capacidad de contexto",
                recommendations=[
                    "Ideal para documentos de código",
                    "Requiere GPU",
                    "Muy buena para RAG técnico"
                ]
            ),
            
            # Modelos Medianos con Buena Optimización
            "qwen3:4b": ModelSpec(
                name="qwen3:4b",
                memory_gb=2.5,
                parent_retrieval_supported=True,
                max_context_chunks=3,
                recommended_k=2,
                score_threshold=0.6,
                model_type="reasoning",
                supports_thinking=True,
                context_window=128000,
                supports_function_calling=True,
                notes="Arquitectura Qwen3 optimizada para eficiencia",
                recommendations=[
                    "Buen balance tamaño/rendimiento",
                    "Ideal para recursos limitados",
                    "Recomendado para RAG general"
                ]
            ),
            
            "gemma3:4b": ModelSpec(
                name="gemma3:4b",
                memory_gb=3.3,
                parent_retrieval_supported=False,  # NO RECOMENDADO para parent retrieval
                max_context_chunks=2,
                recommended_k=2,
                score_threshold=0.5,
                model_type="reasoning",
                supports_thinking=True,
                context_window=128000,
                supports_function_calling=True,
                notes="Buen balance, pero necesita GPU para buen performance",
                recommendations=[
                    "Desactivar parent retrieval",
                    "Usar solo con GPU",
                    "Cuidado con contextos largos en CPU"
                ]
            ),
            
            # Modelos MODERADOS para Parent Retrieval
            "qwen3:1.7b": ModelSpec(
                name="qwen3:1.7b",
                memory_gb=1.4,
                parent_retrieval_supported=True,
                max_context_chunks=2,
                recommended_k=2,
                score_threshold=0.6,
                model_type="reasoning",
                supports_thinking=True,
                context_window=128000,
                supports_function_calling=True,
                notes="Pequeño pero eficiente, usar con k=2 máximo",
                recommendations=[
                    "Solo para contextos cortos",
                    "Limitar a 2 chunks",
                    "Bueno para recursos limitados"
                ]
            ),
            
            "deepseek-r1:1.5b": ModelSpec(
                name="deepseek-r1:1.5b",
                memory_gb=1.1,
                parent_retrieval_supported=False,  # NO RECOMENDADO para parent retrieval
                max_context_chunks=1,
                recommended_k=1,
                score_threshold=0.5,
                model_type="reasoning",
                supports_thinking=True,
                context_window=128000,
                supports_function_calling=True,
                notes="Muy pequeño, solo para contextos muy cortos",
                recommendations=[
                    "Desactivar parent retrieval",
                    "Solo para tareas simples",
                    "Muy limitado con contextos largos"
                ]
            ),
            
            "qwen2.5:3b": ModelSpec(
                name="qwen2.5:3b",
                memory_gb=1.9,
                parent_retrieval_supported=False,  # NO RECOMENDADO para parent retrieval
                max_context_chunks=2,
                recommended_k=2,
                score_threshold=0.5,
                model_type="chat",
                supports_thinking=False,
                context_window=32768,
                supports_function_calling=False,
                notes="Arquitectura antigua, limitado con contextos largos",
                recommendations=[
                    "Desactivar parent retrieval",
                    "Solo para tareas simples",
                    "Limitado con contextos extensos"
                ]
            ),
            
            # Modelos NO RECOMENDADOS para Parent Retrieval
            "gemma3:270m": ModelSpec(
                name="gemma3:270m",
                memory_gb=0.3,
                parent_retrieval_supported=False,
                max_context_chunks=1,
                recommended_k=1,
                score_threshold=0.4,
                model_type="chat",
                supports_thinking=False,
                context_window=8192,
                supports_function_calling=False,
                notes="Demasiado pequeño, se ahoga con contextos largos",
                recommendations=[
                    "Desactivar parent retrieval",
                    "Solo para tareas muy simples",
                    "Muy limitado con contextos largos"
                ]
            ),
            
            "qwen2.5:1.5b": ModelSpec(
                name="qwen2.5:1.5b",
                memory_gb=0.9,
                parent_retrieval_supported=False,
                max_context_chunks=1,
                recommended_k=1,
                score_threshold=0.4,
                model_type="chat",
                supports_thinking=False,
                context_window=32768,
                supports_function_calling=False,
                notes="Muy pequeño para parent retrieval",
                recommendations=[
                    "Desactivar parent retrieval",
                    "Solo para tareas simples",
                    "Muy limitado con contextos largos"
                ]
            ),
            
            "gemma3:1b": ModelSpec(
                name="gemma3:1b",
                memory_gb=0.8,
                parent_retrieval_supported=False,
                max_context_chunks=1,
                recommended_k=1,
                score_threshold=0.4,
                model_type="chat",
                supports_thinking=False,
                context_window=8192,
                supports_function_calling=False,
                notes="Similar al anterior, mejor sin parent retrieval",
                recommendations=[
                    "Desactivar parent retrieval",
                    "Solo para tareas simples",
                    "Muy limitado con contextos largos"
                ]
            ),
            
            "deepseek-coder:latest": ModelSpec(
                name="deepseek-coder:latest",
                memory_gb=0.8,
                parent_retrieval_supported=False,
                max_context_chunks=1,
                recommended_k=1,
                score_threshold=0.4,
                model_type="code",
                supports_thinking=False,
                context_window=16384,
                supports_function_calling=False,
                notes="Modelo de código, no optimizado para RAG extenso",
                recommendations=[
                    "Desactivar parent retrieval",
                    "Solo para tareas de código",
                    "No recomendado para RAG general"
                ]
            ),
            
            # Modelos Especializados
            "qwen3-vl:4b": ModelSpec(
                name="qwen3-vl:4b",
                memory_gb=3.3,
                parent_retrieval_supported=False,
                max_context_chunks=2,
                recommended_k=2,
                score_threshold=0.5,
                model_type="multimodal",
                supports_thinking=True,
                context_window=128000,
                supports_function_calling=True,
                is_multimodal=True,
                notes="Modelo multimodal, no optimizado para texto puro",
                recommendations=[
                    "Desactivar parent retrieval",
                    "Solo para documentos multimodales",
                    "No recomendado para RAG de texto"
                ]
            ),
            
            # Modelos de OpenAI (para referencia)
            "gpt-4": ModelSpec(
                name="gpt-4",
                provider="openai",
                cpu_supported=False,
                gpu_required=True,
                parent_retrieval_supported=True,
                max_context_chunks=10,
                recommended_k=5,
                score_threshold=0.7,
                model_type="reasoning",
                supports_thinking=True,
                context_window=128000,
                supports_function_calling=True,
                notes="Modelo en la nube, excelente para RAG",
                recommendations=[
                    "Excelente para RAG avanzado",
                    "No requiere configuración local",
                    "Costo por uso"
                ]
            ),
            
            "gpt-3.5-turbo": ModelSpec(
                name="gpt-3.5-turbo",
                provider="openai",
                cpu_supported=False,
                gpu_required=True,
                parent_retrieval_supported=True,
                max_context_chunks=8,
                recommended_k=4,
                score_threshold=0.6,
                model_type="chat",
                supports_thinking=False,
                context_window=16385,
                supports_function_calling=True,
                notes="Modelo en la nube, buen balance costo/rendimiento",
                recommendations=[
                    "Bueno para RAG general",
                    "Costo más bajo que GPT-4",
                    "No requiere configuración local"
                ]
            ),
            
            # Modelos de Anthropic (para referencia)
            "claude-3-5-sonnet-20241022": ModelSpec(
                name="claude-3-5-sonnet-20241022",
                provider="anthropic",
                cpu_supported=False,
                gpu_required=True,
                parent_retrieval_supported=True,
                max_context_chunks=10,
                recommended_k=5,
                score_threshold=0.7,
                model_type="reasoning",
                supports_thinking=True,
                context_window=200000,
                supports_function_calling=True,
                notes="Modelo en la nube, excelente para RAG",
                recommendations=[
                    "Excelente para RAG avanzado",
                    "Contexto muy largo",
                    "No requiere configuración local"
                ]
            )
        })
    
    def get_model_spec(self, model_name: str) -> Optional[ModelSpec]:
        """Obtiene la especificación de un modelo por nombre."""
        return self.models.get(model_name.lower())
    
    def get_models_by_capability(self, capability: str, value: bool = True) -> List[ModelSpec]:
        """Obtiene modelos por capacidad específica."""
        return [spec for spec in self.models.values() if getattr(spec, capability, False) == value]
    
    def get_excellent_parent_retrieval_models(self) -> List[ModelSpec]:
        """Obtiene modelos excelentes para parent retrieval."""
        return self.get_models_by_capability("parent_retrieval_supported", True)
    
    def get_not_recommended_parent_retrieval_models(self) -> List[ModelSpec]:
        """Obtiene modelos no recomendados para parent retrieval."""
        return self.get_models_by_capability("parent_retrieval_supported", False)
    
    def get_cpu_supported_models(self) -> List[ModelSpec]:
        """Obtiene modelos compatibles con CPU."""
        return self.get_models_by_capability("cpu_supported", True)
    
    def get_gpu_required_models(self) -> List[ModelSpec]:
        """Obtiene modelos que requieren GPU."""
        return self.get_models_by_capability("gpu_required", True)
    
    def get_model_recommendations(self, model_name: str) -> List[str]:
        """Obtiene recomendaciones para un modelo específico."""
        spec = self.get_model_spec(model_name)
        if spec:
            return spec.recommendations
        return []
    
    def get_all_model_names(self) -> List[str]:
        """Obtiene todos los nombres de modelos en el catálogo."""
        return list(self.models.keys())
    
    def is_parent_retrieval_recommended(self, model_name: str) -> bool:
        """Indica si se recomienda parent retrieval para un modelo."""
        spec = self.get_model_spec(model_name)
        return spec.parent_retrieval_supported if spec else False
    
    def get_recommended_config(self, model_name: str) -> Dict:
        """Obtiene la configuración recomendada para un modelo."""
        spec = self.get_model_spec(model_name)
        if spec:
            return {
                "k": spec.recommended_k,
                "score_threshold": spec.score_threshold,
                "max_context_chunks": spec.max_context_chunks,
                "parent_retrieval_supported": spec.parent_retrieval_supported
            }
        return {
            "k": 3,
            "score_threshold": 0.6,
            "max_context_chunks": 5,
            "parent_retrieval_supported": True
        }


# Instancia global del catálogo
model_catalog = ModelCatalog()