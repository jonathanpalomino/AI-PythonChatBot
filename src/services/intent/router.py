# =============================================================================
# src/services/intent/router.py
# Intent Router - Unified Intent Classification System
# =============================================================================
"""
IntentRouter: Clasificador híbrido de intents que combina:
1. Fast path: Embeddings con cosine similarity (40ms, 85% accuracy)
2. Smart fallback: LLM local (200ms, 95% accuracy)
3. Intelligent cache: LRU + TTL para queries repetidas

Este módulo REEMPLAZA:
- IntentClassifier (embeddings only)
- IntentClassifierLLM (LLM only)
- _detect_code_intent() en tool_executor

Arquitectura:
    User Query → Cache? → Embeddings → Threshold? → LLM → Result
                   ↓          ↓           ↓          ↓
                 <5ms       40ms        ✓/✗       200ms
User Query
    ↓
┌───────────────────────────────────┐
│  IntentRouter.classify()          │
└───────────────────────────────────┘
    ↓
┌───────────────┐
│ 1. Cache?     │ → HIT (< 5ms) ──────────→ Return
└───────────────┘
    ↓ MISS
┌───────────────────────────────────┐
│ 2. Embeddings + Cosine Similarity │
│    (40-60ms, 85% accuracy)        │
└───────────────────────────────────┘
    ↓
┌───────────────┐
│ Threshold OK? │ → YES (≥0.65) ───────────→ Return
└───────────────┘
    ↓ NO (confidence < threshold)
┌───────────────────────────────────┐
│ 3. LLM Fallback (qwen2.5:3b)     │
│    (180-250ms, 95% accuracy)      │
└───────────────────────────────────┘
    ↓
┌───────────────────────────────────┐
│ 4. Extract Target (si aplica)    │
└───────────────────────────────────┘
    ↓
┌───────────────────────────────────┐
│ 5. Cache Result & Return          │
└───────────────────────────────────┘

Performance esperado:
- Cache hit: <5ms
- Embedding hit: 40-60ms
- LLM fallback: 180-250ms
- P95 overall: <100ms (con cache warm)

CAMBIOS v2 (fixes de performance):
- FIX 1: __init__ ya NO carga el modelo — inicialización async via initialize()
          Evita bloquear el event loop ~1s en cada arranque del singleton.
- FIX 2: encode() ejecutado en ThreadPoolExecutor — libera el event loop
          durante los ~12-40ms de inferencia CPU del modelo de embeddings.
- FIX 3: score_tools_for_query pre-calienta el cache de classify() con el
          intent ganador — evita doble encode en path legacy.
- FIX 4: LLM_NEAR_MISS_THRESHOLD subido de 0.45 → 0.52 — elimina ~80% de
          los LLM fallbacks innecesarios en scores que ya son ruido puro.
"""

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List

import numpy as np

# =============================================================================
# PARCHE CRÍTICO para entornos con Zscaler/proxy corporativo
# Este parche debe aplicarse ANTES de importar sentence_transformers
# para evitar que transformers intente verificar modelos con HuggingFace
# =============================================================================
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Parchear la función is_base_mistral ANTES de que se importe transformers
# El error viene de tokenization_utils_tokenizers, no de tokenization_utils
try:
    # Intentar importar ambos posibles módulos
    try:
        import transformers.tokenization_utils_tokenizers as tok_tokenizers
        tok_tokenizers.is_base_mistral = lambda *args, **kwargs: False
        print("[INTENT ROUTER] Patched tokenization_utils_tokenizers.is_base_mistral")
    except ImportError:
        pass

    try:
        import transformers.utils.tokenization_utils as tok_utils
        tok_utils.is_base_mistral = lambda *args, **kwargs: False
        print("[INTENT ROUTER] Patched utils.tokenization_utils.is_base_mistral")
    except ImportError:
        pass

except Exception as e:
    print(f"[INTENT ROUTER] Warning: Could not patch is_base_mistral: {e}")

# Ahora importar sentence_transformers
from sentence_transformers import SentenceTransformer

from src.config.settings import settings
from src.providers.manager import ChatMessage
from src.providers.manager import provider_manager
from src.services.intent.cache import IntentCache
from src.services.intent.config import (
    INTENT_REGISTRY,
    IntentDefinition,
    get_all_training_examples,
    get_intents_by_registered_tool,
)
from src.services.intent.extractors import TargetExtractor
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ThreadPoolExecutor compartido para operaciones CPU-bound de embeddings.
# max_workers=2 es suficiente: el modelo es single-threaded internamente,
# y tener más workers solo aumenta contención en CPU.
_embedding_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="intent_embed")

# =============================================================================
# Result Dataclasses
# =============================================================================

@dataclass
class IntentResult:
    """
    Resultado de clasificación de intent.

    Attributes:
        intent_name: Nombre del intent clasificado (ej: "count_methods")
        intent_def: Definición completa del intent desde INTENT_REGISTRY
        confidence: Score de confidence (0.0-1.0)
        target: Símbolo extraído si requires_target=True (ej: "authenticate")
        reasoning: Explicación de cómo se clasificó (para debugging)
        method: Método usado ("cache", "embeddings", "llm", "fallback")
        execution_time_ms: Tiempo de ejecución en milisegundos
    """
    intent_name: str
    intent_def: IntentDefinition
    confidence: float
    target: Optional[str]
    reasoning: str
    method: str
    execution_time_ms: float


@dataclass
class ToolScore:
    """
    Resultado de scoring de una tool para una query específica.

    Retornado por IntentRouter.score_tools_for_query().
    El Orchestrator usa esto para decidir qué tools ejecutar.
    """
    tool_name: str
    score: float                          # Similitud máxima encontrada (0.0 - 1.0)
    best_intent: str                      # Intent con mayor score
    best_intent_action: Optional[str]     # action_name del intent ganador
    passes_threshold: bool                # True si score >= confidence_threshold
    requires_target: bool                 # El intent ganador necesita extraer target
    confidence_threshold: float           # Umbral aplicado
    default_params: Dict[str, Any] = field(default_factory=dict)
    target: Optional[str] = None          # Target extraído si requires_target=True
    method: str = "embeddings"            # Siempre embeddings, sin LLM fallback


# =============================================================================
# Intent Router Class
# =============================================================================

class IntentRouter:
    """
    Clasificador híbrido unificado de intents.

    Flow de clasificación:
    1. Genera cache key (query + context hash)
    2. Busca en cache → hit? return
    3. Genera embedding de query (en executor thread, no bloquea event loop)
    4. Calcula cosine similarity con training examples
    5. Best match >= threshold? → return
    6. Fallback a LLM classification
    7. Extrae target si requires_target=True
    8. Guarda en cache y retorna

    IMPORTANTE: Usar siempre via get_intent_router() (singleton async-safe).
    El constructor NO carga el modelo — llamar await initialize() antes de usar.

    Example:
        router = await get_intent_router()
        result = await router.classify(
            query="cuántos métodos tiene",
            context={"attached_files": [...]}
        )
        print(f"{result.intent_name}: {result.confidence:.2f}")
        count_methods: 0.87
    """

    def __init__(
        self,
        embedding_model_name: Optional[str] = None,
        cache_size: Optional[int] = None,
        cache_ttl: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ):
        """
        Constructor liviano: solo asigna atributos, NO carga el modelo.

        La carga del modelo (CPU-bound, ~1s) se hace en initialize() para
        no bloquear el event loop de asyncio durante el arranque del singleton.

        Args:
            embedding_model_name: Nombre del modelo de SentenceTransformers (default: from settings)
            cache_size: Tamaño máximo del cache LRU (default: from settings)
            cache_ttl: Tiempo de vida del cache en segundos (default: from settings)
            similarity_threshold: Umbral mínimo para considerar match válido (default: from settings)
        """
        # Usar settings defaults si no se proveen
        self.embedding_model_name = embedding_model_name or settings.INTENT_EMBEDDING_MODEL
        self._cache_size = cache_size or settings.INTENT_CACHE_SIZE
        self._cache_ttl = cache_ttl or settings.INTENT_CACHE_TTL
        self.similarity_threshold = similarity_threshold or settings.INTENT_SIMILARITY_THRESHOLD

        # Estos se populan en initialize()
        self.embedding_model: Optional[SentenceTransformer] = None
        self.intent_embeddings: Dict[str, np.ndarray] = {}

        # Target extractor (sin estado, inicialización inmediata OK)
        self.target_extractor = TargetExtractor()

        # Cache
        self.cache = IntentCache(max_size=self._cache_size, ttl_seconds=self._cache_ttl)

        # Statistics
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "embedding_hits": 0,
            "llm_fallbacks": 0,
            "errors": 0
        }

        # Flag de inicialización completa
        self._initialized = False

    async def initialize(self) -> None:
        """
        Inicialización async del modelo y embeddings.

        Ejecuta la carga del modelo SentenceTransformer y el pre-cómputo
        de embeddings en un ThreadPoolExecutor para no bloquear el event loop.

        Llamar UNA VEZ desde get_intent_router() antes de usar el router.
        Es idempotente: llamadas adicionales son no-op.
        """
        if self._initialized:
            return

        loop = asyncio.get_event_loop()

        # FIX 1: Carga del modelo en executor — CPU-bound, ~1s, no debe bloquear el loop
        self.embedding_model = await loop.run_in_executor(
            _embedding_executor,
            self._load_embedding_model_sync,
            self.embedding_model_name
        )

        # FIX 1: Pre-cómputo de embeddings también en executor
        # (puede tardar 200-500ms con muchos ejemplos)
        self.intent_embeddings = await loop.run_in_executor(
            _embedding_executor,
            self._precompute_embeddings_sync
        )

        self._initialized = True

        logger.info(
            "IntentRouter initialized",
            extra={
                "intents_loaded": len(INTENT_REGISTRY),
                "embedding_model": self.embedding_model_name,
                "cache_size": self._cache_size,
                "similarity_threshold": self.similarity_threshold
            }
        )

    def _load_embedding_model_sync(self, model_name: str) -> SentenceTransformer:
        """
        Carga el modelo de embeddings (síncrono, ejecutar en thread).

        Args:
            model_name: Nombre del modelo (HuggingFace)

        Returns:
            SentenceTransformer instance
        """
        try:
            import os
            from pathlib import Path

            # Cache persistente local (from settings)
            cache_dir = settings.INTENT_MODELS_CACHE_DIR
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Buscar si el modelo existe en alguna de las carpetas de caché
            model_cache_name = f"models--sentence-transformers--{model_name.replace('/', '--')}"
            resolved_cache_dir = cache_dir
            local_files_exist = False

            # Buscar primero en la personalizada y luego en la del sistema por defecto
            for c_dir in [cache_dir, Path.home() / ".cache" / "huggingface" / "hub"]:
                p = c_dir / model_cache_name
                if p.exists() and (p / "snapshots").exists():
                    resolved_cache_dir = c_dir
                    local_files_exist = True
                    logger.info(f"Model found in cache directory: {c_dir}")
                    break

            # Configurar variables de entorno ANTES de cargar con el cache resuelto
            os.environ['TRANSFORMERS_CACHE'] = str(resolved_cache_dir)
            os.environ['HF_HOME'] = str(resolved_cache_dir.parent)

            # Configurar proxy si existe
            if 'HTTP_PROXY' in os.environ:
                os.environ['http_proxy'] = os.environ['HTTP_PROXY']
                os.environ['https_proxy'] = os.environ.get('HTTPS_PROXY', os.environ['HTTP_PROXY'])

            logger.info(f"Loading embedding model: {model_name}")
            logger.info(f"Cache directory being used: {resolved_cache_dir}")

            if local_files_exist:
                model_cache_path = resolved_cache_dir / model_cache_name
                logger.info(f"Model found in local cache: {model_cache_path}")
                # FORZAR modo offline absoluto
                try:
                    model = SentenceTransformer(
                        str(model_cache_path),  # Usar path absoluto al cache
                        cache_folder=resolved_cache_dir,
                        device='cpu'
                    )
                    logger.info("Embedding model loaded from local cache (verified path)")
                    return model
                except Exception as load_error:
                    logger.warning(f"Failed to load from cache path, trying model name: {load_error}")
                    # Si falla, intentar con el nombre del modelo en modo offline
                    pass

            # Intentar cargar primero en modo offline (sin conexión a internet).
            # Esto evita errores SSL cuando el modelo ya está en caché local.
            try:
                model = SentenceTransformer(
                    model_name,
                    cache_folder=resolved_cache_dir,
                    local_files_only=True
                )
                logger.info("Embedding model loaded from local cache (offline mode)")
                return model
            except Exception as local_error:
                # Si no está en caché o falló offline, intentamos descargarlo online
                logger.warning(f"Model not loaded offline: {local_error}. Retrying online...")
                
                # Desactivar temporalmente variables de entorno offline
                old_hf_offline = os.environ.get('HF_HUB_OFFLINE')
                old_tf_offline = os.environ.get('TRANSFORMERS_OFFLINE')
                os.environ['HF_HUB_OFFLINE'] = '0'
                os.environ['TRANSFORMERS_OFFLINE'] = '0'
                
                try:
                    model = SentenceTransformer(
                        model_name,
                        cache_folder=resolved_cache_dir,
                        local_files_only=False
                    )
                    logger.info("Embedding model successfully downloaded and loaded online.")
                    return model
                except Exception as online_error:
                    logger.error(f"Failed to download/load embedding model online: {online_error}")
                    raise RuntimeError(
                        f"Cannot load embedding model. Both offline cache load and online download failed.\n"
                        f"Offline error: {local_error}\n"
                        f"Online error: {online_error}"
                    )
                finally:
                    # Restaurar variables de entorno offline
                    if old_hf_offline is not None:
                        os.environ['HF_HUB_OFFLINE'] = old_hf_offline
                    else:
                        del os.environ['HF_HUB_OFFLINE']
                    if old_tf_offline is not None:
                        os.environ['TRANSFORMERS_OFFLINE'] = old_tf_offline
                    else:
                        del os.environ['TRANSFORMERS_OFFLINE']

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            raise RuntimeError(f"Cannot initialize IntentRouter without embedding model: {e}")

    def _precompute_embeddings_sync(self) -> Dict[str, np.ndarray]:
        """
        Pre-computa embeddings para todos los training examples (síncrono, ejecutar en thread).

        Esto se hace UNA VEZ al inicializar para optimizar performance en runtime.
        El encode() batch aquí es intencional: es más eficiente que encode individual.

        Returns:
            Dict[intent_name, embeddings_array]
            donde embeddings_array shape: (num_examples, embedding_dim)
        """
        logger.info("Pre-computing intent embeddings...")

        training_examples = get_all_training_examples()
        embeddings = {}
        total_examples = 0

        for intent_name, examples in training_examples.items():
            if not examples:
                logger.warning(f"Intent '{intent_name}' has no training examples")
                continue

            # Obtener el IntentDefinition para acceder a sus patterns si requiere target
            intent_def = INTENT_REGISTRY.get(intent_name)

            processed_examples = examples
            # Si el intent extrae variables, enmascarar los ejemplos de entrenamiento
            if intent_def and intent_def.requires_target and intent_def.target_patterns:
                processed_examples = []
                for example in examples:
                    masked, _ = self.target_extractor.extract_and_mask(
                        example, intent_def.target_patterns
                    )
                    processed_examples.append(masked)

            # Encode batch de todos los ejemplos procesados de este intent
            emb = self.embedding_model.encode(
                processed_examples,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            embeddings[intent_name] = emb
            total_examples += len(examples)

            logger.debug(
                f"Precomputed embeddings for '{intent_name}': {len(examples)} examples"
            )

        logger.info(
            f"Embeddings ready: {len(embeddings)} intents, {total_examples} examples"
        )
        return embeddings

    async def classify(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None
    ) -> IntentResult:
        """
        Clasifica un query del usuario.

        Args:
            query: Query del usuario (ej: "cuántos métodos tiene")
            context: Contexto adicional (archivos adjuntos, conversación, etc.)
            llm_provider: Proveedor LLM (ej: "local", "openrouter")
            llm_model: Modelo LLM a usar (ej: "qwen3:4b")

        Returns:
            IntentResult con el intent clasificado

        Raises:
            ValueError: Si query está vacío
        """
        start_time = time.time()

        # Validación de inputs
        if not query or not isinstance(query, str):
            raise ValueError(f"Query must be non-empty string, got: {type(query)}")

        if context is None:
            context = {}

        self.stats["total_requests"] += 1

        # 1. Cache lookup
        cached = self.cache.get(query, context)
        if cached:
            self.stats["cache_hits"] += 1
            # Crear copia para no modificar el objeto cacheado
            result = deepcopy(cached)
            result.method = "cache"
            result.execution_time_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"Cache HIT: {result.intent_name} (conf={result.confidence:.2f})"
            )
            return result

        # 2. Fast path: Embeddings
        embedding_result = await self._classify_with_embeddings(query, context)

        if embedding_result and embedding_result.confidence >= self.similarity_threshold:
            self.stats["embedding_hits"] += 1
            embedding_result.execution_time_ms = (time.time() - start_time) * 1000
            # Guardar en cache
            self.cache.set(query, embedding_result, context)
            logger.info(
                f"Embedding HIT: {embedding_result.intent_name} "
                f"(conf={embedding_result.confidence:.2f}, "
                f"time={embedding_result.execution_time_ms:.0f}ms)"
            )
            return embedding_result

        logger.info(
            f"Embeddings low confidence ({embedding_result.confidence:.2f}), "
            f"trying LLM: {llm_provider}/{llm_model}"
        )

        # 3. Fallback: LLM
        self.stats["llm_fallbacks"] += 1
        llm_result = await self._classify_with_llm(
            query,
            context,
            provider=llm_provider,
            model=llm_model
        )
        llm_result.execution_time_ms = (time.time() - start_time) * 1000

        # Guardar en cache
        self.cache.set(query, llm_result, context)

        logger.info(
            f"LLM fallback: {llm_result.intent_name} "
            f"(conf={llm_result.confidence:.2f}, "
            f"time={llm_result.execution_time_ms:.0f}ms, "
            f"embedding_conf={embedding_result.confidence if embedding_result else 0:.2f})"
        )
        return llm_result

    async def score_tools_for_query(
        self,
        query: str,
        enabled_tool_names: List[str],
        context: Optional[Dict[str, Any]] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None
    ) -> Dict[str, ToolScore]:
        """
        Puntúa TODAS las tools habilitadas contra la query del usuario.

        Para cada tool en enabled_tool_names:
        - Busca sus intents en INTENT_REGISTRY (via get_intents_by_registered_tool)
        - Calcula similitud coseno entre la query y los ejemplos de cada intent
        - Asigna el score máximo encontrado
        - Determina si supera el confidence_threshold del intent ganador

        NO usa LLM fallback en el scoring base. Si embeddings no alcanzan
        umbral → passes_threshold=False. El near-miss LLM se activa solo
        si NINGUNA tool pasa y hay un candidato con score ≥ LLM_NEAR_MISS_THRESHOLD.

        El Orchestrator aplica la regla de fallback a RAGTool si ninguna pasa.

        Args:
            query:              Query del usuario
            enabled_tool_names: Tools que el usuario habilitó en el frontal
            context:            Contexto adicional (no altera scoring, usado para logs)
            llm_provider:       Proveedor LLM para near-miss fallback
            llm_model:          Modelo LLM para near-miss fallback

        Returns:
            Dict[tool_name, ToolScore] para TODAS las tools evaluadas.
            Incluye las que no pasaron el umbral (passes_threshold=False).
        """
        if not query or not enabled_tool_names:
            return {}

        start_time = time.time()

        # Cache key incluye enabled_tool_names para evitar cross-contamination
        cache_context = {
            **(context or {}),
            "_tools": sorted(enabled_tool_names)  # sorted para determinismo
        }
        cache_key_query = f"[SCORE]{query}"
        cached = self.cache.get(cache_key_query, cache_context)
        if cached and isinstance(cached, dict):
            logger.debug(f"ToolScore cache HIT: {query[:50]}")
            return cached

        # Aplicar Entity Masking global antes del embedding para scoring general:
        # Puesto que no sabemos aún el intent ganador, enmascaramos usando los patterns
        # de todos los intents de las tools habilitadas para maximizar similitud.
        masked_query_for_scoring = query
        extracted_global_target = None

        all_patterns_for_scoring = []
        for tool_name in enabled_tool_names:
            for intent_def in get_intents_by_registered_tool(tool_name):
                if intent_def.requires_target and intent_def.target_patterns:
                    all_patterns_for_scoring.extend(intent_def.target_patterns)

        if all_patterns_for_scoring:
            masked_query_for_scoring, extracted_global_target = self.target_extractor.extract_and_mask(
                query, all_patterns_for_scoring
            )

        # FIX 2: encode() en executor — libera event loop durante inferencia CPU
        loop = asyncio.get_event_loop()
        query_embedding = await loop.run_in_executor(
            _embedding_executor,
            lambda: self.embedding_model.encode(
                [masked_query_for_scoring],
                convert_to_numpy=True,
                show_progress_bar=False
            )[0]
        )

        results: Dict[str, ToolScore] = {}

        for tool_name in enabled_tool_names:
            tool_intents = get_intents_by_registered_tool(tool_name)

            if not tool_intents:
                # Tool sin intents declarados: score=0, no pasa
                results[tool_name] = ToolScore(
                    tool_name=tool_name,
                    score=0.0,
                    best_intent="none",
                    best_intent_action=None,
                    passes_threshold=False,
                    requires_target=False,
                    confidence_threshold=0.65,
                    method="no_intents_registered"
                )
                logger.debug(f"Tool '{tool_name}': no intents in registry, score=0.0")
                continue

            best_score = 0.0
            best_intent_def: Optional[IntentDefinition] = None

            for intent_def in tool_intents:
                if intent_def.name not in self.intent_embeddings:
                    continue

                # intent_embeddings[name] shape: (n_examples, embedding_dim)
                intent_embs = self.intent_embeddings[intent_def.name]
                similarities = self._cosine_similarity_batch(query_embedding, intent_embs)
                max_sim = float(similarities.max())

                if max_sim > best_score:
                    best_score = max_sim
                    best_intent_def = intent_def

            if best_intent_def is None:
                results[tool_name] = ToolScore(
                    tool_name=tool_name,
                    score=0.0,
                    best_intent="none",
                    best_intent_action=None,
                    passes_threshold=False,
                    requires_target=False,
                    confidence_threshold=0.65,
                    method="embeddings_miss"
                )
                continue

            threshold = best_intent_def.confidence_threshold
            passes = best_score >= threshold

            # Extraer target usando el preferido globalmente si hubo, o localmente
            target = extracted_global_target
            if passes and best_intent_def.requires_target and best_intent_def.target_patterns and not target:
                # Si falló globalmente, volver a intentar con el query original (fallback)
                target = self.target_extractor.extract(query, best_intent_def.target_patterns)

            # best_intent_action: la CodebaseAction canónica a ejecutar.
            # Se prefiere codebase_action (ej: "basic_analyze_file") sobre action_name
            # (ej: "count_methods") porque codebase_action es la acción real del tool.
            # El intent granular (best_intent_def.name = "count_methods") se propaga
            # por separado como "intent_name" hint desde el orchestrator para que
            # tool_executor lo use como sub_action en extract_tool_parameters.
            best_resolved_action = (
                best_intent_def.codebase_action
                if best_intent_def.codebase_action
                else best_intent_def.action_name
            )

            results[tool_name] = ToolScore(
                tool_name=tool_name,
                score=round(best_score, 4),
                best_intent=best_intent_def.name,
                best_intent_action=best_resolved_action,
                passes_threshold=passes,
                requires_target=best_intent_def.requires_target,
                confidence_threshold=threshold,
                default_params=best_intent_def.default_params or {},
                target=target,
                method="embeddings"
            )

        elapsed_ms = (time.time() - start_time) * 1000

        # FIX 3: Pre-calentar cache de classify() con el intent ganador.
        # Evita que una llamada posterior a classify() para la misma query
        # vuelva a hacer encode + similarity (path de compatibilidad legacy).
        winning_tool = max(
            (ts for ts in results.values() if ts.passes_threshold),
            key=lambda ts: ts.score,
            default=None
        )
        if winning_tool and winning_tool.best_intent and winning_tool.best_intent in INTENT_REGISTRY:
            intent_def = INTENT_REGISTRY[winning_tool.best_intent]
            synthetic_result = IntentResult(
                intent_name=winning_tool.best_intent,
                intent_def=intent_def,
                confidence=winning_tool.score,
                target=winning_tool.target,
                reasoning="Pre-cached from score_tools_for_query",
                method="embeddings",
                execution_time_ms=elapsed_ms
            )
            self.cache.set(query, synthetic_result, context or {})
            logger.debug(
                f"Pre-cached classify result: {winning_tool.best_intent} "
                f"(conf={winning_tool.score:.2f})"
            )

        # ─── LLM Near-Miss Fallback ────────────────────────────────────────────
        # Si NINGÚN tool superó su umbral pero hay un near-miss
        # (score ≥ LLM_NEAR_MISS_THRESHOLD), invocar LLM para decidir.
        #
        # FIX 4: Threshold subido de 0.45 → 0.52.
        # Con all-MiniLM-L6-v2 y 17 intents, scores < 0.52 son ruido estadístico
        # que el LLM tampoco puede resolver con más precisión, pero sí cuesta ~10s.
        # El rango [0.52, threshold) representa señal real que justifica el fallback.
        LLM_NEAR_MISS_THRESHOLD = 0.52  # era 0.45 — reduce ~80% de LLM fallbacks innecesarios
        no_tool_passed = not any(ts.passes_threshold for ts in results.values())
        near_miss_candidates = [
            ts for ts in results.values()
            if not ts.passes_threshold and ts.score >= LLM_NEAR_MISS_THRESHOLD
        ]

        if no_tool_passed and near_miss_candidates:
            logger.info(
                f"Near-miss detected (best score={max(ts.score for ts in near_miss_candidates):.3f}), "
                f"invoking LLM fallback for query='{query[:50]}'"
            )
            try:
                llm_result = await self.classify(
                    query, context,
                    llm_provider=llm_provider,
                    llm_model=llm_model
                )
                intent_name = llm_result.intent_name
                if intent_name in INTENT_REGISTRY:
                    tool_name = INTENT_REGISTRY[intent_name].target_tool
                    intent_def = INTENT_REGISTRY[intent_name]
                    if tool_name in results:
                        ts = results[tool_name]
                        # Promover a "pasa": LLM tiene precedencia en near-miss
                        ts.passes_threshold = True
                        ts.method = "llm_near_miss"
                        ts.target = llm_result.target or ts.target
                        ts.best_intent = intent_name
                        # Preferir codebase_action sobre action_name (mismo criterio que scoring)
                        ts.best_intent_action = (
                            intent_def.codebase_action
                            if intent_def.codebase_action
                            else intent_def.action_name
                        )
                        logger.info(
                            f"LLM near-miss promoted: tool={tool_name}, "
                            f"intent={intent_name}, target={ts.target}"
                        )
            except Exception as e:
                logger.warning(f"LLM near-miss fallback failed, continuing without it: {e}")
        # ──────────────────────────────────────────────────────────────────────

        # Log resumen de scoring
        scores_summary = ", ".join(
            f"{name}={ts.score:.2f}({'✓' if ts.passes_threshold else '✗'})"
            for name, ts in sorted(results.items(), key=lambda x: x[1].score, reverse=True)
        )
        logger.info(
            f"Tool scoring: [{scores_summary}] | "
            f"query='{query[:50]}' | time={elapsed_ms:.1f}ms"
        )

        # Cachear resultado de scoring separado del cache de classify()
        # para no contaminar el path de clasificación individual.
        self.cache._cache[
            self.cache._generate_key(cache_key_query, cache_context)
        ] = (results, __import__('datetime').datetime.now())

        return results

    def get_tools_above_threshold(
        self,
        tool_scores: Dict[str, ToolScore],
        rag_tool_name: str = "rag_search"
    ) -> List[ToolScore]:
        """
        Filtra y ordena tools que superaron su umbral.

        Regla de fallback: si NINGUNA tool pasa el umbral y rag_search
        está habilitado, retorna RAGTool como fallback universal.

        Args:
            tool_scores:   Resultado de score_tools_for_query()
            rag_tool_name: Nombre registrado del RAGTool

        Returns:
            Lista ordenada por score desc. Mínimo [rag_tool] si está disponible.
        """
        passing = [ts for ts in tool_scores.values() if ts.passes_threshold]
        passing.sort(key=lambda x: x.score, reverse=True)

        if passing:
            return passing

        # Fallback: RAGTool universal si está habilitado
        if rag_tool_name in tool_scores:
            rag_score = tool_scores[rag_tool_name]
            logger.info(
                f"No tools passed threshold. Falling back to '{rag_tool_name}' "
                f"(score={rag_score.score:.2f}) as universal fallback."
            )
            # Retornar RAGTool aunque no pasó el umbral
            return [rag_score]

        logger.warning(
            "No tools passed threshold and RAGTool not available. "
            "LLM will respond from conversation history only."
        )
        return []

    async def _classify_with_embeddings(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> Optional[IntentResult]:
        """
        Clasificación rápida usando embeddings + cosine similarity.

        FIX 2: encode() ejecutado en ThreadPoolExecutor para no bloquear
        el event loop durante los ~12-40ms de inferencia CPU.

        Args:
            query: Query del usuario
            context: Contexto

        Returns:
            IntentResult o None si falla
        """
        try:
            # En _classify_with_embeddings no sabemos el intent a priori pero tenemos el registro global

            # Recolectar patterns de todos los intents
            all_patterns = []
            for intent_def in INTENT_REGISTRY.values():
                if intent_def.requires_target and intent_def.target_patterns:
                    all_patterns.extend(intent_def.target_patterns)

            masked_query, extracted_target = self.target_extractor.extract_and_mask(
                query, all_patterns
            ) if all_patterns else (query, None)

            # FIX 2: encode en executor — libera event loop
            loop = asyncio.get_event_loop()
            query_emb = await loop.run_in_executor(
                _embedding_executor,
                lambda: self.embedding_model.encode(
                    [masked_query],
                    convert_to_numpy=True,
                    show_progress_bar=False
                )[0]
            )

            # Calcular similarity con todos los intents
            best_intent = None
            best_score = 0.0

            for intent_name, pattern_embeddings in self.intent_embeddings.items():
                # Cosine similarity con todos los ejemplos de este intent
                similarities = self._cosine_similarity_batch(query_emb, pattern_embeddings)
                max_sim = similarities.max()

                if max_sim > best_score:
                    best_score = float(max_sim)
                    best_intent = intent_name

            if not best_intent:
                return None

            # Get intent definition
            intent_def = INTENT_REGISTRY[best_intent]

            # Extract target si es necesario
            target = extracted_target
            if intent_def.requires_target and not target:
                target = self.target_extractor.extract(query, intent_def.target_patterns)

            return IntentResult(
                intent_name=best_intent,
                intent_def=intent_def,
                confidence=best_score,
                target=target,
                reasoning=f"Embedding similarity: {best_score:.3f}",
                method="embeddings",
                execution_time_ms=0  # Set by caller
            )

        except Exception as e:
            logger.warning(f"Embedding classification failed: {e}", exc_info=True)
            return None

    def _cosine_similarity_batch(
        self,
        query_emb: np.ndarray,
        pattern_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calcula cosine similarity eficientemente (vectorizado).

        Args:
            query_emb: Embedding de la query (shape: embedding_dim)
            pattern_embeddings: Embeddings de patterns (shape: num_patterns, embedding_dim)

        Returns:
            Array de similarities (shape: num_patterns)
        """
        # Normalizar vectores
        query_norm = query_emb / np.linalg.norm(query_emb)
        pattern_norms = pattern_embeddings / np.linalg.norm(
            pattern_embeddings,
            axis=1,
            keepdims=True
        )

        # Dot product (cosine similarity)
        similarities = np.dot(pattern_norms, query_norm)

        return similarities

    async def _classify_with_llm(
        self,
        query: str,
        context: Dict[str, Any],
        provider: str,
        model: str = "qwen2.5:3b"
    ) -> IntentResult:
        """
        Clasificación con LLM (fallback cuando embeddings no alcanzan umbral).

        Args:
            query: Query del usuario
            context: Contexto
            provider: Proveedor LLM (ej: "local", "openrouter")
            model: Modelo LLM a usar

        Returns:
            IntentResult
        """
        prompt = self._build_llm_prompt(query, context)

        try:
            # Resolver provider dinámicamente
            llm_provider = provider_manager.get_provider(provider)

            logger.debug(f"LLM classification: {provider}/{model}")

            # Llamar al LLM
            response = await llm_provider.chat(
                messages=[ChatMessage(role="user", content=prompt)],
                model=model,
                temperature=0.1,
                max_tokens=300
            )

            # Parse respuesta
            intent_name, confidence, target, reasoning = self._parse_llm_response(
                response.content
            )

            # Validar que el intent existe en el registry
            intent_def = INTENT_REGISTRY.get(intent_name)
            if not intent_def:
                logger.warning(
                    f"LLM returned unknown intent: '{intent_name}', "
                    f"falling back to rag_search"
                )
                intent_name = "rag_search"
                intent_def = INTENT_REGISTRY["rag_search"]
                confidence = 0.5

            # Si requiere target pero LLM no lo extrajo, intentar con regex
            if intent_def.requires_target and not target:
                target = self.target_extractor.extract(query, intent_def.target_patterns)

            return IntentResult(
                intent_name=intent_name,
                intent_def=intent_def,
                confidence=confidence,
                target=target,
                reasoning=reasoning or "LLM classification",
                method="llm",
                execution_time_ms=0  # Set by caller
            )

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"LLM classification failed: {e}", exc_info=True)
            return self._ultimate_fallback(query, context, error_msg=str(e))

    def _build_llm_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """
        Construye el prompt para el LLM.

        Args:
            query: Query del usuario
            context: Contexto

        Returns:
            Prompt string
        """
        # Construir lista de intents disponibles
        intent_descriptions = []
        for name, intent_def in INTENT_REGISTRY.items():
            intent_descriptions.append(f"  - {name}: {intent_def.description}")
        intents_list = "\n".join(intent_descriptions)

        # Construir contexto
        context_lines = []
        if context.get("attached_files"):
            count = len(context["attached_files"])
            context_lines.append(f"- Usuario adjuntó {count} archivo(s)")
        if context.get("file_names"):
            names = context["file_names"][:3]  # Primeros 3
            context_lines.append(f"- Archivos: {', '.join(names)}")
        if context.get("previous_files"):
            count = len(context["previous_files"])
            context_lines.append(f"- Conversación previa sobre {count} archivo(s)")

        context_str = "\n".join(context_lines) if context_lines else "- Sin contexto adicional"

        prompt = f"""Clasifica esta query en UNO de los intents disponibles.

**Query del usuario:** "{query}"

**Contexto:**
{context_str}

**Intents disponibles:**
{intents_list}

**Instrucciones:**
1. Selecciona el intent MÁS ESPECÍFICO que coincida con la query
2. Si menciona "contar/cuántos" → count_methods, count_classes
3. Si pide "listar/mostrar" → list_methods, list_classes
4. Si pide "código de X" → get_method_content, get_class_content
5. Si pide "buscar símbolo" → search_symbol
6. Si pide "analizar calidad" → analyze_quality
7. Si es búsqueda general → rag_search

**Output (JSON únicamente):**
{{
  "intent": "<intent_name>",
  "confidence": <0.0-1.0>,
  "target": "<symbol si aplica, null si no>",
  "reasoning": "<breve explicación>"
}}

Responde SOLO con el JSON, sin markdown ni explicaciones adicionales.
"""
        return prompt

    def _parse_llm_response(
        self,
        response: str
    ) -> Tuple[str, float, Optional[str], str]:
        """
        Parse respuesta del LLM.

        Args:
            response: Respuesta raw del LLM

        Returns:
            (intent_name, confidence, target, reasoning)
        """
        try:
            # Extraer JSON (puede venir con markdown ```json...)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)

                intent = data.get("intent", "rag_search")
                confidence = float(data.get("confidence", 0.6))
                target = data.get("target")
                reasoning = data.get("reasoning", "LLM classification")

                return intent, confidence, target, reasoning
            else:
                raise ValueError("No JSON found in LLM response")

        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            logger.debug(f"Response was: {response[:200]}")
            # Fallback parsing con heurística simple
            return self._heuristic_parse(response)

    def _heuristic_parse(self, response: str) -> Tuple[str, float, Optional[str], str]:
        """
        Heurística simple si JSON parse falla.

        Args:
            response: Respuesta del LLM

        Returns:
            (intent_name, confidence, target, reasoning)
        """
        response_lower = response.lower()

        # Buscar nombres de intents en la respuesta
        for intent_name in INTENT_REGISTRY.keys():
            if intent_name in response_lower:
                return intent_name, 0.5, None, "Heuristic parse from LLM response"

        # Default a rag_search
        return "rag_search", 0.4, None, "Fallback after parse failure"

    def _ultimate_fallback(
        self,
        query: str,
        context: Dict[str, Any],
        error_msg: str
    ) -> IntentResult:
        """
        Fallback final cuando todo falla.

        Args:
            query: Query original
            context: Contexto
            error_msg: Mensaje de error

        Returns:
            IntentResult con rag_search
        """
        return IntentResult(
            intent_name="rag_search",
            intent_def=INTENT_REGISTRY["rag_search"],
            confidence=0.3,
            target=None,
            reasoning=f"Ultimate fallback after error: {error_msg[:50]}",
            method="fallback",
            execution_time_ms=0
        )
    async def register_custom_tool(self, tool_name: str, examples: List[str], tool_type: str = "custom", intent_actions: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """
        Registra o actualiza dinámicamente un Custom Tool y calcula sus embeddings.

        Args:
            tool_name: Nombre de la herramienta
            examples: Lista de frases de ejemplo (legacy, para compatibilidad)
            tool_type: Tipo de herramienta (para el CodebaseAction si aplica)
            intent_actions: Mapping opcional de actions con sus examples y params
                         Formato: {"action_name": {"examples": [...], "default_params": {...}}}
        """
        # Si se proporciona intent_actions, registrar múltiples intents (uno por action)
        if intent_actions:
            for action_name, action_config in intent_actions.items():
                action_examples = action_config.get("examples", [])
                default_params = action_config.get("default_params", {})

                if action_examples:
                    await self._register_single_intent(
                        intent_name=f"custom_{tool_name}_{action_name}",
                        tool_name=tool_name,
                        examples=action_examples,
                        default_params=default_params,
                        description=f"Dynamic intent for {tool_name} - {action_name}",
                        action_name=action_name
                    )
            logger.info(f"Registered {len(intent_actions)} intents for custom tool '{tool_name}'")
            return

        # Legacy: single intent registration
        if not examples:
            logger.debug(f"Skipping registration for tool '{tool_name}': no examples provided")
            return

        await self._register_single_intent(
            intent_name=f"custom_{tool_name}",
            tool_name=tool_name,
            examples=examples,
            description=f"Dynamic intent for {tool_name}"
        )

    async def _register_single_intent(self, intent_name: str, tool_name: str, examples: List[str],
                                    default_params: Optional[Dict[str, Any]] = None,
                                    description: str = "",
                                    action_name: Optional[str] = None) -> None:
        """Registra un intent individual con sus embeddings"""
        from src.services.intent.config import IntentDefinition, IntentCategory, INTENT_REGISTRY

        intent_def = IntentDefinition(
            name=intent_name,
            category=IntentCategory.CONTENT,
            description=description or f"Dynamic intent for {tool_name}",
            target_tool=tool_name,
            action_name=action_name or tool_name,
            examples_es=examples,
            default_params=default_params or {},
            requires_thinking=False
        )

        # Registrar globalmente
        INTENT_REGISTRY[intent_name] = intent_def

        # Calcular embeddings (en executor)
        loop = asyncio.get_event_loop()
        emb = await loop.run_in_executor(
            _embedding_executor,
            lambda: self.embedding_model.encode(
                examples,
                convert_to_numpy=True,
                show_progress_bar=False
            )
        )

        # Guardar en memoria
        self.intent_embeddings[intent_name] = emb
        logger.info(f"Registered dynamic intent '{intent_name}' with {len(examples)} examples")

    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de uso del router.

        Returns:
            Dict con estadísticas de performance
        """
        total = self.stats["total_requests"]

        if total == 0:
            return {
                **self.stats,
                "cache_hit_rate": "0%",
                "embedding_hit_rate": "0%",
                "llm_fallback_rate": "0%",
                "error_rate": "0%"
            }

        return {
            **self.stats,
            "cache_hit_rate": f"{self.stats['cache_hits'] / total * 100:.1f}%",
            "embedding_hit_rate": f"{self.stats['embedding_hits'] / total * 100:.1f}%",
            "llm_fallback_rate": f"{self.stats['llm_fallbacks'] / total * 100:.1f}%",
            "error_rate": f"{self.stats['errors'] / total * 100:.1f}%",
            "cache_size": self.cache.size(),
            "intents_loaded": len(self.intent_embeddings)
        }

    def clear_cache(self) -> None:
        """Limpia el cache."""
        self.cache.clear()
        logger.info("Intent cache cleared")

    def reset_stats(self) -> None:
        """Resetea estadísticas."""
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "embedding_hits": 0,
            "llm_fallbacks": 0,
            "errors": 0
        }
        logger.info("Intent router stats reset")


# =============================================================================
# Singleton Factory
# =============================================================================

_router: Optional[IntentRouter] = None
_router_lock = asyncio.Lock()


async def get_intent_router() -> IntentRouter:
    """
    Obtiene o crea la instancia singleton de IntentRouter.

    Thread-safe: usa asyncio.Lock con double-check pattern para garantizar
    una sola instancia incluso bajo requests concurrentes al arranque.

    La inicialización (carga del modelo + pre-cómputo de embeddings) se
    ejecuta en un ThreadPoolExecutor para no bloquear el event loop.

    Returns:
        IntentRouter instance (ya inicializado)

    Example:
        router = await get_intent_router()
        result = await router.classify("cuántos métodos tiene")
    """
    global _router

    # Fast path: ya inicializado (99% de los casos en runtime)
    if _router is not None:
        return _router

    async with _router_lock:
        # Double-check obligatorio: otro task puede haber inicializado
        # mientras esperábamos el lock
        if _router is not None:
            return _router

        instance = IntentRouter()
        await instance.initialize()   # FIX 1: async, no bloquea event loop
        _router = instance
        logger.info("IntentRouter singleton initialized")

    return _router
