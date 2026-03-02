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
"""

import asyncio
import json
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config.settings import settings
from src.providers.manager import provider_manager, ChatMessage
from src.services.intent.cache import IntentCache
from src.services.intent.config import (
    INTENT_REGISTRY,
    IntentDefinition,
    get_all_training_examples,
    get_intents_by_registered_tool,  # ← nuevo
    # ← nuevo
)
from src.services.intent.extractors import TargetExtractor
from src.utils.logger import get_logger

logger = get_logger(__name__)

# =============================================================================
# Result Dataclass
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
    3. Genera embedding de query
    4. Calcula cosine similarity con training examples
    5. Best match >= threshold? → return
    6. Fallback a LLM classification
    7. Extrae target si requires_target=True
    8. Guarda en cache y retorna

    Example:
        router = IntentRouter()
        result = await router.classify(
        ...     query="cuántos métodos tiene",
        ...     context={"attached_files": [...]}
        ... )
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
        Inicializa el Intent Router.

        Args:
            embedding_model_name: Nombre del modelo de SentenceTransformers (default: from settings)
            cache_size: Tamaño máximo del cache LRU (default: from settings)
            cache_ttl: Tiempo de vida del cache en segundos (default: from settings)
            similarity_threshold: Umbral mínimo para considerar match válido (default: from settings)
        """
        # Use settings defaults if not provided
        embedding_model_name = embedding_model_name or settings.INTENT_EMBEDDING_MODEL
        cache_size = cache_size or settings.INTENT_CACHE_SIZE
        cache_ttl = cache_ttl or settings.INTENT_CACHE_TTL
        similarity_threshold = similarity_threshold or settings.INTENT_SIMILARITY_THRESHOLD

        # Embedding model (local, fast)
        self.embedding_model = self._load_embedding_model(embedding_model_name)
        self.embedding_model_name = embedding_model_name

        # Target extractor
        self.target_extractor = TargetExtractor()

        # Cache
        self.cache = IntentCache(max_size=cache_size, ttl_seconds=cache_ttl)

        # Similarity threshold
        self.similarity_threshold = similarity_threshold

        # Pre-compute embeddings de training examples
        self.intent_embeddings = self._precompute_embeddings()

        # Statistics
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "embedding_hits": 0,
            "llm_fallbacks": 0,
            "errors": 0
        }

        logger.info(
            "IntentRouter initialized",
            extra={
                "intents_loaded": len(INTENT_REGISTRY),
                "embedding_model": embedding_model_name,
                "cache_size": cache_size,
                "similarity_threshold": similarity_threshold
            }
        )

    def _load_embedding_model(self, model_name: str) -> SentenceTransformer:
        """
        Carga el modelo de embeddings.

        Args:
            model_name: Nombre del modelo (HuggingFace)

        Returns:
            SentenceTransformer instance
        """
        try:
            import os
            from pathlib import Path

            # ✅ Cache persistente local (from settings)
            cache_dir = settings.INTENT_MODELS_CACHE_DIR
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Configurar variables de entorno ANTES de cargar
            os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
            os.environ['HF_HOME'] = str(cache_dir.parent)

            # ✅ Configurar proxy si existe
            if 'HTTP_PROXY' in os.environ:
                os.environ['http_proxy'] = os.environ['HTTP_PROXY']
                os.environ['https_proxy'] = os.environ.get('HTTPS_PROXY', os.environ['HTTP_PROXY'])

            logger.info(f"Loading embedding model: {model_name}")
            logger.info(f"Cache directory: {cache_dir}")

            # ✅ Intentar cargar primero en modo offline (sin conexión a internet)
            # Esto evita errores SSL cuando el modelo ya está en caché local
            try:
                model = SentenceTransformer(
                    model_name,
                    cache_folder=cache_dir,
                    local_files_only=True
                )
                logger.info("Embedding model loaded from local cache (offline mode)")
                return model
            except Exception as local_error:
                # Si falla la carga local, intentar con conexión
                logger.warning(
                    f"Model not found in local cache, attempting download: {local_error}"
                )
                model = SentenceTransformer(model_name, cache_folder=cache_dir)
                logger.info("Embedding model downloaded and loaded successfully")
                return model

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            raise RuntimeError(f"Cannot initialize IntentRouter without embedding model: {e}")

    def _precompute_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Pre-computa embeddings para todos los training examples.

        Esto se hace UNA VEZ al inicializar para optimizar performance.

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

            # Encode todos los ejemplos de este intent
            emb = self.embedding_model.encode(
                examples,
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
        llm_provider: Optional[str] = None,  # ✅ Desde settings
        llm_model: Optional[str] = None      # ✅ Desde settings
    ) -> IntentResult:
        """
        Clasifica un query del usuario.

        Args:
            query: Query del usuario (ej: "cuántos métodos tiene")
            context: Contexto adicional (archivos adjuntos, conversación, etc.)

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
            # Crear copia para no modificar el cache
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
            provider=llm_provider,  # ✅ Pasar provider
            model=llm_model  # ✅ Pasar model
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
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, ToolScore]:
        """
        Puntúa TODAS las tools habilitadas contra la query del usuario.

        Para cada tool en enabled_tool_names:
        - Busca sus intents en INTENT_REGISTRY (via get_intents_by_registered_tool)
        - Calcula similitud coseno entre la query y los ejemplos de cada intent
        - Asigna el score máximo encontrado
        - Determina si supera el confidence_threshold del intent ganador

        NO usa LLM fallback. Si embeddings no alcanzan umbral → passes_threshold=False.
        El Orchestrator aplica la regla de fallback a RAGTool si ninguna pasa.

        Args:
            query:              Query del usuario
            enabled_tool_names: Tools que el usuario habilitó en el frontal
            context:            Contexto adicional (no altera scoring, usado para logs)

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

        # Encode query (reutiliza modelo ya cargado)
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False
        )[0]

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

            # Extraer target solo si el intent lo requiere y el score pasó
            target = None
            if passes and best_intent_def.requires_target and best_intent_def.target_patterns:
                target = self.target_extractor.extract(query, best_intent_def.target_patterns)

            results[tool_name] = ToolScore(
                tool_name=tool_name,
                score=round(best_score, 4),
                best_intent=best_intent_def.name,
                best_intent_action=best_intent_def.action_name,
                passes_threshold=passes,
                requires_target=best_intent_def.requires_target,
                confidence_threshold=threshold,
                default_params=best_intent_def.default_params or {},
                target=target,
                method="embeddings"
            )

        elapsed_ms = (time.time() - start_time) * 1000

        # ─── LLM Near-Miss Fallback ────────────────────────────────────────────
        # Si NINGÚN tool superó su umbral pero hay un near-miss (score ≥ 0.45),
        # convocamos la clasificación LLM (que sí usa fallback) para decidir.
        # Esto cubre queries como "dame el codigo de create_template" (score~0.56)
        # sin overhead de LLM en queries ya claras.
        LLM_NEAR_MISS_THRESHOLD = 0.45
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
                llm_result = await self.classify(query, context)
                intent_name = llm_result.intent_name
                if intent_name in INTENT_REGISTRY:
                    tool_name = INTENT_REGISTRY[intent_name].target_tool
                    if tool_name in results:
                        ts = results[tool_name]
                        # Promover a "pasa": LLM tiene precedencia en near-miss
                        ts.passes_threshold = True
                        ts.method = "llm_near_miss"
                        ts.target = llm_result.target or ts.target
                        ts.best_intent = intent_name
                        ts.best_intent_action = INTENT_REGISTRY[intent_name].action_name
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

        # Cachear resultado
        # Nota: IntentCache.set espera IntentResult, usamos cache directamente
        # para no contaminar el cache de classify()
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

        Args:
            query: Query del usuario
            context: Contexto

        Returns:
            IntentResult o None si falla
        """
        try:
            # Generar embedding de la query
            query_emb = self.embedding_model.encode(
                [query],
                convert_to_numpy=True,
                show_progress_bar=False
            )[0]

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
            target = None
            if intent_def.requires_target:
                target = self.target_extractor.extract(query, intent_def.target_patterns)

            return IntentResult(
                intent_name=best_intent,
                intent_def=intent_def,
                confidence=best_score,
                target=target,
                reasoning=f"Embedding similarity: {best_score:.3f}",
                method="embeddings",
                execution_time_ms=0  # Will be set by caller
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
        provider: str = "local",    # ✅ NUEVO: Provider dinámico
        model: str = "qwen2.5:3b"   # ✅ NUEVO: Modelo dinámico
    ) -> IntentResult:
        """
        Clasificación con LLM (fallback cuando embeddings fallan).

        Args:
            query: Query del usuario
            context: Contexto

        Returns:
            IntentResult
        """
        prompt = self._build_llm_prompt(query, context)

        try:
            # ✅ Resolver provider dinámicamente
            llm_provider = provider_manager.get_provider(provider)

            logger.debug(f"LLM classification: {provider}/{model}")

            # Llamar al LLM
            response = await llm_provider.chat(
                messages=[ChatMessage(role="user", content=prompt)],  # ✅ Objeto ChatMessage
                model=model,
                temperature=0.1,
                max_tokens=300
            )

            # Parse respuesta
            intent_name, confidence, target, reasoning = self._parse_llm_response(
                response.content
            )

            # Validar que el intent existe
            intent_def = INTENT_REGISTRY.get(intent_name)
            if not intent_def:
                logger.warning(
                    f"LLM returned unknown intent: {intent_name}, "
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
                reasoning=reasoning or f"LLM classification",
                method="llm",
                execution_time_ms=0  # Will be set by caller
            )

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"LLM classification failed: {e}", exc_info=True)

            # Ultimate fallback: rag_search
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

        # Construir prompt
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

    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del router.

        Returns:
            Dict con estadísticas
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

    Esta función es thread-safe y garantiza una sola instancia.

    Returns:
        IntentRouter instance

    Example:
         router = await get_intent_router()
         result = await router.classify("cuántos métodos tiene")
    """
    global _router

    if _router is not None:
        return _router

    async with _router_lock:
        # Double-check después del lock
        if _router is None:
            _router = IntentRouter()
            logger.info("IntentRouter singleton initialized")

        return _router
