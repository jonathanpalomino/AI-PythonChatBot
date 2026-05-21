# =============================================================================
# src/services/chat/tool_executor.py
# Tool Executor - Tool execution logic with Intent Router Integration
# =============================================================================
"""
ToolExecutor: Responsable de ejecutar herramientas y gestionar resultados.

REFACTORED: Sistema híbrido de extracción de parámetros

Responsabilidades (SRP):
- Ejecutar herramientas por nombre
- Gestionar configuraciones de herramientas
- Determinar estrategias de ejecución (USANDO IntentRouter)
- Extraer parámetros:
  - Tools conocidos → IntentRouter (embeddings + LLM fallback)
  - Custom tools → ParameterExtractor (LLM genérico)
- Formatear resultados de herramientas
- Manejar errores de ejecución

ARQUITECTURA ACTUAL:
- IntentRouter: Clasificador híbrido para tools conocidos
  - Cache integrado (< 5ms hit)
  - Embeddings fast-path (40-60ms, 85% accuracy)
  - LLM fallback (180-250ms, 95% accuracy)
  - Target extraction con regex patterns
- ParameterExtractor: Extracción genérica para custom tools
  - LLM con prompts optimizados
  - Cache de extracciones
  - Validación de parámetros
"""

import json
from typing import List, Dict, Any, Optional
from uuid import UUID

from src.config.intent_patterns import CodebaseAction
from src.models.models import Conversation
from src.repositories import FileRepository, CustomToolRepository, MessageRepository
from src.schemas.schemas import ConversationSettings
from src.services.intent.config import get_intent_to_codebase_action_cached
from src.services.intent.llm_classifier import LLMClassifier, get_llm_classifier
from src.services.intent.parameter_extractor import get_parameter_extractor
from src.services.intent.router import get_intent_router, IntentRouter
from src.tools.base_tool import tool_registry, BaseTool, ToolResult
from src.tools.custom_tool import CustomToolExecutor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ToolExecutor:
    """
    Responsable de ejecutar herramientas y gestionar sus resultados.

    REFACTORED: Usa IntentRouter para clasificación unificada.
    """

    def __init__(
        self,
        file_repo: FileRepository,
        custom_tool_repo: CustomToolRepository,
        message_repo: Optional[MessageRepository] = None
    ):
        self.file_repo = file_repo
        self.custom_tool_repo = custom_tool_repo
        self.message_repo = message_repo
        self.logger = get_logger(__name__)

        # NUEVO: Intent router (lazy loaded)
        self._intent_router: Optional[IntentRouter] = None

        # Custom tools cache
        self._custom_tools_cache: Dict[UUID, CustomToolExecutor] = {}

        self._llm_classifier: Optional[LLMClassifier] = None

    async def _get_intent_router(self) -> IntentRouter:
        """Lazy load intent router (singleton)."""
        if self._intent_router is None:
            self._intent_router = await get_intent_router()
            self.logger.info("Intent router initialized")
        return self._intent_router

    async def _get_llm_classifier(self) -> LLMClassifier:
        if self._llm_classifier is None:
            self._llm_classifier = await get_llm_classifier()
        return self._llm_classifier

    # =============================================================================
    # Tool Execution
    # =============================================================================

    async def execute_tool(
        self,
        tool_name: str,
        conversation: Conversation,
        collection_name: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute
            conversation: Conversation context
            collection_name: Optional collection name for RAG
            **kwargs: Tool-specific arguments

        Returns:
            ToolResult with execution outcome
        """
        # Get tool from registry
        tool = tool_registry.get(tool_name)
        if tool is None:
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool '{tool_name}' not found in registry"
            )

        try:
            # Inject dependencies based on tool type
            if hasattr(tool, 'file_repo') and tool.file_repo is None:
                tool.file_repo = self.file_repo

            # Inject file_repo in kwargs for tools that need it
            if 'file_repo' not in kwargs:
                kwargs['file_repo'] = self.file_repo

            # Execute tool
            result = await tool.execute(**kwargs)
            self.logger.info(
                f"Tool executed: {tool_name}",
                extra={"success": result.success}
            )
            return result

        except Exception as e:
            self.logger.error(f"Error executing tool {tool_name}: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=str(e)
            )

    async def execute_tool_with_context(
        self,
        tool: BaseTool,
        conversation: Conversation,
        collection_name: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """
        Centralized tool execution with context injection via reflection.

        Args:
            tool: Tool instance
            conversation: Conversation context
            collection_name: Optional collection name
            **kwargs: Tool arguments

        Returns:
            ToolResult
        """
        # ✅ Reflexión dinámica: Inyectar dependencias basadas en atributos de la tool
        tool_name = tool.name  # Derive for logging purposes
        common_deps = {
            'file_repo': self.file_repo,
            'conversation': conversation,
            'message_repo': self.message_repo,
            'collection_name': collection_name
        }

        for attr_name, dep_value in common_deps.items():
            if hasattr(tool, attr_name) and getattr(tool, attr_name, None) is None:
                setattr(tool, attr_name, dep_value)
                self.logger.debug(f"Injected {attr_name} into {tool_name}")

        # ✅ Inyectar en kwargs solo lo que la tool declara
        allowed_params = {param.name for param in tool.get_parameters()}

        # Para CustomToolExecutor: no filtrar por allowed_params — pasar todos los kwargs
        # para que la herramienta física reciba el contexto completo.
        from src.tools.custom_tool import CustomToolExecutor
        is_custom_wrapper = isinstance(tool, CustomToolExecutor)

        if is_custom_wrapper:
            # Custom tools: pasar todos los exec params; la herramienta física valida
            self.logger.info(
                f"[CustomToolFilter] tool={tool_name} | "
                f"allowed_params={allowed_params} | "
                f"passing {len(kwargs)} kwargs to physical tool"
            )
        else:
            # Herramientas físicas: filtrar por allowed_params
            before_keys = set(kwargs.keys())
            kwargs = {k: v for k, v in kwargs.items() if k in allowed_params or k in common_deps}
            dropped = before_keys - set(kwargs.keys())
            if dropped:
                self.logger.warning(
                    f"[ToolFilter] tool={tool_name} dropped keys: {sorted(dropped)} | "
                    f"allowed={sorted(allowed_params)}"
                )


        # Execute the tool
        return await tool.execute(**kwargs)

    async def execute_tools_batch(
        self,
        tool_calls: List[Dict[str, Any]],
        conversation: Conversation,
        collection_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple tool calls in batch.

        Args:
            tool_calls: List of tool call dicts with 'name', 'arguments', 'id'
            conversation: Conversation context
            collection_name: Optional collection name

        Returns:
            List of tool result dicts
        """
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("arguments", {})
            tool_id = tool_call.get("id", "")

            # Parse arguments if string
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except json.JSONDecodeError:
                    tool_args = {}

            # Get tool
            tool = tool_registry.get(tool_name)
            if tool:
                try:
                    result = await self.execute_tool_with_context(
                        tool=tool,
                        conversation=conversation,
                        collection_name=collection_name,
                        **tool_args
                    )
                    results.append({
                        "tool_call_id": tool_id,
                        "tool_name": tool_name,
                        "result": self.format_tool_result(result),
                        "success": result.success
                    })
                except Exception as e:
                    results.append({
                        "tool_call_id": tool_id,
                        "tool_name": tool_name,
                        "result": f"Error: {str(e)}",
                        "success": False
                    })
            else:
                results.append({
                    "tool_call_id": tool_id,
                    "tool_name": tool_name,
                    "result": f"Error: Tool '{tool_name}' not found in registry",
                    "success": False
                })

        return results

    # =============================================================================
    # Tool Discovery
    # =============================================================================

    def get_available_tools(
        self,
        settings: ConversationSettings
    ) -> List[str]:
        """
        Get list of available tools based on settings.

        Args:
            settings: Conversation settings

        Returns:
            List of tool names
        """
        enabled = settings.available_tools or settings.enabled_tools or []
        return list(enabled)

    def get_tool_definitions(
        self,
        tool_names: List[str],
        provider: str
    ) -> List[Dict[str, Any]]:
        """
        Get tool definitions in provider-specific format.

        Args:
            tool_names: List of tool names
            provider: Provider name (openai, anthropic, etc.)

        Returns:
            List of tool definitions
        """
        if provider == "openai":
            return tool_registry.to_openai_functions(tool_names)
        elif provider == "anthropic":
            return tool_registry.to_anthropic_tools(tool_names)
        else:
            # Return OpenAI format as default
            return tool_registry.to_openai_functions(tool_names)

    # =============================================================================
    # Execution Strategy (REFACTORED con IntentRouter)
    # =============================================================================

    async def determine_execution_strategy(
        self,
        tool: BaseTool,
        tool_name: str,
        user_message: str,
        rag_metadata: Dict[str, Any],
        file_ids: Optional[List[UUID]] = None,
        target_file_id: Optional[UUID] = None,
        settings: Optional[ConversationSettings] = None
    ) -> Dict[str, Any]:
        """
        Determina estrategia de ejecución usando IntentRouter.

        Args:
            tool: Tool instance
            tool_name: Tool name
            user_message: User query
            rag_metadata: RAG metadata (files, symbols, etc.) / execution_strategy del caller
            file_ids: Attached file IDs
            target_file_id: Target file ID for sticky context
            settings: ConversationSettings con provider/model del usuario

        Returns:
            Dict con:
            - use_fast_path: bool
            - action: Optional[str]
            - target: Optional[str]
            - file_ids: Optional[List[str]]
            - context_enrichment: Dict
            - intent_result: Optional[IntentResult]
        """
        strategy = {
            "use_fast_path": False,
            "action": None,
            "target": None,
            "file_ids": None,
            "context_enrichment": {},
            "intent_result": None
        }

        self.logger.info(
            "🔍 determine_execution_strategy (IntentRouter)",
            extra={
                "tool_name": tool_name,
                "file_ids_count": len(file_ids) if file_ids else 0,
                "has_target_file": bool(target_file_id)
            }
        )

        # Custom tools siempre usan LLM extraction
        if isinstance(tool, CustomToolExecutor):
            self.logger.debug(f"Custom tool detected: {tool_name} - extracting parameters")
            
            # Incorporate precomputed default params from the intent (e.g. action="list_repositories")
            precomputed_params = rag_metadata.get("intent_default_params") or {}
            if precomputed_params:
                strategy.setdefault("execution_params", {}).update(precomputed_params)
            
            # For custom tools, extract parameters using ParameterExtractor
            try:
                from src.services.intent.parameter_extractor import get_parameter_extractor
                extractor = await get_parameter_extractor()
                config = await tool._load_custom_tool_config()
                
                # Convertir config_schema properties a lista de parámetros
                tool_config_dict = config.configuration or {}
                properties = tool_config_dict.get("config_schema", {}).get("properties", {})
                parameters_list = []
                for prop_name, prop_data in properties.items():
                    param_def = {"name": prop_name}
                    param_def.update(prop_data)
                    parameters_list.append(param_def)
                
                extracted = await extractor.extract(
                    user_message=user_message,
                    parameters=parameters_list
                )
                strategy.setdefault("execution_params", {}).update(extracted)
                self.logger.debug(f"Extracted parameters for custom tool: {list(extracted.keys())}")
            except Exception as e:
                self.logger.warning(f"Failed to extract parameters for custom tool {tool_name}: {e}")
            return strategy

        # =========================================================
        # FAST-PATH PRECOMPUTADO: El orquestador ya clasificó el intent
        # en _plan_tool_execution (FASE 1). Si rag_metadata contiene
        # intent_action inyectado por _execute_tools_context (step 3c),
        # usarlo directamente y saltarse IntentRouter + LLM classifier.
        # Esto evita ~20s de re-clasificación redundante por turno.
        #
        # El orquestador inyecta estos valores en execution_strategy
        # con setdefault antes de llamar a este método:
        #   execution_strategy.setdefault("intent_action", tool_score.best_intent_action)
        #   execution_strategy.setdefault("intent_target", tool_score.target)
        #   execution_strategy.setdefault("intent_default_params", tool_score.default_params)
        # =========================================================
        precomputed_action = rag_metadata.get("intent_action")
        precomputed_target = rag_metadata.get("intent_target")
        precomputed_params = rag_metadata.get("intent_default_params") or {}

        if precomputed_action:
            from src.config.intent_patterns import is_valid_action
            self.logger.info(
                f"⚡ Fast-path from precomputed plan: "
                f"action={precomputed_action}, target={precomputed_target}"
            )
            # Si el intent_action es un CodebaseAction válido (ej: "basic_analyze_file",
            # "get_method_content", etc.), usarlo directo como action.
            # Si es un hint granular (ej: "count_methods") que no coincide con
            # ninguna CodebaseAction, mapear a BASIC_ANALYZE_FILE.
            if is_valid_action(precomputed_action):
                strategy["action"] = precomputed_action
            else:
                # Intents granulares (count_*, list_*, file_summary) → BASIC_ANALYZE_FILE
                # El intent detallado se pasa como sub_action al codebase_tool
                strategy["action"] = CodebaseAction.BASIC_ANALYZE_FILE

            strategy["use_fast_path"] = True
            strategy["target"] = precomputed_target
            strategy["intent_action"] = precomputed_action
            strategy["intent_target"] = precomputed_target

            # intent_name: nombre granular del intent (ej: "count_methods", "list_methods")
            # Se usa como sub_action en extract_tool_parameters cuando intent_action
            # es una CodebaseAction directa (ej: "basic_analyze_file").
            precomputed_intent_name = rag_metadata.get("intent_name")
            if precomputed_intent_name:
                strategy["intent_name"] = precomputed_intent_name

            if file_ids:
                strategy["file_ids"] = [str(fid) for fid in file_ids]
            elif target_file_id:
                strategy["file_ids"] = [str(target_file_id)]

            for k, v in precomputed_params.items():
                strategy.setdefault(k, v)

            return strategy  # Sale aquí — IntentRouter no se invoca

        # =========================================================
        # CLASIFICACIÓN EN TIEMPO REAL: Solo si no hay plan precomputado
        # (compatibilidad con agent mode o llamadas legacy sin plan)
        # =========================================================
        try:
            router = await self._get_intent_router()

            # Build context for router
            context = {
                "attached_files": file_ids if file_ids else [],
                "file_names": rag_metadata.get("files", []),
                "previous_files": rag_metadata.get("symbols", []),
                "target_file_id": target_file_id
            }

            # Extraer provider/model desde settings para usar el LLM del usuario
            llm_provider = None
            llm_model = None
            if settings:
                llm_provider = settings.provider
                llm_model = settings.model
                self.logger.debug(
                    f"Using user's LLM config: {llm_provider}/{llm_model}"
                )

            intent_result = await router.classify(
                user_message,
                context,
                llm_provider=llm_provider,
                llm_model=llm_model
            )

            # Store result in strategy
            strategy["intent_result"] = intent_result

            self.logger.info(
                f"Intent classified: {intent_result.intent_name} "
                f"(conf={intent_result.confidence:.2f}, "
                f"method={intent_result.method})"
            )

            # Map intent to CodebaseAction
            # Usa el mapeo centralizado desde INTENT_REGISTRY (ver src/services/intent/config.py)
            intent_to_codebase = get_intent_to_codebase_action_cached()

            action = intent_to_codebase.get(intent_result.intent_name)
            if action:
                strategy["use_fast_path"] = True
                strategy["action"] = action
                strategy[
                    "intent_action"] = intent_result.intent_name  # propaga intent granular para sub_action
                strategy["target"] = intent_result.target

                # Determinar file_ids según contexto
                if file_ids:
                    strategy["file_ids"] = [str(fid) for fid in file_ids]
                elif target_file_id:
                    strategy["file_ids"] = [str(target_file_id)]
                elif rag_metadata.get("files"):
                    strategy["context_enrichment"]["context_files"] = rag_metadata["files"]

                self.logger.info(
                    f"⚡ Fast-path via IntentRouter: "
                    f"intent={intent_result.intent_name}, "
                    f"action={action}, "
                    f"target={intent_result.target}, "
                    f"confidence={intent_result.confidence:.2f}"
                )

                return strategy
            else:
                # Embeddings no clasificaron con suficiente confianza → LLM classifier
                self.logger.info(
                    f"Intent '{intent_result.intent_name}' no mapeado o baja confianza "
                    f"({intent_result.confidence:.2f}) → escalando a LLM classifier"
                )
                try:
                    classifier = await self._get_llm_classifier()
                    llm_result = await classifier.classify(
                        user_message=user_message,
                        provider=llm_provider,
                        model=llm_model
                    )
                    llm_action = get_intent_to_codebase_action_cached().get(llm_result["intent"])
                    if llm_action:
                        strategy["use_fast_path"] = True
                        strategy["action"] = llm_action
                        strategy["intent_action"] = llm_result[
                            "intent"]  # propaga intent granular para sub_action
                        strategy["target"] = llm_result.get("target")
                        if file_ids:
                            strategy["file_ids"] = [str(fid) for fid in file_ids]
                        elif target_file_id:
                            strategy["file_ids"] = [str(target_file_id)]
                        self.logger.info(
                            f"⚡ LLM Classifier fast-path: "
                            f"intent={llm_result['intent']}, "
                            f"target={llm_result.get('target')}, "
                            f"confidence={llm_result.get('confidence', 0):.2f}"
                        )
                except Exception as e:
                    self.logger.warning(f"LLM classifier failed: {e}", exc_info=True)

        except Exception as e:
            self.logger.warning(f"IntentRouter classification failed: {e}", exc_info=True)
            # Continuar con fallback legacy si falla IntentRouter

        # PRIORIDAD 2: File_ids fallback (SOLO si NO hay action del IntentRouter)
        if not strategy["action"] and file_ids and len(file_ids) > 0 and tool_name in (
        "codebase_tool"):
            strategy["use_fast_path"] = True
            strategy["action"] = CodebaseAction.ANALYZE_FILE
            strategy["file_ids"] = [str(fid) for fid in file_ids]
            self.logger.info(f"📂 File_ids fallback: {len(file_ids)} files → ANALYZE_FILE")

        # PRIORIDAD 3: Sticky context (solo si NO hay action)
        elif not strategy["action"] and target_file_id and tool_name in ("codebase_tool"):
            strategy["use_fast_path"] = True
            strategy["action"] = CodebaseAction.ANALYZE_FILE
            strategy["file_ids"] = [str(target_file_id)]
            self.logger.info(f"🎯 Sticky fallback: {str(target_file_id)[:8]}...")

        return strategy

    # =============================================================================
    # Parameter Extraction
    # =============================================================================

    async def extract_tool_parameters(
        self,
        tool: BaseTool,
        tool_name: str,
        user_message: str,
        conversation: Conversation,
        settings: ConversationSettings,
        execution_strategy: Dict[str, Any],
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extrae parámetros usando fast-path o LLM según estrategia.

        Args:
            tool: Tool instance
            tool_name: Tool name
            user_message: User message
            conversation: Conversation
            settings: Settings
            execution_strategy: Execution strategy with action/target/file_ids
            collection_name: Optional collection name for RAG

        Returns:
            Dict con parámetros extraídos
        """
        params_to_extract = tool.get_parameters()

        self.logger.info(f"Extracting parameters for {tool_name}")
        self.logger.debug(f"execution_strategy: {execution_strategy}")

        # =========================================================
        # CASO 1: Fast-path activado
        # =========================================================
        if execution_strategy.get("use_fast_path"):
            extracted_params = {
                "action": execution_strategy["action"],
                "target": execution_strategy["target"]
            }

            # Agregar file_ids si están disponibles
            if execution_strategy.get("file_ids"):
                extracted_params["file_ids"] = execution_strategy["file_ids"]
                self.logger.info(f"Fast-path with {len(execution_strategy['file_ids'])} file_ids")

            # Agregar context enrichment
            extracted_params.update(execution_strategy.get("context_enrichment", {}))

            # Override con intent info más específico del orchestrator.
            # intent_action (ej: "basic_analyze_file") es la CodebaseAction canónica.
            # intent_name (ej: "count_methods") es el hint granular para sub_action.
            if execution_strategy.get("intent_action"):
                intent_act = execution_strategy["intent_action"]
                from src.config.intent_patterns import is_valid_action
                if is_valid_action(intent_act):
                    # Coincide con CodebaseAction directamente (basic_analyze_file, get_method_content, etc.)
                    extracted_params["action"] = intent_act
                    # Si hay un intent_name granular (count_methods, list_methods, etc.),
                    # usarlo como sub_action para que llm_formatter responda con granularidad.
                    intent_name_hint = execution_strategy.get("intent_name")
                    if intent_name_hint and intent_name_hint != intent_act:
                        extracted_params["sub_action"] = intent_name_hint
                else:
                    # Es un intent granular pasado directamente — pasa como hint de sub_action
                    extracted_params["sub_action"] = intent_act
            if execution_strategy.get("intent_target") and not extracted_params.get("target"):
                extracted_params["target"] = execution_strategy["intent_target"]
            if execution_strategy.get("intent_default_params"):
                for k, v in execution_strategy["intent_default_params"].items():
                    extracted_params.setdefault(k, v)

            # Agregar intent_result para debugging/logging
            if execution_strategy.get("intent_result"):
                intent_result = execution_strategy["intent_result"]
                self.logger.info(
                    f"Fast-path from intent: {intent_result.intent_name} "
                    f"(method={intent_result.method}, "
                    f"confidence={intent_result.confidence:.2f})"
                )

            self.logger.info(f"Fast-path extraction: {extracted_params}")

            # =====================================================================
            # EXPORT_REFACTORED: enrichment from conversation history.
            # new_content and method_name are NOT in the user's current message;
            # they live in the assistant's previous turn (the suggested code block).
            # =====================================================================
            from src.config.intent_patterns import CodebaseAction as _CA
            if extracted_params.get("action") == _CA.EXPORT_REFACTORED:
                await self._enrich_export_refactored_params(
                    extracted_params=extracted_params,
                    conversation=conversation
                )

            # Para RAG: asegurar query y collections
            if tool_name == "rag_search":
                if "query" not in extracted_params:
                    extracted_params["query"] = user_message

                if "collections" not in extracted_params:
                    extracted_params["collections"] = await self._resolve_rag_collections(
                        collection_name=collection_name,
                        conversation=conversation
                    )

                # Propagar filters inyectados por el orquestador (target_file_id → file_id filter)
                if "filters" not in extracted_params and execution_strategy.get("filters"):
                    extracted_params["filters"] = execution_strategy["filters"]

                self.logger.info(
                    f"RAG params added: query={extracted_params['query']}..., "
                    f"collections={extracted_params['collections']}"
                )

            return extracted_params

        # =========================================================
        # CASO 2: Sin parámetros para extraer
        # =========================================================
        if not params_to_extract:
            # Para RAG: asegurar que query y collections estén presentes
            if tool_name == "rag_search":
                collections = await self._resolve_rag_collections(
                    collection_name=collection_name,
                    conversation=conversation
                )
                extracted_params = {
                    "query": user_message,
                    "collections": collections
                }
                # Propagar filters si existen
                if execution_strategy.get("filters"):
                    extracted_params["filters"] = execution_strategy["filters"]

                self.logger.info(
                    f"RAG params (no params to extract): "
                    f"query={extracted_params['query'][:30]}..., "
                    f"collections={extracted_params['collections']}"
                )
                return extracted_params

            # Pasar filters si existen para otras tools
            if execution_strategy.get("filters"):
                self.logger.info(f"No params to extract, passing filters")
                return {"filters": execution_strategy["filters"]}

            # Custom tools sin config_schema: devolver execution_params + intent_default_params
            # (evita que el CASO 2 devuelva {} y descarte action, repository, etc.)
            if isinstance(tool, CustomToolExecutor):
                extracted_params = dict(execution_strategy.get("execution_params") or {})
                for k, v in (execution_strategy.get("intent_default_params") or {}).items():
                    extracted_params.setdefault(k, v)
                self.logger.info(
                    f"Custom tool (no config_schema): returning intent params "
                    f"{list(extracted_params.keys())}"
                )
                return extracted_params

            return {}

        # =========================================================
        # CASO 3: Custom Tools
        # =========================================================
        is_custom_tool = isinstance(tool, CustomToolExecutor)
        if is_custom_tool:
            self.logger.info(f"Retrieving pre-extracted parameters for custom tool: {tool_name}")
            # Base variables from FASE 1 (config_schema and intent_default_params)
            extracted_params = dict(execution_strategy.get("execution_params") or {})
            
            # Fallback/Complementary extraction for implicit parameters (like {{tags}} templates)
            if params_to_extract:
                self.logger.info(f"Using ParameterExtractor for custom tool implicit params: {tool_name}")
                try:
                    from src.services.intent.parameter_extractor import get_parameter_extractor
                    extractor = await get_parameter_extractor()
                    params_dict = []
                    for p in params_to_extract:
                        param_dict = {
                            "name": p.name,
                            "type": p.type,
                            "description": p.description,
                            "required": p.required
                        }
                        if p.enum: param_dict["enum"] = p.enum
                        if p.default is not None: param_dict["default"] = p.default
                        params_dict.append(param_dict)

                    llm_extracted = await extractor.extract(
                        user_message=user_message,
                        parameters=params_dict,
                        provider=settings.provider if settings else "local",
                        model=settings.model if settings else "qwen2.5:3b"
                    )
                    # Use setdefault: LLM extraction CANNOT override intent_default_params
                    # (e.g. action=list_repositories must not be overwritten by action=diff)
                    for k, v in llm_extracted.items():
                        extracted_params.setdefault(k, v)
                    self.logger.debug(
                        f"LLM extracted (non-overriding): {list(llm_extracted.keys())} | "
                        f"final action={extracted_params.get('action')}"
                    )
                except Exception as e:
                    self.logger.error(f"ParameterExtractor failed for implicit params: {e}", exc_info=True)

            # Safety net: intent_default_params always win over LLM extraction
            for k, v in (execution_strategy.get("intent_default_params") or {}).items():
                extracted_params.setdefault(k, v)

            if execution_strategy.get("filters"):
                extracted_params["filters"] = execution_strategy["filters"]
                
            if tool_name == "rag_search":
                if "query" not in extracted_params:
                    extracted_params["query"] = user_message
                if "collections" not in extracted_params:
                    extracted_params["collections"] = await self._resolve_rag_collections(
                        collection_name=collection_name,
                        conversation=conversation
                    )
            return extracted_params
        # CASO 4: Tools conocidos — Usar IntentRouter
        # IntentRouter ya clasificó el intent y extrajo target si aplica
        # =========================================================
        self.logger.info(f"Using IntentRouter for parameter extraction: {tool_name}")
        try:
            # Usar intent_result si está disponible (ya clasificado en determine_execution_strategy)
            intent_result = execution_strategy.get("intent_result")
            if intent_result:
                # Construir parámetros desde el intent clasificado
                extracted_params = {}

                # Si el intent tiene action_name, usarlo
                if intent_result.intent_def.action_name:
                    extracted_params["action"] = intent_result.intent_def.action_name

                # Si el intent tiene target, usarlo
                if intent_result.target:
                    extracted_params["target"] = intent_result.target

                # Agregar parámetros por defecto del intent
                if intent_result.intent_def.default_params:
                    extracted_params.update(intent_result.intent_def.default_params)

                self.logger.info(
                    f"IntentRouter extracted params: {extracted_params} "
                    f"(intent={intent_result.intent_name}, conf={intent_result.confidence:.2f})"
                )

            else:
                # Si no hay intent_result, clasificar ahora (path de compatibilidad)
                router = await self._get_intent_router()
                context = {
                    "attached_files": [],
                    "file_names": [],
                    "previous_files": [],
                }
                intent_result = await router.classify(
                    user_message,
                    context,
                    llm_provider=settings.provider,
                    llm_model=settings.model
                )
                extracted_params = {}
                if intent_result.intent_def.action_name:
                    extracted_params["action"] = intent_result.intent_def.action_name
                if intent_result.target:
                    extracted_params["target"] = intent_result.target
                if intent_result.intent_def.default_params:
                    extracted_params.update(intent_result.intent_def.default_params)

                self.logger.info(
                    f"IntentRouter late extraction: {extracted_params} "
                    f"(intent={intent_result.intent_name})"
                )

            # Complementar con intent info del orchestrator si falta action.
            # Cubre el caso donde intent_result no tiene action_name pero el
            # orchestrator sí recibió best_intent_action de score_tools_for_query.
            if execution_strategy.get("intent_action"):
                intent_act = execution_strategy["intent_action"]
                from src.config.intent_patterns import is_valid_action
                if is_valid_action(intent_act) and not extracted_params.get("action"):
                    extracted_params["action"] = intent_act
                elif not is_valid_action(intent_act):
                    extracted_params.setdefault("sub_action", intent_act)
            if execution_strategy.get("intent_target") and not extracted_params.get("target"):
                extracted_params["target"] = execution_strategy["intent_target"]
            if execution_strategy.get("intent_default_params"):
                for k, v in execution_strategy["intent_default_params"].items():
                    extracted_params.setdefault(k, v)

            # Agregar filters si existen en strategy
            if execution_strategy.get("filters"):
                extracted_params["filters"] = execution_strategy["filters"]

            # Enriquecer con target_file_id si existe
            if execution_strategy.get("target_file_id"):
                target_id = str(execution_strategy["target_file_id"])
                # Verificar si el tool acepta file_ids
                param_names = [p.name for p in params_to_extract]
                if 'file_ids' in param_names:
                    # Si no hay file_ids, usar target_file_id
                    if 'file_ids' not in extracted_params or not extracted_params.get('file_ids'):
                        extracted_params['file_ids'] = [target_id]
                        self.logger.info(
                            f"🎯 Enriched with target_file_id: {target_id[:8]}..."
                        )

            # Enriquecer con file_ids de la conversación si es necesario
            await self._enrich_params_with_file_ids(
                extracted_params=extracted_params,
                params_schema=params_to_extract,
                conversation=conversation
            )

            # =========================================================
            # RAG: Asegurar query y collections (igual que CASO 1)
            # =========================================================
            if tool_name == "rag_search":
                if "query" not in extracted_params:
                    extracted_params["query"] = user_message

                if "collections" not in extracted_params:
                    extracted_params["collections"] = await self._resolve_rag_collections(
                        collection_name=collection_name,
                        conversation=conversation
                    )

                self.logger.info(
                    f"RAG params added in IntentRouter path: query={extracted_params['query'][:30]}..., "
                    f"collections={extracted_params['collections']}"
                )

            return extracted_params

        except Exception as e:
            self.logger.error(
                f"Parameter extraction failed for {tool_name}: {e}",
                exc_info=True
            )

        # FALLBACK: Retornar params básicos
        fallback_params = {}
        # Intentar usar target_file_id en fallback
        if execution_strategy.get("target_file_id"):
            fallback_params['file_ids'] = [str(execution_strategy["target_file_id"])]
            self.logger.info(f"Fallback using target_file_id")
        else:
            # Intentar enriquecer con file_ids de BD
            try:
                files = await self.file_repo.get_by_conversation(conversation.id)
                file_ids_list = [str(file.id) for file in files]
                if file_ids_list:
                    fallback_params['file_ids'] = file_ids_list
                    self.logger.info(
                        f"Fallback enriched with {len(file_ids_list)} file_ids from DB")
            except Exception as enrich_error:
                self.logger.warning(f"Fallback enrichment failed: {enrich_error}")

        return fallback_params

    async def _enrich_export_refactored_params(
        self,
        extracted_params: dict,
        conversation,
    ) -> None:
        """
        Auto-extract `new_content` and `method_name` for the export_refactored action.

        Scans the last N assistant messages for a fenced code block (```...```).
        The most recent code block containing Python code is treated as the
        refactored content. The method/class name is inferred from the leading
        `def` or `class` line inside it.

        Args:
            extracted_params: Dict mutated in-place.
            conversation:     Active conversation (used to query history).
        """
        import re
        from src.models.models import MessageRole

        if not self.message_repo:
            self.logger.warning(
                "_enrich_export_refactored_params: message_repo not available, skipping"
            )
            return

        # Already populated – nothing to do.
        if extracted_params.get("new_content") and extracted_params.get("method_name"):
            return

        try:
            messages = await self.message_repo.get_last_n_messages(
                conversation_id=conversation.id,
                n=20
            )
        except Exception as e:
            self.logger.warning(f"_enrich_export_refactored_params: DB error: {e}")
            return

        self.logger.info(
            f"_enrich_export_refactored_params: scanning {len(messages)} messages "
            f"for code blocks"
        )

        # Walk messages newest-first so we use the most recent code block.
        code_block = None
        all_names = []

        for msg in reversed(messages):
            role = getattr(msg, 'role', None)
            content = getattr(msg, 'content', '') or ''

            # MessageRole is a str enum — compare via .value
            role_value = role.value if hasattr(role, 'value') else str(role)
            if role_value != MessageRole.ASSISTANT.value:
                continue

            self.logger.debug(
                f"_enrich: scanning assistant msg ({len(content)} chars)"
            )

            # ── Strategy 1: explicit fenced code blocks ──────────────────────
            blocks = re.findall(
                r'```(?:[a-zA-Z]*)?\n(.*?)```',
                content,
                re.DOTALL
            )

            for block in reversed(blocks):
                block = block.strip()
                if not block:
                    continue
                # Accept only blocks that look like Python code
                if re.search(r'(def |class |async def )', block):
                    code_block = block
                    break

            # ── Strategy 2: whole message if it starts with def/class ────────
            if not code_block:
                stripped = content.strip()
                if re.match(r'^(def |async def |class )', stripped):
                    code_block = stripped

            if code_block:
                # Infer ALL method/class names in the block
                # We'll try them in order until one matches in the file.
                matches = re.finditer(
                    r'^[ \t]*(async\s+)?(?:def|class)\s+(\w+)',
                    code_block,
                    re.MULTILINE
                )
                all_names = [m.group(2) for m in matches]
                break  # stop after first message with valid code

        if code_block:
            extracted_params.setdefault("new_content", code_block)
            self.logger.info(
                f"_enrich_export_refactored_params: new_content extracted "
                f"({len(code_block)} chars)"
            )
        else:
            self.logger.warning(
                "_enrich_export_refactored_params: no code block found in history"
            )

        if all_names:
            # We pass the first name as primary, but we'll tell the tool it's a list
            # by storing the whole list in extracted_params if possible, 
            # or joining them if the tool expects a string.
            extracted_params["method_name"] = ",".join(all_names)
            self.logger.info(
                f"_enrich_export_refactored_params: potential names={all_names}"
            )
        else:
            self.logger.warning(
                "_enrich_export_refactored_params: could not infer any names from code block"
            )

    async def _resolve_rag_collections(
        self,
        collection_name: Optional[str],
        conversation: Conversation
    ) -> List[str]:
        """
        Resuelve la lista de colecciones Qdrant para una búsqueda RAG.

        Prioridad de resolución:
        1. collection_name explícito del caller (más confiable, viene del request)
        2. extra_metadata de la conversación (configurado desde frontend)
        3. project_id de la conversación → "project_{id}"
        4. collection_name del primer archivo adjunto en BD
        5. "chat_{conversation.id}" como último recurso específico
        "default" eliminado — no existe en Qdrant y genera 404 en cascada.

        Args:
            collection_name: Nombre de colección explícito (puede ser None)
            conversation:    Conversación activa

        Returns:
            Lista con al menos una colección resuelta
        """
        if collection_name:
            return [collection_name]

        collections = []

        # Intentar desde extra_metadata de la conversación
        if hasattr(conversation, 'extra_metadata') and conversation.extra_metadata:
            cols = conversation.extra_metadata.get("collections", [])
            if isinstance(cols, list):
                collections.extend(cols)
            elif isinstance(cols, str):
                collections.append(cols)
            col = conversation.extra_metadata.get("collection_name")
            if col and col not in collections:
                collections.append(col)

        if not collections and hasattr(conversation, 'project_id') and conversation.project_id:
            collections.append(f"project_{conversation.project_id}")

        # Resolver desde archivos adjuntos en BD
        # Cubre el caso de mensajes de seguimiento donde file_ids viene del historial
        if not collections:
            try:
                files = await self.file_repo.get_by_conversation(conversation.id)
                if files:
                    first_col = getattr(files[0], 'collection_name', None)
                    if first_col:
                        collections = [first_col]
            except Exception as e:
                self.logger.warning(f"Could not resolve collection from files: {e}")

        # Último recurso: chat_{id} — al menos es específico de esta conversación
        if not collections:
            collections = [f"chat_{conversation.id}"]
            self.logger.warning(
                f"No collection resolved for conversation {conversation.id}, "
                f"using fallback: {collections[0]}"
            )

        return collections

    async def _enrich_params_with_file_ids(
        self,
        extracted_params: Dict[str, Any],
        params_schema: List[Any],
        conversation: Conversation
    ) -> None:
        """
        Enriquece parámetros con file_ids de BD si:
        1. El tool acepta file_ids
        2. No se extrajeron file_ids pero hay action
        3. La conversación tiene archivos

        Args:
            extracted_params: Parámetros extraídos (mutado in-place)
            params_schema: Schema de parámetros del tool
            conversation: Conversación actual
        """
        # Verificar si el tool acepta file_ids
        param_names = [p.name for p in params_schema] if params_schema else []
        if 'file_ids' not in param_names:
            return

        # Verificar si hay action que requiera archivos
        action = extracted_params.get('action')
        if not action:
            return

        # Acciones que requieren análisis de archivos
        file_dependent_actions = {
            # CodebaseAction values (legacy)
            'analyze_file',
            'basic_analyze_file',
            'analyze_quality',
            'explain',
            'find_definition',
            'find_references',
            'get_callers',
            'get_dependencies',
            'get_method_content',
            'get_class_content',
            'modify_method',
            'export_refactored',    # NEW: requires the target file to apply refactoring
            # NUEVO: action_names de INTENT_REGISTRY
            'count_methods',
            'count_classes',
            'list_methods',
            'list_classes',
            'get_method',       # action_name de get_method_content intent
            'get_class',        # action_name de get_class_content intent
            'search_symbol',
            'file_summary',
        }

        # Normalizar: si action es un enum, usar su value
        action_str = action.value if hasattr(action, 'value') else str(action)

        if action_str not in file_dependent_actions:
            return

        # Si ya hay file_ids, no sobrescribir
        if extracted_params.get('file_ids'):
            return

        # Obtener file_ids de la conversación
        try:
            files = await self.file_repo.get_by_conversation(conversation.id)
            file_ids = [str(f.id) for f in files]
            if file_ids:
                self.logger.info(f"Enriching params with {len(file_ids)} file_ids from conversation")
                extracted_params['file_ids'] = file_ids
            else:
                self.logger.debug("No files attached to conversation")
        except Exception as e:
            self.logger.warning(f"Failed to get file_ids from database: {e}")

    # =============================================================================
    # Result Formatting
    # =============================================================================

    def format_tool_result(self, result: ToolResult) -> str:
        """
        Format tool result for LLM consumption.

        Args:
            result: Tool execution result

        Returns:
            Formatted string
        """
        if not result.success:
            return f"Error: {result.error}"

        if result.data is None:
            return "Tool executed successfully (no data returned)"

        # Handle different data types
        if isinstance(result.data, str):
            return result.data

        if isinstance(result.data, (list, dict)):
            try:
                return json.dumps(result.data, indent=2, ensure_ascii=False)
            except Exception:
                return str(result.data)

        return str(result.data)

    def format_tool_results_for_llm(
        self,
        tool_results: List[Dict[str, Any]]
    ) -> str:
        """
        Format tool results for LLM consumption.

        Args:
            tool_results: List of tool result dicts

        Returns:
            Formatted string for LLM
        """
        parts = []
        for result in tool_results:
            tool_name = result.get("tool_name", "unknown")
            success = result.get("success", False)
            content = result.get("result", "")

            if success:
                parts.append(f"**{tool_name}**:\n{content}")
            else:
                parts.append(f"**{tool_name}** (Error):\n{content}")

        return "\n\n---\n\n".join(parts)

    # =============================================================================
    # Custom Tools
    # =============================================================================

    async def get_custom_tool(self, tool_id: UUID) -> Optional[CustomToolExecutor]:
        """
        Get or create custom tool executor.

        Args:
            tool_id: Custom tool UUID

        Returns:
            CustomToolExecutor or None
        """
        if tool_id in self._custom_tools_cache:
            return self._custom_tools_cache[tool_id]

        try:
            custom_tool = await self.custom_tool_repo.get_by_id(tool_id)
            if not custom_tool:
                return None

            executor = CustomToolExecutor(
                custom_tool_id=tool_id,
                file_repo=self.file_repo,
                custom_tool_repo=self.custom_tool_repo
            )
            self._custom_tools_cache[tool_id] = executor
            return executor

        except Exception as e:
            self.logger.error(f"Error loading custom tool {tool_id}: {e}")
            return None

    async def get_custom_rag_tools(self) -> List[Any]:
        """Get all active custom RAG tool instances."""
        try:
            custom_tools = await self.custom_tool_repo.get_rag_instances()
            return custom_tools
        except Exception as e:
            self.logger.error(f"Error fetching custom RAG tools: {e}")
            return []

    # =============================================================================
    # Tool Configuration
    # =============================================================================

    async def get_active_tool_configurations(self, conversation_id: UUID) -> Dict[str, Any]:
        """Get all active tool configurations for a conversation."""
        # Note: Requires conversation_repo
        return {}

    def get_tool_config(self, conversation: Conversation, tool_name: str):
        """Get tool configuration for conversation."""
        for config in conversation.tool_configurations:
            if config.tool_name == tool_name and config.is_active:
                return config
        return None

    # =============================================================================
    # Statistics & Monitoring (NUEVO)
    # =============================================================================

    async def get_intent_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del IntentRouter.

        Returns:
            Dict con estadísticas de performance
        """
        try:
            router = await self._get_intent_router()
            return router.get_stats()
        except Exception as e:
            self.logger.warning(f"Failed to get intent stats: {e}")
            return {}

    async def clear_intent_cache(self) -> None:
        """Limpia el cache del IntentRouter."""
        try:
            router = await self._get_intent_router()
            router.clear_cache()
            self.logger.info("Intent cache cleared")
        except Exception as e:
            self.logger.warning(f"Failed to clear intent cache: {e}")
