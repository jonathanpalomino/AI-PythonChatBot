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
from src.services.intent.router import get_intent_router, IntentRouter
from src.services.intent.parameter_extractor import get_parameter_extractor, ParameterExtractor
from src.tools.base_tool import tool_registry, BaseTool, ToolResult, ToolCategory
from src.tools.custom_tool import CustomToolExecutor
from src.utils.logger import get_logger
from src.services.intent.llm_classifier import LLMClassifier, get_llm_classifier

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
        Centralized tool execution with context injection.

        Args:
            tool: Tool instance
            conversation: Conversation context
            collection_name: Optional collection name
            **kwargs: Tool arguments

        Returns:
            ToolResult
        """
        # Inject file_repo if needed
        if hasattr(tool, 'file_repo') and tool.file_repo is None:
            tool.file_repo = self.file_repo
            self.logger.info(f"Injected file_repo into {tool.name}")

        # Inject file_repo in kwargs
        if 'file_repo' not in kwargs:
            kwargs['file_repo'] = self.file_repo

        # Special handling for RAG-type tools
        is_rag = False
        if tool.name == "rag_search":
            is_rag = True
        elif hasattr(tool, 'tool_type') and tool.tool_type == "rag_search":
            is_rag = True
        elif tool.category == ToolCategory.RAG:
            is_rag = True

        if is_rag:
            # RAG tools should be executed via ContextBuilder
            return ToolResult(
                success=False,
                data=None,
                error="RAG tools should be executed via ContextBuilder for proper context handling"
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
        settings: Optional[ConversationSettings] = None  # ✅ NUEVO parámetro
    ) -> Dict[str, Any]:
        """
        Determina estrategia de ejecución usando IntentRouter.

        Args:
            tool: Tool instance
            tool_name: Tool name
            user_message: User query
            rag_metadata: RAG metadata (files, symbols, etc.)
            file_ids: Attached file IDs
            target_file_id: Target file ID for sticky context
            settings: ConversationSettings con provider/model del usuario # ✅ NUEVO

        Returns:
            Dict con:
            - use_fast_path: bool
            - action: Optional[str]
            - target: Optional[str]
            - file_ids: Optional[List[str]]
            - context_enrichment: Dict
            - intent_result: Optional[IntentResult] # NUEVO
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
            self.logger.debug(f"Custom tool detected: {tool_name} - using LLM extraction")
            return strategy

        try:
            router = await self._get_intent_router()

            # Build context for router
            context = {
                "attached_files": file_ids if file_ids else [],
                "file_names": rag_metadata.get("files", []),
                "previous_files": rag_metadata.get("symbols", []),
                "target_file_id": target_file_id
            }

            # ✅ NUEVO: Extraer provider/model desde settings
            llm_provider = None
            llm_model = None
            if settings:
                llm_provider = settings.provider  # "local", "openrouter", etc.
                llm_model = settings.model         # "qwen2.5:3b", "claude-3.5-sonnet"
                self.logger.debug(
                    f"Using user's LLM config: {llm_provider}/{llm_model}"
                )

            # ✅ NUEVO: Pasar provider/model a classify()
            intent_result = await router.classify(
                user_message,
                context,
                llm_provider=llm_provider,  # ✅ Del usuario
                llm_model=llm_model          # ✅ Del usuario
            )

            # Store result in strategy
            strategy["intent_result"] = intent_result

            self.logger.info(
                f"Intent classified: {intent_result.intent_name} "
                f"(conf={intent_result.confidence:.2f}, "
                f"method={intent_result.method})"
            )

            # Map intent to CodebaseAction
            # ⚠️ CUIDADO al editar: los valores DEBEN coincidir con CodebaseAction
            # (ver src/config/intent_patterns.py) y con los action_name del INTENT_REGISTRY
            # (ver src/services/intent/config.py).
            action_mapping = {
                # Structure queries (BASIC) — no necesitan código, solo estructura
                "count_methods":     CodebaseAction.BASIC_ANALYZE_FILE,
                "count_classes":     CodebaseAction.BASIC_ANALYZE_FILE,
                "list_methods":      CodebaseAction.BASIC_ANALYZE_FILE,
                "list_classes":      CodebaseAction.BASIC_ANALYZE_FILE,
                "file_summary":      CodebaseAction.BASIC_ANALYZE_FILE,
                # Content queries — devuelven el CÓDIGO FUENTE del símbolo solicitado
                "get_method_content": CodebaseAction.GET_METHOD_CONTENT,
                "get_method": CodebaseAction.GET_METHOD_CONTENT,  # ← action_name del intent
                "get_class_content":  CodebaseAction.ANALYZE_FILE,        # sin acción propia; ANALYZE_FILE incluye código
                "explain_code":       CodebaseAction.ANALYZE_FILE,
                "get_class":          CodebaseAction.ANALYZE_FILE,
                # Symbol operations
                "search_symbol":    CodebaseAction.FIND_DEFINITION,
                "find_definition":  CodebaseAction.FIND_DEFINITION,
                "find_references":  CodebaseAction.FIND_REFERENCES,
                "get_callers":      CodebaseAction.GET_CALLERS,
                # Quality & dependencies
                "analyze_quality":  CodebaseAction.ANALYZE_QUALITY,
                "get_dependencies": CodebaseAction.GET_DEPENDENCIES,
                # Modifications
                "modify_code":      CodebaseAction.MODIFY_METHOD,
            }

            action = action_mapping.get(intent_result.intent_name)
            if action:
                strategy["use_fast_path"] = True
                strategy["action"] = action
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
                    llm_action = action_mapping.get(llm_result["intent"])
                    if llm_action:
                        strategy["use_fast_path"] = True
                        strategy["action"] = llm_action
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
            # Continue con fallback legacy si falla IntentRouter

        # PRIORIDAD 2: File_ids fallback (SOLO si NO hay action del IntentRouter)
        if not strategy["action"] and file_ids and len(file_ids) > 0 and tool_name in (
            "codebase_tool"):  # ← Bug fix: era "codebasetool"
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
        execution_strategy: Dict[str, Any]
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

        Returns:
            Dict con parámetros extraídos
        """
        params_to_extract = tool.get_parameters()

        self.logger.info(f"Extracting parameters for {tool_name}")
        self.logger.debug(f"execution_strategy: {execution_strategy}")

        # CASO 1: Fast-path activado
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

            # NUEVO: Override con intent info más específico del orchestrator.
            # intent_action (ej: "count_methods") tiene precedencia sobre
            # CodebaseAction.BASIC_ANALYZE_FILE para dar contexto fino al codebase_tool.
            if execution_strategy.get("intent_action"):
                intent_act = execution_strategy["intent_action"]
                from src.config.intent_patterns import is_valid_action
                if is_valid_action(intent_act):
                    # Coincide con CodebaseAction directamente (find_definition, get_callers, etc.)
                    extracted_params["action"] = intent_act
                else:
                    # Es un intent granular (count_methods, list_methods, etc.) — pasa como hint
                    extracted_params["sub_action"] = intent_act
            if execution_strategy.get("intent_target") and not extracted_params.get("target"):
                extracted_params["target"] = execution_strategy["intent_target"]
            if execution_strategy.get("intent_default_params"):
                for k, v in execution_strategy["intent_default_params"].items():
                    extracted_params.setdefault(k, v)

            # NUEVO: Agregar intent_result para debugging/logging
            if execution_strategy.get("intent_result"):
                intent_result = execution_strategy["intent_result"]
                self.logger.info(
                    f"Fast-path from intent: {intent_result.intent_name} "
                    f"(method={intent_result.method}, "
                    f"confidence={intent_result.confidence:.2f})"
                )

            self.logger.info(f"Fast-path extraction: {extracted_params}")
            return extracted_params

        # CASO 2: Sin parámetros para extraer
        if not params_to_extract:
            # Pasar filters si existen
            if execution_strategy.get("filters"):
                self.logger.info(f"No params to extract, passing filters")
                return {"filters": execution_strategy["filters"]}
            return {}

        # CASO 3: Custom Tools - Usar ParameterExtractor (LLM genérico)
        # Los custom tools no están en INTENT_REGISTRY, necesitan extracción LLM
        is_custom_tool = isinstance(tool, CustomToolExecutor)
        if is_custom_tool:
            self.logger.info(f"Using ParameterExtractor for custom tool: {tool_name}")
            try:
                extractor = await get_parameter_extractor()

                # Convertir ToolParameter a dict para el extractor
                params_dict = []
                for p in params_to_extract:
                    param_dict = {
                        "name": p.name,
                        "type": p.type,
                        "description": p.description,
                        "required": p.required
                    }
                    if p.enum:
                        param_dict["enum"] = p.enum
                    if p.default is not None:
                        param_dict["default"] = p.default
                    params_dict.append(param_dict)

                extracted_params = await extractor.extract(
                    user_message=user_message,
                    parameters=params_dict,
                    provider=settings.provider if settings else "local",
                    model=settings.model if settings else "qwen2.5:3b"
                )

                self.logger.info(f"ParameterExtractor result: {extracted_params}")

                # Agregar filters si existen
                if execution_strategy.get("filters"):
                    extracted_params["filters"] = execution_strategy["filters"]

                return extracted_params

            except Exception as e:
                self.logger.error(f"ParameterExtractor failed: {e}", exc_info=True)
                return {}

        # CASO 4: Tools conocidos - Usar IntentRouter
        # IntentRouter ya clasificó el intent y extrajo target si aplica
        self.logger.info(f"Using IntentRouter for parameter extraction: {tool_name}")
        try:
            # Usar intent_result si está disponible (ya clasificado)
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
                # Si no hay intent_result, clasificar ahora
                router = await self._get_intent_router()
                context = {
                    "attached_files": [],
                    "file_names": [],
                    "previous_files": [],
                }
                intent_result = await router.classify(
                    user_message,
                    context,
                    llm_provider=settings.provider if settings else None,
                    llm_model=settings.model if settings else None
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

            # NUEVO: Complementar con intent info del orchestrator si falta action.
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
                    self.logger.info(f"Fallback enriched with {len(file_ids_list)} file_ids from DB")
            except Exception as enrich_error:
                self.logger.warning(f"Fallback enrichment failed: {enrich_error}")

        return fallback_params

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
