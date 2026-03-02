# =============================================================================
# src/tools/base_tool.py
# Base class for all tools
# =============================================================================
"""
Sistema base de tools extensible para el chatbot.

v2.0 — Contratos estandarizados:
- Logger automático por subclase (self.logger)
- run() como Template Method con pipeline estándar
- Hooks opcionales _before_execute / _after_execute
- ExecutionContext: objeto estándar de contexto por request
- required_dependencies: dependencias de infraestructura declaradas
- file_dependent_actions: acciones que requieren file_ids declaradas
- is_relevant(): relevancia contextual (reemplaza lógica hardcodeada en Orchestrator)
- required_context_keys: qué keys de ExecutionContext consume la tool
- llm_hint: hint al LLM cuando la tool está activa
- get_intent_definitions(): intents que maneja la tool (para IntentRouter)
- params_from_intent(): traducción IntentResult → parámetros de execute()
- execution_stats: métricas básicas de uso

COMPATIBILIDAD: Toda la API existente se mantiene sin cambios.
Las subclases existentes (HTTPTool, RAGTool, CodebaseTool) no necesitan
modificarse para seguir funcionando. Los nuevos contratos son opt-in via override.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    # Importados solo para type hints — evita cargar SentenceTransformer
    # y demás dependencias pesadas en runtime al importar base_tool.
    from src.services.intent.router import IntentResult
    from src.services.intent.config import IntentDefinition

# Logger a nivel de módulo (usado por ToolRegistry, no por instancias de tools)
_logger = logging.getLogger(__name__)


# =============================================================================
# Enums y Data Classes
# =============================================================================

class ToolCategory(str, Enum):
    """Tool categories"""
    RAG = "rag"
    CODE = "code"
    DOCUMENT = "document"
    MEMORY = "memory"
    WEB = "web"
    UTILITY = "utility"


@dataclass
class ToolParameter:
    """Tool parameter definition"""
    name: str
    type: str  # "string", "integer", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None
    example: Optional[Any] = None


@dataclass
class ToolResult:
    """Tool execution result"""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# =============================================================================
# ExecutionContext — Contexto estándar de ejecución por request (NUEVO v2.0)
# =============================================================================

@dataclass
class ExecutionContext:
    """
    Contexto de ejecución construido por el Orchestrator una vez por request.

    Se pasa a is_relevant() y a los métodos que necesitan datos del request
    sin que el Orchestrator tenga que conocer los internos de cada tool.

    Attributes:
        user_message:    Mensaje original del usuario.
        conversation_id: UUID de la conversación activa.
        file_ids:        Archivos adjuntos en el request actual.
        target_file_id:  Archivo objetivo detectado (sticky context).
        rag_metadata:    Metadata del resultado RAG previo, si existe.
        collection_name: Colección Qdrant activa.
        provider:        Provider LLM activo (ej: "openai", "anthropic", "local").
        model:           Modelo LLM activo (ej: "gpt-4o", "qwen2.5:3b").
        extra:           Datos adicionales arbitrarios para extensibilidad futura.
    """
    user_message: str
    conversation_id: Optional[UUID] = None
    file_ids: Optional[List[UUID]] = None
    target_file_id: Optional[UUID] = None
    rag_metadata: Optional[Dict[str, Any]] = None
    collection_name: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# BaseTool — Clase base v2.0
# =============================================================================

class BaseTool(ABC):
    """
    Clase base para todas las tools. v2.0.

    NUEVOS CONTRATOS (todos tienen default seguro — opt-in via override):
        - self.logger           : Logger automático, no requiere declaración en subclase.
        - run()                 : Template Method con pipeline estandarizado.
        - required_dependencies : Infraestructura que la tool necesita.
        - file_dependent_actions: Acciones que requieren file_ids.
        - is_relevant()         : Condición de activación contextual.
        - required_context_keys : Keys de ExecutionContext que la tool consume.
        - llm_hint              : Hint de sistema para el LLM.
        - get_intent_definitions: Intents que maneja la tool.
        - params_from_intent()  : Traduce IntentResult → parámetros de execute().
        - execution_stats       : Métricas básicas de uso.

    API EXISTENTE (sin cambios):
        - name, description, category (abstractos)
        - enabled_by_default, requires_context, auto_discover
        - get_parameters(), to_openai_function(), to_anthropic_tool()
        - execute() (abstracto)
        - validate_input(), format_output()
    """

    # Atributo de clase para control de auto-discovery (sin cambios)
    auto_discover: bool = True

    def __init__(self):
        # ---------------------------------------------------------------------
        # Logger automático (NUEVO v2.0)
        # Usa hasattr para respetar el logger que la subclase haya asignado
        # ANTES de llamar a super().__init__() (patrón existente en HTTPTool y RAGTool).
        # Si la subclase llama super().__init__() primero (recomendado para nuevas tools),
        # BaseTool provee el logger usando el módulo y clase de la subclase concreta.
        # ---------------------------------------------------------------------
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(
                f"{self.__class__.__module__}.{self.__class__.__name__}"
            )

        # Estadísticas de ejecución (NUEVO v2.0)
        self._call_count: int = 0
        self._total_execution_ms: float = 0.0
        self._error_count: int = 0

        # Validación de definición (sin cambios)
        self._validate_tool_definition()

    # =========================================================================
    # Tool Metadata — Abstractos obligatorios (sin cambios)
    # =========================================================================

    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre único de la tool (ej: 'rag_search', 'codebase_tool')."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Descripción legible para humanos y LLMs."""
        pass

    @property
    @abstractmethod
    def category(self) -> ToolCategory:
        """Categoría de la tool."""
        pass

    @property
    def enabled_by_default(self) -> bool:
        """Si la tool está habilitada por defecto. Default: False."""
        return False

    @property
    def requires_context(self) -> List[str]:
        """
        Dependencias requeridas (ej: ["qdrant", "files"]).

        DEPRECATED desde v2.0: usar required_dependencies.
        Mantenido por compatibilidad con RAGTool existente.
        """
        return []

    # =========================================================================
    # NUEVOS CONTRATOS DECLARATIVOS (v2.0) — todos con default seguro
    # =========================================================================

    @property
    def required_dependencies(self) -> List[str]:
        """
        Dependencias de infraestructura que esta tool necesita para funcionar.

        El sistema las inyecta al registrar la tool. Si una dependencia falta,
        la tool no se activa en lugar de fallar en runtime.

        Los nombres deben coincidir con los atributos del contenedor de
        dependencias (file_repo, qdrant, etc.).

        Ejemplos:
            CodebaseTool → return ["file_repo"]
            RAGTool      → return ["qdrant"]
            HTTPTool     → return []
        """
        return []

    @property
    def file_dependent_actions(self) -> Set[str]:
        """
        Acciones de esta tool que requieren file_ids para ejecutarse.

        Reemplaza el set hardcodeado en ToolExecutor._enrich_params_with_file_ids.
        El executor lee esto en lugar de mantener una lista central por tool.

        Retorna set vacío para tools de acción única o sin dependencia de archivos.

        Ejemplo (CodebaseTool):
            return {
                "analyze_file", "basic_analyze_file", "analyze_quality",
                "explain", "get_method_content", "get_class_content",
                "modify_method"
            }
        """
        return set()

    async def is_relevant(self, context: ExecutionContext) -> bool:
        """
        Determina si esta tool debe activarse para el contexto dado.

        Reemplaza el conocimiento hardcodeado en el Orchestrator sobre
        cuándo activar cada tool (ej: inyección de hints por nombre de tool).

        Default: True — siempre relevante si está habilitada.
        Override para lógica específica de la tool.

        Args:
            context: ExecutionContext con los datos del request actual.

        Returns:
            True si la tool debe ejecutarse en este contexto.

        Ejemplos:
            CodebaseTool → return bool(context.file_ids or context.target_file_id)
            HTTPTool     → return True
            RAGTool      → return bool(context.file_ids or context.collection_name)
        """
        return True

    @property
    def required_context_keys(self) -> List[str]:
        """
        Keys de ExecutionContext que esta tool consume en is_relevant() y execute().

        El Orchestrator puede usar esto para validar que el contexto está
        completo antes de invocar la tool, sin conocer sus internos.

        Los nombres deben ser atributos válidos de ExecutionContext.

        Ejemplo (CodebaseTool):
            return ["file_ids", "target_file_id", "rag_metadata"]

        Ejemplo (RAGTool):
            return ["file_ids", "collection_name"]
        """
        return []

    @property
    def llm_hint(self) -> Optional[str]:
        """
        Hint de sistema para el LLM cuando esta tool está activa en agent mode.

        Reemplaza los strings hardcodeados en _agent_mode y _agent_mode_stream
        del Orchestrator. El Orchestrator agrega los hints de TODAS las tools
        activas sin conocer su contenido.

        Retorna None si la tool no necesita guiar al LLM explícitamente.

        Ejemplo (CodebaseTool):
            return (
                "Source code files are available. Use 'codebase_tool' for "
                "structural analysis, finding definitions, or understanding "
                "code logic."
            )
        """
        return None

    def get_intent_definitions(self) -> Dict[str, Any]:
        """
        Definiciones de intents que esta tool maneja.

        Permite que el INTENT_REGISTRY global se construya automáticamente
        al registrar tools, en lugar de mantenerse como config estática separada.
        Resuelve la desincronización entre ToolRegistry e INTENT_REGISTRY.

        Retorna dict vacío para tools de acción única (no intent-driven).

        El valor de cada entrada es un dict compatible con IntentDefinition.
        Se usa Dict[str, Any] en lugar de Dict[str, IntentDefinition] para
        evitar imports circulares en runtime (IntentDefinition carga
        SentenceTransformer y otros modelos pesados).

        Formato de cada entrada:
            {
                "description":    str,          # Para el LLM prompt del router
                "action_name":    str,          # Valor para el parámetro "action" de execute()
                "requires_target": bool,        # Si necesita extraer un símbolo/nombre
                "target_patterns": List[str],   # Regex patterns para extraer target (si aplica)
                "examples":       List[str],    # Frases de entrenamiento para embeddings
                "default_params": Dict[str, Any] # Parámetros extra con valores fijos
            }

        Ejemplo (CodebaseTool):
            return {
                "count_methods": {
                    "description": "Contar métodos en un archivo de código",
                    "action_name": "basic_analyze_file",
                    "requires_target": False,
                    "target_patterns": [],
                    "examples": ["cuántos métodos tiene", "cuenta los métodos"],
                    "default_params": {}
                },
                "get_method_content": {
                    "description": "Obtener el código fuente de un método específico",
                    "action_name": "analyze_file",
                    "requires_target": True,
                    "target_patterns": ["(?:método|función|def)\\s+(\\w+)"],
                    "examples": ["muéstrame el código de authenticate"],
                    "default_params": {}
                }
            }
        """
        return {}

    def params_from_intent(self, intent_result: Any) -> Dict[str, Any]:
        """
        Traduce un IntentResult a parámetros para execute().

        Reemplaza el action_mapping hardcodeado en ToolExecutor.
        La tool misma sabe cómo convertir el intent detectado en parámetros
        concretos para su execute().

        Args:
            intent_result: IntentResult de IntentRouter.
                           Tipado como Any para evitar import circular en runtime.
                           En runtime tendrá los atributos:
                               intent_result.intent_name       : str
                               intent_result.intent_def        : IntentDefinition
                               intent_result.intent_def.action_name   : str
                               intent_result.intent_def.default_params: Dict
                               intent_result.target            : Optional[str]
                               intent_result.confidence        : float

        Returns:
            Dict con parámetros listos para pasar a execute(**params).
            Retorna {} si la tool no soporta traducción desde intent.

        Ejemplo (CodebaseTool):
            params = {"action": intent_result.intent_def.action_name}
            if intent_result.target:
                params["target"] = intent_result.target
            if intent_result.intent_def.default_params:
                params.update(intent_result.intent_def.default_params)
            return params
        """
        return {}

    # =========================================================================
    # Tool Definition para LLM (sin cambios)
    # =========================================================================

    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """Retorna la definición de parámetros de la tool."""
        pass

    def _build_parameter_properties(self, parameters: List[ToolParameter]) -> tuple:
        """
        Construye properties y required list desde la lista de ToolParameter.

        Returns:
            Tuple de (properties dict, required list)
        """
        properties = {}
        required = []

        for param in parameters:
            param_def = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                param_def["enum"] = param.enum
            properties[param.name] = param_def
            if param.required:
                required.append(param.name)

        return properties, required

    def to_openai_function(self) -> Dict[str, Any]:
        """Convierte la tool al formato de function calling de OpenAI."""
        parameters = self.get_parameters()
        properties, required = self._build_parameter_properties(parameters)
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }

    def to_anthropic_tool(self) -> Dict[str, Any]:
        """Convierte la tool al formato de tool use de Anthropic."""
        parameters = self.get_parameters()
        properties, required = self._build_parameter_properties(parameters)
        input_schema = {
            "type": "object",
            "properties": properties,
            "required": required
        }
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": input_schema
        }

    # =========================================================================
    # Template Method run() — Pipeline estandarizado (NUEVO v2.0)
    # =========================================================================

    async def run(self, **kwargs) -> ToolResult:
        """
        Template Method: pipeline estandarizado de ejecución.

        Las tools que quieran logging, timing y error handling unificados
        llaman run() en lugar de execute() directamente.

        Pipeline:
            1. validate_input()    — validación estándar (existente)
            2. _before_execute()   — hook pre-ejecución (override opcional)
            3. execute()           — lógica de negocio (override obligatorio)
            4. _after_execute()    — hook post-ejecución (override opcional)
            5. Inyecta execution_ms y tool_name en result.metadata

        COMPATIBILIDAD: execute() sigue siendo el método de implementación.
        Los callers existentes que llaman execute() directamente no se ven
        afectados. run() es una capa adicional opt-in.

        Returns:
            ToolResult con metadata["execution_ms"] y metadata["tool_name"] inyectados.
        """
        start = time.perf_counter()
        self._call_count += 1

        self.logger.info(
            f"[{self.name}] START",
            extra={"kwargs_keys": list(kwargs.keys()), "call_count": self._call_count}
        )

        try:
            await self.validate_input(**kwargs)
            await self._before_execute(**kwargs)
            result = await self.execute(**kwargs)
            result = await self._after_execute(result, **kwargs)

            elapsed_ms = (time.perf_counter() - start) * 1000
            self._total_execution_ms += elapsed_ms

            self.logger.info(
                f"[{self.name}] END",
                extra={"success": result.success, "elapsed_ms": round(elapsed_ms, 2)}
            )

            # Inyecta sin sobreescribir keys que execute() haya puesto
            result.metadata.setdefault("execution_ms", round(elapsed_ms, 2))
            result.metadata.setdefault("tool_name", self.name)
            return result

        except ValueError as e:
            # Errores de validación: warning sin stack trace
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._error_count += 1
            self.logger.warning(f"[{self.name}] Validation error: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                metadata={"execution_ms": round(elapsed_ms, 2), "tool_name": self.name}
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._error_count += 1
            self.logger.error(
                f"[{self.name}] Unhandled error: {e}",
                exc_info=True,
                extra={"elapsed_ms": round(elapsed_ms, 2)}
            )
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                metadata={"execution_ms": round(elapsed_ms, 2), "tool_name": self.name}
            )

    async def _before_execute(self, **kwargs) -> None:
        """
        Hook pre-ejecución. Override para lógica antes de execute().

        Llamado por run() después de validate_input() y antes de execute().
        No necesita retornar nada. Para interrumpir la ejecución, lanzar excepción.

        Ejemplos de uso:
            - Verificar disponibilidad de dependencias externas
            - Rate limiting (RateLimitedMixin puede implementarlo aquí)
            - Sanitización adicional de inputs específica de la tool
        """
        pass

    async def _after_execute(self, result: ToolResult, **kwargs) -> ToolResult:
        """
        Hook post-ejecución. Override para transformar o enriquecer el resultado.

        Llamado por run() después de execute(). Debe retornar un ToolResult
        (puede ser el mismo modificado o uno nuevo).

        Args:
            result:   ToolResult retornado por execute().
            **kwargs: Los mismos kwargs pasados a execute().

        Returns:
            ToolResult (modificado o el mismo).

        Ejemplos de uso:
            - Guardar resultado en caché (CacheableMixin puede implementarlo aquí)
            - Enriquecer metadata con datos adicionales
            - Formatear data para consumo específico
        """
        return result

    # =========================================================================
    # Tool Execution — Contrato existente (sin cambios)
    # =========================================================================

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        Implementa la lógica de negocio de la tool.

        Args:
            **kwargs: Parámetros específicos de la tool.

        Returns:
            ToolResult con success, data, error y metadata.
        """
        pass

    async def validate_input(self, **kwargs) -> bool:
        """
        Valida los parámetros de entrada contra la definición de get_parameters().

        Verifica:
            - Parámetros requeridos presentes
            - Tipos básicos (string, integer, boolean, array)
            - Valores dentro de enum si está definido

        Returns:
            True si válido.

        Raises:
            ValueError: Si falta un parámetro requerido, el tipo es incorrecto,
                        o el valor no está en el enum permitido.
        """
        parameters = self.get_parameters()

        for param in parameters:
            if param.required and param.name not in kwargs:
                raise ValueError(f"Missing required parameter: {param.name}")

            if param.name in kwargs:
                value = kwargs[param.name]

                # None es válido para parámetros opcionales
                if value is None:
                    continue

                if param.type == "string" and not isinstance(value, str):
                    raise ValueError(f"Parameter {param.name} must be string")
                elif param.type == "integer" and not isinstance(value, int):
                    raise ValueError(f"Parameter {param.name} must be integer")
                elif param.type == "boolean" and not isinstance(value, bool):
                    raise ValueError(f"Parameter {param.name} must be boolean")
                elif param.type == "array" and not isinstance(value, list):
                    raise ValueError(f"Parameter {param.name} must be array")

                if param.enum and value not in param.enum:
                    raise ValueError(
                        f"Parameter {param.name} must be one of {param.enum}"
                    )

        return True

    def format_output(self, result: ToolResult) -> str:
        """
        Formatea el resultado para consumo del LLM.

        Override para formato personalizado.

        Args:
            result: ToolResult de execute().

        Returns:
            String formateado.
        """
        if not result.success:
            return f"Error executing {self.name}: {result.error}"
        return str(result.data)

    # =========================================================================
    # Estadísticas de ejecución (NUEVO v2.0)
    # =========================================================================

    @property
    def execution_stats(self) -> Dict[str, Any]:
        """
        Estadísticas de ejecución acumuladas desde la inicialización.

        Solo cuenta llamadas a run(). Las llamadas directas a execute()
        no se contabilizan aquí (por diseño — run() es el punto de medición).

        Returns:
            Dict con call_count, error_count, avg_execution_ms, total_execution_ms.
        """
        avg_ms = (
            self._total_execution_ms / self._call_count
            if self._call_count > 0 else 0.0
        )
        return {
            "tool_name": self.name,
            "call_count": self._call_count,
            "error_count": self._error_count,
            "avg_execution_ms": round(avg_ms, 2),
            "total_execution_ms": round(self._total_execution_ms, 2),
        }

    def reset_stats(self) -> None:
        """Resetea las estadísticas de ejecución a cero."""
        self._call_count = 0
        self._total_execution_ms = 0.0
        self._error_count = 0

    # =========================================================================
    # Utility Methods (sin cambios)
    # =========================================================================

    def _validate_tool_definition(self):
        """Valida que la tool esté correctamente definida al instanciar."""
        if not self.name:
            raise ValueError("Tool must have a name")
        if not self.description:
            raise ValueError("Tool must have a description")
        if not self.category:
            raise ValueError("Tool must have a category")

    def __repr__(self):
        return f"<{self.__class__.__name__}(name={self.name})>"


# =============================================================================
# ToolRegistry — Extendido para soportar contratos v2.0
# =============================================================================

class ToolRegistry:
    """
    Registry para gestionar las tools disponibles.

    NUEVO en v2.0:
        - register() construye automáticamente el mapa intent → tool
        - unregister() limpia intents de la tool al desregistrarla
        - get_all_intents(): vista agregada de intents de todas las tools
        - get_tool_for_intent(): tool responsable de un intent
        - get_tools_with_llm_hints(): tools con hint para el LLM
        - get_relevant_tools(): filtra tools activas por relevancia contextual

    API EXISTENTE sin cambios:
        - register, unregister, get, get_all, get_by_category
        - get_enabled_by_default, list_names
        - to_openai_functions, to_anthropic_tools
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        # Mapa intent_name → tool_name, construido automáticamente en register()
        self._intent_to_tool: Dict[str, str] = {}

    def register(self, tool: BaseTool) -> None:
        """
        Registra una tool.

        Adicionalmente registra los intents declarados en get_intent_definitions()
        en el mapa interno intent → tool_name.

        Args:
            tool: Instancia de BaseTool a registrar.

        Raises:
            ValueError: Si el nombre de la tool ya está registrado.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name} already registered")

        self._tools[tool.name] = tool

        # Registrar intents de la tool (NUEVO v2.0)
        intent_defs = tool.get_intent_definitions()
        if intent_defs:
            for intent_name in intent_defs:
                if intent_name in self._intent_to_tool:
                    _logger.warning(
                        f"Intent '{intent_name}' ya registrado por "
                        f"'{self._intent_to_tool[intent_name]}', "
                        f"reemplazado por '{tool.name}'"
                    )
                self._intent_to_tool[intent_name] = tool.name
            print(f"✅ Tool registered: {tool.name} ({len(intent_defs)} intents)")
        else:
            print(f"✅ Tool registered: {tool.name}")

    def unregister(self, tool_name: str) -> None:
        """
        Desregistra una tool y elimina sus intents del mapa.

        Args:
            tool_name: Nombre de la tool a desregistrar.
        """
        if tool_name not in self._tools:
            return

        # Limpiar intents de la tool (NUEVO v2.0)
        tool = self._tools[tool_name]
        intent_defs = tool.get_intent_definitions()
        for intent_name in intent_defs:
            if self._intent_to_tool.get(intent_name) == tool_name:
                del self._intent_to_tool[intent_name]

        del self._tools[tool_name]
        print(f"🗑️ Tool unregistered: {tool_name}")

    def get(self, tool_name: str) -> Optional[BaseTool]:
        """Retorna una tool por nombre. None si no existe."""
        return self._tools.get(tool_name)

    def get_all(self) -> List[BaseTool]:
        """Retorna todas las tools registradas."""
        return list(self._tools.values())

    def get_by_category(self, category: ToolCategory) -> List[BaseTool]:
        """Retorna tools filtradas por categoría."""
        return [
            tool for tool in self._tools.values()
            if tool.category == category
        ]

    def get_enabled_by_default(self) -> List[BaseTool]:
        """Retorna tools habilitadas por defecto."""
        return [
            tool for tool in self._tools.values()
            if tool.enabled_by_default
        ]

    def list_names(self) -> List[str]:
        """Retorna los nombres de todas las tools registradas."""
        return list(self._tools.keys())

    # -------------------------------------------------------------------------
    # NUEVOS métodos v2.0 — Contratos declarativos
    # -------------------------------------------------------------------------

    def get_all_intents(self) -> Dict[str, Any]:
        """
        Vista agregada de los intents de todas las tools registradas.

        Puede usarse como fuente de verdad dinámica para el IntentRouter,
        reemplazando al INTENT_REGISTRY estático de intent/config.py.
        El IntentRouter puede llamar esto en lugar de importar la config.

        Returns:
            Dict[intent_name, intent_definition_dict]
        """
        all_intents: Dict[str, Any] = {}
        for tool in self._tools.values():
            all_intents.update(tool.get_intent_definitions())
        return all_intents

    def get_tool_for_intent(self, intent_name: str) -> Optional[BaseTool]:
        """
        Retorna la tool responsable de manejar un intent dado.

        Args:
            intent_name: Nombre del intent (ej: "count_methods").

        Returns:
            BaseTool o None si el intent no está registrado en ninguna tool.
        """
        tool_name = self._intent_to_tool.get(intent_name)
        if tool_name:
            return self._tools.get(tool_name)
        return None

    def get_tools_with_llm_hints(self) -> List[Dict[str, str]]:
        """
        Retorna las tools que tienen llm_hint definido.

        El Orchestrator usa esto para construir el contexto del LLM en
        agent mode sin conocer el contenido de los hints.

        Returns:
            List de dicts con keys 'tool_name' y 'hint'.
        """
        result = []
        for tool in self._tools.values():
            if tool.llm_hint is not None:
                result.append({
                    "tool_name": tool.name,
                    "hint": tool.llm_hint
                })
        return result

    async def get_relevant_tools(
        self,
        context: ExecutionContext,
        enabled_tool_names: List[str]
    ) -> List[BaseTool]:
        """
        Filtra tools habilitadas según su relevancia para el contexto actual.

        Reemplaza la lógica de activación condicional hardcodeada en el
        Orchestrator (ej: if file_ids and "rag_search" in available_tools).
        Cada tool decide su propia relevancia via is_relevant().

        Args:
            context:            ExecutionContext del request actual.
            enabled_tool_names: Nombres de tools habilitadas en settings.

        Returns:
            Lista de tools habilitadas Y relevantes para este contexto,
            en el mismo orden que enabled_tool_names.
        """
        relevant: List[BaseTool] = []

        for tool_name in enabled_tool_names:
            tool = self._tools.get(tool_name)
            if tool is None:
                _logger.warning(f"Tool '{tool_name}' habilitada pero no registrada")
                continue
            try:
                if await tool.is_relevant(context):
                    relevant.append(tool)
            except Exception as e:
                # Si is_relevant() falla, incluir la tool por defecto (fail-open)
                _logger.warning(
                    f"is_relevant() falló para '{tool_name}': {e} — incluida por defecto"
                )
                relevant.append(tool)

        return relevant

    # -------------------------------------------------------------------------
    # Métodos existentes para LLM (sin cambios)
    # -------------------------------------------------------------------------

    def to_openai_functions(self, tool_names: Optional[List[str]] = None) -> List[Dict]:
        """
        Convierte tools al formato de function calling de OpenAI.

        Args:
            tool_names: Tools específicas a convertir. None = todas.
        """
        if tool_names is None:
            tools = self._tools.values()
        else:
            tools = [self._tools[name] for name in tool_names if name in self._tools]
        return [tool.to_openai_function() for tool in tools]

    def to_anthropic_tools(self, tool_names: Optional[List[str]] = None) -> List[Dict]:
        """
        Convierte tools al formato de tool use de Anthropic.

        Args:
            tool_names: Tools específicas a convertir. None = todas.
        """
        if tool_names is None:
            tools = self._tools.values()
        else:
            tools = [self._tools[name] for name in tool_names if name in self._tools]
        return [tool.to_anthropic_tool() for tool in tools]

    def __repr__(self):
        return f"<ToolRegistry tools={list(self._tools.keys())}>"


# =============================================================================
# Global Registry Instance (sin cambios)
# =============================================================================

tool_registry = ToolRegistry()
