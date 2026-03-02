# =============================================================================
# src/services/intent/config.py
# Intent Configuration - Single Source of Truth
# =============================================================================
"""
Este módulo define TODOS los intents del sistema.
Es la única fuente de verdad para:
- Definiciones de intents
- Training examples (embeddings)
- Tool routing
- Parameter defaults

Para agregar un nuevo intent:
1. Crear IntentDefinition en INTENT_REGISTRY
2. Listo. El sistema automáticamente:
   - Genera embeddings
   - Crea prompts LLM
   - Configura routing
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

# =============================================================================
# Enums y Tipos Base
# =============================================================================

class IntentCategory(Enum):
    """Categorías de alto nivel para organización"""
    CODE_QUERY = "code_query"          # Queries atómicas (count, list, get)
    CODE_ANALYSIS = "code_analysis"    # Análisis profundo (quality, smells)
    CODE_SEARCH = "code_search"        # Búsqueda de definiciones/referencias
    CONTENT = "content"                # RAG general, conversational

# =============================================================================
# Dataclass Principal
# =============================================================================

@dataclass
class IntentDefinition:
    """
    Definición completa de un intent.

    Attributes:
        name: Identificador único (snake_case)
        category: Categoría del intent
        description: Descripción para humanos y LLM
        target_tool: Nombre del tool a ejecutar
        action_name: Action específica del tool (opcional)
        examples_es: Ejemplos en español para embeddings
        examples_en: Ejemplos en inglés para embeddings
        requires_target: Si necesita extraer un símbolo/target
        target_patterns: Regex patterns para extraer target
        default_params: Parámetros por defecto para el tool
        confidence_threshold: Umbral mínimo de confidence
        priority: Mayor = más prioritario en desempates (0-10)
    """

    # =========================================================================
    # CAMPOS OBLIGATORIOS (sin default) - DEBEN IR PRIMERO
    # =========================================================================
    name: str
    category: IntentCategory
    description: str
    target_tool: str

    # =========================================================================
    # CAMPOS OPCIONALES (con default) - DEBEN IR DESPUÉS
    # =========================================================================
    action_name: Optional[str] = None
    examples_es: List[str] = field(default_factory=list)
    examples_en: List[str] = field(default_factory=list)
    requires_target: bool = False
    target_patterns: List[str] = field(default_factory=list)
    default_params: Dict[str, Any] = field(default_factory=dict)
    confidence_threshold: float = 0.65
    priority: int = 5  # 0-10, mayor = más prioritario

# =============================================================================
# REGISTRO DE INTENTS (Source of Truth)
# =============================================================================

INTENT_REGISTRY: Dict[str, IntentDefinition] = {

    # =========================================================================
    # CODE QUERY INTENTS (para CodeQueryTool)
    # =========================================================================

    "count_methods": IntentDefinition(
        name="count_methods",
        category=IntentCategory.CODE_QUERY,
        description="Contar número de métodos o funciones en archivos",
        target_tool="codebase_tool",
        action_name="count_methods",
        examples_es=[
            "cuántos métodos tiene",
            "cuántas funciones hay",
            "número de métodos",
            "cuenta los métodos",
            "cantidad de funciones def",
            "contar funciones",
            "cuántos def hay",
            "total de métodos",
        ],
        examples_en=[
            "how many methods",
            "count methods",
            "number of functions",
            "how many functions",
            "count the methods",
            "total methods",
        ],
        default_params={"format": "markdown"},
        confidence_threshold=0.70,
        priority=10
    ),

    "count_classes": IntentDefinition(
        name="count_classes",
        category=IntentCategory.CODE_QUERY,
        description="Contar número de clases en archivos",
        target_tool="codebase_tool",
        action_name="count_classes",
        examples_es=[
            "cuántas clases tiene",
            "número de clases",
            "cuenta las clases",
            "cantidad de clases",
            "total de clases",
        ],
        examples_en=[
            "how many classes",
            "count classes",
            "number of classes",
            "total classes",
        ],
        default_params={"format": "markdown"},
        confidence_threshold=0.70,
        priority=10
    ),

    "list_methods": IntentDefinition(
        name="list_methods",
        category=IntentCategory.CODE_QUERY,
        description="Listar nombres de métodos/funciones",
        target_tool="codebase_tool",
        action_name="list_methods",
        examples_es=[
            "lista los métodos",
            "enumera las funciones",
            "qué métodos tiene",
            "muestra los métodos",
            "dame los nombres de los métodos",
            "listame las funciones",
            "cuáles son los métodos",
        ],
        examples_en=[
            "list methods",
            "show methods",
            "enumerate functions",
            "what methods are there",
            "list all functions",
            "give me method names",
        ],
        default_params={"include_docstrings": False, "format": "markdown"},
        confidence_threshold=0.68,
        priority=9
    ),

    "list_classes": IntentDefinition(
        name="list_classes",
        category=IntentCategory.CODE_QUERY,
        description="Listar nombres de clases",
        target_tool="codebase_tool",
        action_name="list_classes",
        examples_es=[
            "lista las clases",
            "qué clases tiene",
            "muestra las clases",
            "enumera las clases",
            "dame las clases",
            "cuáles son las clases",
        ],
        examples_en=[
            "list classes",
            "show classes",
            "what classes",
            "enumerate classes",
            "give me class names",
        ],
        default_params={"format": "markdown"},
        confidence_threshold=0.68,
        priority=9
    ),

    "get_method_content": IntentDefinition(
        name="get_method_content",
        category=IntentCategory.CODE_QUERY,
        description="Obtener código fuente de método o función específica",
        target_tool="codebase_tool",
        action_name="get_method",
        requires_target=True,
        examples_es=[
            "muéstrame el método authenticate",
            "dame el código de validate_user",
            "dame el contenido de validate_user",
            "contenido del método process_data",
            "código de la función calculate",
            "cómo está implementado get_user",
            "implementación de send_email",
            "ver el método create_order",
            "muestra la función parse_input",
            "dame el codigo de",  # sin tilde — cobertura coloquial
            "muéstrame el codigo de",
            "dame el codigo del método",
            "mostrame el código del método",
            "quiero ver el código de",
            "ver el código de",
            "cómo está hecho",
            "dame la implementación de",
            # Variantes con nombres concretos de métodos (snake_case)
            "dame el codigo de create_template",
            "dame el codigo de delete_template",
            "dame el codigo de list_templates",
            "dame el codigo de get_template",
            "dame el codigo de update_template",
            "dame el codigo de validate_user",
            "dame el codigo de create_order",
            "dame el codigo de process_payment",
            "dame el codigo de send_email",
            "dame el codigo de get_user",
            "muéstrame el codigo de create_template",
            "quiero ver el codigo de list_templates",
            "mostrame la función create_template",
            "dame el método create_template",
            "ver el código de create_template",
            "código de create_template",
        ],
        examples_en=[
            "show method authenticate",
            "get method validate_user",
            "code of process_data",
            "method content calculate",
            "implementation of get_user",
            "show function send_email",
        ],
        target_patterns=[
            r"(?:método|metodo|función|funcion|function|method)\s+(\w+)",
            r"(?:de|del|of)\s+([\w_]+)",          # ← [\w_]+ para snake_case
            r"implementa(?:ción|cion|do|tion)?\s+(?:de|del|of)?\s*([\w_]+)",
            r"(?:codigo|código)\s+(?:de|del)?\s*([\w_]+)",  # ← NUEVO: "codigo de X"
            r"([\w_]+)$",                          # última palabra del mensaje
        ],
        default_params={"format": "markdown", "include_docstrings": True},
        confidence_threshold=0.55,  # Bajado de 0.60 → 0.55 para cubrir variantes con nombres concretos
        priority=10
    ),

    "get_class_content": IntentDefinition(
        name="get_class_content",
        category=IntentCategory.CODE_QUERY,
        description="Obtener código fuente de clase específica",
        target_tool="codebase_tool",
        action_name="get_class",
        requires_target=True,
        examples_es=[
            "muéstrame la clase User",
            "dame el código de OrderService",
            "contenido de la clase DatabaseManager",
            "código de ApiClient",
            "implementación de EmailSender",
        ],
        examples_en=[
            "show class User",
            "get class OrderService",
            "code of DatabaseManager",
            "implementation of ApiClient",
        ],
        target_patterns=[
            r"(?:clase|class)\s+(\w+)",
            r"(?:de|del|of)\s+(\w+)",
        ],
        default_params={"format": "markdown", "include_docstrings": True},
        confidence_threshold=0.70,
        priority=9
    ),

    "search_symbol": IntentDefinition(
        name="search_symbol",
        category=IntentCategory.CODE_QUERY,
        description="Búsqueda fuzzy de símbolos (métodos/clases/funciones)",
        target_tool="codebase_tool",
        action_name="search_symbol",
        requires_target=True,
        examples_es=[
            "busca el símbolo auth",
            "encuentra la clase User",
            "dónde está process",
            "localiza validate",
        ],
        examples_en=[
            "find symbol auth",
            "search for User",
            "locate process",
            "where is validate",
        ],
        target_patterns=[
            r"(?:busca|encuentra|localiza|find|search|locate)\s+(?:el|la|los|las)?\s*(?:símbolo|symbol)?\s*(\w+)",
        ],
        default_params={"format": "markdown"},
        confidence_threshold=0.65,
        priority=7
    ),

    "file_summary": IntentDefinition(
        name="file_summary",
        category=IntentCategory.CODE_QUERY,
        description="Resumen estructural de archivo (clases, métodos, LOC, imports)",
        target_tool="codebase_tool",
        action_name="file_summary",
        examples_es=[
            "resume el archivo",
            "estructura del archivo",
            "qué contiene el archivo",
            "overview del código",
            "resumen de la estructura",
        ],
        examples_en=[
            "summarize file",
            "file structure",
            "file overview",
            "what's in the file",
            "code summary",
        ],
        default_params={"format": "markdown"},
        confidence_threshold=0.65,
        priority=6
    ),

    # =========================================================================
    # CODE ANALYSIS INTENTS (para CodebaseTool - mantener funcionalidad avanzada)
    # =========================================================================

    "analyze_quality": IntentDefinition(
        name="analyze_quality",
        category=IntentCategory.CODE_ANALYSIS,
        description="Análisis profundo de calidad de código (metrics, smells, security)",
        target_tool="codebase_tool",
        action_name="analyze_quality",
        requires_target=True,
        examples_es=[
            "analiza la calidad del código",
            "sugiere mejoras para el método",
            "problemas de código",
            "code smells",
            "revisa la calidad",
            "vulnerabilidades de seguridad",
            "métricas de complejidad",
        ],
        examples_en=[
            "analyze quality",
            "suggest improvements",
            "code problems",
            "quality issues",
            "security vulnerabilities",
        ],
        target_patterns=[
            r"(?:de|del|of|para)\s+(?:el|la)?\s*(?:método|metodo|función|funcion)?\s*(\w+)",
        ],
        default_params={},
        confidence_threshold=0.68,
        priority=8
    ),

    "explain_code": IntentDefinition(
        name="explain_code",
        category=IntentCategory.CODE_ANALYSIS,
        description="Explicación detallada de qué hace un código",
        target_tool="codebase_tool",
        action_name="explain",
        requires_target=True,
        examples_es=[
            "qué hace este código",
            "explica el método",
            "describe la función",
            "para qué sirve",
            "cómo funciona",
        ],
        examples_en=[
            "what does this do",
            "explain method",
            "describe function",
            "how does it work",
        ],
        target_patterns=[
            r"(?:qué hace|que hace|what does|explica|explain|describe)\s+(?:el|la)?\s*(\w+)",
        ],
        default_params={},
        confidence_threshold=0.65,
        priority=7
    ),

    # =========================================================================
    # CODE SEARCH INTENTS (para CodebaseTool - graph-based search)
    # =========================================================================

    "find_definition": IntentDefinition(
        name="find_definition",
        category=IntentCategory.CODE_SEARCH,
        description="Encontrar definición de símbolo (dónde está definido)",
        target_tool="codebase_tool",
        action_name="find_definition",
        requires_target=True,
        examples_es=[
            "dónde está definido authenticate",
            "definición de User",
            "encuentra la definición de process",
            "ubicación de validate",
        ],
        examples_en=[
            "where is authenticate defined",
            "definition of User",
            "find definition of process",
            "locate validate",
        ],
        target_patterns=[
            r"(?:definido|defined|definición|definition|ubicación|location)\s+(?:de|del|of)?\s*(\w+)",
        ],
        default_params={},
        confidence_threshold=0.70,
        priority=9
    ),

    "find_references": IntentDefinition(
        name="find_references",
        category=IntentCategory.CODE_SEARCH,
        description="Encontrar referencias/usos de un símbolo",
        target_tool="codebase_tool",
        action_name="find_references",
        requires_target=True,
        examples_es=[
            "dónde se usa authenticate",
            "quién llama a validate",
            "referencias de User",
            "usos de process_data",
        ],
        examples_en=[
            "where is authenticate used",
            "who calls validate",
            "references to User",
            "usages of process_data",
        ],
        target_patterns=[
            r"(?:usa|used|llama|calls|referencias|references)\s+(?:a|de|to)?\s*(\w+)",
        ],
        default_params={},
        confidence_threshold=0.68,
        priority=8
    ),

    "get_callers": IntentDefinition(
        name="get_callers",
        category=IntentCategory.CODE_SEARCH,
        description="Obtener qué funciones llaman a un símbolo",
        target_tool="codebase_tool",
        action_name="get_callers",
        requires_target=True,
        examples_es=[
            "qué llama a authenticate",
            "quién invoca validate",
            "callers de process_data",
        ],
        examples_en=[
            "what calls authenticate",
            "who invokes validate",
            "callers of process_data",
        ],
        target_patterns=[
            r"(?:llama|calls|invoca|invokes)\s+(?:a)?\s*(\w+)",
        ],
        default_params={},
        confidence_threshold=0.68,
        priority=7
    ),

    "get_dependencies": IntentDefinition(
        name="get_dependencies",
        category=IntentCategory.CODE_SEARCH,
        description="Obtener dependencias de un archivo o módulo",
        target_tool="codebase_tool",
        action_name="get_dependencies",
        examples_es=[
            "qué importa el archivo",
            "dependencias del módulo",
            "módulos requeridos",
            "imports del archivo",
        ],
        examples_en=[
            "what imports the file",
            "module dependencies",
            "required modules",
            "file imports",
        ],
        default_params={},
        confidence_threshold=0.65,
        priority=6
    ),

    # =========================================================================
    # CONTENT INTENTS (para RAGTool - búsqueda semántica general)
    # =========================================================================

    "rag_search": IntentDefinition(
        name="rag_search",
        category=IntentCategory.CONTENT,
        description="Búsqueda semántica general en documentación/código",
        target_tool="rag_search",
        examples_es=[
            "busca información sobre",
            "qué dice sobre autenticación",
            "explica el proceso de",
            "háblame de",
            "cómo se hace",
        ],
        examples_en=[
            "search for information about",
            "find information about",
            "explain process",
            "tell me about",
            "how to",
        ],
        default_params={"k": 10, "score_threshold": 0.3},
        confidence_threshold=0.60,
        priority=5
    ),

    "conversational": IntentDefinition(
        name="conversational",
        category=IntentCategory.CONTENT,
        description="Pregunta conversacional o seguimiento de contexto",
        target_tool="rag_search",
        examples_es=[
            "y eso cómo funciona",
            "puedes explicar más",
            "qué significa eso",
            "y el anterior",
        ],
        examples_en=[
            "and how does that work",
            "can you explain more",
            "what does that mean",
            "and the previous one",
        ],
        default_params={"k": 5, "score_threshold": 0.4},
        confidence_threshold=0.55,
        priority=4
    ),
    # =============================================================================
    # AGREGAR en config.py - INTENT_REGISTRY
    # =============================================================================

    "contextual_reference": IntentDefinition(
        name="contextual_reference",
        category=IntentCategory.CODE_QUERY,
        description="Referencia contextual a archivo procesado anteriormente",
        target_tool="codebase_tool",
        examples_es=[
            "y el otro archivo",
            "y en el otro",
            "y el anterior",
            "qué hay en el otro archivo",
            "y el archivo anterior",
            "y en el previo",
            "muestra el otro",
            "analiza el anterior",
            "cuántos métodos tiene el otro",
            "y en el otro archivo cuántos métodos",
        ],
        examples_en=[
            "and the other file",
            "and the other",
            "and the previous",
            "what about the other file",
            "and the previous file",
            "show the other",
            "analyze the previous",
            "how many methods in the other",
        ],
        confidence_threshold=0.65,
        priority=8
    ),

}

def get_intents_by_registered_tool(registered_name: str) -> List[IntentDefinition]:
    """
    Obtiene todos los intents cuyo target_tool resuelve al nombre registrado.

    Ejemplo:
        get_intents_by_registered_tool("codebase_tool")
        → retorna intents con target_tool="codebase_tool" Y target_tool="codebase_tool"

    Usado por IntentRouter.score_tools_for_query() para agregar scores por tool.
    """
    return [
        intent for intent in INTENT_REGISTRY.values()
        if intent.target_tool == registered_name  # directo, sin alias
    ]

# =============================================================================
# Helper Functions
# =============================================================================

def get_intent(name: str) -> Optional[IntentDefinition]:
    """
    Obtener definición de intent por nombre.

    Args:
        name: Nombre del intent (ej: "count_methods")

    Returns:
        IntentDefinition o None si no existe
    """
    return INTENT_REGISTRY.get(name)

def get_intents_by_category(category: IntentCategory) -> List[IntentDefinition]:
    """
    Obtener todos los intents de una categoría.

    Args:
        category: Categoría (IntentCategory enum)

    Returns:
        Lista de IntentDefinition
    """
    return [
        intent for intent in INTENT_REGISTRY.values()
        if intent.category == category
    ]

def get_intents_by_tool(tool_name: str) -> List[IntentDefinition]:
    """
    Obtener todos los intents que usan un tool específico.

    Args:
        tool_name: Nombre del tool (ej: "codebase_tool")

    Returns:
        Lista de IntentDefinition
    """
    return [
        intent for intent in INTENT_REGISTRY.values()
        if intent.target_tool == tool_name
    ]

def get_all_training_examples() -> Dict[str, List[str]]:
    """
    Extrae todos los training examples de INTENT_REGISTRY.

    Returns:
        Dict[intent_name, List[examples]]
    """
    result = {}
    for intent_name, intent_def in INTENT_REGISTRY.items():
        examples = []

        # Combinar ejemplos en español
        if hasattr(intent_def, 'examples_es') and intent_def.examples_es:
            examples.extend(intent_def.examples_es)

        # Combinar ejemplos en inglés
        if hasattr(intent_def, 'examples_en') and intent_def.examples_en:
            examples.extend(intent_def.examples_en)

        # Fallback a campo antiguo si existe
        if not examples and hasattr(intent_def, 'training_examples'):
            examples = intent_def.training_examples

        if examples:
            result[intent_name] = examples
        #else:
            #logger.warning(f"Intent '{intent_name}' has no training examples")

    return result


def get_all_intent_names() -> List[str]:
    """Obtener lista de todos los nombres de intents."""
    return list(INTENT_REGISTRY.keys())

def validate_intent_registry() -> List[str]:
    """
    Validar consistencia del registry.

    Returns:
        Lista de warnings/errores encontrados
    """
    issues = []

    for name, intent in INTENT_REGISTRY.items():
        # Check name consistency
        if intent.name != name:
            issues.append(f"Intent '{name}': name mismatch ('{intent.name}')")

        # Check training examples
        if not intent.examples_es and not intent.examples_en:
            issues.append(f"Intent '{name}': no training examples")

        # Check target extraction
        if intent.requires_target and not intent.target_patterns:
            issues.append(f"Intent '{name}': requires_target but no patterns")

    return issues

# Validación al importar
_validation_issues = validate_intent_registry()
if _validation_issues:
    import warnings
    for issue in _validation_issues:
        warnings.warn(f"IntentRegistry validation: {issue}")
