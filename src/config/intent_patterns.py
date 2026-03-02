# =============================================================================
# src/config/intent_patterns.py
# CodebaseAction Definitions (Cleaned)
# =============================================================================
"""
Codebase tool action definitions.

Este módulo define ÚNICAMENTE las acciones disponibles para CodebaseTool.

NOTA: Los patrones de intents (INTENT_PATTERNS) han sido movidos a:
      src/services/intent/config.py (IntentRouter system)

Este archivo se mantiene solo por backward compatibility con:
- CodebaseTool.execute(action=...)
- tool_executor.determine_execution_strategy()
- chat_orchestrator.py
"""
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass

# =============================================================================
# CODEBASE TOOL ACTIONS
# =============================================================================
@dataclass
class IntentStrategy:
    tools: List[str]
    skip_extraction: bool
    priority: Optional[str] = None
    extracted_params: Optional[Callable[[], Dict]] = None
    reasoning: str = ""

class CodebaseAction:
    """
    Definición centralizada de todas las acciones de codebase_tool.

    Estas acciones son usadas por:
    - CodebaseTool.execute(action=CodebaseAction.ANALYZE_FILE)
    - tool_executor.determine_execution_strategy()
    - orchestrator al construir parámetros

    Uses snake_case para estandarización.
    """

    # Action definitions
    ANALYZE_FILE = "analyze_file"              # Full analysis (quality + structure)
    BASIC_ANALYZE_FILE = "basic_analyze_file"  # Structure only (no quality metrics)
    FIND_DEFINITION = "find_definition"
    FIND_REFERENCES = "find_references"
    GET_CALLERS = "get_callers"
    GET_DEPENDENCIES = "get_dependencies"
    REFRESH_GRAPH = "refresh_graph"
    ANALYZE_QUALITY = "analyze_quality"
    EXPLAIN = "explain"
    GET_METHOD_CONTENT = "get_method_content"
    MODIFY_METHOD = "modify_method"

    # All available actions
    ALL_ACTIONS = [
        ANALYZE_FILE,
        BASIC_ANALYZE_FILE,  # Structure-only analysis
        FIND_DEFINITION,
        FIND_REFERENCES,
        GET_CALLERS,
        GET_DEPENDENCIES,
        REFRESH_GRAPH,
        ANALYZE_QUALITY,
        EXPLAIN,
        GET_METHOD_CONTENT,
        MODIFY_METHOD,
    ]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def create_intent_strategy(intent_type: str, target: Optional[str] = None,
                           confidence: float = 0.0) -> IntentStrategy:
    """Fábrica de estrategias por tipo de intent. SINGLE SOURCE OF TRUTH."""

    strategies = {
        # ⭐ CONTEO ESTRUCTURA (ligero)
        "count_structure": IntentStrategy(
            tools=["codebasetool"],
            skip_extraction=True,
            priority="codebasetool",
            extracted_params=lambda: {"action": CodebaseAction.BASIC_ANALYZE_FILE,
                                      "target": target},
            reasoning=f"Count structure {target or 'file'} conf={confidence:.2f}"
        ),

        # ⭐ ANÁLISIS COMPLETO (pesado)
        "full_analysis": IntentStrategy(
            tools=["codebasetool", "ragsearch"],
            skip_extraction=True,
            priority="codebasetool",
            extracted_params=lambda: {"action": CodebaseAction.ANALYZE_FILE, "target": target},
            reasoning=f"Full analysis {target or 'file'} conf={confidence:.2f}"
        ),

        # Análisis de calidad específico
        "analyze_quality": IntentStrategy(
            tools=["codebasetool"],
            skip_extraction=True,
            priority="codebasetool",
            extracted_params=lambda: {"action": CodebaseAction.ANALYZE_QUALITY, "target": target},
            reasoning=f"Quality analysis {target} conf={confidence:.2f}"
        ),

        # Búsqueda definición
        "find_definition": IntentStrategy(
            tools=["codebasetool"],
            skip_extraction=True,
            priority="codebasetool",
            extracted_params=lambda: {"action": CodebaseAction.FIND_DEFINITION, "target": target},
            reasoning=f"Find definition {target} conf={confidence:.2f}"
        ),

        # Contenido general (RAG)
        "retrieve_content": IntentStrategy(
            tools=["ragsearch"],
            skip_extraction=False,
            priority="ragsearch",
            extracted_params=lambda: {"k": 10, "score_threshold": 0.2},
            reasoning=f"Retrieve content for {target} conf={confidence:.2f}"
        ),

        # Query general
        "general_query": IntentStrategy(
            tools=["ragsearch"],
            skip_extraction=False,
            priority="ragsearch",
            extracted_params=lambda: {"k": 15, "score_threshold": 0.3},
            reasoning=f"General query conf={confidence:.2f}"
        ),
    }

    return strategies.get(intent_type, strategies["general_query"])


# Patrones para detect_code_intent (solo nombres de intents)
COUNT_PATTERNS = [
    r"cuantos?\s+(métodos?|metodos?|funciones?|clases?)",
    r"cuántos?\s+(métodos?|metodos?|funciones?|clases?)",
    r"lista?\s+(métodos?|metodos?|funciones?|clases?)",
    r"cuántas?\s+(clases?|funciones?)",
]

QUALITY_PATTERNS = [
    r"(calidad|issues|smells|mejora|refactor)",
]
def is_valid_action(action: str) -> bool:
    """
    Verifica si una acción es válida para codebase_tool.

    Args:
        action: Nombre de la acción

    Returns:
        True si la acción es válida, False si no
    """
    return action in CodebaseAction.ALL_ACTIONS

def get_all_actions() -> list:
    """
    Obtiene todas las acciones disponibles.

    Returns:
        Lista de nombres de acciones
    """
    return CodebaseAction.ALL_ACTIONS.copy()
