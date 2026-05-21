# =============================================================================
# src/config/intent_patterns.py
# CodebaseAction Definitions
# =============================================================================
"""
Define las acciones disponibles para CodebaseTool.

Usado directamente por:
- CodebaseTool.execute(action=...)
- tool_executor.determine_execution_strategy()
- codebase_tool/core.py, llm_formatter.py
"""

# =============================================================================

class CodebaseAction:
    """
    Constantes de acción para codebase_tool.

    Cada constante corresponde al valor del parámetro `action`
    que recibe CodebaseTool.execute().
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
    GET_CLASS_CONTENT ="get_class_content"
    EXPORT_REFACTORED = "export_refactored"   # NEW: Generate safe versioned copy with fixes

    MODIFY_METHOD_ALL_CALLERS = "modify_method_all_callers" # Propagate changes to callers

    # All available actions (used for parameter validation in get_parameters())
    ALL_ACTIONS = [
        ANALYZE_FILE,
        BASIC_ANALYZE_FILE,
        FIND_DEFINITION,
        FIND_REFERENCES,
        GET_CALLERS,
        GET_DEPENDENCIES,
        REFRESH_GRAPH,
        ANALYZE_QUALITY,
        EXPLAIN,
        GET_METHOD_CONTENT,
        MODIFY_METHOD,
        GET_CLASS_CONTENT,
        EXPORT_REFACTORED,
        MODIFY_METHOD_ALL_CALLERS,
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
