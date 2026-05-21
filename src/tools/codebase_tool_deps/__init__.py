# =============================================================================
# src/tools/codebase_tool/__init__.py
# Professional Codebase Analysis Tool
# =============================================================================
"""
Professional Codebase Analysis Tool

A comprehensive static code analysis tool that provides:
- Advanced code quality metrics
- Code smell detection
- Security vulnerability scanning
- Refactoring suggestions
- Dependency analysis
- Multi-language support
- SARIF report generation
- LLM-optimized output

Usage:
    from src.tools.codebase_tool import CodebaseTool
    from src.config.intent_patterns import CodebaseAction

    tool = CodebaseTool()
    result = await tool.execute(
        action=CodebaseAction.ANALYZE_FILE,
        file_ids=["uuid"]
    )
"""

from .models import (
    CodeMetrics,
    CodeSmell,
    SecurityIssue,
    RefactoringSuggestion,
    AnalysisResult,
    FileAnalysisResult,
    SymbolInfo,
    CodeLocation,
    SeverityLevel,
    RefactoringType,
    SolidPrinciple,
    EffortEstimate,
    ImpactScope
)
from .metrics import MetricsCalculator
from .code_smells import CodeSmellDetector
from .security import SecurityAnalyzer
from .refactoring import RefactoringSuggester
from .sarif import SarifGenerator
from .llm_formatter import LLMFormatter

__version__ = "2.0.0"
__all__ = [
    # Models
    "CodeMetrics",
    "CodeSmell",
    "SecurityIssue",
    "RefactoringSuggestion",
    "AnalysisResult",
    "FileAnalysisResult",
    "SymbolInfo",
    "CodeLocation",
    "SeverityLevel",
    "RefactoringType",
    "SolidPrinciple",
    "EffortEstimate",
    "ImpactScope",
    # Components
    "MetricsCalculator",
    "CodeSmellDetector",
    "SecurityAnalyzer",
    "RefactoringSuggester",
    "SarifGenerator",
    "LLMFormatter"
]
