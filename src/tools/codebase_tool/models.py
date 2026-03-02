# =============================================================================
# src/tools/codebase_tool/models.py
# Professional Data Models for Codebase Analysis
# =============================================================================
"""
Professional data models for codebase analysis results.
Provides type-safe, well-documented structures for all analysis outputs.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Set
from datetime import datetime


# =============================================================================
# Enums
# =============================================================================

class SeverityLevel(Enum):
    """Severity levels for issues and suggestions."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class CodeSmellType(Enum):
    """Types of code smells."""
    DUPLICATED_CODE = "duplicated_code"
    LONG_METHOD = "long_method"
    LARGE_CLASS = "large_class"
    FEATURE_ENVY = "feature_envvy"
    INAPPROPRIATE_INTIMACY = "inappropriate_intimacy"
    LAZY_CLASS = "lazy_class"
    DATA_CLUMPS = "data_clumps"
    PRIMITIVE_OBSESSION = "primitive_obsession"
    SWITCH_STATEMENTS = "switch_statements"
    TEMPORARY_FIELD = "temporary_field"
    REFUSED_BEQUEST = "refused_bequest"
    COMMENTS = "comments"
    GOD_OBJECT = "god_object"
    SPAGHETTI_CODE = "spaghetti_code"


class SecurityIssueType(Enum):
    """Types of security vulnerabilities."""
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    HARDCODED_CREDENTIALS = "hardcoded_credentials"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    PATH_TRAVERSAL = "path_traversal"
    WEAK_CRYPTOGRAPHY = "weak_cryptography"
    COMMAND_INJECTION = "command_injection"
    INSECURE_RANDOM = "insecure_random"
    OPEN_REDIRECT = "open_redirect"
    SSRF = "ssrf"


class RefactoringType(Enum):
    """Types of refactoring operations."""
    EXTRACT_METHOD = "extract_method"
    EXTRACT_CLASS = "extract_class"
    INLINE_METHOD = "inline_method"
    REPLACE_CONDITIONAL_WITH_POLYMORPHISM = "replace_conditional_with_polymorphism"
    DECOMPOSE_CONDITIONAL = "decompose_conditional"
    CONSOLIDATE_CONDITIONAL = "consolidate_conditional"
    REPLACE_MAGIC_NUMBER = "replace_magic_number"
    INTRODUCE_PARAMETER_OBJECT = "introduce_parameter_object"
    PRESERVE_WHOLE_OBJECT = "preserve_whole_object"
    REPLACE_TYPE_CODE_WITH_CLASS = "replace_type_code_with_class"
    RENAME = "rename"
    MOVE_METHOD = "move_method"
    MOVE_FIELD = "move_field"
    EXTRACT_INTERFACE = "extract_interface"


class SolidPrinciple(Enum):
    """SOLID principles."""
    SRP = "single_responsibility"
    OCP = "open_closed"
    LSP = "liskov_substitution"
    ISP = "interface_segregation"
    DIP = "dependency_inversion"


class EffortEstimate(Enum):
    """Effort estimates for refactoring."""
    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"
    WEEKS = "weeks"


class ImpactScope(Enum):
    """Scope of impact for changes."""
    LOCAL = "local"
    MODULE = "module"
    SYSTEM = "system"


# =============================================================================
# Location Models
# =============================================================================

@dataclass
class CodeLocation:
    """Represents a location in source code."""
    file_path: str
    start_line: int
    end_line: int
    start_column: Optional[int] = None
    end_column: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "start_column": self.start_column,
            "end_column": self.end_column
        }


# =============================================================================
# Metrics Models
# =============================================================================

@dataclass
class ComplexityMetrics:
    """Complexity metrics for code analysis."""
    cyclomatic_complexity: int
    cognitive_complexity: int
    nesting_depth: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "cognitive_complexity": self.cognitive_complexity,
            "nesting_depth": self.nesting_depth
        }


@dataclass
class SizeMetrics:
    """Size metrics for code analysis."""
    lines_of_code: int
    lines_of_comments: int
    blank_lines: int
    comment_ratio: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "lines_of_code": self.lines_of_code,
            "lines_of_comments": self.lines_of_comments,
            "blank_lines": self.blank_lines,
            "comment_ratio": self.comment_ratio
        }


@dataclass
class MaintainabilityMetrics:
    """Maintainability metrics."""
    maintainability_index: float
    technical_debt_hours: float
    technical_debt_ratio: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "maintainability_index": self.maintainability_index,
            "technical_debt_hours": self.technical_debt_hours,
            "technical_debt_ratio": self.technical_debt_ratio
        }


@dataclass
class CouplingMetrics:
    """Coupling metrics."""
    afferent_coupling: int  # Ca: classes that depend on this
    efferent_coupling: int  # Ce: classes this depends on
    instability: float  # Ce / (Ca + Ce)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "afferent_coupling": self.afferent_coupling,
            "efferent_coupling": self.efferent_coupling,
            "instability": self.instability
        }


@dataclass
class CohesionMetrics:
    """Cohesion metrics."""
    lack_of_cohesion_methods: float  # LCOM4
    cohesion_ratio: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "lack_of_cohesion_methods": self.lack_of_cohesion_methods,
            "cohesion_ratio": self.cohesion_ratio
        }


@dataclass
class DuplicationMetrics:
    """Code duplication metrics."""
    duplication_ratio: float
    duplicated_lines: int
    duplicated_blocks: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "duplication_ratio": self.duplication_ratio,
            "duplicated_lines": self.duplicated_lines,
            "duplicated_blocks": self.duplicated_blocks
        }


@dataclass
class CodeMetrics:
    """Comprehensive code metrics."""
    complexity: ComplexityMetrics
    size: SizeMetrics
    maintainability: MaintainabilityMetrics
    coupling: CouplingMetrics
    cohesion: CohesionMetrics
    duplication: DuplicationMetrics
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "complexity": self.complexity.to_dict(),
            "size": self.size.to_dict(),
            "maintainability": self.maintainability.to_dict(),
            "coupling": self.coupling.to_dict(),
            "cohesion": self.cohesion.to_dict(),
            "duplication": self.duplication.to_dict()
        }


# =============================================================================
# Issue Models
# =============================================================================

@dataclass
class CodeSmell:
    """Represents a code smell detected in the code."""
    smell_type: CodeSmellType
    severity: SeverityLevel
    location: CodeLocation
    description: str
    rationale: str
    affected_symbols: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.smell_type.value,
            "severity": self.severity.value,
            "location": self.location.to_dict(),
            "description": self.description,
            "rationale": self.rationale,
            "affected_symbols": self.affected_symbols
        }


@dataclass
class SecurityIssue:
    """Represents a security vulnerability."""
    issue_type: SecurityIssueType
    severity: SeverityLevel
    location: CodeLocation
    description: str
    cwe_id: Optional[str] = None  # CWE identifier
    owasp_category: Optional[str] = None  # OWASP category
    remediation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.issue_type.value,
            "severity": self.severity.value,
            "location": self.location.to_dict(),
            "description": self.description,
            "cwe_id": self.cwe_id,
            "owasp_category": self.owasp_category,
            "remediation": self.remediation
        }


@dataclass
class SolidViolation:
    """Represents a violation of SOLID principles."""
    principle: SolidPrinciple
    severity: SeverityLevel
    location: CodeLocation
    description: str
    explanation: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "principle": self.principle.value,
            "severity": self.severity.value,
            "location": self.location.to_dict(),
            "description": self.description,
            "explanation": self.explanation
        }


# =============================================================================
# Refactoring Models
# =============================================================================

@dataclass
class RefactoringSuggestion:
    """Represents a refactoring suggestion."""
    refactoring_type: RefactoringType
    severity: SeverityLevel
    location: CodeLocation
    description: str
    rationale: str
    suggested_code: Optional[str] = None
    effort: EffortEstimate = EffortEstimate.HOURS
    impact: ImpactScope = ImpactScope.LOCAL
    benefits: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.refactoring_type.value,
            "severity": self.severity.value,
            "location": self.location.to_dict(),
            "description": self.description,
            "rationale": self.rationale,
            "suggested_code": self.suggested_code,
            "effort": self.effort.value,
            "impact": self.impact.value,
            "benefits": self.benefits
        }


# =============================================================================
# Analysis Result Models
# =============================================================================

@dataclass
class SymbolInfo:
    """Information about a code symbol (class, function, method)."""
    name: str
    symbol_type: str  # 'class', 'function', 'method'
    location: CodeLocation
    docstring: Optional[str]
    decorators: List[str]
    dependencies: Set[str]
    parent: Optional[str]
    content: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.symbol_type,
            "location": self.location.to_dict(),
            "docstring": self.docstring,
            "decorators": self.decorators,
            "dependencies": list(self.dependencies),
            "parent": self.parent,
            "content": self.content
        }


@dataclass
class FileAnalysisResult:
    """Result of analyzing a single file."""
    file_id: Optional[str]
    file_path: str
    language: str
    
    # Structure
    symbols: List[SymbolInfo] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    
    # Metrics
    metrics: Optional[CodeMetrics] = None
    
    # Issues
    code_smells: List[CodeSmell] = field(default_factory=list)
    security_issues: List[SecurityIssue] = field(default_factory=list)
    solid_violations: List[SolidViolation] = field(default_factory=list)
    
    # Suggestions
    refactoring_suggestions: List[RefactoringSuggestion] = field(default_factory=list)
    
    # Content
    content: str = ""
    
    # Metadata
    analyzed_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_id": self.file_id,
            "file_path": self.file_path,
            "language": self.language,
            "symbols": [s.to_dict() for s in self.symbols],
            "classes": self.classes,
            "functions": self.functions,
            "imports": self.imports,
            "metrics": self.metrics.to_dict(),
            "code_smells": [cs.to_dict() for cs in self.code_smells],
            "security_issues": [si.to_dict() for si in self.security_issues],
            "solid_violations": [sv.to_dict() for sv in self.solid_violations],
            "refactoring_suggestions": [rs.to_dict() for rs in self.refactoring_suggestions],
            "content": self.content,
            "analyzed_at": self.analyzed_at.isoformat()
        }


@dataclass
class AnalysisResult:
    """Complete result of codebase analysis."""
    action: str
    target: Optional[str]
    files_analyzed: int
    results: List[FileAnalysisResult]
    
    # Summary metrics
    total_code_smells: int = 0
    total_security_issues: int = 0
    total_solid_violations: int = 0
    total_refactoring_suggestions: int = 0
    
    # Severity breakdown
    critical_issues: int = 0
    high_issues: int = 0
    medium_issues: int = 0
    low_issues: int = 0
    
    def calculate_summary(self):
        """Calculate summary statistics from results."""
        for result in self.results:
            self.total_code_smells += len(result.code_smells)
            self.total_security_issues += len(result.security_issues)
            self.total_solid_violations += len(result.solid_violations)
            self.total_refactoring_suggestions += len(result.refactoring_suggestions)
            
            for issue in result.code_smells + result.security_issues:
                if issue.severity == SeverityLevel.CRITICAL:
                    self.critical_issues += 1
                elif issue.severity == SeverityLevel.HIGH:
                    self.high_issues += 1
                elif issue.severity == SeverityLevel.MEDIUM:
                    self.medium_issues += 1
                elif issue.severity == SeverityLevel.LOW:
                    self.low_issues += 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "target": self.target,
            "files_analyzed": self.files_analyzed,
            "results": [r.to_dict() for r in self.results],
            "summary": {
                "total_code_smells": self.total_code_smells,
                "total_security_issues": self.total_security_issues,
                "total_solid_violations": self.total_solid_violations,
                "total_refactoring_suggestions": self.total_refactoring_suggestions,
                "severity_breakdown": {
                    "critical": self.critical_issues,
                    "high": self.high_issues,
                    "medium": self.medium_issues,
                    "low": self.low_issues
                }
            }
        }


# =============================================================================
# Dependency Models
# =============================================================================

@dataclass
class DependencyInfo:
    """Information about a dependency."""
    name: str
    type: str  # 'internal', 'external', 'standard_library'
    location: CodeLocation
    usage_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "location": self.location.to_dict(),
            "usage_count": self.usage_count
        }


@dataclass
class CircularDependency:
    """Represents a circular dependency."""
    cycle: List[str]  # List of symbols in the cycle
    severity: SeverityLevel
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle": self.cycle,
            "severity": self.severity.value,
            "description": self.description
        }


@dataclass
class DependencyAnalysisResult:
    """Result of dependency analysis."""
    dependencies: List[DependencyInfo]
    circular_dependencies: List[CircularDependency]
    coupling_metrics: CouplingMetrics
    cohesion_metrics: CohesionMetrics
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dependencies": [d.to_dict() for d in self.dependencies],
            "circular_dependencies": [cd.to_dict() for cd in self.circular_dependencies],
            "coupling_metrics": self.coupling_metrics.to_dict(),
            "cohesion_metrics": self.cohesion_metrics.to_dict()
        }
