# =============================================================================
# src/tools/codebase_tool/llm_formatter.py
# Professional LLM-Optimized Output Formatter
# =============================================================================
"""
Professional LLM-optimized output formatter.
Formats analysis results for optimal consumption by Large Language Models.
"""

from datetime import datetime
from typing import List, Optional

from src.config.intent_patterns import CodebaseAction
from .models import (
    AnalysisResult,
    FileAnalysisResult,
    CodeMetrics,
    CodeSmell,
    SecurityIssue,
    RefactoringSuggestion,
    SeverityLevel
)


class LLMFormatter:
    """
    Professional LLM output formatter.

    Formats analysis results for optimal LLM consumption:
    - Structured markdown output
    - Rich context for analysis
    - Actionable suggestions
    - Clear severity indicators
    """

    def __init__(self, include_code: bool = True, max_code_length: int = 500):
        self.include_code = include_code
        self.max_code_length = max_code_length

    # =========================================================================
    # Main Formatting Methods
    # =========================================================================

    def format_analysis_result(
        self,
        analysis_result: AnalysisResult,
        format_type: str = "detailed",
        sub_action: Optional[str] = None   # ← NUEVO
    ) -> str:
        """
        Format complete analysis result for LLM consumption.

        Args:
            analysis_result: Complete analysis result
            format_type: "detailed", "summary", or "compact"

        Returns:
            Formatted markdown string
        """
        if format_type == "basic":
            return self._format_basic(analysis_result, sub_action=sub_action)  # pasa sub_action

        # Check action for specialized formatting
        if analysis_result.action == CodebaseAction.BASIC_ANALYZE_FILE or format_type == "basic":
            return self._format_structural_report(analysis_result)

        if format_type == "summary":
            return self._format_summary(analysis_result)
        elif format_type == "compact":
            return self._format_compact(analysis_result)
        else:
            return self._format_detailed(analysis_result)

    def format_file_result(
        self,
        file_result: FileAnalysisResult,
        format_type: str = "detailed",
        action: Optional[str] = None
    ) -> str:
        """
        Format single file analysis result.

        Args:
            file_result: File analysis result
            format_type: "detailed", "summary", or "compact"

        Returns:
            Formatted markdown string
        """
        if action == CodebaseAction.BASIC_ANALYZE_FILE or format_type == "basic":
            return self._format_file_structural_only(file_result)

        if format_type == "summary":
            return self._format_file_summary(file_result)
        elif format_type == "compact":
            return self._format_file_compact(file_result)
        else:
            return self._format_file_detailed(file_result)

    # =========================================================================
    # Detailed Format
    # =========================================================================

    def _format_detailed(self, analysis_result: AnalysisResult) -> str:
        """Format detailed analysis result."""
        # Calculate summary if not already done
        if analysis_result.total_code_smells == 0:
            analysis_result.calculate_summary()

        sections = []

        # Header
        sections.append(self._format_header(analysis_result))

        # Executive Summary
        sections.append(self._format_executive_summary(analysis_result))

        # File-by-file analysis
        for file_result in analysis_result.results:
            sections.append(self._format_file_detailed(file_result))

        # Overall recommendations
        sections.append(self._format_overall_recommendations(analysis_result))

        return "\n\n".join(sections)

    def _format_header(self, analysis_result: AnalysisResult) -> str:
        """Format analysis header."""
        return f"""# Codebase Analysis Report

**Action:** {analysis_result.action}
**Target:** {analysis_result.target or 'N/A'}
**Files Analyzed:** {analysis_result.files_analyzed}
**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}

---
"""

    def _format_executive_summary(self, analysis_result: AnalysisResult) -> str:
        """Format executive summary."""
        summary = analysis_result.to_dict().get('summary', {})

        return f"""## Executive Summary

### Issue Overview
- **Code Smells:** {summary.get('total_code_smells', 0)}
- **Security Issues:** {summary.get('total_security_issues', 0)}
- **SOLID Violations:** {summary.get('total_solid_violations', 0)}
- **Refactoring Suggestions:** {summary.get('total_refactoring_suggestions', 0)}

### Severity Breakdown
- 🔴 **Critical:** {summary.get('severity_breakdown', {}).get('critical', 0)}
- 🟠 **High:** {summary.get('severity_breakdown', {}).get('high', 0)}
- 🟡 **Medium:** {summary.get('severity_breakdown', {}).get('medium', 0)}
- 🟢 **Low:** {summary.get('severity_breakdown', {}).get('low', 0)}

---
"""

    def _format_file_detailed(self, file_result: FileAnalysisResult) -> str:
        """Format detailed file analysis."""
        sections = []

        # File header
        sections.append(f"## 📄 {file_result.file_path}")
        sections.append(f"**Language:** {file_result.language}")
        sections.append("")

        # Metrics
        sections.append(self._format_metrics(file_result.metrics))

        # Structure
        sections.append(self._format_structure(file_result))

        # Code smells
        if file_result.code_smells:
            sections.append(self._format_code_smells(file_result.code_smells))

        # Security issues
        if file_result.security_issues:
            sections.append(self._format_security_issues(file_result.security_issues))

        # Refactoring suggestions
        if file_result.refactoring_suggestions:
            sections.append(self._format_refactoring_suggestions(file_result.refactoring_suggestions))

        # Code snippet
        if self.include_code and file_result.content:
            sections.append(self._format_code_snippet(file_result))

        return "\n".join(sections)

    def _format_metrics(self, metrics: CodeMetrics) -> str:
        """Format code metrics."""
        return f"""### 📊 Metrics

**Complexity:**
- Cyclomatic: {metrics.complexity.cyclomatic_complexity}
- Cognitive: {metrics.complexity.cognitive_complexity}
- Nesting Depth: {metrics.complexity.nesting_depth}

**Size:**
- Lines of Code: {metrics.size.lines_of_code}
- Lines of Comments: {metrics.size.lines_of_comments}
- Comment Ratio: {metrics.size.comment_ratio:.1%}

**Maintainability:**
- Maintainability Index: {metrics.maintainability.maintainability_index:.1f}
- Technical Debt: {metrics.maintainability.technical_debt_hours:.1f} hours
- Technical Debt Ratio: {metrics.maintainability.technical_debt_ratio:.1%}

**Coupling:**
- Afferent (Ca): {metrics.coupling.afferent_coupling}
- Efferent (Ce): {metrics.coupling.efferent_coupling}
- Instability: {metrics.coupling.instability:.2f}

**Cohesion:**
- LCOM4: {metrics.cohesion.lack_of_cohesion_methods:.1f}
- Cohesion Ratio: {metrics.cohesion.cohesion_ratio:.1%}

**Duplication:**
- Duplication Ratio: {metrics.duplication.duplication_ratio:.1%}
- Duplicated Lines: {metrics.duplication.duplicated_lines}

---
"""

    def _format_structure(self, file_result: FileAnalysisResult) -> str:
        """Format code structure."""
        sections = []

        sections.append("### Structure")

        if file_result.classes:
            sections.append(f"**Classes ({len(file_result.classes)}):**")
            for cls in file_result.classes:
                sections.append(f"  - `{cls}`")
            sections.append("")

        if file_result.functions:
            sections.append(f"**Functions ({len(file_result.functions)}):**")
            for func in file_result.functions:
                sections.append(f"  - `{func}`")
            sections.append("")

        if file_result.imports:
            sections.append(f"**Imports ({len(file_result.imports)}):**")
            for imp in file_result.imports[:10]:  # Limit to 10
                sections.append(f"  - `{imp}`")
            if len(file_result.imports) > 10:
                sections.append(f"  - ... and {len(file_result.imports) - 10} more")
            sections.append("")

        sections.append("---")

        return "\n".join(sections)

    def _format_code_smells(self, code_smells: List[CodeSmell]) -> str:
        """Format code smells."""
        sections = []

        sections.append("### 🚨 Code Smells")

        for smell in code_smells:
            severity_emoji = self._get_severity_emoji(smell.severity)
            sections.append(f"{severity_emoji} **{smell.smell_type.value.replace('_', ' ').title()}** (Line {smell.location.start_line})")
            sections.append(f"  - **Description:** {smell.description}")
            sections.append(f"  - **Rationale:** {smell.rationale}")
            sections.append("")

        sections.append("---")

        return "\n".join(sections)

    def _format_security_issues(self, security_issues: List[SecurityIssue]) -> str:
        """Format security issues."""
        sections = []

        sections.append("### 🔒 Security Issues")

        for issue in security_issues:
            severity_emoji = self._get_severity_emoji(issue.severity)
            sections.append(f"{severity_emoji} **{issue.issue_type.value.replace('_', ' ').title()}** (Line {issue.location.start_line})")
            sections.append(f"  - **Description:** {issue.description}")

            if issue.cwe_id:
                sections.append(f"  - **CWE:** {issue.cwe_id}")
            if issue.owasp_category:
                sections.append(f"  - **OWASP:** {issue.owasp_category}")
            if issue.remediation:
                sections.append(f"  - **Remediation:** {issue.remediation}")

            sections.append("")

        sections.append("---")

        return "\n".join(sections)

    def _format_refactoring_suggestions(self, suggestions: List[RefactoringSuggestion]) -> str:
        """Format refactoring suggestions."""
        sections = []

        sections.append("### 💡 Refactoring Suggestions")

        for suggestion in suggestions:
            severity_emoji = self._get_severity_emoji(suggestion.severity)
            sections.append(f"{severity_emoji} **{suggestion.refactoring_type.value.replace('_', ' ').title()}** (Line {suggestion.location.start_line})")
            sections.append(f"  - **Description:** {suggestion.description}")
            sections.append(f"  - **Rationale:** {suggestion.rationale}")
            sections.append(f"  - **Effort:** {suggestion.effort.value}")
            sections.append(f"  - **Impact:** {suggestion.impact.value}")

            if suggestion.benefits:
                sections.append(f"  - **Benefits:**")
                for benefit in suggestion.benefits:
                    sections.append(f"    - {benefit}")

            if suggestion.suggested_code:
                sections.append(f"  - **Suggested Code:**")
                sections.append(f"```python")
                sections.append(suggestion.suggested_code)
                sections.append(f"```")

            sections.append("")

        sections.append("---")

        return "\n".join(sections)

    def _format_code_snippet(self, file_result: FileAnalysisResult) -> str:
        """Format code snippet."""
        content = file_result.content

        if len(content) > self.max_code_length:
            content = content[:self.max_code_length] + "\n... (truncated)"

        return f"""### 📝 Code Snippet

```{file_result.language}
{content}
```

---
"""

    def _format_overall_recommendations(self, analysis_result: AnalysisResult) -> str:
        """Format overall recommendations."""
        sections = []

        sections.append("## 📋 Overall Recommendations")

        # Collect all critical and high issues
        critical_issues = []
        high_issues = []

        for file_result in analysis_result.results:
            for issue in file_result.code_smells + file_result.security_issues:
                if issue.severity == SeverityLevel.CRITICAL:
                    critical_issues.append((file_result.file_path, issue))
                elif issue.severity == SeverityLevel.HIGH:
                    high_issues.append((file_result.file_path, issue))

        if critical_issues:
            sections.append("### 🔴 Critical Issues (Address Immediately)")
            for file_path, issue in critical_issues:
                sections.append(f"- **{file_path}:{issue.location.start_line}** - {issue.description}")
            sections.append("")

        if high_issues:
            sections.append("### 🟠 High Priority Issues")
            for file_path, issue in high_issues:
                sections.append(f"- **{file_path}:{issue.location.start_line}** - {issue.description}")
            sections.append("")

        sections.append("### 📊 Next Steps")
        sections.append("1. Address all critical security issues immediately")
        sections.append("2. Prioritize high-severity code smells")
        sections.append("3. Implement refactoring suggestions to improve maintainability")
        sections.append("4. Consider setting up automated code quality checks")

        return "\n".join(sections)

    # =========================================================================
    # Summary Format
    # =========================================================================

    def _format_summary(self, analysis_result: AnalysisResult) -> str:
        """Format summary analysis result."""
        if analysis_result.total_code_smells == 0:
            analysis_result.calculate_summary()

        summary = analysis_result.to_dict().get('summary', {})

        return f"""# Codebase Analysis Summary

**Files Analyzed:** {analysis_result.files_analyzed}

## Issue Overview
- **Code Smells:** {summary.get('total_code_smells', 0)}
- **Security Issues:** {summary.get('total_security_issues', 0)}
- **SOLID Violations:** {summary.get('total_solid_violations', 0)}
- **Refactoring Suggestions:** {summary.get('total_refactoring_suggestions', 0)}

## Severity Breakdown
- 🔴 **Critical:** {summary.get('severity_breakdown', {}).get('critical', 0)}
- 🟠 **High:** {summary.get('severity_breakdown', {}).get('high', 0)}
- 🟡 **Medium:** {summary.get('severity_breakdown', {}).get('medium', 0)}
- 🟢 **Low:** {summary.get('severity_breakdown', {}).get('low', 0)}

## Files with Issues
"""

    def _format_file_summary(self, file_result: FileAnalysisResult) -> str:
        """Format file summary."""
        critical_count = sum(1 for i in file_result.code_smells + file_result.security_issues if i.severity == SeverityLevel.CRITICAL)
        high_count = sum(1 for i in file_result.code_smells + file_result.security_issues if i.severity == SeverityLevel.HIGH)

        return f"""### 📄 {file_result.file_path}
**Language:** {file_result.language} | **Critical:** {critical_count} | **High:** {high_count}

**Metrics:**
- Maintainability Index: {file_result.metrics.maintainability.maintainability_index:.1f}
- Cyclomatic Complexity: {file_result.metrics.complexity.cyclomatic_complexity}
- Technical Debt: {file_result.metrics.maintainability.technical_debt_hours:.1f}h

**Issues:**
- Code Smells: {len(file_result.code_smells)}
- Security Issues: {len(file_result.security_issues)}
- Refactoring Suggestions: {len(file_result.refactoring_suggestions)}
"""

    # =========================================================================
    # Compact Format
    # =========================================================================

    def _format_compact(self, analysis_result: AnalysisResult) -> str:
        """Format compact analysis result."""
        if analysis_result.total_code_smells == 0:
            analysis_result.calculate_summary()

        summary = analysis_result.to_dict().get('summary', {})

        lines = []
        lines.append(f"Codebase Analysis: {analysis_result.files_analyzed} files")
        lines.append(f"Critical: {summary.get('severity_breakdown', {}).get('critical', 0)} | "
                   f"High: {summary.get('severity_breakdown', {}).get('high', 0)} | "
                   f"Medium: {summary.get('severity_breakdown', {}).get('medium', 0)} | "
                   f"Low: {summary.get('severity_breakdown', {}).get('low', 0)}")
        lines.append("")

        for file_result in analysis_result.results:
            lines.append(self._format_file_compact(file_result))

        return "\n".join(lines)

    def _format_file_compact(self, file_result: FileAnalysisResult) -> str:
        """Format file compact."""
        critical_count = sum(1 for i in file_result.code_smells + file_result.security_issues if i.severity == SeverityLevel.CRITICAL)
        high_count = sum(1 for i in file_result.code_smells + file_result.security_issues if i.severity == SeverityLevel.HIGH)

        return (f"📄 {file_result.file_path} [{file_result.language}] "
                f"MI:{file_result.metrics.maintainability.maintainability_index:.0f} "
                f"CC:{file_result.metrics.complexity.cyclomatic_complexity} "
                f"🔴{critical_count} 🟠{high_count}")

    # =========================================================================
    # Structural Only Format (New)
    # =========================================================================

    def _format_file_structural_only(self, file_result: FileAnalysisResult) -> str:
        """
        Format file with ONLY structural information.
        NO quality metrics, NO code smells, NO complexity.

        Used for basic_analyze_file queries like "cuantos metodos tiene".
        """
        sections = []

        # File header
        sections.append(f"## 📄 {file_result.file_path}")
        sections.append(f"**Language:** {file_result.language}")
        sections.append("")

        # Structure ONLY
        sections.append("### Structure")

        if file_result.classes:
            sections.append(f"**Classes ({len(file_result.classes)}):**")
            for cls in file_result.classes:
                sections.append(f"  - `{cls}`")
            sections.append("")

        if file_result.functions:
            sections.append(f"**Functions ({len(file_result.functions)}):**")
            for func in file_result.functions:
                sections.append(f"  - `{func}`")
            sections.append("")

        # Summary line
        class_count = len(file_result.classes) if file_result.classes else 0
        func_count = len(file_result.functions) if file_result.functions else 0
        sections.append(f"**Total Summary:** {class_count} classes, {func_count} functions")

        return "\n".join(sections)

    def _format_structural_report(self, analysis_result: AnalysisResult) -> str:
        """Format structural-only report for basic_analyze_file."""
        sections = []

        # Header (Simple)
        sections.append(f"# Codebase Structure Report")
        sections.append(f"**Files Analyzed:** {analysis_result.files_analyzed}")
        sections.append("")

        # File-by-file structural analysis
        for file_result in analysis_result.results:
            sections.append(self._format_file_structural_only(file_result))

        return "\n\n".join(sections)

    # =========================================================================
    # Prompt Templates
    # =========================================================================

    def format_for_refactoring_prompt(self, file_result: FileAnalysisResult) -> str:
        """Format file result for refactoring prompt."""
        return f"""# Refactoring Request

## File: {file_result.file_path}
**Language:** {file_result.language}

## Current Code
```{file_result.language}
{file_result.content}
```

## Issues Found
{self._format_issues_for_prompt(file_result)}

## Metrics
- Maintainability Index: {file_result.metrics.maintainability.maintainability_index:.1f}
- Cyclomatic Complexity: {file_result.metrics.complexity.cyclomatic_complexity}
- Technical Debt: {file_result.metrics.maintainability.technical_debt_hours:.1f}h

## Request
Please refactor the code to address the issues found above. Provide:
1. Refactored code
2. Explanation of changes
3. Benefits of the refactoring
"""

    def _format_issues_for_prompt(self, file_result: FileAnalysisResult) -> str:
        """Format issues for prompt."""
        sections = []

        if file_result.code_smells:
            sections.append("### Code Smells")
            for smell in file_result.code_smells:
                sections.append(f"- {smell.description} (Line {smell.location.start_line})")

        if file_result.security_issues:
            sections.append("### Security Issues")
            for issue in file_result.security_issues:
                sections.append(f"- {issue.description} (Line {issue.location.start_line})")

        if file_result.refactoring_suggestions:
            sections.append("### Refactoring Suggestions")
            for suggestion in file_result.refactoring_suggestions:
                sections.append(f"- {suggestion.description} (Line {suggestion.location.start_line})")

        return "\n".join(sections)

    def format_for_explanation_prompt(self, file_result: FileAnalysisResult) -> str:
        """Format file result for explanation prompt."""
        return f"""# Code Explanation Request

## File: {file_result.file_path}
**Language:** {file_result.language}

## Code to Explain
```{file_result.language}
{file_result.content}
```

## Context
- **Classes:** {', '.join(file_result.classes) if file_result.classes else 'None'}
- **Functions:** {', '.join(file_result.functions) if file_result.functions else 'None'}
- **Complexity:** {file_result.metrics.complexity.cyclomatic_complexity}
- **Maintainability Index:** {file_result.metrics.maintainability.maintainability_index:.1f}

## Request
Please explain this code in detail:
1. What is the purpose of this code?
2. How does it work?
3. What are the key components?
4. Are there any potential issues or improvements?
"""

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_severity_emoji(self, severity: SeverityLevel) -> str:
        """Get emoji for severity level."""
        mapping = {
            SeverityLevel.CRITICAL: "🔴",
            SeverityLevel.HIGH: "🟠",
            SeverityLevel.MEDIUM: "🟡",
            SeverityLevel.LOW: "🟢",
            SeverityLevel.INFO: "ℹ️"
        }
        return mapping.get(severity, "⚪")

    def _format_basic(self, result: AnalysisResult, sub_action: Optional[str] = None) -> str:
        # Lógica existente de basic...

        # NUEVO: salida granular según sub_action
        if sub_action == "count_methods":
            total = sum(len(f.functions) for f in result.results)
            names = [n for f in result.results for n in f.functions]
            return f"El archivo contiene **{total} métodos/funciones**:\n" + \
                "\n".join(f"- `{n}`" for n in names)

        elif sub_action == "count_classes":
            total = sum(len(f.classes) for f in result.results)
            names = [n for f in result.results for n in f.classes]
            return f"El archivo contiene **{total} clases**:\n" + \
                "\n".join(f"- `{n}`" for n in names)

        elif sub_action == "list_methods":
            names = [n for f in result.results for n in f.functions]
            return "**Métodos y funciones:**\n" + "\n".join(f"- `{n}`" for n in names)

        elif sub_action == "list_classes":
            names = [n for f in result.results for n in f.classes]
            return "**Clases:**\n" + "\n".join(f"- `{n}`" for n in names)

        elif sub_action == "file_summary":
            # Resumen estructural completo: clases + métodos + imports
            # (usar lógica existente de basic sin filtro)
            return self._format_structural_report(result)   # o una función especializada

        # Default: comportamiento actual de basic
        return self._format_structural_report(result) # o como se llame actualmente
