# =============================================================================
# src/tools/codebase_tool/metrics.py
# Professional Code Metrics Calculator
# =============================================================================
"""
Professional code metrics calculator.
Implements comprehensive metrics for code quality assessment.
"""

import ast
import math
import re
from typing import List, Dict, Any, Optional, Set

from .models import (
    ComplexityMetrics,
    SizeMetrics,
    MaintainabilityMetrics,
    CouplingMetrics,
    CohesionMetrics,
    DuplicationMetrics,
    CodeMetrics
)


class MetricsCalculator:
    """
    Professional metrics calculator for code analysis.

    Calculates comprehensive metrics including:
    - Complexity metrics (cyclomatic, cognitive, nesting)
    - Size metrics (LOC, comments, blank lines)
    - Maintainability metrics (MI, technical debt)
    - Coupling metrics (Ca, Ce, instability)
    - Cohesion metrics (LCOM4)
    - Duplication metrics
    """

    def __init__(self):
        self._complexity_visitor = None
        self._size_analyzer = None

    # =========================================================================
    # Complexity Metrics
    # =========================================================================

    def calculate_complexity_metrics(
        self,
        content: str,
        language: str = "python"
    ) -> ComplexityMetrics:
        """
        Calculate complexity metrics for code.

        Args:
            content: Source code content
            language: Programming language

        Returns:
            ComplexityMetrics object
        """
        if language == "python":
            return self._calculate_python_complexity(content)
        else:
            return self._calculate_regex_complexity(content, language)

    def _calculate_python_complexity(self, content: str) -> ComplexityMetrics:
        """Calculate complexity metrics for Python code using AST."""
        try:
            tree = ast.parse(content)
            visitor = PythonComplexityVisitor()
            visitor.visit(tree)

            return ComplexityMetrics(
                cyclomatic_complexity=visitor.cyclomatic_complexity,
                cognitive_complexity=visitor.cognitive_complexity,
                nesting_depth=visitor.max_nesting_depth
            )
        except SyntaxError:
            # Fallback to regex-based calculation
            return self._calculate_regex_complexity(content, "python")

    def _calculate_regex_complexity(
        self,
        content: str,
        language: str
    ) -> ComplexityMetrics:
        """Calculate complexity metrics using regex patterns."""
        cyclomatic = 1  # Base complexity
        cognitive = 1
        nesting_depth = 0
        current_nesting = 0

        lines = content.split('\n')

        for line in lines:
            stripped = line.strip()

            # Count control structures for cyclomatic complexity
            if any(keyword in stripped for keyword in [
                'if ', 'elif ', 'for ', 'while ', 'case ', 'catch ',
                '&& ', '|| ', ' and ', ' or '
            ]):
                cyclomatic += 1

            # Count nesting
            if any(keyword in stripped for keyword in [
                'if ', 'for ', 'while ', 'try:', 'case ', 'switch',
                'def ', 'class ', 'function ', '{'
            ]):
                current_nesting += 1
                nesting_depth = max(nesting_depth, current_nesting)

            # Count closing braces/brackets for nesting
            if stripped.endswith('}') or stripped.endswith('end'):
                current_nesting -= 1

            # Cognitive complexity: nested structures + logical operators
            if any(keyword in stripped for keyword in ['if ', 'elif ', 'for ', 'while ']):
                cognitive += 1 + current_nesting

            # Logical operators add to cognitive complexity
            cognitive += stripped.count(' and ') + stripped.count(' or ')
            cognitive += stripped.count('&&') + stripped.count('||')

        return ComplexityMetrics(
            cyclomatic_complexity=cyclomatic,
            cognitive_complexity=cognitive,
            nesting_depth=nesting_depth
        )

    # =========================================================================
    # Size Metrics
    # =========================================================================

    def calculate_size_metrics(self, content: str) -> SizeMetrics:
        """
        Calculate size metrics for code.

        Args:
            content: Source code content

        Returns:
            SizeMetrics object
        """
        lines = content.split('\n')

        lines_of_code = 0
        lines_of_comments = 0
        blank_lines = 0

        in_multiline_comment = False
        multiline_start = None

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Check for multiline comments
            if '"""' in line or "'''" in line:
                if not in_multiline_comment:
                    in_multiline_comment = True
                    multiline_start = i
                else:
                    in_multiline_comment = False
                    lines_of_comments += (i - multiline_start + 1)
                    continue

            if in_multiline_comment:
                continue

            # Check for single-line comments
            if stripped.startswith('#') or stripped.startswith('//') or stripped.startswith('/*'):
                lines_of_comments += 1
            elif not stripped:
                blank_lines += 1
            else:
                lines_of_code += 1

        total_lines = lines_of_code + lines_of_comments + blank_lines
        comment_ratio = lines_of_comments / total_lines if total_lines > 0 else 0.0

        return SizeMetrics(
            lines_of_code=lines_of_code,
            lines_of_comments=lines_of_comments,
            blank_lines=blank_lines,
            comment_ratio=comment_ratio
        )

    # =========================================================================
    # Maintainability Metrics
    # =========================================================================

    def calculate_maintainability_metrics(
        self,
        complexity: ComplexityMetrics,
        size: SizeMetrics,
        volume: Optional[float] = None
    ) -> MaintainabilityMetrics:
        """
        Calculate maintainability metrics.

        Uses the Microsoft Maintainability Index formula:
        MI = MAX(0, (171 - 5.2 * ln(V) - 0.23 * CC - 16.2 * ln(L)) * 100 / 171)

        Where:
        - V = Volume (Halstead volume)
        - CC = Cyclomatic Complexity
        - L = Lines of Code

        Args:
            complexity: Complexity metrics
            size: Size metrics
            volume: Optional Halstead volume (calculated if not provided)

        Returns:
            MaintainabilityMetrics object
        """
        # Calculate volume if not provided
        if volume is None:
            volume = self._calculate_halstead_volume(size.lines_of_code)

        # Calculate Maintainability Index
        if size.lines_of_code > 0:
            mi = max(0, (
                171 - 5.2 * math.log(volume) - 0.23 * complexity.cyclomatic_complexity
                - 16.2 * math.log(size.lines_of_code)
            ) * 100 / 171)
        else:
            mi = 100.0

        # Calculate technical debt (simplified model)
        # Based on complexity and code smells
        technical_debt_hours = self._estimate_technical_debt(
            complexity.cyclomatic_complexity,
            size.lines_of_code
        )

        # Technical debt ratio (TD / (TD + cost to fix))
        # Simplified: TD ratio based on complexity threshold
        td_ratio = min(1.0, technical_debt_hours / (size.lines_of_code * 0.5))

        return MaintainabilityMetrics(
            maintainability_index=round(mi, 2),
            technical_debt_hours=round(technical_debt_hours, 2),
            technical_debt_ratio=round(td_ratio, 2)
        )

    def _calculate_halstead_volume(self, loc: int) -> float:
        """
        Calculate Halstead volume (simplified).

        V = n1 * log2(n1) + n2 * log2(n2)

        Where:
        - n1 = number of distinct operators
        - n2 = number of distinct operands

        Simplified: V ≈ LOC * log2(LOC)
        """
        if loc <= 0:
            return 1.0
        return loc * math.log2(loc) if loc > 1 else 1.0

    def _estimate_technical_debt(self, complexity: int, loc: int) -> float:
        """
        Estimate technical debt in hours.

        Simplified model based on complexity and LOC.
        """
        # Base debt: 1 hour per 100 LOC
        base_debt = loc / 100.0

        # Complexity multiplier
        complexity_multiplier = max(1.0, complexity / 10.0)

        return base_debt * complexity_multiplier

    # =========================================================================
    # Coupling Metrics
    # =========================================================================

    def calculate_coupling_metrics(
        self,
        afferent_coupling: int,
        efferent_coupling: int
    ) -> CouplingMetrics:
        """
        Calculate coupling metrics.

        Args:
            afferent_coupling: Number of classes that depend on this class (Ca)
            efferent_coupling: Number of classes this class depends on (Ce)

        Returns:
            CouplingMetrics object
        """
        # Calculate instability: I = Ce / (Ca + Ce)
        # Range: 0 (stable) to 1 (unstable)
        total = afferent_coupling + efferent_coupling
        instability = efferent_coupling / total if total > 0 else 0.0

        return CouplingMetrics(
            afferent_coupling=afferent_coupling,
            efferent_coupling=efferent_coupling,
            instability=round(instability, 2)
        )

    # =========================================================================
    # Cohesion Metrics
    # =========================================================================

    def calculate_cohesion_metrics(
        self,
        methods: List[str],
        attributes: List[str],
        method_attribute_access: Dict[str, Set[str]]
    ) -> CohesionMetrics:
        """
        Calculate cohesion metrics using LCOM4.

        LCOM4: Lack of Cohesion of Methods
        Measures how many disconnected groups of methods exist.

        Args:
            methods: List of method names
            attributes: List of attribute names
            method_attribute_access: Dict mapping method -> set of attributes accessed

        Returns:
            CohesionMetrics object
        """
        if not methods:
            return CohesionMetrics(
                lack_of_cohesion_methods=0.0,
                cohesion_ratio=1.0
            )

        # Build graph where nodes are methods
        # Edge exists if two methods share at least one attribute
        graph = {method: set() for method in methods}

        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                attrs1 = method_attribute_access.get(method1, set())
                attrs2 = method_attribute_access.get(method2, set())

                if attrs1 & attrs2:  # Intersection not empty
                    graph[method1].add(method2)
                    graph[method2].add(method1)

        # Count connected components (LCOM4)
        visited = set()
        components = 0

        for method in methods:
            if method not in visited:
                components += 1
                self._dfs_visit(method, graph, visited)

        lcom4 = components - 1 if components > 1 else 0

        # Cohesion ratio: 1 - (LCOM4 / (methods - 1))
        cohesion_ratio = 1.0 - (lcom4 / (len(methods) - 1)) if len(methods) > 1 else 1.0

        return CohesionMetrics(
            lack_of_cohesion_methods=float(lcom4),
            cohesion_ratio=round(cohesion_ratio, 2)
        )

    def _dfs_visit(self, node: str, graph: Dict[str, Set[str]], visited: Set[str]):
        """Depth-first search for connected components."""
        visited.add(node)
        for neighbor in graph.get(node, set()):
            if neighbor not in visited:
                self._dfs_visit(neighbor, graph, visited)

    # =========================================================================
    # Duplication Metrics
    # =========================================================================

    def calculate_duplication_metrics(
        self,
        content: str,
        min_block_size: int = 6
    ) -> DuplicationMetrics:
        """
        Calculate code duplication metrics.

        Args:
            content: Source code content
            min_block_size: Minimum number of lines for a duplicated block

        Returns:
            DuplicationMetrics object
        """
        lines = content.split('\n')
        total_lines = len(lines)

        # Find duplicated blocks
        duplicated_blocks = self._find_duplicated_blocks(lines, min_block_size)

        # Calculate duplicated lines
        duplicated_lines = sum(
            len(block['lines']) for block in duplicated_blocks
        )

        # Duplication ratio
        duplication_ratio = duplicated_lines / total_lines if total_lines > 0 else 0.0

        return DuplicationMetrics(
            duplication_ratio=round(duplication_ratio, 2),
            duplicated_lines=duplicated_lines,
            duplicated_blocks=len(duplicated_blocks)
        )

    def _find_duplicated_blocks(
        self,
        lines: List[str],
        min_block_size: int
    ) -> List[Dict[str, Any]]:
        """
        Find duplicated code blocks using Rabin-Karp algorithm.

        Args:
            lines: List of code lines
            min_block_size: Minimum block size

        Returns:
            List of duplicated blocks
        """
        # Simplified implementation
        # In production, use a more sophisticated algorithm
        duplicated_blocks = []

        # Normalize lines (remove whitespace, comments)
        normalized = [self._normalize_line(line) for line in lines]

        # Find duplicate sequences
        for start in range(len(lines) - min_block_size):
            block = normalized[start:start + min_block_size]
            block_str = '\n'.join(block)

            # Search for this block elsewhere
            for other_start in range(start + min_block_size, len(lines) - min_block_size):
                other_block = normalized[other_start:other_start + min_block_size]
                other_block_str = '\n'.join(other_block)

                if block_str == other_block_str:
                    # Found a duplicate
                    duplicated_blocks.append({
                        'start_line': start + 1,
                        'end_line': start + min_block_size,
                        'duplicate_start_line': other_start + 1,
                        'duplicate_end_line': other_start + min_block_size,
                        'lines': lines[start:start + min_block_size]
                    })
                    break

        return duplicated_blocks

    def _normalize_line(self, line: str) -> str:
        """Normalize a line for comparison."""
        # Remove leading/trailing whitespace
        normalized = line.strip()

        # Remove inline comments
        for comment_marker in ['#', '//']:
            if comment_marker in normalized:
                normalized = normalized.split(comment_marker)[0].strip()

        # Collapse multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized)

        return normalized

    # =========================================================================
    # Comprehensive Metrics
    # =========================================================================

    def calculate_all_metrics(
        self,
        content: str,
        language: str = "python",
        coupling_data: Optional[Dict[str, Any]] = None,
        cohesion_data: Optional[Dict[str, Any]] = None
    ) -> CodeMetrics:
        """
        Calculate all metrics for code.

        Args:
            content: Source code content
            language: Programming language
            coupling_data: Optional coupling data (afferent/efferent coupling)
            cohesion_data: Optional cohesion data (methods, attributes, access patterns)

        Returns:
            CodeMetrics object with all metrics
        """
        # Calculate complexity
        complexity = self.calculate_complexity_metrics(content, language)

        # Calculate size
        size = self.calculate_size_metrics(content)

        # Calculate maintainability
        maintainability = self.calculate_maintainability_metrics(complexity, size)

        # Calculate coupling (if data provided)
        if coupling_data:
            coupling = self.calculate_coupling_metrics(
                coupling_data.get('afferent', 0),
                coupling_data.get('efferent', 0)
            )
        else:
            coupling = CouplingMetrics(
                afferent_coupling=0,
                efferent_coupling=0,
                instability=0.0
            )

        # Calculate cohesion (if data provided)
        if cohesion_data:
            cohesion = self.calculate_cohesion_metrics(
                cohesion_data.get('methods', []),
                cohesion_data.get('attributes', []),
                cohesion_data.get('access_patterns', {})
            )
        else:
            cohesion = CohesionMetrics(
                lack_of_cohesion_methods=0.0,
                cohesion_ratio=1.0
            )

        # Calculate duplication
        duplication = self.calculate_duplication_metrics(content)

        return CodeMetrics(
            complexity=complexity,
            size=size,
            maintainability=maintainability,
            coupling=coupling,
            cohesion=cohesion,
            duplication=duplication
        )


# =============================================================================
# Python AST Visitor for Complexity
# =============================================================================

class PythonComplexityVisitor(ast.NodeVisitor):
    """AST visitor for calculating Python code complexity."""

    def __init__(self):
        self.cyclomatic_complexity = 1  # Base complexity
        self.cognitive_complexity = 1
        self.max_nesting_depth = 0
        self.current_nesting_depth = 0
        self.nesting_stack = []

    def _enter_nesting(self):
        """Enter a nested block."""
        self.current_nesting_depth += 1
        self.max_nesting_depth = max(self.max_nesting_depth, self.current_nesting_depth)
        self.nesting_stack.append(self.current_nesting_depth)

    def _exit_nesting(self):
        """Exit a nested block."""
        self.current_nesting_depth -= 1
        if self.nesting_stack:
            self.nesting_stack.pop()

    def visit_If(self, node: ast.If):
        self.cyclomatic_complexity += 1
        self.cognitive_complexity += 1 + self.current_nesting_depth
        self._enter_nesting()
        self.generic_visit(node)
        self._exit_nesting()

    def visit_For(self, node: ast.For):
        self.cyclomatic_complexity += 1
        self.cognitive_complexity += 1 + self.current_nesting_depth
        self._enter_nesting()
        self.generic_visit(node)
        self._exit_nesting()

    def visit_AsyncFor(self, node: ast.AsyncFor):
        self.cyclomatic_complexity += 1
        self.cognitive_complexity += 1 + self.current_nesting_depth
        self._enter_nesting()
        self.generic_visit(node)
        self._exit_nesting()

    def visit_While(self, node: ast.While):
        self.cyclomatic_complexity += 1
        self.cognitive_complexity += 1 + self.current_nesting_depth
        self._enter_nesting()
        self.generic_visit(node)
        self._exit_nesting()

    def visit_Try(self, node: ast.Try):
        self.cyclomatic_complexity += 1
        self.cognitive_complexity += 1 + self.current_nesting_depth
        self._enter_nesting()
        self.generic_visit(node)
        self._exit_nesting()

    def visit_BoolOp(self, node: ast.BoolOp):
        self.cyclomatic_complexity += len(node.values) - 1
        self.cognitive_complexity += len(node.values) - 1
        self.generic_visit(node)

    def visit_With(self, node: ast.With):
        self._enter_nesting()
        self.generic_visit(node)
        self._exit_nesting()

    def visit_AsyncWith(self, node: ast.AsyncWith):
        self._enter_nesting()
        self.generic_visit(node)
        self._exit_nesting()

    def visit_ListComp(self, node: ast.ListComp):
        self.cognitive_complexity += 1
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp):
        self.cognitive_complexity += 1
        self.generic_visit(node)

    def visit_DictComp(self, node: ast.DictComp):
        self.cognitive_complexity += 1
        self.generic_visit(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp):
        self.cognitive_complexity += 1
        self.generic_visit(node)
