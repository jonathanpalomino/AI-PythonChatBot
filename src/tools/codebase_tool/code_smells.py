# =============================================================================
# src/tools/codebase_tool/code_smells.py
# Professional Code Smell Detection
# =============================================================================
"""
Professional code smell detection system.
Implements comprehensive detection of code smells and anti-patterns.
"""

import re
from typing import List, Dict, Any, Optional, Tuple

from .models import (
    CodeSmell,
    CodeSmellType,
    SeverityLevel,
    CodeLocation,
    SymbolInfo
)


class CodeSmellDetector:
    """
    Professional code smell detector.

    Detects various code smells including:
    - Duplicated code
    - Long methods
    - Large classes
    - Feature envy
    - Inappropriate intimacy
    - Lazy classes
    - Data clumps
    - Primitive obsession
    - Switch statements
    - Temporary fields
    - Refused bequest
    - Comments
    - God objects
    - Spaghetti code
    """

    def __init__(self):
        self._patterns = self._initialize_patterns()

    def _initialize_patterns(self) -> Dict[str, Any]:
        """Initialize detection patterns."""
        return {
            'sql_injection': [
                re.compile(r'["\'].*?\+.*?["\'].*?(SELECT|INSERT|UPDATE|DELETE)', re.IGNORECASE),
                re.compile(r'f["\'].*?\{.*?\}.*?(SELECT|INSERT|UPDATE|DELETE)', re.IGNORECASE),
                re.compile(r'%s.*?(SELECT|INSERT|UPDATE|DELETE)', re.IGNORECASE),
            ],
            'hardcoded_credentials': [
                re.compile(r'(password|passwd|pwd|api_key|secret|token)\s*=\s*["\'][^"\']{8,}["\']', re.IGNORECASE),
                re.compile(r'(password|passwd|pwd|api_key|secret|token)\s*=\s*[a-zA-Z0-9+/]{20,}', re.IGNORECASE),
            ],
            'magic_numbers': [
                re.compile(r'\b\d{3,}\b'),  # Numbers with 3+ digits
            ],
            'todo_comments': [
                re.compile(r'#\s*(TODO|FIXME|HACK|XXX)', re.IGNORECASE),
                re.compile(r'//\s*(TODO|FIXME|HACK|XXX)', re.IGNORECASE),
            ],
        }

    # =========================================================================
    # Main Detection Method
    # =========================================================================

    def detect_all_smells(
        self,
        content: str,
        file_path: str,
        language: str = "python",
        symbols: Optional[List[SymbolInfo]] = None
    ) -> List[CodeSmell]:
        """
        Detect all code smells in the given content.

        Args:
            content: Source code content
            file_path: Path to the file
            language: Programming language
            symbols: Optional list of symbols (classes, functions, methods)

        Returns:
            List of detected code smells
        """
        smells = []

        # Detect duplicated code
        smells.extend(self._detect_duplicated_code(content, file_path))

        # Detect long methods
        if symbols:
            smells.extend(self._detect_long_methods(content, file_path, symbols))

        # Detect large classes
        if symbols:
            smells.extend(self._detect_large_classes(content, file_path, symbols))

        # Detect feature envy
        if symbols:
            smells.extend(self._detect_feature_envy(content, file_path, symbols))

        # Detect inappropriate intimacy
        if symbols:
            smells.extend(self._detect_inappropriate_intimacy(content, file_path, symbols))

        # Detect lazy classes
        if symbols:
            smells.extend(self._detect_lazy_classes(content, file_path, symbols))

        # Detect data clumps
        if symbols:
            smells.extend(self._detect_data_clumps(content, file_path, symbols))

        # Detect primitive obsession
        if symbols:
            smells.extend(self._detect_primitive_obsession(content, file_path, symbols))

        # Detect switch statements
        smells.extend(self._detect_switch_statements(content, file_path, language))

        # Detect temporary fields
        if symbols:
            smells.extend(self._detect_temporary_fields(content, file_path, symbols))

        # Detect refused bequest
        if symbols:
            smells.extend(self._detect_refused_bequest(content, file_path, symbols))

        # Detect comments
        smells.extend(self._detect_comments(content, file_path))

        # Detect god objects
        if symbols:
            smells.extend(self._detect_god_objects(content, file_path, symbols))

        # Detect spaghetti code
        smells.extend(self._detect_spaghetti_code(content, file_path))

        return smells

    # =========================================================================
    # Individual Smell Detectors
    # =========================================================================

    def _detect_duplicated_code(
        self,
        content: str,
        file_path: str
    ) -> List[CodeSmell]:
        """Detect duplicated code blocks."""
        smells = []
        lines = content.split('\n')
        min_block_size = 6

        # Find duplicated blocks
        duplicated_blocks = self._find_duplicated_blocks(lines, min_block_size)

        for block in duplicated_blocks:
            location = CodeLocation(
                file_path=file_path,
                start_line=block['start_line'],
                end_line=block['end_line']
            )

            smells.append(CodeSmell(
                smell_type=CodeSmellType.DUPLICATED_CODE,
                severity=SeverityLevel.MEDIUM,
                location=location,
                description=f"Duplicated code block ({block['end_line'] - block['start_line'] + 1} lines) also found at lines {block['duplicate_start_line']}-{block['duplicate_end_line']}",
                rationale="Duplicated code increases maintenance burden and can lead to inconsistencies when changes are made in one place but not the other."
            ))

        return smells

    def _find_duplicated_blocks(
        self,
        lines: List[str],
        min_block_size: int
    ) -> List[Dict[str, Any]]:
        """Find duplicated code blocks."""
        duplicated_blocks = []
        normalized = [self._normalize_line(line) for line in lines]

        for start in range(len(lines) - min_block_size):
            block = normalized[start:start + min_block_size]
            block_str = '\n'.join(block)

            for other_start in range(start + min_block_size, len(lines) - min_block_size):
                other_block = normalized[other_start:other_start + min_block_size]
                other_block_str = '\n'.join(other_block)

                if block_str == other_block_str:
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
        normalized = line.strip()

        for comment_marker in ['#', '//']:
            if comment_marker in normalized:
                normalized = normalized.split(comment_marker)[0].strip()

        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized

    def _detect_long_methods(
        self,
        content: str,
        file_path: str,
        symbols: List[SymbolInfo]
    ) -> List[CodeSmell]:
        """Detect methods that are too long."""
        smells = []
        lines = content.split('\n')

        for symbol in symbols:
            if symbol.symbol_type in ('function', 'method'):
                start_line = symbol.location.start_line
                end_line = symbol.location.end_line
                line_count = end_line - start_line + 1

                if line_count > 50:
                    severity = SeverityLevel.HIGH if line_count > 100 else SeverityLevel.MEDIUM

                    location = CodeLocation(
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line
                    )

                    smells.append(CodeSmell(
                        smell_type=CodeSmellType.LONG_METHOD,
                        severity=severity,
                        location=location,
                        description=f"Method '{symbol.name}' is too long ({line_count} lines, recommended < 50)",
                        rationale="Long methods are harder to understand, test, and maintain. Consider breaking it down into smaller, more focused methods."
                    ))

        return smells

    def _detect_large_classes(
        self,
        content: str,
        file_path: str,
        symbols: List[SymbolInfo]
    ) -> List[CodeSmell]:
        """Detect classes that are too large."""
        smells = []

        # Group methods by class
        class_methods: Dict[str, List[SymbolInfo]] = {}
        class_lines: Dict[str, Tuple[int, int]] = {}

        for symbol in symbols:
            if symbol.symbol_type == 'class':
                class_methods[symbol.name] = []
                class_lines[symbol.name] = (symbol.location.start_line, symbol.location.end_line)
            elif symbol.symbol_type == 'method':
                parent = symbol.parent
                if parent:
                    if parent not in class_methods:
                        class_methods[parent] = []
                    class_methods[parent].append(symbol)

        for class_name, methods in class_methods.items():
            method_count = len(methods)

            if method_count > 15:
                severity = SeverityLevel.HIGH if method_count > 25 else SeverityLevel.MEDIUM

                start_line, end_line = class_lines.get(class_name, (1, 1))
                location = CodeLocation(
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line
                )

                smells.append(CodeSmell(
                    smell_type=CodeSmellType.LARGE_CLASS,
                    severity=severity,
                    location=location,
                    description=f"Class '{class_name}' has too many methods ({method_count}, recommended < 15)",
                    rationale="Large classes often have multiple responsibilities and violate the Single Responsibility Principle. Consider splitting into smaller, more focused classes."
                ))

        return smells

    def _detect_feature_envy(
        self,
        content: str,
        file_path: str,
        symbols: List[SymbolInfo]
    ) -> List[CodeSmell]:
        """Detect methods that are more interested in other classes' data."""
        smells = []

        for symbol in symbols:
            if symbol.symbol_type == 'method':
                content_snippet = symbol.content

                # Count accesses to other objects
                other_object_accesses = 0
                self_accesses = 0

                # Simple heuristic: count accesses to other objects vs self
                other_object_accesses += content_snippet.count('.')
                self_accesses += content_snippet.count('self.')

                if other_object_accesses > self_accesses * 2 and other_object_accesses > 5:
                    location = CodeLocation(
                        file_path=file_path,
                        start_line=symbol.location.start_line,
                        end_line=symbol.location.end_line
                    )

                    smells.append(CodeSmell(
                        smell_type=CodeSmellType.FEATURE_ENVY,
                        severity=SeverityLevel.MEDIUM,
                        location=location,
                        description=f"Method '{symbol.name}' shows feature envy - accesses other objects' data more than its own",
                        rationale="Feature envy indicates that a method might be in the wrong class. Consider moving it to the class it's most interested in."
                    ))

        return smells

    def _detect_inappropriate_intimacy(
        self,
        content: str,
        file_path: str,
        symbols: List[SymbolInfo]
    ) -> List[CodeSmell]:
        """Detect classes that know too much about each other's internals."""
        smells = []

        # This is a simplified detection
        # In production, would need more sophisticated analysis
        for symbol in symbols:
            if symbol.symbol_type == 'method':
                content_snippet = symbol.content

                # Detect direct access to private members of other classes
                private_access_pattern = re.compile(r'\w+\._\w+')
                matches = private_access_pattern.findall(content_snippet)

                if len(matches) > 3:
                    location = CodeLocation(
                        file_path=file_path,
                        start_line=symbol.location.start_line,
                        end_line=symbol.location.end_line
                    )

                    smells.append(CodeSmell(
                        smell_type=CodeSmellType.INAPPROPRIATE_INTIMACY,
                        severity=SeverityLevel.MEDIUM,
                        location=location,
                        description=f"Method '{symbol.name}' shows inappropriate intimacy by accessing private members of other classes",
                        rationale="Inappropriate intimacy creates tight coupling between classes. Consider using public methods or refactoring to reduce coupling."
                    ))

        return smells

    def _detect_lazy_classes(
        self,
        content: str,
        file_path: str,
        symbols: List[SymbolInfo]
    ) -> List[CodeSmell]:
        """Detect classes that do very little."""
        smells = []

        # Group methods by class
        class_methods: Dict[str, List[SymbolInfo]] = {}
        class_lines: Dict[str, Tuple[int, int]] = {}

        for symbol in symbols:
            if symbol.symbol_type == 'class':
                class_methods[symbol.name] = []
                class_lines[symbol.name] = (symbol.location.start_line, symbol.location.end_line)
            elif symbol.symbol_type == 'method':
                parent = symbol.parent
                if parent:
                    if parent not in class_methods:
                        class_methods[parent] = []
                    class_methods[parent].append(symbol)

        for class_name, methods in class_methods.items():
            method_count = len(methods)
            start_line, end_line = class_lines.get(class_name, (1, 1))
            line_count = end_line - start_line + 1

            if method_count < 3 and line_count < 50:
                location = CodeLocation(
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line
                )

                smells.append(CodeSmell(
                    smell_type=CodeSmellType.LAZY_CLASS,
                    severity=SeverityLevel.LOW,
                    location=location,
                    description=f"Class '{class_name}' does very little ({method_count} methods, {line_count} lines)",
                    rationale="Lazy classes add unnecessary complexity. Consider merging with another class or removing if not needed."
                ))

        return smells

    def _detect_data_clumps(
        self,
        content: str,
        file_path: str,
        symbols: List[SymbolInfo]
    ) -> List[CodeSmell]:
        """Detect groups of parameters that always appear together."""
        smells = []

        # Collect parameter lists from all methods
        param_lists = []

        for symbol in symbols:
            if symbol.symbol_type in ('function', 'method'):
                content_snippet = symbol.content

                # Extract parameters
                param_match = re.search(r'def\s+\w+\s*\((.*?)\)', content_snippet)
                if param_match:
                    params = [p.strip() for p in param_match.group(1).split(',') if p.strip() and p.strip() != 'self']
                    if len(params) >= 2:
                        param_lists.append(params)

        # Find parameter groups that appear together multiple times
        param_groups: Dict[Tuple[str, ...], int] = {}

        for params in param_lists:
            for i in range(len(params) - 1):
                for j in range(i + 2, min(i + 4, len(params) + 1)):
                    group = tuple(params[i:j])
                    param_groups[group] = param_groups.get(group, 0) + 1

        # Report groups that appear 3+ times
        for group, count in param_groups.items():
            if count >= 3:
                location = CodeLocation(
                    file_path=file_path,
                    start_line=1,
                    end_line=len(content.split('\n'))
                )

                smells.append(CodeSmell(
                    smell_type=CodeSmellType.DATA_CLUMPS,
                    severity=SeverityLevel.MEDIUM,
                    location=location,
                    description=f"Parameters {', '.join(group)} appear together in {count} methods",
                    rationale="Data clumps suggest that these parameters should be grouped into a single object or data structure."
                ))

        return smells

    def _detect_primitive_obsession(
        self,
        content: str,
        file_path: str,
        symbols: List[SymbolInfo]
    ) -> List[CodeSmell]:
        """Detect excessive use of primitive types."""
        smells = []

        for symbol in symbols:
            if symbol.symbol_type in ('function', 'method'):
                content_snippet = symbol.content

                # Extract parameters
                param_match = re.search(r'def\s+\w+\s*\((.*?)\)', content_snippet)
                if param_match:
                    params = [p.strip() for p in param_match.group(1).split(',') if p.strip() and p.strip() != 'self']

                    # Count primitive types (simplified heuristic)
                    primitive_count = 0
                    for param in params:
                        # Check if parameter name suggests it should be an object
                        if any(keyword in param.lower() for keyword in ['id', 'name', 'email', 'phone', 'address']):
                            primitive_count += 1

                    if primitive_count >= 3:
                        location = CodeLocation(
                            file_path=file_path,
                            start_line=symbol.location.start_line,
                            end_line=symbol.location.end_line
                        )

                        smells.append(CodeSmell(
                            smell_type=CodeSmellType.PRIMITIVE_OBSESSION,
                            severity=SeverityLevel.MEDIUM,
                            location=location,
                            description=f"Method '{symbol.name}' uses multiple primitive parameters that could be grouped into an object",
                            rationale="Primitive obsession leads to code duplication and makes it harder to add behavior. Consider creating a value object or data class."
                        ))

        return smells

    def _detect_switch_statements(
        self,
        content: str,
        file_path: str,
        language: str
    ) -> List[CodeSmell]:
        """Detect excessive use of switch/case statements."""
        smells = []

        if language == "python":
            # Python doesn't have switch, but has if-elif chains
            lines = content.split('\n')
            consecutive_elif = 0
            start_line = 0

            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith('elif '):
                    if consecutive_elif == 0:
                        start_line = i + 1
                    consecutive_elif += 1
                elif consecutive_elif > 0:
                    if consecutive_elif >= 5:
                        location = CodeLocation(
                            file_path=file_path,
                            start_line=start_line,
                            end_line=i
                        )

                        smells.append(CodeSmell(
                            smell_type=CodeSmellType.SWITCH_STATEMENTS,
                            severity=SeverityLevel.MEDIUM,
                            location=location,
                            description=f"Long if-elif chain ({consecutive_elif} branches) could be replaced with polymorphism",
                            rationale="Long conditional chains are hard to maintain and extend. Consider using polymorphism or a strategy pattern."
                        ))
                    consecutive_elif = 0
        else:
            # For other languages with switch/case
            switch_pattern = re.compile(r'switch\s*\([^)]+\)\s*\{', re.IGNORECASE)
            matches = list(switch_pattern.finditer(content))

            if len(matches) >= 3:
                location = CodeLocation(
                    file_path=file_path,
                    start_line=1,
                    end_line=len(content.split('\n'))
                )

                smells.append(CodeSmell(
                    smell_type=CodeSmellType.SWITCH_STATEMENTS,
                    severity=SeverityLevel.MEDIUM,
                    location=location,
                    description=f"Multiple switch statements ({len(matches)}) detected in file",
                    rationale="Multiple switch statements suggest that polymorphism could be used instead."
                ))

        return smells

    def _detect_temporary_fields(
        self,
        content: str,
        file_path: str,
        symbols: List[SymbolInfo]
    ) -> List[CodeSmell]:
        """Detect fields that are only used in certain contexts."""
        smells = []

        # This is a simplified detection
        # In production, would need to track field usage across methods
        for symbol in symbols:
            if symbol.symbol_type == 'class':
                content_snippet = symbol.content

                # Find field definitions
                field_pattern = re.compile(r'self\.(\w+)\s*=')
                fields = set(field_pattern.findall(content_snippet))

                # Check if fields are used in all methods
                for field in fields:
                    field_usage = content_snippet.count(f'self.{field}')

                    if field_usage == 1:  # Only defined, never used
                        location = CodeLocation(
                            file_path=file_path,
                            start_line=symbol.location.start_line,
                            end_line=symbol.location.end_line
                        )

                        smells.append(CodeSmell(
                            smell_type=CodeSmellType.TEMPORARY_FIELD,
                            severity=SeverityLevel.LOW,
                            location=location,
                            description=f"Field '{field}' is defined but never used",
                            rationale="Temporary fields add confusion and should be removed or properly used."
                        ))

        return smells

    def _detect_refused_bequest(
        self,
        content: str,
        file_path: str,
        symbols: List[SymbolInfo]
    ) -> List[CodeSmell]:
        """Detect subclasses that don't use inherited methods."""
        smells = []

        # This is a simplified detection
        # In production, would need to compare parent and child class methods
        for symbol in symbols:
            if symbol.symbol_type == 'class':
                content_snippet = symbol.content

                # Check for pass statements in methods (empty overrides)
                pass_pattern = re.compile(r'def\s+\w+\s*\(.*?\):\s*pass')
                matches = pass_pattern.findall(content_snippet)

                if len(matches) >= 2:
                    location = CodeLocation(
                        file_path=file_path,
                        start_line=symbol.location.start_line,
                        end_line=symbol.location.end_line
                    )

                    smells.append(CodeSmell(
                        smell_type=CodeSmellType.REFUSED_BEQUEST,
                        severity=SeverityLevel.MEDIUM,
                        location=location,
                        description=f"Class '{symbol.name}' has {len(matches)} empty method overrides",
                        rationale="Refused bequest indicates that the inheritance relationship might not be appropriate. Consider composition instead of inheritance."
                    ))

        return smells

    def _detect_comments(
        self,
        content: str,
        file_path: str
    ) -> List[CodeSmell]:
        """Detect comments that explain complex code."""
        smells = []
        lines = content.split('\n')

        for i, line in enumerate(lines):
            # Check for TODO/FIXME comments
            for pattern in self._patterns['todo_comments']:
                if pattern.search(line):
                    location = CodeLocation(
                        file_path=file_path,
                        start_line=i + 1,
                        end_line=i + 1
                    )

                    smells.append(CodeSmell(
                        smell_type=CodeSmellType.COMMENTS,
                        severity=SeverityLevel.LOW,
                        location=location,
                        description=f"TODO/FIXME comment found: {line.strip()}",
                        rationale="TODO and FIXME comments indicate incomplete work. Address them or create proper issues."
                    ))

        return smells

    def _detect_god_objects(
        self,
        content: str,
        file_path: str,
        symbols: List[SymbolInfo]
    ) -> List[CodeSmell]:
        """Detect classes that know too much or do too much."""
        smells = []

        # Group methods by class
        class_methods: Dict[str, List[SymbolInfo]] = {}
        class_lines: Dict[str, Tuple[int, int]] = {}

        for symbol in symbols:
            if symbol.symbol_type == 'class':
                class_methods[symbol.name] = []
                class_lines[symbol.name] = (symbol.location.start_line, symbol.location.end_line)
            elif symbol.symbol_type == 'method':
                parent = symbol.parent
                if parent:
                    if parent not in class_methods:
                        class_methods[parent] = []
                    class_methods[parent].append(symbol)

        for class_name, methods in class_methods.items():
            method_count = len(methods)
            start_line, end_line = class_lines.get(class_name, (1, 1))
            line_count = end_line - start_line + 1

            # God object: many methods AND many lines
            if method_count > 20 and line_count > 500:
                location = CodeLocation(
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line
                )

                smells.append(CodeSmell(
                    smell_type=CodeSmellType.GOD_OBJECT,
                    severity=SeverityLevel.HIGH,
                    location=location,
                    description=f"Class '{class_name}' is a God object ({method_count} methods, {line_count} lines)",
                    rationale="God objects know too much and do too much. They are hard to maintain, test, and understand. Consider breaking down into smaller, more focused classes."
                ))

        return smells

    def _detect_spaghetti_code(
        self,
        content: str,
        file_path: str
    ) -> List[CodeSmell]:
        """Detect code with complex control flow."""
        smells = []
        lines = content.split('\n')

        # Calculate nesting depth for each line
        nesting_depths = []
        current_depth = 0

        for line in lines:
            stripped = line.strip()

            if any(keyword in stripped for keyword in ['if ', 'for ', 'while ', 'try:', 'def ', 'class ', '{']):
                current_depth += 1
            elif stripped.endswith('}') or stripped.endswith('end'):
                current_depth = max(0, current_depth - 1)

            nesting_depths.append(current_depth)

        # Find sections with high nesting
        max_nesting = max(nesting_depths) if nesting_depths else 0

        if max_nesting >= 5:
            # Find the line with maximum nesting
            max_line = nesting_depths.index(max_nesting) + 1

            location = CodeLocation(
                file_path=file_path,
                start_line=max(1, max_line - 10),
                end_line=min(len(lines), max_line + 10)
            )

            smells.append(CodeSmell(
                smell_type=CodeSmellType.SPAGHETTI_CODE,
                severity=SeverityLevel.HIGH,
                location=location,
                description=f"Code has deep nesting (depth {max_nesting}) around line {max_line}",
                rationale="Deep nesting makes code hard to read and understand. Consider extracting methods or using guard clauses to reduce nesting."
            ))

        return smells
