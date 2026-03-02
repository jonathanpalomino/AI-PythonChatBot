# =============================================================================
# src/tools/codebase_tool/refactoring.py
# Professional Refactoring Suggestion System
# =============================================================================
"""
Professional refactoring suggestion system.
Implements comprehensive refactoring recommendations with code generation.
"""

import re
from typing import List, Dict, Any, Optional

from .models import (
    RefactoringSuggestion,
    RefactoringType,
    SeverityLevel,
    CodeLocation,
    EffortEstimate,
    ImpactScope,
    SymbolInfo
)


class RefactoringSuggester:
    """
    Professional refactoring suggestion generator.

    Generates refactoring suggestions including:
    - Extract method
    - Extract class
    - Inline method
    - Replace conditional with polymorphism
    - Decompose conditional
    - Consolidate conditional
    - Replace magic number with constant
    - Introduce parameter object
    - Preserve whole object
    - Replace type code with class
    - Rename
    - Move method
    - Move field
    - Extract interface
    """

    def __init__(self):
        self._patterns = self._initialize_patterns()

    def _initialize_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Initialize refactoring detection patterns."""
        return {
            'magic_numbers': [
                re.compile(r'\b\d{3,}\b'),  # Numbers with 3+ digits
            ],
            'long_parameter_list': [
                re.compile(r'def\s+\w+\s*\([^)]{50,}\)'),  # Long parameter lists
            ],
            'duplicate_code': [
                re.compile(r'.{50,}'),  # Long repeated patterns
            ],
        }

    # =========================================================================
    # Main Suggestion Method
    # =========================================================================

    def generate_all_suggestions(
        self,
        content: str,
        file_path: str,
        language: str = "python",
        symbols: Optional[List[SymbolInfo]] = None,
        code_smells: Optional[List[Any]] = None,
        security_issues: Optional[List[Any]] = None
    ) -> List[RefactoringSuggestion]:
        """
        Generate all refactoring suggestions for the given content.

        Args:
            content: Source code content
            file_path: Path to the file
            language: Programming language
            symbols: Optional list of symbols
            code_smells: Optional list of code smells
            security_issues: Optional list of security issues

        Returns:
            List of refactoring suggestions
        """
        suggestions = []

        # Generate suggestions from code smells
        if code_smells:
            suggestions.extend(self._generate_suggestions_from_smells(
                content, file_path, code_smells
            ))

        # Generate suggestions from security issues
        if security_issues:
            suggestions.extend(self._generate_suggestions_from_security(
                content, file_path, security_issues
            ))

        # Generate suggestions from symbols
        if symbols:
            suggestions.extend(self._generate_suggestions_from_symbols(
                content, file_path, language, symbols
            ))

        # Generate general suggestions
        suggestions.extend(self._generate_general_suggestions(
            content, file_path, language
        ))

        return suggestions

    # =========================================================================
    # Suggestion Generators from Issues
    # =========================================================================

    def _generate_suggestions_from_smells(
        self,
        content: str,
        file_path: str,
        code_smells: List[Any]
    ) -> List[RefactoringSuggestion]:
        """Generate refactoring suggestions from code smells."""
        suggestions = []

        for smell in code_smells:
            smell_type = getattr(smell, 'smell_type', None)
            location = getattr(smell, 'location', None)

            if not location:
                continue

            if smell_type and hasattr(smell_type, 'value'):
                smell_type_value = smell_type.value
            else:
                smell_type_value = str(smell_type) if smell_type else 'unknown'

            # Map code smells to refactoring suggestions
            if 'long_method' in smell_type_value:
                suggestions.append(self._suggest_extract_method(
                    content, file_path, location
                ))
            elif 'large_class' in smell_type_value:
                suggestions.append(self._suggest_extract_class(
                    content, file_path, location
                ))
            elif 'duplicated_code' in smell_type_value:
                suggestions.append(self._suggest_extract_method(
                    content, file_path, location
                ))
            elif 'data_clumps' in smell_type_value:
                suggestions.append(self._suggest_introduce_parameter_object(
                    content, file_path, location
                ))
            elif 'primitive_obsession' in smell_type_value:
                suggestions.append(self._suggest_replace_type_code_with_class(
                    content, file_path, location
                ))
            elif 'switch_statements' in smell_type_value:
                suggestions.append(self._suggest_replace_conditional_with_polymorphism(
                    content, file_path, location
                ))

        return suggestions

    def _generate_suggestions_from_security(
        self,
        content: str,
        file_path: str,
        security_issues: List[Any]
    ) -> List[RefactoringSuggestion]:
        """Generate refactoring suggestions from security issues."""
        suggestions = []

        for issue in security_issues:
            issue_type = getattr(issue, 'issue_type', None)
            location = getattr(issue, 'location', None)

            if not location:
                continue

            if issue_type and hasattr(issue_type, 'value'):
                issue_type_value = issue_type.value
            else:
                issue_type_value = str(issue_type) if issue_type else 'unknown'

            # Map security issues to refactoring suggestions
            if 'sql_injection' in issue_type_value:
                suggestions.append(self._suggest_fix_sql_injection(
                    content, file_path, location
                ))
            elif 'command_injection' in issue_type_value:
                suggestions.append(self._suggest_fix_command_injection(
                    content, file_path, location
                ))
            elif 'hardcoded_credentials' in issue_type_value:
                suggestions.append(self._suggest_move_credentials_to_env(
                    content, file_path, location
                ))

        return suggestions

    def _generate_suggestions_from_symbols(
        self,
        content: str,
        file_path: str,
        language: str,
        symbols: List[SymbolInfo]
    ) -> List[RefactoringSuggestion]:
        """Generate refactoring suggestions from symbols."""
        suggestions = []

        for symbol in symbols:
            symbol_type = symbol.symbol_type
            symbol_name = symbol.name
            start_line = symbol.location.start_line
            end_line = symbol.location.end_line

            location = CodeLocation(
                file_path=file_path,
                start_line=start_line,
                end_line=end_line
            )

            if symbol_type in ('function', 'method'):
                content_snippet = symbol.content

                # Check for long parameter lists
                param_match = re.search(r'def\s+\w+\s*\((.*?)\)', content_snippet)
                if param_match:
                    params = [p.strip() for p in param_match.group(1).split(',') if p.strip() and p.strip() != 'self']
                    if len(params) > 5:
                        suggestions.append(self._suggest_introduce_parameter_object(
                            content, file_path, location
                        ))

                # Check for magic numbers
                magic_numbers = self._find_magic_numbers(content_snippet)
                if magic_numbers:
                    suggestions.append(self._suggest_replace_magic_numbers(
                        content, file_path, location, magic_numbers
                    ))

            elif symbol_type == 'class':
                # Check for class that could be an interface
                methods = [s for s in symbols if s.symbol_type == 'method' and s.parent == symbol_name]
                if methods:
                    suggestions.append(self._suggest_extract_interface(
                        content, file_path, location, symbol_name, methods
                    ))

        return suggestions

    def _generate_general_suggestions(
        self,
        content: str,
        file_path: str,
        language: str
    ) -> List[RefactoringSuggestion]:
        """Generate general refactoring suggestions."""
        suggestions = []

        # Check for magic numbers in entire file
        magic_numbers = self._find_magic_numbers(content)
        if magic_numbers:
            location = CodeLocation(
                file_path=file_path,
                start_line=1,
                end_line=len(content.split('\n'))
            )
            suggestions.append(self._suggest_replace_magic_numbers(
                content, file_path, location, magic_numbers
            ))

        return suggestions

    # =========================================================================
    # Individual Suggestion Generators
    # =========================================================================

    def _suggest_extract_method(
        self,
        content: str,
        file_path: str,
        location: CodeLocation
    ) -> RefactoringSuggestion:
        """Suggest extracting a method."""
        lines = content.split('\n')
        start_idx = location.start_line - 1
        end_idx = min(location.end_line, len(lines))
        code_block = '\n'.join(lines[start_idx:end_idx])

        return RefactoringSuggestion(
            refactoring_type=RefactoringType.EXTRACT_METHOD,
            severity=SeverityLevel.HIGH,
            location=location,
            description="Extract this code block into a separate method",
            rationale="Extracting methods improves readability, reusability, and testability. It also makes the code easier to understand and maintain.",
            suggested_code=self._generate_extract_method_code(code_block),
            effort=EffortEstimate.MINUTES,
            impact=ImpactScope.LOCAL,
            benefits=[
                "Improved readability",
                "Better code organization",
                "Easier to test",
                "Potential for reuse"
            ]
        )

    def _suggest_extract_class(
        self,
        content: str,
        file_path: str,
        location: CodeLocation
    ) -> RefactoringSuggestion:
        """Suggest extracting a class."""
        return RefactoringSuggestion(
            refactoring_type=RefactoringType.EXTRACT_CLASS,
            severity=SeverityLevel.HIGH,
            location=location,
            description="Extract related functionality into a separate class",
            rationale="Large classes often have multiple responsibilities. Extracting a class helps follow the Single Responsibility Principle and improves maintainability.",
            effort=EffortEstimate.HOURS,
            impact=ImpactScope.MODULE,
            benefits=[
                "Better separation of concerns",
                "Improved testability",
                "Easier to understand",
                "Follows Single Responsibility Principle"
            ]
        )

    def _suggest_replace_conditional_with_polymorphism(
        self,
        content: str,
        file_path: str,
        location: CodeLocation
    ) -> RefactoringSuggestion:
        """Suggest replacing conditional with polymorphism."""
        return RefactoringSuggestion(
            refactoring_type=RefactoringType.REPLACE_CONDITIONAL_WITH_POLYMORPHISM,
            severity=SeverityLevel.MEDIUM,
            location=location,
            description="Replace conditional logic with polymorphism",
            rationale="Polymorphism eliminates complex conditional statements and makes the code more extensible. New types can be added without modifying existing code.",
            effort=EffortEstimate.HOURS,
            impact=ImpactScope.MODULE,
            benefits=[
                "Eliminates complex conditionals",
                "More extensible",
                "Follows Open/Closed Principle",
                "Easier to add new types"
            ]
        )

    def _suggest_decompose_conditional(
        self,
        content: str,
        file_path: str,
        location: CodeLocation
    ) -> RefactoringSuggestion:
        """Suggest decomposing complex conditionals."""
        return RefactoringSuggestion(
            refactoring_type=RefactoringType.DECOMPOSE_CONDITIONAL,
            severity=SeverityLevel.MEDIUM,
            location=location,
            description="Decompose complex conditional into separate methods",
            rationale="Complex conditionals are hard to understand and maintain. Extracting them into well-named methods improves readability.",
            effort=EffortEstimate.MINUTES,
            impact=ImpactScope.LOCAL,
            benefits=[
                "Improved readability",
                "Self-documenting code",
                "Easier to test conditions"
            ]
        )

    def _suggest_consolidate_conditional(
        self,
        content: str,
        file_path: str,
        location: CodeLocation
    ) -> RefactoringSuggestion:
        """Suggest consolidating duplicate conditionals."""
        return RefactoringSuggestion(
            refactoring_type=RefactoringType.CONSOLIDATE_CONDITIONAL,
            severity=SeverityLevel.LOW,
            location=location,
            description="Consolidate duplicate conditional expressions",
            rationale="Duplicate conditionals increase maintenance burden. Consolidating them reduces duplication and makes changes easier.",
            effort=EffortEstimate.MINUTES,
            impact=ImpactScope.LOCAL,
            benefits=[
                "Reduced duplication",
                "Easier to maintain",
                "Single source of truth"
            ]
        )

    def _suggest_replace_magic_numbers(
        self,
        content: str,
        file_path: str,
        location: CodeLocation,
        magic_numbers: List[int]
    ) -> RefactoringSuggestion:
        """Suggest replacing magic numbers with constants."""
        constants_code = self._generate_constants_code(magic_numbers)

        return RefactoringSuggestion(
            refactoring_type=RefactoringType.REPLACE_MAGIC_NUMBER,
            severity=SeverityLevel.MEDIUM,
            location=location,
            description=f"Replace {len(magic_numbers)} magic number(s) with named constants",
            rationale="Magic numbers make code hard to understand and maintain. Named constants improve readability and make it easier to change values.",
            suggested_code=constants_code,
            effort=EffortEstimate.MINUTES,
            impact=ImpactScope.LOCAL,
            benefits=[
                "Improved readability",
                "Self-documenting code",
                "Easier to maintain",
                "Single source of truth"
            ]
        )

    def _suggest_introduce_parameter_object(
        self,
        content: str,
        file_path: str,
        location: CodeLocation
    ) -> RefactoringSuggestion:
        """Suggest introducing a parameter object."""
        return RefactoringSuggestion(
            refactoring_type=RefactoringType.INTRODUCE_PARAMETER_OBJECT,
            severity=SeverityLevel.MEDIUM,
            location=location,
            description="Introduce a parameter object to group related parameters",
            rationale="Long parameter lists are hard to work with and often indicate missing abstractions. A parameter object groups related data and makes the code cleaner.",
            effort=EffortEstimate.HOURS,
            impact=ImpactScope.MODULE,
            benefits=[
                "Cleaner method signatures",
                "Groups related data",
                "Easier to extend",
                "Better encapsulation"
            ]
        )

    def _suggest_preserve_whole_object(
        self,
        content: str,
        file_path: str,
        location: CodeLocation
    ) -> RefactoringSuggestion:
        """Suggest preserving whole object instead of individual parameters."""
        return RefactoringSuggestion(
            refactoring_type=RefactoringType.PRESERVE_WHOLE_OBJECT,
            severity=SeverityLevel.LOW,
            location=location,
            description="Pass the whole object instead of individual fields",
            rationale="Passing individual fields from an object breaks encapsulation and makes the code fragile to changes. Passing the whole object is more flexible.",
            effort=EffortEstimate.MINUTES,
            impact=ImpactScope.LOCAL,
            benefits=[
                "Better encapsulation",
                "More flexible",
                "Less fragile to changes"
            ]
        )

    def _suggest_replace_type_code_with_class(
        self,
        content: str,
        file_path: str,
        location: CodeLocation
    ) -> RefactoringSuggestion:
        """Suggest replacing type codes with classes."""
        return RefactoringSuggestion(
            refactoring_type=RefactoringType.REPLACE_TYPE_CODE_WITH_CLASS,
            severity=SeverityLevel.MEDIUM,
            location=location,
            description="Replace type codes with classes or enums",
            rationale="Type codes are error-prone and don't provide type safety. Classes or enums provide better encapsulation and enable polymorphism.",
            effort=EffortEstimate.HOURS,
            impact=ImpactScope.MODULE,
            benefits=[
                "Type safety",
                "Better encapsulation",
                "Enables polymorphism",
                "Self-documenting"
            ]
        )

    def _suggest_rename(
        self,
        content: str,
        file_path: str,
        location: CodeLocation,
        old_name: str,
        suggested_name: str
    ) -> RefactoringSuggestion:
        """Suggest renaming a symbol."""
        return RefactoringSuggestion(
            refactoring_type=RefactoringType.RENAME,
            severity=SeverityLevel.LOW,
            location=location,
            description=f"Rename '{old_name}' to '{suggested_name}'",
            rationale="Good names are essential for code readability. The suggested name better describes the symbol's purpose.",
            effort=EffortEstimate.MINUTES,
            impact=ImpactScope.MODULE,
            benefits=[
                "Improved readability",
                "Self-documenting code",
                "Better understanding"
            ]
        )

    def _suggest_move_method(
        self,
        content: str,
        file_path: str,
        location: CodeLocation
    ) -> RefactoringSuggestion:
        """Suggest moving a method to another class."""
        return RefactoringSuggestion(
            refactoring_type=RefactoringType.MOVE_METHOD,
            severity=SeverityLevel.MEDIUM,
            location=location,
            description="Move this method to a more appropriate class",
            rationale="Methods should be in the class where they are most used or where they have the most knowledge. Moving methods improves cohesion.",
            effort=EffortEstimate.HOURS,
            impact=ImpactScope.MODULE,
            benefits=[
                "Improved cohesion",
                "Better organization",
                "Follows Feature Envy principle"
            ]
        )

    def _suggest_move_field(
        self,
        content: str,
        file_path: str,
        location: CodeLocation
    ) -> RefactoringSuggestion:
        """Suggest moving a field to another class."""
        return RefactoringSuggestion(
            refactoring_type=RefactoringType.MOVE_FIELD,
            severity=SeverityLevel.LOW,
            location=location,
            description="Move this field to a more appropriate class",
            rationale="Fields should be in the class where they are most used. Moving fields improves cohesion and reduces coupling.",
            effort=EffortEstimate.MINUTES,
            impact=ImpactScope.MODULE,
            benefits=[
                "Improved cohesion",
                "Better organization",
                "Reduced coupling"
            ]
        )

    def _suggest_extract_interface(
        self,
        content: str,
        file_path: str,
        location: CodeLocation,
        class_name: str,
        methods: List[Dict[str, Any]]
    ) -> RefactoringSuggestion:
        """Suggest extracting an interface."""
        interface_code = self._generate_interface_code(class_name, methods)

        return RefactoringSuggestion(
            refactoring_type=RefactoringType.EXTRACT_INTERFACE,
            severity=SeverityLevel.MEDIUM,
            location=location,
            description=f"Extract an interface from class '{class_name}'",
            rationale="Interfaces define contracts and enable polymorphism. Extracting an interface makes the code more flexible and testable.",
            suggested_code=interface_code,
            effort=EffortEstimate.HOURS,
            impact=ImpactScope.MODULE,
            benefits=[
                "Enables polymorphism",
                "Better testability",
                "Loose coupling",
                "Clear contracts"
            ]
        )

    # =========================================================================
    # Security-specific Suggestions
    # =========================================================================

    def _suggest_fix_sql_injection(
        self,
        content: str,
        file_path: str,
        location: CodeLocation
    ) -> RefactoringSuggestion:
        """Suggest fixing SQL injection vulnerability."""
        return RefactoringSuggestion(
            refactoring_type=RefactoringType.EXTRACT_METHOD,
            severity=SeverityLevel.CRITICAL,
            location=location,
            description="Fix SQL injection vulnerability by using parameterized queries",
            rationale="SQL injection is a critical security vulnerability. Parameterized queries prevent attackers from manipulating SQL statements.",
            suggested_code=self._generate_parameterized_query_code(),
            effort=EffortEstimate.MINUTES,
            impact=ImpactScope.LOCAL,
            benefits=[
                "Prevents SQL injection",
                "Better security",
                "Cleaner code"
            ]
        )

    def _suggest_fix_command_injection(
        self,
        content: str,
        file_path: str,
        location: CodeLocation
    ) -> RefactoringSuggestion:
        """Suggest fixing command injection vulnerability."""
        return RefactoringSuggestion(
            refactoring_type=RefactoringType.EXTRACT_METHOD,
            severity=SeverityLevel.CRITICAL,
            location=location,
            description="Fix command injection vulnerability by avoiding shell=True",
            rationale="Command injection allows attackers to execute arbitrary commands. Using subprocess without shell=True prevents this.",
            suggested_code=self._generate_safe_subprocess_code(),
            effort=EffortEstimate.MINUTES,
            impact=ImpactScope.LOCAL,
            benefits=[
                "Prevents command injection",
                "Better security",
                "More predictable behavior"
            ]
        )

    def _suggest_move_credentials_to_env(
        self,
        content: str,
        file_path: str,
        location: CodeLocation
    ) -> RefactoringSuggestion:
        """Suggest moving hardcoded credentials to environment variables."""
        return RefactoringSuggestion(
            refactoring_type=RefactoringType.EXTRACT_METHOD,
            severity=SeverityLevel.CRITICAL,
            location=location,
            description="Move hardcoded credentials to environment variables",
            rationale="Hardcoded credentials in source code are a major security risk. Environment variables provide a secure way to manage secrets.",
            suggested_code=self._generate_env_var_code(),
            effort=EffortEstimate.MINUTES,
            impact=ImpactScope.LOCAL,
            benefits=[
                "Improved security",
                "No credentials in source code",
                "Easier to manage in different environments"
            ]
        )

    # =========================================================================
    # Code Generation Helpers
    # =========================================================================

    def _generate_extract_method_code(self, code_block: str) -> str:
        """Generate code for extracted method."""
        return f"""
# Extract this code block into a separate method:
def extracted_method():
    {code_block}

# Then call it:
extracted_method()
"""

    def _generate_constants_code(self, magic_numbers: List[int]) -> str:
        """Generate code for constants."""
        constants = []
        for num in magic_numbers:
            constants.append(f"{self._suggest_constant_name(num)} = {num}")

        return f"""
# Define constants at module level:
{chr(10).join(constants)}

# Then use the constants instead of magic numbers
"""

    def _suggest_constant_name(self, number: int) -> str:
        """Suggest a name for a constant based on its value."""
        # This is a simplified heuristic
        # In production, would use more sophisticated analysis
        if number == 0:
            return "ZERO"
        elif number == 1:
            return "ONE"
        elif number == 100:
            return "PERCENTAGE"
        elif number == 60:
            return "SECONDS_PER_MINUTE"
        elif number == 3600:
            return "SECONDS_PER_HOUR"
        elif number == 86400:
            return "SECONDS_PER_DAY"
        else:
            return f"CONSTANT_{number}"

    def _generate_interface_code(self, class_name: str, methods: List[Any]) -> str:
        """Generate code for extracted interface."""
        method_signatures = []
        for method in methods:
            # Handle both SymbolInfo objects and dicts
            if hasattr(method, 'name'):
                method_name = method.name
            else:
                method_name = method.get('name', 'unknown')
            method_signatures.append(f"    def {method_name}(self): ...")

        return f"""
class I{class_name}:
    \"\"\"Interface for {class_name}\"\"\"
{chr(10).join(method_signatures)}
"""

    def _generate_parameterized_query_code(self) -> str:
        """Generate code for parameterized queries."""
        return """
# Instead of:
# cursor.execute(f"SELECT * FROM users WHERE name = '{user_input}'")

# Use parameterized queries:
cursor.execute("SELECT * FROM users WHERE name = %s", (user_input,))
"""

    def _generate_safe_subprocess_code(self) -> str:
        """Generate code for safe subprocess usage."""
        return """
# Instead of:
# subprocess.run(f"command {user_input}", shell=True)

# Use list of arguments:
subprocess.run(["command", user_input], check=True)
"""

    def _generate_env_var_code(self) -> str:
        """Generate code for environment variables."""
        return """
import os

# Instead of:
# password = "hardcoded_password"

# Use environment variables:
password = os.getenv('PASSWORD')
if not password:
    raise ValueError("PASSWORD environment variable not set")
"""

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _find_magic_numbers(self, content: str) -> List[int]:
        """Find magic numbers in code."""
        magic_numbers = []

        # Find numbers with 3+ digits
        pattern = re.compile(r'\b(\d{3,})\b')
        matches = pattern.findall(content)

        # Filter out common numbers that aren't really "magic"
        common_numbers = {100, 1000, 1024, 2048, 4096}

        for num_str in matches:
            num = int(num_str)
            if num not in common_numbers:
                magic_numbers.append(num)

        return list(set(magic_numbers))  # Remove duplicates
