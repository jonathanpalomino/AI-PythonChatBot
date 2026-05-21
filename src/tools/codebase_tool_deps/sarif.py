# =============================================================================
# src/tools/codebase_tool/sarif.py
# Professional SARIF Report Generation
# =============================================================================
"""
Professional SARIF (Static Analysis Results Interchange Format) report generator.
Implements SARIF 2.1.0 standard for integration with CI/CD tools.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from .models import (
    AnalysisResult,
    CodeSmell,
    SecurityIssue,
    RefactoringSuggestion,
    SeverityLevel
)


class SarifGenerator:
    """
    Professional SARIF report generator.

    Generates SARIF 2.1.0 compliant reports for:
    - Code smells
    - Security vulnerabilities
    - Refactoring suggestions
    - General analysis results
    """

    def __init__(self, tool_name: str = "codebase_tool", tool_version: str = "2.0.0"):
        self.tool_name = tool_name
        self.tool_version = tool_version
        self.schema_url = "https://json.schemastore.org/sarif-2.1.0.json"

    # =========================================================================
    # Main Generation Methods
    # =========================================================================

    def generate_sarif_report(
        self,
        analysis_result: AnalysisResult,
        repository_root: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a complete SARIF report from analysis results.

        Args:
            analysis_result: Complete analysis result
            repository_root: Optional repository root path for relative paths

        Returns:
            SARIF report as dictionary
        """
        # Calculate summary if not already done
        if analysis_result.total_code_smells == 0:
            analysis_result.calculate_summary()

        # Build SARIF report
        report = {
            "version": "2.1.0",
            "$schema": self.schema_url,
            "runs": [
                self._build_run(
                    analysis_result,
                    repository_root
                )
            ]
        }

        return report

    def generate_sarif_json(
        self,
        analysis_result: AnalysisResult,
        repository_root: Optional[str] = None,
        indent: int = 2
    ) -> str:
        """
        Generate SARIF report as JSON string.

        Args:
            analysis_result: Complete analysis result
            repository_root: Optional repository root path
            indent: JSON indentation

        Returns:
            SARIF report as JSON string
        """
        report = self.generate_sarif_report(analysis_result, repository_root)
        return json.dumps(report, indent=indent, default=str)

    def save_sarif_report(
        self,
        analysis_result: AnalysisResult,
        output_path: str,
        repository_root: Optional[str] = None
    ) -> None:
        """
        Save SARIF report to file.

        Args:
            analysis_result: Complete analysis result
            output_path: Path to save the report
            repository_root: Optional repository root path
        """
        report_json = self.generate_sarif_json(analysis_result, repository_root)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(report_json, encoding='utf-8')

    # =========================================================================
    # SARIF Run Builder
    # =========================================================================

    def _build_run(
        self,
        analysis_result: AnalysisResult,
        repository_root: Optional[str]
    ) -> Dict[str, Any]:
        """Build a SARIF run object."""
        return {
            "tool": self._build_tool_info(),
            "invocations": [self._build_invocation()],
            "results": self._build_results(analysis_result, repository_root),
            "artifacts": self._build_artifacts(analysis_result),
            "columnKind": "utf16CodeUnits"
        }

    def _build_tool_info(self) -> Dict[str, Any]:
        """Build SARIF tool information."""
        return {
            "driver": {
                "name": self.tool_name,
                "version": self.tool_version,
                "informationUri": "https://github.com/your-repo/codebase_tool",
                "fullName": "Professional Codebase Analysis Tool",
                "shortDescription": {
                    "text": "Comprehensive static code analysis tool"
                },
                "rules": self._build_rules()
            }
        }

    def _build_rules(self) -> List[Dict[str, Any]]:
        """Build SARIF rules (rule definitions)."""
        rules = []

        # Code smell rules
        rules.extend(self._build_code_smell_rules())

        # Security issue rules
        rules.extend(self._build_security_rules())

        # Refactoring suggestion rules
        rules.extend(self._build_refactoring_rules())

        return rules

    def _build_code_smell_rules(self) -> List[Dict[str, Any]]:
        """Build code smell rules."""
        return [
            {
                "id": "duplicated_code",
                "name": "Duplicated Code",
                "shortDescription": {
                    "text": "Code is duplicated in multiple locations"
                },
                "fullDescription": {
                    "text": "Duplicated code increases maintenance burden and can lead to inconsistencies when changes are made in one place but not the other."
                },
                "help": {
                    "text": "Extract the duplicated code into a method or function that can be called from multiple locations."
                },
                "properties": {
                    "category": "Maintainability",
                    "precision": "high"
                }
            },
            {
                "id": "long_method",
                "name": "Long Method",
                "shortDescription": {
                    "text": "Method is too long"
                },
                "fullDescription": {
                    "text": "Long methods are harder to understand, test, and maintain. Consider breaking it down into smaller, more focused methods."
                },
                "help": {
                    "text": "Break the method into smaller, more focused methods. Each method should do one thing well."
                },
                "properties": {
                    "category": "Maintainability",
                    "precision": "high"
                }
            },
            {
                "id": "large_class",
                "name": "Large Class",
                "shortDescription": {
                    "text": "Class has too many methods"
                },
                "fullDescription": {
                    "text": "Large classes often have multiple responsibilities and violate the Single Responsibility Principle. Consider splitting into smaller, more focused classes."
                },
                "help": {
                    "text": "Split the class into smaller classes, each with a single responsibility."
                },
                "properties": {
                    "category": "Maintainability",
                    "precision": "high"
                }
            },
            {
                "id": "feature_envy",
                "name": "Feature Envy",
                "shortDescription": {
                    "text": "Method is more interested in other classes' data"
                },
                "fullDescription": {
                    "text": "Feature envy indicates that a method might be in the wrong class. Consider moving it to the class it's most interested in."
                },
                "help": {
                    "text": "Move the method to the class whose data it accesses most frequently."
                },
                "properties": {
                    "category": "Design",
                    "precision": "medium"
                }
            },
            {
                "id": "god_object",
                "name": "God Object",
                "shortDescription": {
                    "text": "Class knows too much or does too much"
                },
                "fullDescription": {
                    "text": "God objects know too much and do too much. They are hard to maintain, test, and understand. Consider breaking down into smaller, more focused classes."
                },
                "help": {
                    "text": "Break down the class into smaller, more focused classes following the Single Responsibility Principle."
                },
                "properties": {
                    "category": "Design",
                    "precision": "high"
                }
            }
        ]

    def _build_security_rules(self) -> List[Dict[str, Any]]:
        """Build security issue rules."""
        return [
            {
                "id": "sql_injection",
                "name": "SQL Injection",
                "shortDescription": {
                    "text": "Potential SQL injection vulnerability"
                },
                "fullDescription": {
                    "text": "SQL injection allows attackers to manipulate database queries through user input."
                },
                "help": {
                    "text": "Use parameterized queries or prepared statements instead of string concatenation."
                },
                "defaultConfiguration": {
                    "level": "error"
                },
                "properties": {
                    "category": "Security",
                    "precision": "high",
                    "tags": ["CWE-89", "OWASP-A03"]
                }
            },
            {
                "id": "xss",
                "name": "Cross-Site Scripting (XSS)",
                "shortDescription": {
                    "text": "Potential XSS vulnerability"
                },
                "fullDescription": {
                    "text": "XSS allows attackers to inject malicious scripts into web pages viewed by other users."
                },
                "help": {
                    "text": "Sanitize and escape user input before rendering. Use template engines with auto-escaping."
                },
                "defaultConfiguration": {
                    "level": "error"
                },
                "properties": {
                    "category": "Security",
                    "precision": "high",
                    "tags": ["CWE-79", "OWASP-A03"]
                }
            },
            {
                "id": "hardcoded_credentials",
                "name": "Hardcoded Credentials",
                "shortDescription": {
                    "text": "Credentials are hardcoded in source code"
                },
                "fullDescription": {
                    "text": "Hardcoded credentials in source code are a major security risk."
                },
                "help": {
                    "text": "Move credentials to environment variables or a secure configuration management system."
                },
                "defaultConfiguration": {
                    "level": "error"
                },
                "properties": {
                    "category": "Security",
                    "precision": "high",
                    "tags": ["CWE-798", "OWASP-A07"]
                }
            },
            {
                "id": "command_injection",
                "name": "Command Injection",
                "shortDescription": {
                    "text": "Potential command injection vulnerability"
                },
                "fullDescription": {
                    "text": "Command injection allows attackers to execute arbitrary commands on the system."
                },
                "help": {
                    "text": "Avoid shell=True in subprocess calls. Use subprocess.run with list of arguments instead."
                },
                "defaultConfiguration": {
                    "level": "error"
                },
                "properties": {
                    "category": "Security",
                    "precision": "high",
                    "tags": ["CWE-78", "OWASP-A03"]
                }
            },
            {
                "id": "weak_cryptography",
                "name": "Weak Cryptography",
                "shortDescription": {
                    "text": "Weak cryptographic algorithm detected"
                },
                "fullDescription": {
                    "text": "Weak cryptographic algorithms are vulnerable to attacks and should not be used."
                },
                "help": {
                    "text": "Use strong cryptographic algorithms (e.g., SHA-256, AES-256, bcrypt for passwords)."
                },
                "defaultConfiguration": {
                    "level": "warning"
                },
                "properties": {
                    "category": "Security",
                    "precision": "high",
                    "tags": ["CWE-327", "OWASP-A02"]
                }
            }
        ]

    def _build_refactoring_rules(self) -> List[Dict[str, Any]]:
        """Build refactoring suggestion rules."""
        return [
            {
                "id": "extract_method",
                "name": "Extract Method",
                "shortDescription": {
                    "text": "Extract code block into a separate method"
                },
                "fullDescription": {
                    "text": "Extracting methods improves readability, reusability, and testability."
                },
                "help": {
                    "text": "Extract the code block into a well-named method that describes what it does."
                },
                "defaultConfiguration": {
                    "level": "warning"
                },
                "properties": {
                    "category": "Refactoring",
                    "precision": "medium"
                }
            },
            {
                "id": "extract_class",
                "name": "Extract Class",
                "shortDescription": {
                    "text": "Extract related functionality into a separate class"
                },
                "fullDescription": {
                    "text": "Extracting a class helps follow the Single Responsibility Principle and improves maintainability."
                },
                "help": {
                    "text": "Identify related functionality and extract it into a new class with a single responsibility."
                },
                "defaultConfiguration": {
                    "level": "warning"
                },
                "properties": {
                    "category": "Refactoring",
                    "precision": "medium"
                }
            },
            {
                "id": "replace_magic_number",
                "name": "Replace Magic Number",
                "shortDescription": {
                    "text": "Replace magic numbers with named constants"
                },
                "fullDescription": {
                    "text": "Magic numbers make code hard to understand and maintain."
                },
                "help": {
                    "text": "Define named constants for magic numbers and use them instead."
                },
                "defaultConfiguration": {
                    "level": "note"
                },
                "properties": {
                    "category": "Refactoring",
                    "precision": "high"
                }
            }
        ]

    def _build_invocation(self) -> Dict[str, Any]:
        """Build SARIF invocation object."""
        return {
            "startTimeUtc": datetime.utcnow().isoformat() + "Z",
            "endTimeUtc": datetime.utcnow().isoformat() + "Z",
            "machine": "localhost",
            "account": "codebase_tool"
        }

    def _build_results(
        self,
        analysis_result: AnalysisResult,
        repository_root: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Build SARIF results from analysis."""
        results = []

        for file_result in analysis_result.results:
            # Add code smells
            for smell in file_result.code_smells:
                results.append(self._build_code_smell_result(smell, repository_root))

            # Add security issues
            for issue in file_result.security_issues:
                results.append(self._build_security_result(issue, repository_root))

            # Add refactoring suggestions
            for suggestion in file_result.refactoring_suggestions:
                results.append(self._build_refactoring_result(suggestion, repository_root))

        return results

    def _build_code_smell_result(
        self,
        smell: CodeSmell,
        repository_root: Optional[str]
    ) -> Dict[str, Any]:
        """Build SARIF result for code smell."""
        rule_id = smell.smell_type.value

        return {
            "ruleId": rule_id,
            "level": self._severity_to_level(smell.severity),
            "message": {
                "text": smell.description
            },
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": self._get_relative_path(smell.location.file_path, repository_root)
                        },
                        "region": {
                            "startLine": smell.location.start_line,
                            "endLine": smell.location.end_line
                        }
                    }
                }
            ],
            "relatedLocations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": self._get_relative_path(smell.location.file_path, repository_root)
                        },
                        "region": {
                            "startLine": smell.location.start_line,
                            "endLine": smell.location.end_line
                        }
                    },
                    "message": {
                        "text": smell.rationale
                    }
                }
            ]
        }

    def _build_security_result(
        self,
        issue: SecurityIssue,
        repository_root: Optional[str]
    ) -> Dict[str, Any]:
        """Build SARIF result for security issue."""
        rule_id = issue.issue_type.value

        result = {
            "ruleId": rule_id,
            "level": self._severity_to_level(issue.severity),
            "message": {
                "text": issue.description
            },
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": self._get_relative_path(issue.location.file_path, repository_root)
                        },
                        "region": {
                            "startLine": issue.location.start_line,
                            "endLine": issue.location.end_line
                        }
                    }
                }
            ]
        }

        # Add CWE and OWASP tags if available
        if issue.cwe_id or issue.owasp_category:
            result["properties"] = {}
            if issue.cwe_id:
                result["properties"]["cweId"] = issue.cwe_id
            if issue.owasp_category:
                result["properties"]["owaspCategory"] = issue.owasp_category

        # Add remediation if available
        if issue.remediation:
            result["fixes"] = [
                {
                    "description": {
                        "text": issue.remediation
                    }
                }
            ]

        return result

    def _build_refactoring_result(
        self,
        suggestion: RefactoringSuggestion,
        repository_root: Optional[str]
    ) -> Dict[str, Any]:
        """Build SARIF result for refactoring suggestion."""
        rule_id = suggestion.refactoring_type.value

        result = {
            "ruleId": rule_id,
            "level": self._severity_to_level(suggestion.severity),
            "message": {
                "text": suggestion.description
            },
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": self._get_relative_path(suggestion.location.file_path, repository_root)
                        },
                        "region": {
                            "startLine": suggestion.location.start_line,
                            "endLine": suggestion.location.end_line
                        }
                    }
                }
            ]
        }

        # Add rationale
        result["relatedLocations"] = [
            {
                "physicalLocation": {
                    "artifactLocation": {
                        "uri": self._get_relative_path(suggestion.location.file_path, repository_root)
                    },
                    "region": {
                        "startLine": suggestion.location.start_line,
                        "endLine": suggestion.location.end_line
                    }
                },
                "message": {
                    "text": suggestion.rationale
                }
            }
        ]

        # Add suggested code if available
        if suggestion.suggested_code:
            result["fixes"] = [
                {
                    "description": {
                        "text": "Suggested refactoring:"
                    },
                    "artifactChanges": [
                        {
                            "artifactLocation": {
                                "uri": self._get_relative_path(suggestion.location.file_path, repository_root)
                            },
                            "replacements": [
                                {
                                    "deletedRegion": {
                                        "startLine": suggestion.location.start_line,
                                        "endLine": suggestion.location.end_line
                                    },
                                    "insertedContent": {
                                        "text": suggestion.suggested_code
                                    }
                                }
                            ]
                        }
                    ]
                }
            ]

        return result

    def _build_artifacts(self, analysis_result: AnalysisResult) -> List[Dict[str, Any]]:
        """Build SARIF artifacts (analyzed files)."""
        artifacts = []

        for file_result in analysis_result.results:
            artifacts.append({
                "location": {
                    "uri": file_result.file_path
                },
                "length": len(file_result.content),
                "language": file_result.language
            })

        return artifacts

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _severity_to_level(self, severity: SeverityLevel) -> str:
        """Convert severity level to SARIF level."""
        mapping = {
            SeverityLevel.CRITICAL: "error",
            SeverityLevel.HIGH: "error",
            SeverityLevel.MEDIUM: "warning",
            SeverityLevel.LOW: "note",
            SeverityLevel.INFO: "note"
        }
        return mapping.get(severity, "note")

    def _get_relative_path(self, file_path: str, repository_root: Optional[str]) -> str:
        """Get relative path from repository root."""
        if repository_root:
            try:
                path = Path(file_path)
                root = Path(repository_root)
                return str(path.relative_to(root))
            except ValueError:
                return file_path
        return file_path
