# =============================================================================
# src/tools/codebase_tool/security.py
# Professional Security Vulnerability Detection
# =============================================================================
"""
Professional security vulnerability detection system.
Implements comprehensive security analysis for code.
"""

import ast
import re
from typing import List, Dict

from .models import (
    SecurityIssue,
    SecurityIssueType,
    SeverityLevel,
    CodeLocation
)


class SecurityAnalyzer:
    """
    Professional security vulnerability analyzer.

    Detects various security vulnerabilities including:
    - SQL Injection
    - XSS (Cross-Site Scripting)
    - Hardcoded credentials
    - Insecure deserialization
    - Path traversal
    - Weak cryptography
    - Command injection
    - Insecure random number generation
    - Open redirect
    - SSRF (Server-Side Request Forgery)
    """

    def __init__(self):
        self._patterns = self._initialize_patterns()
        self._cwe_mapping = self._initialize_cwe_mapping()
        self._owasp_mapping = self._initialize_owasp_mapping()

    def _initialize_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Initialize security detection patterns."""
        return {
            'sql_injection': [
                # String concatenation in SQL queries
                re.compile(r'["\'].*?\+.*?["\'].*?(SELECT|INSERT|UPDATE|DELETE|DROP|ALTER)', re.IGNORECASE),
                # f-strings with SQL
                re.compile(r'f["\'].*?\{.*?\}.*?(SELECT|INSERT|UPDATE|DELETE|DROP|ALTER)', re.IGNORECASE),
                # % formatting with SQL
                re.compile(r'%s.*?(SELECT|INSERT|UPDATE|DELETE|DROP|ALTER)', re.IGNORECASE),
                # .format() with SQL
                re.compile(r'\.format\(.*?\).*?(SELECT|INSERT|UPDATE|DELETE|DROP|ALTER)', re.IGNORECASE),
                # execute/executemany with string concatenation
                re.compile(r'(execute|executemany)\s*\(\s*["\'].*?\+.*?["\']', re.IGNORECASE),
            ],
            'xss': [
                # innerHTML with user input
                re.compile(r'innerHTML\s*=\s*.*?(request|form|input|params|query)', re.IGNORECASE),
                # document.write with user input
                re.compile(r'document\.write\s*\(\s*.*?(request|form|input|params|query)', re.IGNORECASE),
                # eval with user input
                re.compile(r'eval\s*\(\s*.*?(request|form|input|params|query)', re.IGNORECASE),
            ],
            'hardcoded_credentials': [
                # Password assignments
                re.compile(r'(password|passwd|pwd|api_key|secret|token|private_key)\s*=\s*["\'][^"\']{8,}["\']', re.IGNORECASE),
                # Base64 encoded credentials (simplified)
                re.compile(r'(password|passwd|pwd|api_key|secret|token)\s*=\s*[a-zA-Z0-9+/]{20,}={0,2}', re.IGNORECASE),
                # Connection strings with credentials
                re.compile(r'(mongodb://|mysql://|postgresql://|redis://)[^:]+:[^@]+@', re.IGNORECASE),
            ],
            'insecure_deserialization': [
                # pickle.loads
                re.compile(r'pickle\.loads\s*\(', re.IGNORECASE),
                # yaml.load without safe_load
                re.compile(r'yaml\.load\s*\(', re.IGNORECASE),
                # marshal.loads
                re.compile(r'marshal\.loads\s*\(', re.IGNORECASE),
                # shelve.open without protocol restriction
                re.compile(r'shelve\.open\s*\(', re.IGNORECASE),
            ],
            'path_traversal': [
                # os.path.join with user input
                re.compile(r'os\.path\.join\s*\([^)]*?(request|form|input|params|query)', re.IGNORECASE),
                # open() with user input
                re.compile(r'open\s*\(\s*["\'].*?\.\.[/\\]', re.IGNORECASE),
                # Path operations with user input
                re.compile(r'(Path|pathlib\.Path)\s*\([^)]*?(request|form|input|params|query)', re.IGNORECASE),
            ],
            'weak_cryptography': [
                # MD5
                re.compile(r'(hashlib\.md5|md5\.new|Crypto\.Hash\.MD5)', re.IGNORECASE),
                # SHA1
                re.compile(r'(hashlib\.sha1|sha1\.new|Crypto\.Hash\.SHA1)', re.IGNORECASE),
                # DES
                re.compile(r'(DES|des)\.new', re.IGNORECASE),
                # RC4
                re.compile(r'(ARC4|RC4|rc4)\.new', re.IGNORECASE),
            ],
            'command_injection': [
                # os.system with user input
                re.compile(r'os\.system\s*\(\s*.*?(request|form|input|params|query)', re.IGNORECASE),
                # subprocess.call with shell=True
                re.compile(r'subprocess\.(call|run|Popen)\s*\([^)]*shell\s*=\s*True', re.IGNORECASE),
                # exec with user input
                re.compile(r'exec\s*\(\s*.*?(request|form|input|params|query)', re.IGNORECASE),
            ],
            'insecure_random': [
                # random module (not secrets)
                re.compile(r'import\s+random\s*$', re.MULTILINE),
                re.compile(r'from\s+random\s+import', re.MULTILINE),
                # random.random, random.randint, etc.
                re.compile(r'random\.(random|randint|choice|shuffle)', re.IGNORECASE),
            ],
            'open_redirect': [
                # redirect with user input
                re.compile(r'(redirect|url_for)\s*\(\s*.*?(request|form|input|params|query)', re.IGNORECASE),
                # Response with user-controlled URL
                re.compile(r'Response\s*\(\s*.*?(request|form|input|params|query)', re.IGNORECASE),
            ],
            'ssrf': [
                # requests.get with user input
                re.compile(r'requests\.(get|post|put|delete)\s*\(\s*.*?(request|form|input|params|query)', re.IGNORECASE),
                # urllib with user input
                re.compile(r'urllib\.(request|urlopen)\s*\(\s*.*?(request|form|input|params|query)', re.IGNORECASE),
                # httpx with user input
                re.compile(r'httpx\.(get|post|put|delete)\s*\(\s*.*?(request|form|input|params|query)', re.IGNORECASE),
            ],
        }

    def _initialize_cwe_mapping(self) -> Dict[str, str]:
        """Initialize CWE (Common Weakness Enumeration) mapping."""
        return {
            'sql_injection': 'CWE-89',
            'xss': 'CWE-79',
            'hardcoded_credentials': 'CWE-798',
            'insecure_deserialization': 'CWE-502',
            'path_traversal': 'CWE-22',
            'weak_cryptography': 'CWE-327',
            'command_injection': 'CWE-78',
            'insecure_random': 'CWE-338',
            'open_redirect': 'CWE-601',
            'ssrf': 'CWE-918',
        }

    def _initialize_owasp_mapping(self) -> Dict[str, str]:
        """Initialize OWASP Top 10 category mapping."""
        return {
            'sql_injection': 'A03:2021 - Injection',
            'xss': 'A03:2021 - Injection',
            'hardcoded_credentials': 'A07:2021 - Identification and Authentication Failures',
            'insecure_deserialization': 'A08:2021 - Software and Data Integrity Failures',
            'path_traversal': 'A01:2021 - Broken Access Control',
            'weak_cryptography': 'A02:2021 - Cryptographic Failures',
            'command_injection': 'A03:2021 - Injection',
            'insecure_random': 'A02:2021 - Cryptographic Failures',
            'open_redirect': 'A01:2021 - Broken Access Control',
            'ssrf': 'A01:2021 - Broken Access Control',
        }

    # =========================================================================
    # Main Detection Method
    # =========================================================================

    def detect_all_vulnerabilities(
        self,
        content: str,
        file_path: str,
        language: str = "python"
    ) -> List[SecurityIssue]:
        """
        Detect all security vulnerabilities in the given content.

        Args:
            content: Source code content
            file_path: Path to the file
            language: Programming language

        Returns:
            List of detected security issues
        """
        issues = []

        # Detect SQL injection
        issues.extend(self._detect_sql_injection(content, file_path, language))

        # Detect XSS
        issues.extend(self._detect_xss(content, file_path, language))

        # Detect hardcoded credentials
        issues.extend(self._detect_hardcoded_credentials(content, file_path))

        # Detect insecure deserialization
        issues.extend(self._detect_insecure_deserialization(content, file_path, language))

        # Detect path traversal
        issues.extend(self._detect_path_traversal(content, file_path, language))

        # Detect weak cryptography
        issues.extend(self._detect_weak_cryptography(content, file_path, language))

        # Detect command injection
        issues.extend(self._detect_command_injection(content, file_path, language))

        # Detect insecure random
        issues.extend(self._detect_insecure_random(content, file_path, language))

        # Detect open redirect
        issues.extend(self._detect_open_redirect(content, file_path, language))

        # Detect SSRF
        issues.extend(self._detect_ssrf(content, file_path, language))

        return issues

    # =========================================================================
    # Individual Vulnerability Detectors
    # =========================================================================

    def _detect_sql_injection(
        self,
        content: str,
        file_path: str,
        language: str
    ) -> List[SecurityIssue]:
        """Detect SQL injection vulnerabilities."""
        issues = []
        lines = content.split('\n')

        for i, line in enumerate(lines):
            for pattern in self._patterns['sql_injection']:
                if pattern.search(line):
                    location = CodeLocation(
                        file_path=file_path,
                        start_line=i + 1,
                        end_line=i + 1
                    )

                    issues.append(SecurityIssue(
                        issue_type=SecurityIssueType.SQL_INJECTION,
                        severity=SeverityLevel.CRITICAL,
                        location=location,
                        description="Potential SQL injection vulnerability detected",
                        cwe_id=self._cwe_mapping['sql_injection'],
                        owasp_category=self._owasp_mapping['sql_injection'],
                        remediation="Use parameterized queries or prepared statements instead of string concatenation."
                    ))
                    break  # Avoid duplicate issues for the same line

        return issues

    def _detect_xss(
        self,
        content: str,
        file_path: str,
        language: str
    ) -> List[SecurityIssue]:
        """Detect XSS (Cross-Site Scripting) vulnerabilities."""
        issues = []
        lines = content.split('\n')

        for i, line in enumerate(lines):
            for pattern in self._patterns['xss']:
                if pattern.search(line):
                    location = CodeLocation(
                        file_path=file_path,
                        start_line=i + 1,
                        end_line=i + 1
                    )

                    issues.append(SecurityIssue(
                        issue_type=SecurityIssueType.XSS,
                        severity=SeverityLevel.HIGH,
                        location=location,
                        description="Potential XSS vulnerability detected",
                        cwe_id=self._cwe_mapping['xss'],
                        owasp_category=self._owasp_mapping['xss'],
                        remediation="Sanitize and escape user input before rendering. Use template engines with auto-escaping."
                    ))
                    break

        return issues

    def _detect_hardcoded_credentials(
        self,
        content: str,
        file_path: str
    ) -> List[SecurityIssue]:
        """Detect hardcoded credentials."""
        issues = []
        lines = content.split('\n')

        for i, line in enumerate(lines):
            for pattern in self._patterns['hardcoded_credentials']:
                match = pattern.search(line)
                if match:
                    location = CodeLocation(
                        file_path=file_path,
                        start_line=i + 1,
                        end_line=i + 1
                    )

                    # Extract the credential type
                    credential_type = match.group(1) if match.groups() else "credential"

                    issues.append(SecurityIssue(
                        issue_type=SecurityIssueType.HARDCODED_CREDENTIALS,
                        severity=SeverityLevel.CRITICAL,
                        location=location,
                        description=f"Hardcoded {credential_type} detected",
                        cwe_id=self._cwe_mapping['hardcoded_credentials'],
                        owasp_category=self._owasp_mapping['hardcoded_credentials'],
                        remediation="Move credentials to environment variables or a secure configuration management system."
                    ))
                    break

        return issues

    def _detect_insecure_deserialization(
        self,
        content: str,
        file_path: str,
        language: str
    ) -> List[SecurityIssue]:
        """Detect insecure deserialization vulnerabilities."""
        issues = []
        lines = content.split('\n')

        for i, line in enumerate(lines):
            for pattern in self._patterns['insecure_deserialization']:
                if pattern.search(line):
                    location = CodeLocation(
                        file_path=file_path,
                        start_line=i + 1,
                        end_line=i + 1
                    )

                    issues.append(SecurityIssue(
                        issue_type=SecurityIssueType.INSECURE_DESERIALIZATION,
                        severity=SeverityLevel.HIGH,
                        location=location,
                        description="Insecure deserialization detected",
                        cwe_id=self._cwe_mapping['insecure_deserialization'],
                        owasp_category=self._owasp_mapping['insecure_deserialization'],
                        remediation="Use safe deserialization methods (e.g., yaml.safe_load, pickle with restricted protocols)."
                    ))
                    break

        return issues

    def _detect_path_traversal(
        self,
        content: str,
        file_path: str,
        language: str
    ) -> List[SecurityIssue]:
        """Detect path traversal vulnerabilities."""
        issues = []
        lines = content.split('\n')

        for i, line in enumerate(lines):
            for pattern in self._patterns['path_traversal']:
                if pattern.search(line):
                    location = CodeLocation(
                        file_path=file_path,
                        start_line=i + 1,
                        end_line=i + 1
                    )

                    issues.append(SecurityIssue(
                        issue_type=SecurityIssueType.PATH_TRAVERSAL,
                        severity=SeverityLevel.HIGH,
                        location=location,
                        description="Potential path traversal vulnerability detected",
                        cwe_id=self._cwe_mapping['path_traversal'],
                        owasp_category=self._owasp_mapping['path_traversal'],
                        remediation="Validate and sanitize file paths. Use os.path.abspath() and check if path is within allowed directory."
                    ))
                    break

        return issues

    def _detect_weak_cryptography(
        self,
        content: str,
        file_path: str,
        language: str
    ) -> List[SecurityIssue]:
        """Detect weak cryptography usage."""
        issues = []
        lines = content.split('\n')

        for i, line in enumerate(lines):
            for pattern in self._patterns['weak_cryptography']:
                match = pattern.search(line)
                if match:
                    location = CodeLocation(
                        file_path=file_path,
                        start_line=i + 1,
                        end_line=i + 1
                    )

                    # Extract the algorithm
                    algorithm = match.group(1) if match.groups() else "weak algorithm"

                    issues.append(SecurityIssue(
                        issue_type=SecurityIssueType.WEAK_CRYPTOGRAPHY,
                        severity=SeverityLevel.MEDIUM,
                        location=location,
                        description=f"Weak cryptography algorithm detected: {algorithm}",
                        cwe_id=self._cwe_mapping['weak_cryptography'],
                        owasp_category=self._owasp_mapping['weak_cryptography'],
                        remediation="Use strong cryptographic algorithms (e.g., SHA-256, AES-256, bcrypt for passwords)."
                    ))
                    break

        return issues

    def _detect_command_injection(
        self,
        content: str,
        file_path: str,
        language: str
    ) -> List[SecurityIssue]:
        """Detect command injection vulnerabilities."""
        issues = []
        lines = content.split('\n')

        for i, line in enumerate(lines):
            for pattern in self._patterns['command_injection']:
                if pattern.search(line):
                    location = CodeLocation(
                        file_path=file_path,
                        start_line=i + 1,
                        end_line=i + 1
                    )

                    issues.append(SecurityIssue(
                        issue_type=SecurityIssueType.COMMAND_INJECTION,
                        severity=SeverityLevel.CRITICAL,
                        location=location,
                        description="Potential command injection vulnerability detected",
                        cwe_id=self._cwe_mapping['command_injection'],
                        owasp_category=self._owasp_mapping['command_injection'],
                        remediation="Avoid shell=True in subprocess calls. Use subprocess.run with list of arguments instead."
                    ))
                    break

        return issues

    def _detect_insecure_random(
        self,
        content: str,
        file_path: str,
        language: str
    ) -> List[SecurityIssue]:
        """Detect insecure random number generation."""
        issues = []
        lines = content.split('\n')

        for i, line in enumerate(lines):
            for pattern in self._patterns['insecure_random']:
                if pattern.search(line):
                    location = CodeLocation(
                        file_path=file_path,
                        start_line=i + 1,
                        end_line=i + 1
                    )

                    issues.append(SecurityIssue(
                        issue_type=SecurityIssueType.INSECURE_RANDOM,
                        severity=SeverityLevel.MEDIUM,
                        location=location,
                        description="Insecure random number generation detected",
                        cwe_id=self._cwe_mapping['insecure_random'],
                        owasp_category=self._owasp_mapping['insecure_random'],
                        remediation="Use secrets module for cryptographic random numbers (e.g., secrets.randbelow, secrets.token_hex)."
                    ))
                    break

        return issues

    def _detect_open_redirect(
        self,
        content: str,
        file_path: str,
        language: str
    ) -> List[SecurityIssue]:
        """Detect open redirect vulnerabilities."""
        issues = []
        lines = content.split('\n')

        for i, line in enumerate(lines):
            for pattern in self._patterns['open_redirect']:
                if pattern.search(line):
                    location = CodeLocation(
                        file_path=file_path,
                        start_line=i + 1,
                        end_line=i + 1
                    )

                    issues.append(SecurityIssue(
                        issue_type=SecurityIssueType.OPEN_REDIRECT,
                        severity=SeverityLevel.MEDIUM,
                        location=location,
                        description="Potential open redirect vulnerability detected",
                        cwe_id=self._cwe_mapping['open_redirect'],
                        owasp_category=self._owasp_mapping['open_redirect'],
                        remediation="Validate redirect URLs against a whitelist of allowed domains."
                    ))
                    break

        return issues

    def _detect_ssrf(
        self,
        content: str,
        file_path: str,
        language: str
    ) -> List[SecurityIssue]:
        """Detect SSRF (Server-Side Request Forgery) vulnerabilities."""
        issues = []
        lines = content.split('\n')

        for i, line in enumerate(lines):
            for pattern in self._patterns['ssrf']:
                if pattern.search(line):
                    location = CodeLocation(
                        file_path=file_path,
                        start_line=i + 1,
                        end_line=i + 1
                    )

                    issues.append(SecurityIssue(
                        issue_type=SecurityIssueType.SSRF,
                        severity=SeverityLevel.HIGH,
                        location=location,
                        description="Potential SSRF vulnerability detected",
                        cwe_id=self._cwe_mapping['ssrf'],
                        owasp_category=self._owasp_mapping['ssrf'],
                        remediation="Validate and sanitize URLs. Use a whitelist of allowed domains and protocols."
                    ))
                    break

        return issues

    # =========================================================================
    # Python-specific Security Analysis
    # =========================================================================

    def analyze_python_security(self, content: str, file_path: str) -> List[SecurityIssue]:
        """
        Perform Python-specific security analysis using AST.

        Args:
            content: Python source code
            file_path: Path to the file

        Returns:
            List of security issues
        """
        issues = []

        try:
            tree = ast.parse(content)
            visitor = PythonSecurityVisitor(file_path)
            visitor.visit(tree)
            issues.extend(visitor.issues)
        except SyntaxError:
            # Fall back to regex-based detection
            pass

        return issues


class PythonSecurityVisitor(ast.NodeVisitor):
    """AST visitor for Python-specific security analysis."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.issues: List[SecurityIssue] = []

    def visit_Call(self, node: ast.Call):
        """Visit function calls for security issues."""

        # Check for dangerous function calls
        if isinstance(node.func, ast.Attribute):
            # pickle.loads
            if (isinstance(node.func.value, ast.Name) and
                node.func.value.id == 'pickle' and
                node.func.attr == 'loads'):
                self._add_issue(
                    SecurityIssueType.INSECURE_DESERIALIZATION,
                    SeverityLevel.HIGH,
                    node.lineno,
                    "Insecure deserialization using pickle.loads"
                )

            # yaml.load
            elif (isinstance(node.func.value, ast.Name) and
                  node.func.value.id == 'yaml' and
                  node.func.attr == 'load'):
                self._add_issue(
                    SecurityIssueType.INSECURE_DESERIALIZATION,
                    SeverityLevel.HIGH,
                    node.lineno,
                    "Insecure deserialization using yaml.load (use yaml.safe_load instead)"
                )

            # os.system
            elif (isinstance(node.func.value, ast.Name) and
                  node.func.value.id == 'os' and
                  node.func.attr == 'system'):
                self._add_issue(
                    SecurityIssueType.COMMAND_INJECTION,
                    SeverityLevel.CRITICAL,
                    node.lineno,
                    "Potential command injection using os.system"
                )

            # subprocess with shell=True
            elif (isinstance(node.func.value, ast.Name) and
                  node.func.value.id == 'subprocess' and
                  node.func.attr in ('call', 'run', 'Popen')):
                # Check for shell=True keyword argument
                for keyword in node.keywords:
                    if keyword.arg == 'shell' and isinstance(keyword.value, ast.Constant):
                        if keyword.value.value is True:
                            self._add_issue(
                                SecurityIssueType.COMMAND_INJECTION,
                                SeverityLevel.CRITICAL,
                                node.lineno,
                                "Potential command injection using subprocess with shell=True"
                            )
                            break

        # Check for eval/exec
        elif isinstance(node.func, ast.Name):
            if node.func.id in ('eval', 'exec'):
                self._add_issue(
                    SecurityIssueType.COMMAND_INJECTION,
                    SeverityLevel.HIGH,
                    node.lineno,
                    f"Potential code injection using {node.func.id}"
                )

        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        """Check for dangerous imports."""
        for alias in node.names:
            if alias.name == 'pickle':
                self._add_issue(
                    SecurityIssueType.INSECURE_DESERIALIZATION,
                    SeverityLevel.MEDIUM,
                    node.lineno,
                    "Import of pickle module (ensure safe usage)"
                )
            elif alias.name == 'random':
                self._add_issue(
                    SecurityIssueType.INSECURE_RANDOM,
                    SeverityLevel.MEDIUM,
                    node.lineno,
                    "Import of random module (use secrets module for cryptographic randomness)"
                )

        self.generic_visit(node)

    def _add_issue(
        self,
        issue_type: SecurityIssueType,
        severity: SeverityLevel,
        line: int,
        description: str
    ):
        """Add a security issue to the list."""
        location = CodeLocation(
            file_path=self.file_path,
            start_line=line,
            end_line=line
        )

        self.issues.append(SecurityIssue(
            issue_type=issue_type,
            severity=severity,
            location=location,
            description=description,
            cwe_id=self._get_cwe_id(issue_type),
            owasp_category=self._get_owasp_category(issue_type)
        ))

    def _get_cwe_id(self, issue_type: SecurityIssueType) -> str:
        """Get CWE ID for issue type."""
        mapping = {
            SecurityIssueType.INSECURE_DESERIALIZATION: 'CWE-502',
            SecurityIssueType.COMMAND_INJECTION: 'CWE-78',
            SecurityIssueType.INSECURE_RANDOM: 'CWE-338',
        }
        return mapping.get(issue_type, '')

    def _get_owasp_category(self, issue_type: SecurityIssueType) -> str:
        """Get OWASP category for issue type."""
        mapping = {
            SecurityIssueType.INSECURE_DESERIALIZATION: 'A08:2021 - Software and Data Integrity Failures',
            SecurityIssueType.COMMAND_INJECTION: 'A03:2021 - Injection',
            SecurityIssueType.INSECURE_RANDOM: 'A02:2021 - Cryptographic Failures',
        }
        return mapping.get(issue_type, '')
