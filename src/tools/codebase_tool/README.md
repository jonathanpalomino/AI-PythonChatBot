# Professional Codebase Analysis Tool

## Overview

The Professional Codebase Analysis Tool is a comprehensive static code analysis system that provides:

- **Advanced Code Quality Metrics**: Cyclomatic complexity, cognitive complexity, maintainability index, technical debt
- **Code Smell Detection**: 14+ code smells including duplicated code, long methods, large classes, god objects
- **Security Vulnerability Scanning**: 10+ security issues including SQL injection, XSS, hardcoded credentials
- **Refactoring Suggestions**: 14+ refactoring types with code generation
- **Multi-language Support**: Python, Java, JavaScript, TypeScript, C#, Go, and more
- **SARIF Report Generation**: SARIF 2.1.0 compliant reports for CI/CD integration
- **LLM-Optimized Output**: Formatted output for optimal LLM consumption

## Architecture

```
src/tools/codebase_tool/
├── __init__.py           # Package initialization and exports
├── core.py               # Main CodebaseTool implementation
├── models.py             # Data models (enums, dataclasses)
├── metrics.py            # Metrics calculator
├── code_smells.py        # Code smell detector
├── security.py           # Security vulnerability analyzer
├── refactoring.py        # Refactoring suggestion generator
├── sarif.py             # SARIF report generator
└── llm_formatter.py     # LLM-optimized output formatter
```

## Quick Start

### Basic Usage

```python
from src.tools.codebase_tool import CodebaseTool

# Initialize the tool
tool = CodebaseTool()

# Analyze uploaded files
result = await tool.execute(
    action="analyze_file",
    file_ids=["uuid-of-file-1", "uuid-of-file-2"]
)

# Get analysis results
if result.success:
    analysis = result.data
    print(f"Files analyzed: {analysis['files_analyzed']}")
    print(f"Code smells: {analysis['summary']['total_code_smells']}")
    print(f"Security issues: {analysis['summary']['total_security_issues']}")
```

### Generate SARIF Report

```python
from src.tools.codebase_tool import CodebaseTool

tool = CodebaseTool()

result = await tool.execute(
    action="generatesarif",
    file_ids=["uuid-of-file"]
)

if result.success:
    sarif_json = result.data
    # Save to file
    with open("analysis.sarif", "w") as f:
        f.write(sarif_json)
```

### Format for LLM

```python
from src.tools.codebase_tool import CodebaseTool

tool = CodebaseTool()

result = await tool.execute(
    action="formatforllm",
    file_ids=["uuid-of-file"],
    format="detailed"  # or "summary", "compact"
)

if result.success:
    formatted_output = result.data
    print(formatted_output)  # Markdown formatted for LLM
```

## Actions

### `analyze_file`

Analyze uploaded files and return comprehensive analysis results.

**Parameters:**
- `file_ids` (required): List of file UUIDs to analyze

**Returns:**
- Complete analysis with metrics, code smells, security issues, and refactoring suggestions

### `generatesarif`

Generate SARIF 2.1.0 compliant report.

**Parameters:**
- `file_ids` (required): List of file UUIDs to analyze

**Returns:**
- SARIF JSON string

### `formatforllm`

Format analysis results for LLM consumption.

**Parameters:**
- `file_ids` (required): List of file UUIDs to analyze
- `format` (optional): Output format - "detailed", "summary", or "compact" (default: "detailed")

**Returns:**
- Markdown formatted string

## Analysis Results

### Metrics

The tool calculates comprehensive metrics:

**Complexity Metrics:**
- Cyclomatic Complexity: Number of independent paths through code
- Cognitive Complexity: How difficult the code is to understand
- Nesting Depth: Maximum nesting level

**Size Metrics:**
- Lines of Code: Actual code lines
- Lines of Comments: Comment lines
- Comment Ratio: Percentage of comments

**Maintainability Metrics:**
- Maintainability Index: 0-100 score (higher is better)
- Technical Debt: Estimated hours to fix issues
- Technical Debt Ratio: Ratio of technical debt to total effort

**Coupling Metrics:**
- Afferent Coupling (Ca): Classes that depend on this
- Efferent Coupling (Ce): Classes this depends on
- Instability: Ce / (Ca + Ce) - 0 (stable) to 1 (unstable)

**Cohesion Metrics:**
- LCOM4: Lack of Cohesion of Methods
- Cohesion Ratio: 0-1 (higher is better)

**Duplication Metrics:**
- Duplication Ratio: Percentage of duplicated code
- Duplicated Lines: Number of duplicated lines
- Duplicated Blocks: Number of duplicated blocks

### Code Smells

The tool detects 14+ code smells:

1. **Duplicated Code**: Code blocks that appear in multiple locations
2. **Long Method**: Methods with >50 lines
3. **Large Class**: Classes with >15 methods
4. **Feature Envy**: Methods that access other classes' data more than their own
5. **Inappropriate Intimacy**: Classes that know too much about each other
6. **Lazy Class**: Classes that do very little
7. **Data Clumps**: Groups of parameters that always appear together
8. **Primitive Obsession**: Excessive use of primitive types
9. **Switch Statements**: Excessive use of switch/case
10. **Temporary Field**: Fields that are only used in certain contexts
11. **Refused Bequest**: Subclasses that don't use inherited methods
12. **Comments**: TODO/FIXME comments
13. **God Object**: Classes that know too much or do too much
14. **Spaghetti Code**: Code with deep nesting

### Security Issues

The tool detects 10+ security vulnerabilities:

1. **SQL Injection**: String concatenation in SQL queries
2. **XSS**: Cross-Site Scripting vulnerabilities
3. **Hardcoded Credentials**: Passwords/API keys in code
4. **Insecure Deserialization**: Unsafe deserialization
5. **Path Traversal**: Unvalidated file paths
6. **Weak Cryptography**: MD5, SHA1, DES, etc.
7. **Command Injection**: Unsafe command execution
8. **Insecure Random**: Non-cryptographic random numbers
9. **Open Redirect**: Unvalidated redirects
10. **SSRF**: Server-Side Request Forgery

Each security issue includes:
- CWE ID (Common Weakness Enumeration)
- OWASP Top 10 category
- Remediation suggestions

### Refactoring Suggestions

The tool generates 14+ refactoring suggestions:

1. **Extract Method**: Extract code block into separate method
2. **Extract Class**: Extract related functionality into new class
3. **Inline Method**: Simplify trivial methods
4. **Replace Conditional with Polymorphism**: Eliminate complex conditionals
5. **Decompose Conditional**: Simplify complex conditions
6. **Consolidate Conditional**: Merge duplicate conditionals
7. **Replace Magic Number**: Extract magic numbers to constants
8. **Introduce Parameter Object**: Group related parameters
9. **Preserve Whole Object**: Pass object instead of individual fields
10. **Replace Type Code with Class**: Use classes instead of type codes
11. **Rename**: Improve naming
12. **Move Method**: Move method to appropriate class
13. **Move Field**: Move field to appropriate class
14. **Extract Interface**: Define interface for class

Each suggestion includes:
- Description and rationale
- Suggested code (when applicable)
- Effort estimate (minutes, hours, days)
- Impact scope (local, module, system)
- Benefits

## SARIF Integration

The tool generates SARIF 2.1.0 compliant reports that can be integrated with:

- GitHub Advanced Security
- Azure DevOps
- GitLab
- SonarQube
- VS Code SARIF Viewer

### Example SARIF Usage

```bash
# Generate SARIF report
python -c "
from src.tools.codebase_tool import CodebaseTool
import asyncio

async def main():
    tool = CodebaseTool()
    result = await tool.execute(
        action='generatesarif',
        file_ids=['file-uuid']
    )
    if result.success:
        print(result.data)

asyncio.run(main())
" > analysis.sarif

# View in VS Code
code --install-extension ms-sarifvscode.sarif-viewer
code analysis.sarif
```

## LLM Integration

The tool provides LLM-optimized output formats:

### Detailed Format

Comprehensive markdown with:
- Executive summary
- File-by-file analysis
- Metrics breakdown
- Code smells with severity
- Security issues with CWE/OWASP
- Refactoring suggestions with code

### Summary Format

High-level overview with:
- Issue counts
- Severity breakdown
- File summaries

### Compact Format

One-line per file with:
- File path
- Language
- Maintainability Index
- Cyclomatic Complexity
- Critical/High issue counts

### Prompt Templates

The tool includes prompt templates for:

- **Refactoring**: Request code refactoring with context
- **Explanation**: Request code explanation with structure
- **Review**: Request code review with security focus

## Advanced Usage

### Using Individual Components

```python
from src.tools.codebase_tool import (
    MetricsCalculator,
    CodeSmellDetector,
    SecurityAnalyzer,
    RefactoringSuggester
)

# Calculate metrics
calculator = MetricsCalculator()
metrics = calculator.calculate_all_metrics(content, "python")

# Detect code smells
detector = CodeSmellDetector()
smells = detector.detect_all_smells(content, file_path, "python")

# Analyze security
analyzer = SecurityAnalyzer()
issues = analyzer.detect_all_vulnerabilities(content, file_path, "python")

# Generate suggestions
suggester = RefactoringSuggester()
suggestions = suggester.generate_all_suggestions(
    content, file_path, "python", symbols, smells, issues
)
```

### Custom Analysis

```python
from src.tools.codebase_tool import (
    FileAnalysisResult,
    AnalysisResult,
    CodeMetrics
)

# Create custom analysis result
file_result = FileAnalysisResult(
    file_id="uuid",
    file_path="example.py",
    language="python",
    symbols=[],
    classes=[],
    functions=[],
    imports=[],
    metrics=CodeMetrics(...),
    code_smells=[],
    security_issues=[],
    solid_violations=[],
    refactoring_suggestions=[],
    content=content
)

analysis_result = AnalysisResult(
    action="custom",
    target=None,
    files_analyzed=1,
    results=[file_result]
)
```

## Best Practices

1. **Analyze Early and Often**: Run analysis regularly during development
2. **Address Critical Issues First**: Prioritize security vulnerabilities
3. **Use SARIF for CI/CD**: Integrate SARIF reports into your pipeline
4. **Leverage LLM Output**: Use formatted output for AI-assisted refactoring
5. **Track Metrics Over Time**: Monitor maintainability index and technical debt

## Troubleshooting

### File Not Found

If you get "File not found" errors:
- Ensure file IDs are valid UUIDs
- Check that files exist in the database
- Verify file_repo is properly configured

### Analysis Errors

If analysis fails:
- Check file encoding (UTF-8 recommended)
- Verify language detection
- Review logs for specific errors

### Performance Issues

For large codebases:
- Analyze files in batches
- Use summary format for quick overview
- Consider caching results

## Contributing

To extend the tool:

1. Add new metrics in `metrics.py`
2. Add new code smells in `code_smells.py`
3. Add new security checks in `security.py`
4. Add new refactoring types in `refactoring.py`
5. Update models in `models.py`

## License

See project LICENSE file.
