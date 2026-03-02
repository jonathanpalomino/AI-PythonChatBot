# =============================================================================
# src/config/constants.py
# Centralized application constants
# =============================================================================

# Directory exclusions for codebase analysis
IGNORED_DIRS = (
    ".venv", "venv", "__pycache__", "node_modules",
    "dist", ".git", ".idea", ".vscode", "target",
    "build", "bin", ".svn"
)

# File extension exclusions for analysis
IGNORED_EXTENSIONS = (
    ".class", ".pyc", ".pyo", ".exe", ".so",
    ".dll", ".bin", ".obj"
)

# RAG Defaults
DEFAULT_RAG_LIMIT = 5
DEFAULT_SCORE_THRESHOLD = 0.5
CODE_CHUNK_SIZE = 8000  # Optimal chunk size for code files

# Extraction Service - Faster model for intent extraction
EXTRACTION_MODEL = "qwen2.5:1.5b"  # Optimized for fast JSON extraction

# =============================================================================
# Code File Extensions
# =============================================================================

CODE_EXTENSIONS = {
    '.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.cpp', '.c', '.h', '.hpp',
    '.cs', '.go', '.rb', '.php', '.swift', '.kt', '.rs', '.scala', '.sh',
    '.sql', '.r', '.m', '.dart', '.lua', '.pl', '.vim'
}

# =============================================================================
# Physical Tools (built-in tools with fixed names)
# =============================================================================

PHYSICAL_TOOLS = {
    "rag_search",
    "codebase_tool",
    "http_request",
    "sql_query",
    "web_search"
}
