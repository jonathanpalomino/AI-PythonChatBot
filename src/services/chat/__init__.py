# =============================================================================
# src/services/chat/__init__.py
# Chat Service Package - Refactored for SRP
# =============================================================================
"""
Chat service package with separated responsibilities.

SRP REFACTORING COMPLETE:
- ChatOrchestrator: Main coordinator (~300 lines, delegates to specialists)
- ContextBuilder: RAG context construction (~900 lines)
- ToolExecutor: Tool execution logic (~550 lines)
- ResponseFormatter: Response formatting (~350 lines)
- StreamHandler: Streaming support (~350 lines)

BEFORE: ChatOrchestrator had 2725 lines with 8+ responsibilities
AFTER: Each class has ONE responsibility (Single Responsibility Principle)

Usage:
    from src.services.chat import ChatOrchestrator, ContextBuilder, ToolExecutor
    
    # Full orchestration (recommended)
    orchestrator = ChatOrchestrator(conversation_repo, message_repo, file_repo, custom_tool_repo)
    response = await orchestrator.process_message(conversation, user_message)
    
    # Direct specialist access (advanced use cases)
    context_builder = ContextBuilder(conversation_repo, message_repo, file_repo)
    messages = await context_builder.build_message_history(conversation, user_message, settings)
"""

from src.services.chat.context_builder import ContextBuilder
from src.services.chat.tool_executor import ToolExecutor
from src.services.chat.response_formatter import ResponseFormatter
from src.services.chat.stream_handler import StreamHandler
from src.services.chat.orchestrator import ChatOrchestrator

__all__ = [
    "ChatOrchestrator",
    "ContextBuilder",
    "ToolExecutor",
    "ResponseFormatter",
    "StreamHandler",
]
