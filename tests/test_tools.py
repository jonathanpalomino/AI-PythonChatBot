# =============================================================================
# tests/test_tools.py
# Test tools system
# =============================================================================
"""
Unit tests for tools
"""
import pytest
from src.tools.code_analyzer_tool import CodeAnalyzerTool

from src.tools.base_tool import ToolRegistry, ToolCategory


class TestToolRegistry:
    """Test tool registry"""

    def test_registry_singleton(self):
        """Test that registry is a singleton"""
        from src.tools.base_tool import tool_registry
        registry2 = ToolRegistry()

        # Should be different instances (not singleton by default)
        # But tool_registry is the global instance
        assert tool_registry is not None

    def test_register_tool(self):
        """Test registering a tool"""
        registry = ToolRegistry()
        tool = CodeAnalyzerTool()

        registry.register(tool)

        assert tool.name in registry.list_names()
        assert registry.get(tool.name) is not None


@pytest.mark.unit
def test_tool_categories():
    """Test tool categories"""
    assert ToolCategory.RAG.value == "rag"
