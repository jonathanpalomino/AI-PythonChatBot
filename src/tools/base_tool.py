# =============================================================================
# src/tools/base_tool.py
# Base class for all tools
# =============================================================================
"""
Sistema base de tools extensible para el chatbot
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ToolCategory(str, Enum):
    """Tool categories"""
    RAG = "rag"
    CODE = "code"
    DOCUMENT = "document"
    MEMORY = "memory"
    WEB = "web"
    UTILITY = "utility"


@dataclass
class ToolParameter:
    """Tool parameter definition"""
    name: str
    type: str  # "string", "integer", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None
    example: Optional[Any] = None  # Example value for this parameter


@dataclass
class ToolResult:
    """Tool execution result"""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseTool(ABC):
    """Base class for all tools"""

    def __init__(self):
        self._validate_tool_definition()

    # =============================================================================
    # Tool Metadata (must be defined by subclasses)
    # =============================================================================

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name (e.g., 'rag_search', 'code_analyzer')"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable tool description"""
        pass

    @property
    @abstractmethod
    def category(self) -> ToolCategory:
        """Tool category"""
        pass

    @property
    def enabled_by_default(self) -> bool:
        """Whether tool is enabled by default"""
        return False

    @property
    def requires_context(self) -> List[str]:
        """
        Required context/dependencies (e.g., ["qdrant", "files"])
        Returns empty list if no dependencies
        """
        return []

    # =============================================================================
    # Tool Definition (for LLM function calling)
    # =============================================================================

    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """Get tool parameters definition"""
        pass

    def to_openai_function(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format"""
        parameters = self.get_parameters()

        properties = {}
        required = []

        for param in parameters:
            param_def = {
                "type": param.type,
                "description": param.description
            }

            if param.enum:
                param_def["enum"] = param.enum

            properties[param.name] = param_def

            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }

    def to_anthropic_tool(self) -> Dict[str, Any]:
        """Convert to Anthropic tool format"""
        parameters = self.get_parameters()

        input_schema = {
            "type": "object",
            "properties": {},
            "required": []
        }

        for param in parameters:
            param_def = {
                "type": param.type,
                "description": param.description
            }

            if param.enum:
                param_def["enum"] = param.enum

            input_schema["properties"][param.name] = param_def

            if param.required:
                input_schema["required"].append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": input_schema
        }

    # =============================================================================
    # Tool Execution
    # =============================================================================

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            ToolResult with success status and data
        """
        pass

    async def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters

        Returns:
            True if valid, False otherwise
        """
        parameters = self.get_parameters()

        for param in parameters:
            if param.required and param.name not in kwargs:
                raise ValueError(f"Missing required parameter: {param.name}")

            if param.name in kwargs:
                value = kwargs[param.name]

                # Skip validation for None values in optional parameters
                if value is None:
                    continue

                # Type validation (basic)
                if param.type == "string" and not isinstance(value, str):
                    raise ValueError(f"Parameter {param.name} must be string")
                elif param.type == "integer" and not isinstance(value, int):
                    raise ValueError(f"Parameter {param.name} must be integer")
                elif param.type == "boolean" and not isinstance(value, bool):
                    raise ValueError(f"Parameter {param.name} must be boolean")
                elif param.type == "array" and not isinstance(value, list):
                    raise ValueError(f"Parameter {param.name} must be array")

                # Enum validation
                if param.enum and value not in param.enum:
                    raise ValueError(
                        f"Parameter {param.name} must be one of {param.enum}"
                    )

        return True

    def format_output(self, result: ToolResult) -> str:
        """
        Format tool output for LLM consumption
        Can be overridden by subclasses for custom formatting

        Args:
            result: Tool execution result

        Returns:
            Formatted string for LLM
        """
        if not result.success:
            return f"Error executing {self.name}: {result.error}"

        return str(result.data)

    # =============================================================================
    # Utility Methods
    # =============================================================================

    def _validate_tool_definition(self):
        """Validate that tool is properly defined"""
        if not self.name:
            raise ValueError("Tool must have a name")
        if not self.description:
            raise ValueError("Tool must have a description")
        if not self.category:
            raise ValueError("Tool must have a category")

    def __repr__(self):
        return f"<{self.__class__.__name__}(name={self.name})>"


# =============================================================================
# Tool Registry
# =============================================================================

class ToolRegistry:
    """Registry for managing available tools"""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        """Register a tool"""
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name} already registered")

        self._tools[tool.name] = tool
        print(f"✅ Tool registered: {tool.name}")

    def unregister(self, tool_name: str):
        """Unregister a tool"""
        if tool_name in self._tools:
            del self._tools[tool_name]
            print(f"🗑️  Tool unregistered: {tool_name}")

    def get(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name"""
        return self._tools.get(tool_name)

    def get_all(self) -> List[BaseTool]:
        """Get all registered tools"""
        return list(self._tools.values())

    def get_by_category(self, category: ToolCategory) -> List[BaseTool]:
        """Get tools by category"""
        return [
            tool for tool in self._tools.values()
            if tool.category == category
        ]

    def get_enabled_by_default(self) -> List[BaseTool]:
        """Get tools enabled by default"""
        return [
            tool for tool in self._tools.values()
            if tool.enabled_by_default
        ]

    def list_names(self) -> List[str]:
        """List all tool names"""
        return list(self._tools.keys())

    def to_openai_functions(self, tool_names: Optional[List[str]] = None) -> List[Dict]:
        """
        Convert tools to OpenAI function calling format

        Args:
            tool_names: Specific tools to convert, or None for all
        """
        if tool_names is None:
            tools = self._tools.values()
        else:
            tools = [self._tools[name] for name in tool_names if name in self._tools]

        return [tool.to_openai_function() for tool in tools]

    def to_anthropic_tools(self, tool_names: Optional[List[str]] = None) -> List[Dict]:
        """
        Convert tools to Anthropic tool format

        Args:
            tool_names: Specific tools to convert, or None for all
        """
        if tool_names is None:
            tools = self._tools.values()
        else:
            tools = [self._tools[name] for name in tool_names if name in self._tools]

        return [tool.to_anthropic_tool() for tool in tools]

    def __repr__(self):
        return f"<ToolRegistry(tools={len(self._tools)})>"


# =============================================================================
# Global Registry Instance
# =============================================================================

tool_registry = ToolRegistry()
