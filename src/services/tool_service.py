# =============================================================================
# src/services/tool_service.py
# Tool Service - Business Logic
# =============================================================================
"""
Business logic for tool operations

REFACTORED: Service now receives Repositories directly, not UnitOfWork.
This follows the Repository pattern correctly:
    Service → Repository → Session
"""
from typing import List, Optional, Dict
from uuid import UUID

from src.models.models import (
    ToolConfiguration, CustomTool
)
from src.schemas.schemas import (
    ToolConfigurationCreate, ToolConfigurationUpdate, CustomToolCreate, CustomToolUpdate
)
from src.tools.base_tool import tool_registry
from src.utils.logger import get_logger
from src.utils.transactional import transactional

logger = get_logger(__name__)


class ToolService:
    """
    Service for tool business logic.

    REFACTORED: Receives Repositories directly, not UnitOfWork.
    This follows the Repository pattern correctly.
    """

    def __init__(
        self,
        custom_tool_repo,
        tool_configuration_repo,
        conversation_repo,
        file_repo=None
    ):
        """
        Initialize ToolService with repositories.

        Args:
            custom_tool_repo: CustomToolRepository instance
            tool_configuration_repo: ToolConfigurationRepository instance
            conversation_repo: ConversationRepository instance
            file_repo: FileRepository instance (optional, for CustomToolExecutor)
        """
        self.custom_tool_repo = custom_tool_repo
        self.tool_configuration_repo = tool_configuration_repo
        self.conversation_repo = conversation_repo
        self.file_repo = file_repo

    # =============================================================================
    # Tool Registry Operations
    # =============================================================================

    def list_available_tools(self) -> List[dict]:
        """List all available tools in the registry"""
        logger.info("Listing available tools")
        tools = tool_registry.get_all()
        logger.info(f"Found {len(tools)} available tools")

        return [
            {
                "name": tool.name,
                "description": tool.description,
                "category": tool.category.value,
                "enabled_by_default": tool.enabled_by_default,
                "requires_context": tool.requires_context,
                "parameters": [
                    {
                        "name": p.name,
                        "type": p.type,
                        "description": p.description,
                        "required": p.required,
                        "default": p.default,
                        "enum": p.enum
                    }
                    for p in tool.get_parameters()
                ]
            }
            for tool in tools
        ]

    def get_tool_details(self, tool_name: str) -> dict:
        """Get detailed information about a specific tool"""
        logger.info(f"Getting tool details: {tool_name}")
        tool = tool_registry.get(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")

        return {
            "name": tool.name,
            "description": tool.description,
            "category": tool.category.value,
            "enabled_by_default": tool.enabled_by_default,
            "requires_context": tool.requires_context,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                    "enum": p.enum
                }
                for p in tool.get_parameters()
            ],
            "openai_function": tool.to_openai_function(),
            "anthropic_tool": tool.to_anthropic_tool()
        }

    def list_tool_categories(self) -> Dict[str, List[dict]]:
        """List all tool categories"""
        logger.info("Listing tool categories")
        categories = {}
        for tool in tool_registry.get_all():
            category = tool.category.value
            if category not in categories:
                categories[category] = []
            categories[category].append({
                "name": tool.name,
                "description": tool.description
            })

        logger.info(f"Found {len(categories)} tool categories")
        return categories

    async def list_tools_by_mode(
        self,
        mode: str,
        conversation_id: Optional[UUID] = None
    ) -> List[dict]:
        """List available tools based on execution mode"""
        logger.info(f"Listing tools for mode: {mode}")

        if mode not in ["manual", "agent"]:
            raise ValueError("Mode must be 'manual' or 'agent'")

        # Get all physical tools from registry
        physical_tools = tool_registry.get_all()
        logger.info(f"Found {len(physical_tools)} physical tools")

        # Get all custom tool instances
        custom_tool_instances = await self.custom_tool_repo.get_active_tools()
        # Filter out templates manually
        custom_tool_instances = [t for t in custom_tool_instances if not t.is_template]
        logger.info(f"Found {len(custom_tool_instances)} custom tool instances")

        # Get RAG instances specifically
        rag_custom_instances = await self.custom_tool_repo.get_rag_instances()

        available_tools = []

        if mode == "agent":
            # Agent mode: Show all tools (physical + custom instances)
            for tool in physical_tools:
                available_tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "category": tool.category.value,
                    "enabled_by_default": tool.enabled_by_default,
                    "requires_context": tool.requires_context,
                    "type": "physical",
                    "parameters": [
                        {
                            "name": p.name,
                            "type": p.type,
                            "description": p.description,
                            "required": p.required,
                            "default": p.default,
                            "enum": p.enum
                        }
                        for p in tool.get_parameters()
                    ]
                })

            for custom_tool in custom_tool_instances:
                available_tools.append({
                    "name": custom_tool.name,
                    "description": custom_tool.description,
                    "category": "custom",
                    "enabled_by_default": True,
                    "requires_context": [],
                    "type": "custom_instance",
                    "tool_type": custom_tool.tool_type.value if hasattr(custom_tool.tool_type,
                                                                        'value') else str(
                        custom_tool.tool_type),
                    "parameters": custom_tool.config_schema.get("properties",
                                                                {}) if custom_tool.config_schema else {}
                })

        else:  # manual mode
            # Group custom instances by type
            custom_instances_by_type = {}
            for tool in custom_tool_instances:
                tool_type = tool.tool_type
                if tool_type not in custom_instances_by_type:
                    custom_instances_by_type[tool_type] = []
                custom_instances_by_type[tool_type].append(tool)

            # For each physical tool, decide what to show
            for tool in physical_tools:
                if tool.name == "rag_search":
                    # Special case for rag_tool
                    if rag_custom_instances:
                        # Show custom instances
                        for custom_tool in rag_custom_instances:
                            available_tools.append({
                                "name": custom_tool.name,
                                "description": custom_tool.description,
                                "category": "custom",
                                "enabled_by_default": True,
                                "requires_context": [],
                                "type": "custom_instance",
                                "tool_type": custom_tool.tool_type.value if hasattr(
                                    custom_tool.tool_type, 'value') else str(custom_tool.tool_type),
                                "parameters": custom_tool.config_schema.get("properties",
                                                                            {}) if custom_tool.config_schema else {}
                            })
                    else:
                        # Show physical tool
                        available_tools.append({
                            "name": tool.name,
                            "description": tool.description,
                            "category": tool.category.value,
                            "enabled_by_default": tool.enabled_by_default,
                            "requires_context": tool.requires_context,
                            "type": "physical",
                            "parameters": [
                                {
                                    "name": p.name,
                                    "type": p.type,
                                    "description": p.description,
                                    "required": p.required,
                                    "default": p.default,
                                    "enum": p.enum
                                }
                                for p in tool.get_parameters()
                            ]
                        })
                else:
                    # For other tools, check if custom instances exist
                    # Try to match tool_type
                    tool_type_str = tool.name
                    matching_instances = []
                    for tool_type, instances in custom_instances_by_type.items():
                        type_value = tool_type.value if hasattr(tool_type, 'value') else str(
                            tool_type)
                        if type_value == tool_type_str:
                            matching_instances.extend(instances)

                    if matching_instances:
                        # Show custom instances
                        for custom_tool in matching_instances:
                            available_tools.append({
                                "name": custom_tool.name,
                                "description": custom_tool.description,
                                "category": "custom",
                                "enabled_by_default": True,
                                "requires_context": [],
                                "type": "custom_instance",
                                "tool_type": custom_tool.tool_type.value if hasattr(
                                    custom_tool.tool_type, 'value') else str(custom_tool.tool_type),
                                "parameters": custom_tool.config_schema.get("properties",
                                                                            {}) if custom_tool.config_schema else {}
                            })
                    else:
                        # Show physical tool
                        available_tools.append({
                            "name": tool.name,
                            "description": tool.description,
                            "category": tool.category.value,
                            "enabled_by_default": tool.enabled_by_default,
                            "requires_context": tool.requires_context,
                            "type": "physical",
                            "parameters": [
                                {
                                    "name": p.name,
                                    "type": p.type,
                                    "description": p.description,
                                    "required": p.required,
                                    "default": p.default,
                                    "enum": p.enum
                                }
                                for p in tool.get_parameters()
                            ]
                        })

        return available_tools

    async def execute_tool(self, tool_name: str, parameters: dict) -> dict:
        """Execute a tool with given parameters (for testing)"""
        logger.info(f"Executing tool: {tool_name}")
        tool = tool_registry.get(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")

        try:
            # Validate input
            await tool.validate_input(**parameters)

            # Execute
            result = await tool.execute(**parameters)

            if not result.success:
                raise ValueError(f"Tool execution failed: {result.error}")

            logger.info(f"Tool executed successfully: {tool_name}")
            return {
                "tool": tool_name,
                "success": True,
                "data": result.data,
                "metadata": result.metadata,
                "formatted_output": tool.format_output(result)
            }

        except ValueError as e:
            logger.warning(f"Invalid parameters for tool {tool_name}: {e}")
            raise ValueError(f"Invalid parameters: {str(e)}")
        except Exception as e:
            logger.error(f"Tool execution error: {e}", exc_info=True)
            raise ValueError(f"Tool execution failed: {str(e)}")

    # =============================================================================
    # Tool Configuration Operations
    # =============================================================================

    @transactional
    async def create_tool_configuration(
        self,
        data: ToolConfigurationCreate
    ) -> ToolConfiguration:
        """Create a tool configuration for a conversation"""
        logger.info(f"Creating tool configuration for conversation {data.conversation_id}")

        # Validate conversation exists
        conversation = await self.conversation_repo.get_by_id(data.conversation_id)
        if not conversation:
            raise ValueError("Conversation not found")

        # Validate tool exists
        tool = tool_registry.get(data.tool_name)
        if not tool:
            raise ValueError(f"Tool '{data.tool_name}' not found")

        # Check if configuration already exists
        existing = await self.tool_configuration_repo.get_by_conversation_and_tool(
            data.conversation_id,
            data.tool_name
        )
        if existing:
            raise ValueError(
                f"Configuration for tool '{data.tool_name}' already exists for this conversation")

        # Create configuration - repository handles flush/refresh internally
        config = await self.tool_configuration_repo.create(
            conversation_id=data.conversation_id,
            tool_name=data.tool_name,
            config=data.config,
            is_active=data.is_active
        )

        logger.info(f"Tool configuration created: {config.id}")
        return config

    async def list_conversation_tool_configurations(
        self,
        conversation_id: UUID,
        active_only: bool = True
    ) -> List[ToolConfiguration]:
        """List tool configurations for a conversation"""
        logger.info(f"Listing tool configurations for conversation {conversation_id}")
        configs = await self.tool_configuration_repo.get_by_conversation(
            conversation_id,
            active_only
        )
        logger.info(f"Found {len(configs)} tool configurations")
        return configs

    async def get_tool_configuration(self, config_id: UUID) -> Optional[ToolConfiguration]:
        """Get specific tool configuration"""
        return await self.tool_configuration_repo.get_by_id(config_id)

    @transactional
    async def update_tool_configuration(
        self,
        config_id: UUID,
        data: ToolConfigurationUpdate
    ) -> Optional[ToolConfiguration]:
        """Update tool configuration"""
        logger.info(f"Updating tool configuration: {config_id}")
        config = await self.tool_configuration_repo.get_by_id(config_id)
        if not config:
            return None

        # Update fields
        if data.config is not None:
            config.config = data.config
        if data.is_active is not None:
            config.is_active = data.is_active

        # Repository handles flush/refresh internally
        config = await self.tool_configuration_repo.save(config)
        logger.info(f"Tool configuration updated: {config_id}")
        return config

    @transactional
    async def delete_tool_configuration(self, config_id: UUID) -> bool:
        """Delete tool configuration"""
        logger.info(f"Deleting tool configuration: {config_id}")
        config = await self.tool_configuration_repo.get_by_id(config_id)
        if not config:
            return False

        await self.tool_configuration_repo.delete(config_id)
        logger.info(f"Tool configuration deleted: {config_id}")
        return True

    @transactional
    async def bulk_create_tool_configurations(
        self,
        conversation_id: UUID,
        tool_names: List[str],
        default_configs: Optional[dict] = None
    ) -> List[ToolConfiguration]:
        """Create multiple tool configurations at once"""
        logger.info(f"Creating bulk tool configurations for conversation {conversation_id}")

        # Validate conversation
        conversation = await self.conversation_repo.get_by_id(conversation_id)
        if not conversation:
            raise ValueError("Conversation not found")

        configs = []
        for tool_name in tool_names:
            # Check if tool exists
            tool = tool_registry.get(tool_name)
            if not tool:
                logger.warning(f"Tool not found in bulk creation: {tool_name}")
                continue

            # Check if already exists
            existing = await self.tool_configuration_repo.get_by_conversation_and_tool(
                conversation_id,
                tool_name
            )
            if existing:
                logger.info(f"Configuration exists for {tool_name}, using existing")
                configs.append(existing)
                continue

            # Create new - repository handles flush/refresh internally
            config = await self.tool_configuration_repo.create(
                conversation_id=conversation_id,
                tool_name=tool_name,
                config=default_configs or {},
                is_active=True
            )
            configs.append(config)

        logger.info(f"Created {len(configs)} tool configurations")
        return configs

    # =============================================================================
    # Custom Tools Operations
    # =============================================================================

    async def list_tool_types(self) -> Dict[str, Dict]:
        """List all available tool types with their configuration templates"""
        logger.info("Listing tool types")

        # Para obtener templates necesitamos get_active_tools y filtrar manualmente
        all_tools = await self.custom_tool_repo.get_active_tools()
        templates = [t for t in all_tools if t.is_template]

        tool_types = {}
        for tool in templates:
            tool_type = tool.tool_type
            tool_type_key = tool_type.value if hasattr(tool_type, 'value') else str(tool_type)

            if tool_type_key not in tool_types:
                tool_types[tool_type_key] = {
                    "name": tool.name,
                    "description": tool.description or f"{tool_type_key.capitalize()} tool type",
                    "config_schema": tool.config_schema or {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False
                    },
                    "example": tool.example or {}
                }
            else:
                # Merge schemas if multiple templates
                if tool.config_schema:
                    for prop_name, prop_def in tool.config_schema.get("properties", {}).items():
                        if prop_name not in tool_types[tool_type_key]["config_schema"][
                            "properties"]:
                            tool_types[tool_type_key]["config_schema"]["properties"][
                                prop_name] = prop_def

        logger.info(f"Found {len(tool_types)} tool types")
        return tool_types

    @transactional
    async def create_custom_tool(
        self,
        data: CustomToolCreate,
        conversation_id: Optional[UUID] = None
    ) -> CustomTool:
        """Create a custom tool"""
        logger.info(f"Creating custom tool: {data.name}")

        # Check if name already exists within conversation (if provided)
        if conversation_id:
            existing = await self.custom_tool_repo.get_by_name(conversation_id, data.name)
            if existing:
                raise ValueError(
                    f"Custom tool with name '{data.name}' already exists in this conversation")

        # Create custom tool - repository handles flush/refresh internally
        custom_tool = await self.custom_tool_repo.create(
            name=data.name,
            description=data.description,
            tool_type=data.tool_type,
            configuration=data.configuration,
            visibility=data.visibility,
            conversation_id=conversation_id,
            is_active=data.is_active
        )

        # Register in tool registry - pass repositories, not session
        from src.tools.custom_tool import CustomToolExecutor
        custom_tool_executor = CustomToolExecutor(
            custom_tool_id=custom_tool.id,
            file_repo=self.file_repo,
            custom_tool_repo=self.custom_tool_repo
        )
        custom_tool_executor._name = custom_tool.name
        tool_registry.register(custom_tool_executor)

        logger.info(f"Custom tool created and registered: {custom_tool.name}")
        return custom_tool

    async def list_custom_tools(
        self,
        conversation_id: Optional[UUID] = None,
        active_only: bool = True,
        is_template: bool = False
    ) -> List[CustomTool]:
        """List all custom tools"""
        logger.info("Listing custom tools")

        if conversation_id:
            # Get tools for specific conversation
            tools = await self.custom_tool_repo.get_conversation_tools(
                conversation_id=conversation_id,
                active_only=active_only,
                is_template=is_template
            )
        else:
            # Get all active tools (filtered by template status if requested)
            tools = await self.custom_tool_repo.get_active_tools(is_template=is_template)
            if active_only:
                tools = [t for t in tools if t.is_active]

        logger.info(f"Found {len(tools)} custom tools")
        return tools

    async def get_custom_tool(self, tool_id: UUID) -> Optional[CustomTool]:
        """Get custom tool by ID"""
        return await self.custom_tool_repo.get_by_id(tool_id)

    @transactional
    async def update_custom_tool(
        self,
        tool_id: UUID,
        data: CustomToolUpdate
    ) -> Optional[CustomTool]:
        """Update custom tool"""
        logger.info(f"Updating custom tool: {tool_id}")
        tool = await self.custom_tool_repo.get_by_id(tool_id)
        if not tool:
            return None

        # Update fields
        if data.name is not None:
            tool.name = data.name
        if data.description is not None:
            tool.description = data.description
        if data.tool_type is not None:
            tool.tool_type = data.tool_type
        if data.configuration is not None:
            tool.configuration = data.configuration
        if data.visibility is not None:
            tool.visibility = data.visibility
        if data.is_active is not None:
            tool.is_active = data.is_active

        # Repository handles flush/refresh internally
        tool = await self.custom_tool_repo.save(tool)
        logger.info(f"Custom tool updated: {tool_id}")
        return tool

    @transactional
    async def delete_custom_tool(self, tool_id: UUID) -> bool:
        """Delete custom tool"""
        logger.info(f"Deleting custom tool: {tool_id}")
        tool = await self.custom_tool_repo.get_by_id(tool_id)
        if not tool:
            return False

        # Unregister from tool registry
        tool_registry.unregister(tool.name)

        await self.custom_tool_repo.delete(tool_id)
        logger.info(f"Custom tool deleted and unregistered: {tool_id}")
        return True

    # En tool_service.py, agregar este método público:

    async def load_tool_types_for_initialization(self) -> Dict[str, Dict]:
        """
        Load tool types from database for application initialization.
        Used by main.py startup routines.

        Returns:
            Dictionary of tool types with configuration templates
        """
        logger.info("Loading tool types from database for initialization")

        # Get all template tools
        all_tools = await self.custom_tool_repo.get_active_tools()
        templates = [t for t in all_tools if t.is_template]

        # Build tool types dictionary
        tool_types = {}
        for tool in templates:
            tool_type = tool.tool_type
            tool_type_key = tool_type.value if hasattr(tool_type, 'value') else str(tool_type)

            if tool_type_key not in tool_types:
                # Initialize tool type with data from database
                tool_types[tool_type_key] = {
                    "name": tool.name,
                    "description": tool.description or f"{tool_type_key.capitalize()} tool type",
                    "config_schema": tool.config_schema or {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False
                    },
                    "example": tool.example or {}
                }
            else:
                # If tool type already exists, merge the schemas
                if tool.config_schema:
                    for prop_name, prop_def in tool.config_schema.get("properties", {}).items():
                        if prop_name not in tool_types[tool_type_key]["config_schema"]["properties"]:
                            tool_types[tool_type_key]["config_schema"]["properties"][prop_name] = prop_def

        logger.info(f"Loaded {len(tool_types)} tool types from database")
        return tool_types

    # =========================================================================
    # Tool Template Synchronization (for startup)
    # =========================================================================

    async def sync_tool_templates(self, tools_to_sync: List) -> tuple:
        """
        Synchronize tool templates from Python code to database.
        This ensures that database templates match the current code implementation.

        Args:
            tools_to_sync: List of tool instances to synchronize

        Returns:
            Tuple of (created_count, updated_count)
        """
        logger.info(f"Synchronizing {len(tools_to_sync)} tool templates")

        updated_count = 0
        created_count = 0

        for tool_instance in tools_to_sync:
            tool_type = tool_instance.name.lower()

            # Build template data from tool instance
            properties = {}
            example_values = {}

            for param in tool_instance.get_parameters():
                properties[param.name] = {
                    "type": param.type,
                    "description": param.description,
                    "required": param.required,
                    "default": param.default,
                    "enum": param.enum,
                    "example": param.example
                }

                # Generate example values
                if param.example is not None:
                    example_values[param.name] = param.example
                elif param.default is not None:
                    example_values[param.name] = param.default
                elif param.type == "string":
                    example_values[param.name] = f"example_{param.name}"
                elif param.type == "integer":
                    example_values[param.name] = 100
                elif param.type == "boolean":
                    example_values[param.name] = True
                elif param.type == "object":
                    example_values[param.name] = {}
                elif param.type == "array":
                    example_values[param.name] = []

            template_data = {
                "name": tool_instance.name,
                "description": tool_instance.description,
                "tool_type": tool_type,
                "configuration": {},
                "config_schema": {
                    "type": "object",
                    "properties": properties
                },
                "example": example_values
            }

            # Check if template exists
            existing = await self.custom_tool_repo.get_template_by_tool_type(tool_type)

            if existing:
                # Update existing template
                existing.name = template_data["name"]
                existing.description = template_data["description"]
                existing.configuration = template_data["configuration"]
                existing.config_schema = template_data["config_schema"]
                existing.example = template_data["example"]
                await self.custom_tool_repo.save(existing)
                updated_count += 1
            else:
                # Create new template using repository
                await self.custom_tool_repo.create(
                    name=template_data["name"],
                    description=template_data["description"],
                    tool_type=tool_type,
                    is_template=True,
                    is_active=True,
                    configuration=template_data["configuration"],
                    config_schema=template_data["config_schema"],
                    example=template_data["example"]
                )
                created_count += 1

        # Commit all changes at once
        await self.custom_tool_repo.commit()
        logger.info(f"Tool templates synchronized: {created_count} created, {updated_count} updated")
        return created_count, updated_count

    async def get_custom_tools_for_startup(self) -> List[CustomTool]:
        """
        Get all custom tool instances (not templates) for application startup.
        Used by main.py to register custom tools at startup.

        Returns:
            List of custom tool instances
        """
        logger.info("Loading custom tool instances for startup")
        custom_tools = await self.custom_tool_repo.get_active_tools(is_template=False)
        # Double check filter
        actual_custom_tools = [t for t in custom_tools if not t.is_template]
        logger.info(f"Found {len(actual_custom_tools)} custom tool instances")
        return actual_custom_tools
