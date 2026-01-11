# =============================================================================
# api/v1/tools.py
# Tools Management API endpoints
# =============================================================================
"""
API endpoints para gestión de herramientas (tools)
"""
from typing import List, Optional, Dict
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.connection import get_async_db
from src.models.models import ToolConfiguration, Conversation, CustomTool, ToolType
from src.schemas.schemas import (
    ToolConfigurationCreate,
    ToolConfigurationUpdate,
    ToolConfigurationResponse,
    CustomToolCreate,
    CustomToolUpdate,
    CustomToolResponse
)
from src.tools.base_tool import tool_registry
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


# =============================================================================
# Tool Registry Endpoints
# =============================================================================

@router.get("/available")
async def list_available_tools():
    """List all available tools in the registry"""
    logger.info("Iniciando listado de herramientas disponibles")
    logger.debug("Listing available tools")
    tools = tool_registry.get_all()
    logger.info(f"Se encontraron {len(tools)} herramientas disponibles")

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


@router.get("/available/{tool_name}")
async def get_tool_details(tool_name: str):
    """Get detailed information about a specific tool"""
    logger.info(f"Obteniendo detalles de la herramienta: {tool_name}")
    tool = tool_registry.get(tool_name)

    if not tool:
        logger.warning(f"Herramienta no encontrada: {tool_name}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{tool_name}' not found"
        )

    logger.info(f"Detalles de la herramienta {tool_name} obtenidos correctamente")
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


@router.get("/categories")
async def list_tool_categories():
    """List all tool categories"""
    logger.info("Iniciando listado de categorías de herramientas")
    categories = {}

    for tool in tool_registry.get_all():
        category = tool.category.value
        if category not in categories:
            categories[category] = []
        categories[category].append({
            "name": tool.name,
            "description": tool.description
        })

    logger.info(f"Se encontraron {len(categories)} categorías de herramientas")
    return categories


@router.get("/available/mode/{mode}")
async def list_tools_by_mode(
    mode: str,
    db: AsyncSession = Depends(get_async_db)
):
    """
    List available tools based on execution mode.
     
    In manual mode: Show only custom tool instances (configured tools),
    except rag_tool which is physical and used directly if no custom instance exists.
     
    In agent mode: Show all tools (physical + custom instances).
     
    Args:
        mode: Execution mode ('manual' or 'agent')
        db: Database session
    """
    logger.info(f"Listando herramientas para el modo: {mode}")
    logger.debug(f"Listing tools for mode: {mode}")
     
    if mode not in ["manual", "agent"]:
        logger.warning(f"Modo no válido: {mode}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Mode must be 'manual' or 'agent'"
        )
     
    # Get all physical tools from registry
    physical_tools = tool_registry.get_all()
    logger.info(f"Se encontraron {len(physical_tools)} herramientas físicas")
     
    # Get all custom tool instances from database
    result = await db.execute(
        select(CustomTool).filter(
            CustomTool.is_active == True,
            CustomTool.is_template == False  # Only instances, not templates
        )
    )
    custom_tool_instances = result.scalars().all()
    logger.info(f"Se encontraron {len(custom_tool_instances)} instancias de herramientas personalizadas")
     
    # Get custom tool templates for reference
    result = await db.execute(
        select(CustomTool).filter(
            CustomTool.is_active == True,
            CustomTool.is_template == True
        )
    )
    custom_tool_templates = result.scalars().all()
    logger.info(f"Se encontraron {len(custom_tool_templates)} plantillas de herramientas personalizadas")
     
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
                "tool_type": custom_tool.tool_type.value,
                "parameters": custom_tool.config_schema.get("properties", {}) if custom_tool.config_schema else {}
            })
            
    else:  # manual mode
        # Manual mode: Show only custom tool instances, with special handling for rag_tool
        
        # Check if there are any custom instances of rag_tool
        rag_custom_instances = [
            tool for tool in custom_tool_instances 
            if tool.tool_type == ToolType.rag_search
        ]
        
        # Group custom tool instances by type
        custom_instances_by_type = {}
        for tool in custom_tool_instances:
            if tool.tool_type not in custom_instances_by_type:
                custom_instances_by_type[tool.tool_type] = []
            custom_instances_by_type[tool.tool_type].append(tool)
        
        # For each physical tool, decide whether to show it or its custom instances
        for tool in physical_tools:
            if tool.name == "rag_search":
                # Special case for rag_tool
                if rag_custom_instances:
                    # Show custom instances instead of physical tool
                    for custom_tool in rag_custom_instances:
                        available_tools.append({
                            "name": custom_tool.name,
                            "description": custom_tool.description,
                            "category": "custom",
                            "enabled_by_default": True,
                            "requires_context": [],
                            "type": "custom_instance",
                            "tool_type": custom_tool.tool_type.value,
                            "parameters": custom_tool.config_schema.get("properties", {}) if custom_tool.config_schema else {}
                        })
                else:
                    # No custom instances, show physical tool
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
                # For other tools, show custom instances if they exist, otherwise show physical tool
                tool_type = ToolType(tool.name) if tool.name in [t.value for t in ToolType] else None
                
                if tool_type and tool_type in custom_instances_by_type:
                    # Show custom instances instead of physical tool
                    for custom_tool in custom_instances_by_type[tool_type]:
                        available_tools.append({
                            "name": custom_tool.name,
                            "description": custom_tool.description,
                            "category": "custom",
                            "enabled_by_default": True,
                            "requires_context": [],
                            "type": "custom_instance",
                            "tool_type": custom_tool.tool_type.value,
                            "parameters": custom_tool.config_schema.get("properties", {}) if custom_tool.config_schema else {}
                        })
                else:
                    # No custom instances, show physical tool
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


# =============================================================================
# Tool Configuration Endpoints (per Conversation)
# =============================================================================

@router.post("/configurations", response_model=ToolConfigurationResponse,
              status_code=status.HTTP_201_CREATED)
async def create_tool_configuration(
    data: ToolConfigurationCreate,
    db: AsyncSession = Depends(get_async_db)
):
    """Create a tool configuration for a conversation"""
    logger.info(f"Creando configuración de herramienta para la conversación {data.conversation_id}")
    # Validate conversation exists
    result = await db.execute(
        select(Conversation).filter(Conversation.id == data.conversation_id)
    )
    conversation = result.scalars().first()

    if not conversation:
        logger.warning(f"Conversación no encontrada: {data.conversation_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    # Validate tool exists
    tool = tool_registry.get(data.tool_name)
    if not tool:
        logger.warning(f"Herramienta no encontrada: {data.tool_name}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{data.tool_name}' not found"
        )

    # Check if configuration already exists
    result = await db.execute(
        select(ToolConfiguration).filter(
            ToolConfiguration.conversation_id == data.conversation_id,
            ToolConfiguration.tool_name == data.tool_name
        )
    )
    existing = result.scalars().first()

    if existing:
        logger.warning(f"Configuración ya existe para la herramienta {data.tool_name} en la conversación {data.conversation_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Configuration for tool '{data.tool_name}' already exists for this conversation"
        )

    # Create configuration
    config = ToolConfiguration(
        conversation_id=data.conversation_id,
        tool_name=data.tool_name,
        config=data.config,
        is_active=data.is_active
    )

    db.add(config)
    await db.commit()
    await db.refresh(config)
    logger.info(f"Configuración de herramienta creada correctamente: {config.id}")

    return config


@router.get("/configurations/conversation/{conversation_id}",
            response_model=List[ToolConfigurationResponse])
async def list_conversation_tool_configurations(
    conversation_id: UUID,
    active_only: bool = True,
    db: AsyncSession = Depends(get_async_db)
):
    """List tool configurations for a conversation"""
    logger.info(f"Listando configuraciones de herramientas para la conversación {conversation_id}")
    query = select(ToolConfiguration).filter(
        ToolConfiguration.conversation_id == conversation_id
    )

    if active_only:
        query = query.filter(ToolConfiguration.is_active == True)

    result = await db.execute(query)
    configs = result.scalars().all()
    logger.info(f"Se encontraron {len(configs)} configuraciones de herramientas para la conversación {conversation_id}")

    return configs


@router.get("/configurations/{config_id}", response_model=ToolConfigurationResponse)
async def get_tool_configuration(
    config_id: UUID,
    db: AsyncSession = Depends(get_async_db)
):
    """Get a specific tool configuration"""
    logger.info(f"Obteniendo configuración de herramienta con ID: {config_id}")
    result = await db.execute(
        select(ToolConfiguration).filter(ToolConfiguration.id == config_id)
    )
    config = result.scalars().first()

    if not config:
        logger.warning(f"Configuración de herramienta no encontrada: {config_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tool configuration not found"
        )

    logger.info(f"Configuración de herramienta {config_id} obtenida correctamente")
    return config


@router.patch("/configurations/{config_id}", response_model=ToolConfigurationResponse)
async def update_tool_configuration(
    config_id: UUID,
    data: ToolConfigurationUpdate,
    db: AsyncSession = Depends(get_async_db)
):
    """Update a tool configuration"""
    logger.info(f"Actualizando configuración de herramienta con ID: {config_id}")
    logger.debug(f"Datos recibidos para actualización: {data}")
    result = await db.execute(
        select(ToolConfiguration).filter(ToolConfiguration.id == config_id)
    )
    config = result.scalars().first()

    if not config:
        logger.error(f"Configuración de herramienta no encontrada para actualizar: {config_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tool configuration not found"
        )

    # Update fields
    if data.config is not None:
        logger.info(f"Actualizando configuración para la herramienta {config.tool_name}")
        config.config = data.config

    if data.is_active is not None:
        logger.info(f"Actualizando estado activo a {data.is_active} para la herramienta {config.tool_name}")
        config.is_active = data.is_active

    await db.commit()
    await db.refresh(config)
    logger.info(f"Configuración de herramienta {config_id} actualizada correctamente")

    # Devolver las configuraciones actualizadas
    return config


@router.delete("/configurations/{config_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_tool_configuration(
    config_id: UUID,
    db: AsyncSession = Depends(get_async_db)
):
    """Delete a tool configuration"""
    logger.info(f"Eliminando configuración de herramienta con ID: {config_id}")
    result = await db.execute(
        select(ToolConfiguration).filter(ToolConfiguration.id == config_id)
    )
    config = result.scalars().first()

    if not config:
        logger.warning(f"Configuración de herramienta no encontrada para eliminar: {config_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tool configuration not found"
        )

    await db.delete(config)
    await db.commit()
    logger.info(f"Configuración de herramienta {config_id} eliminada correctamente")

    return None


# =============================================================================
# Tool Execution (Testing)
# =============================================================================

@router.post("/execute/{tool_name}")
async def execute_tool(
    tool_name: str,
    parameters: dict
):
    """
    Execute a tool with given parameters (for testing)
    Note: In production, tools are executed through the chat orchestrator
    """
    logger.info(f"Ejecutando herramienta: {tool_name}")
    tool = tool_registry.get(tool_name)

    if not tool:
        logger.error(f"Herramienta no encontrada para ejecución: {tool_name}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{tool_name}' not found"
        )

    try:
        logger.info(f"Executing tool: {tool_name}", extra={"parameters": parameters})
        # Validate input
        await tool.validate_input(**parameters)

        # Execute
        result = await tool.execute(**parameters)

        if not result.success:
            logger.warning(f"Tool execution failed: {result.error}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Tool execution failed: {result.error}"
            )

        logger.info(f"Tool executed successfully: {tool_name}")
        logger.info(f"Herramienta {tool_name} ejecutada correctamente")
        return {
            "tool": tool_name,
            "success": True,
            "data": result.data,
            "metadata": result.metadata,
            "formatted_output": tool.format_output(result)
        }

    except ValueError as e:
        logger.warning(f"Invalid parameters for tool {tool_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid parameters: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Tool execution error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tool execution failed: {str(e)}"
        )


# =============================================================================
# Bulk Operations
# =============================================================================

@router.post("/configurations/conversation/{conversation_id}/bulk",
              response_model=List[ToolConfigurationResponse])
async def bulk_create_tool_configurations(
    conversation_id: UUID,
    tool_names: List[str],
    default_configs: Optional[dict] = None,
    db: AsyncSession = Depends(get_async_db)
):
    """Create multiple tool configurations at once"""
    logger.info(f"Creando configuraciones masivas de herramientas para la conversación {conversation_id}")
    # Validate conversation
    result = await db.execute(
        select(Conversation).filter(Conversation.id == conversation_id)
    )
    conversation = result.scalars().first()

    if not conversation:
        logger.error(f"Conversación no encontrada para configuraciones masivas: {conversation_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    configs = []

    for tool_name in tool_names:
        # Check if tool exists
        tool = tool_registry.get(tool_name)
        if not tool:
            logger.warning(f"Herramienta no encontrada en configuración masiva: {tool_name}")
            continue

        # Check if already exists
        result = await db.execute(
            select(ToolConfiguration).filter(
                ToolConfiguration.conversation_id == conversation_id,
                ToolConfiguration.tool_name == tool_name
            )
        )
        existing = result.scalars().first()

        if existing:
            logger.info(f"Configuración existente encontrada para {tool_name}, usando la existente")
            configs.append(existing)
            continue

        # Create new
        config = ToolConfiguration(
            conversation_id=conversation_id,
            tool_name=tool_name,
            config=default_configs or {},
            is_active=True
        )

        db.add(config)
        configs.append(config)

    await db.commit()
    logger.info(f"Se crearon {len(configs)} configuraciones de herramientas para la conversación {conversation_id}")

    for config in configs:
        await db.refresh(config)

    return configs


# =============================================================================
# Tool Type Templates
# =============================================================================

async def load_tool_types_from_db(db: AsyncSession) -> Dict[str, Dict]:
    """
    Load tool types from the custom_tools table in the database.
    Returns a dictionary of tool types with their configuration templates.
    """
    from sqlalchemy import select
    from src.models.models import CustomTool
    
    # Get all active custom tools from the database
    # Only include template tools (is_template=True)
    result = await db.execute(
        select(CustomTool).filter(
            CustomTool.is_active == True,
            CustomTool.is_template == True
        )
    )
    custom_tools = result.scalars().all()
    
    # Build tool types dictionary from database data
    tool_types = {}
    
    for tool in custom_tools:
        tool_type = tool.tool_type
        
        if tool_type not in tool_types:
            # Initialize tool type with data from database
            tool_types[tool_type] = {
                "name": tool.name,
                "description": tool.description or f"{tool_type.capitalize()} tool type",
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
                # Merge properties from the new tool's schema
                for prop_name, prop_def in tool.config_schema.get("properties", {}).items():
                    if prop_name not in tool_types[tool_type]["config_schema"]["properties"]:
                        tool_types[tool_type]["config_schema"]["properties"][prop_name] = prop_def
    
    return tool_types


@router.get("/types")
async def list_tool_types(db: AsyncSession = Depends(get_async_db)):
    """List all available tool types with their configuration templates"""
    logger.info("Listando tipos de herramientas disponibles")
    tool_types = await load_tool_types_from_db(db)
    logger.info(f"Se encontraron {len(tool_types)} tipos de herramientas")
    return tool_types


# =============================================================================
# Custom Tools Management
# =============================================================================

@router.post("/custom", response_model=CustomToolResponse,
              status_code=status.HTTP_201_CREATED)
async def create_custom_tool(
    data: CustomToolCreate,
    db: AsyncSession = Depends(get_async_db)
):
    """Create a custom tool"""
    logger.info(f"Creando herramienta personalizada: {data.name}")
    # Check if tool name already exists
    result = await db.execute(
        select(CustomTool).filter(CustomTool.name == data.name)
    )
    existing = result.scalars().first()

    if existing:
        logger.warning(f"Nombre de herramienta personalizada ya existe: {data.name}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Custom tool with name '{data.name}' already exists"
        )

    # Create custom tool
    custom_tool = CustomTool(
        name=data.name,
        description=data.description,
        tool_type=data.tool_type,
        configuration=data.configuration,
        visibility=data.visibility,
        is_active=data.is_active
    )

    db.add(custom_tool)
    await db.commit()
    await db.refresh(custom_tool)

    # Register the custom tool in the tool registry for testing
    from src.tools.custom_tool import CustomToolExecutor
    custom_tool_executor = CustomToolExecutor(custom_tool.id, db)
     
    # Override the name to use the tool's actual name instead of ID-based name
    custom_tool_executor._name = custom_tool.name
     
    tool_registry.register(custom_tool_executor)

    logger.info(f"Custom tool registered in registry: {custom_tool.name} (ID: {custom_tool.id})")
    logger.info(f"Tool name in registry: {custom_tool.name}")
    logger.info(f"Herramienta personalizada {data.name} creada correctamente")
     
    return custom_tool


@router.get("/custom", response_model=List[CustomToolResponse])
async def list_custom_tools(
    active_only: bool = True,
    include_templates: bool = False,
    db: AsyncSession = Depends(get_async_db)
):
    """List all custom tools (excludes template tools by default)"""
    logger.info("Listando herramientas personalizadas")
    query = select(CustomTool)
    
    # By default, exclude template tools (only show custom instances)
    if not include_templates:
        query = query.filter(CustomTool.is_template == False)
    
    if active_only:
        query = query.filter(CustomTool.is_active == True)
    
    result = await db.execute(query)
    tools = result.scalars().all()
    logger.info(f"Se encontraron {len(tools)} herramientas personalizadas")
    
    return tools


@router.get("/custom/{tool_id}", response_model=CustomToolResponse)
async def get_custom_tool(
    tool_id: UUID,
    db: AsyncSession = Depends(get_async_db)
):
    """Get a specific custom tool"""
    logger.info(f"Obteniendo herramienta personalizada con ID: {tool_id}")
    result = await db.execute(
        select(CustomTool).filter(CustomTool.id == tool_id)
    )
    tool = result.scalars().first()

    if not tool:
        logger.warning(f"Herramienta personalizada no encontrada: {tool_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Custom tool not found"
        )

    logger.info(f"Herramienta personalizada {tool_id} obtenida correctamente")
    return tool


@router.patch("/custom/{tool_id}", response_model=CustomToolResponse)
async def update_custom_tool(
    tool_id: UUID,
    data: CustomToolUpdate,
    db: AsyncSession = Depends(get_async_db)
):
    """Update a custom tool"""
    logger.info(f"Actualizando herramienta personalizada con ID: {tool_id}")
    result = await db.execute(
        select(CustomTool).filter(CustomTool.id == tool_id)
    )
    tool = result.scalars().first()

    if not tool:
        logger.error(f"Herramienta personalizada no encontrada para actualizar: {tool_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Custom tool not found"
        )

    # Update fields
    if data.name is not None:
        logger.info(f"Actualizando nombre de herramienta personalizada a {data.name}")
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

    await db.commit()
    await db.refresh(tool)
    logger.info(f"Herramienta personalizada {tool_id} actualizada correctamente")

    return tool


@router.delete("/custom/{tool_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_custom_tool(
    tool_id: UUID,
    db: AsyncSession = Depends(get_async_db)
):
    """Delete a custom tool"""
    logger.info(f"Eliminando herramienta personalizada con ID: {tool_id}")
    result = await db.execute(
        select(CustomTool).filter(CustomTool.id == tool_id)
    )
    tool = result.scalars().first()

    if not tool:
        logger.warning(f"Herramienta personalizada no encontrada para eliminar: {tool_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Custom tool not found"
        )

    # Unregister from tool registry using the actual tool name
    tool_registry.unregister(tool.name)

    await db.delete(tool)
    await db.commit()

    logger.info(f"Custom tool unregistered from registry: {tool.name} (ID: {tool.id})")
    logger.info(f"Herramienta personalizada {tool_id} eliminada correctamente")

    return None
