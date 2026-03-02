# =============================================================================
# api/v1/tools.py
# Tools API endpoints
# =============================================================================
"""
API endpoints para gestión de herramientas
"""
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, status
from src.dependencies import get_tool_service
from src.schemas.schemas import (
    ToolConfigurationCreate, ToolConfigurationUpdate, ToolConfigurationResponse,
    CustomToolCreate, CustomToolUpdate, CustomToolResponse
)
from src.services.tool_service import ToolService
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


# =============================================================================
# Tool Registry Endpoints
# =============================================================================

@router.get("/available")
async def list_available_tools(
    service: ToolService = Depends(get_tool_service)
):
    """
    List all available tools in the registry
    Returns physical tools with their parameters and metadata
    """
    return service.list_available_tools()


@router.get("/available/{tool_name}")
async def get_tool_details(
    tool_name: str,
    service: ToolService = Depends(get_tool_service)
):
    """Get detailed information about a specific tool"""
    try:
        return service.get_tool_details(tool_name)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@router.get("/categories")
async def list_tool_categories(
    service: ToolService = Depends(get_tool_service)
):
    """List all tool categories with their tools"""
    return service.list_tool_categories()


@router.get("/by-mode/{mode}")
async def list_tools_by_mode(
    mode: str,
    conversation_id: Optional[UUID] = Query(None, description="Filter by conversation context"),
    service: ToolService = Depends(get_tool_service)
):
    """
    List available tools based on execution mode
    
    Modes:
    - manual: Returns tools for manual selection (shows custom instances when available)
    - agent: Returns all tools for agent autonomous selection
    """
    try:
        return await service.list_tools_by_mode(mode, conversation_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/execute")
async def execute_tool(
    tool_name: str = Query(..., description="Name of the tool to execute"),
    parameters: dict = {},
    service: ToolService = Depends(get_tool_service)
):
    """
    Execute a tool with given parameters (for testing/debugging)
    Use with caution - this directly executes tools
    """
    try:
        return await service.execute_tool(tool_name, parameters)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Tool execution failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# =============================================================================
# Tool Configuration Endpoints
# =============================================================================

@router.post("/configurations", response_model=ToolConfigurationResponse, status_code=status.HTTP_201_CREATED)
async def create_tool_configuration(
    data: ToolConfigurationCreate,
    service: ToolService = Depends(get_tool_service)
):
    """Create a tool configuration for a conversation"""
    try:
        config = await service.create_tool_configuration(data)
        return config
    except ValueError as e:
        status_code = status.HTTP_404_NOT_FOUND if "not found" in str(e).lower() else status.HTTP_400_BAD_REQUEST
        raise HTTPException(status_code=status_code, detail=str(e))


@router.get("/configurations/conversation/{conversation_id}", response_model=List[ToolConfigurationResponse])
async def list_conversation_tool_configurations(
    conversation_id: UUID,
    active_only: bool = Query(True, description="Only return active configurations"),
    service: ToolService = Depends(get_tool_service)
):
    """List all tool configurations for a conversation"""
    configs = await service.list_conversation_tool_configurations(conversation_id, active_only)
    return configs


@router.get("/configurations/{config_id}", response_model=ToolConfigurationResponse)
async def get_tool_configuration(
    config_id: UUID,
    service: ToolService = Depends(get_tool_service)
):
    """Get a specific tool configuration"""
    config = await service.get_tool_configuration(config_id)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tool configuration not found"
        )
    return config


@router.patch("/configurations/{config_id}", response_model=ToolConfigurationResponse)
async def update_tool_configuration(
    config_id: UUID,
    data: ToolConfigurationUpdate,
    service: ToolService = Depends(get_tool_service)
):
    """Update a tool configuration"""
    config = await service.update_tool_configuration(config_id, data)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tool configuration not found"
        )
    return config


@router.delete("/configurations/{config_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_tool_configuration(
    config_id: UUID,
    service: ToolService = Depends(get_tool_service)
):
    """Delete a tool configuration"""
    deleted = await service.delete_tool_configuration(config_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tool configuration not found"
        )
    return None


@router.post("/configurations/bulk", response_model=List[ToolConfigurationResponse])
async def bulk_create_tool_configurations(
    conversation_id: UUID = Query(..., description="Conversation ID"),
    tool_names: List[str] = Query(..., description="List of tool names"),
    default_configs: Optional[dict] = None,
    service: ToolService = Depends(get_tool_service)
):
    """Create multiple tool configurations at once"""
    try:
        configs = await service.bulk_create_tool_configurations(
            conversation_id,
            tool_names,
            default_configs
        )
        return configs
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


# =============================================================================
# Custom Tools Endpoints
# =============================================================================

@router.get("/types")
async def list_tool_types(
    service: ToolService = Depends(get_tool_service)
):
    """
    List all available tool types with their configuration templates
    Used for creating custom tool instances
    """
    return await service.list_tool_types()


@router.post("/custom", response_model=CustomToolResponse, status_code=status.HTTP_201_CREATED)
async def create_custom_tool(
    data: CustomToolCreate,
    conversation_id: Optional[UUID] = Query(None, description="Associate with conversation (for private tools)"),
    service: ToolService = Depends(get_tool_service)
):
    """
    Create a custom tool instance
    Custom tools extend physical tools with specific configurations
    """
    try:
        custom_tool = await service.create_custom_tool(data, conversation_id)
        return custom_tool
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/custom", response_model=List[CustomToolResponse])
async def list_custom_tools(
    conversation_id: Optional[UUID] = Query(None, description="Filter by conversation"),
    active_only: bool = Query(True, description="Only return active tools"),
    service: ToolService = Depends(get_tool_service)
):
    """
    List all custom tools
    Can be filtered by conversation for private tools
    """
    tools = await service.list_custom_tools(conversation_id, active_only,False)
    return tools


@router.get("/custom/{tool_id}", response_model=CustomToolResponse)
async def get_custom_tool(
    tool_id: UUID,
    service: ToolService = Depends(get_tool_service)
):
    """Get a specific custom tool"""
    tool = await service.get_custom_tool(tool_id)
    if not tool:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Custom tool not found"
        )
    return tool


@router.patch("/custom/{tool_id}", response_model=CustomToolResponse)
async def update_custom_tool(
    tool_id: UUID,
    data: CustomToolUpdate,
    service: ToolService = Depends(get_tool_service)
):
    """Update a custom tool"""
    tool = await service.update_custom_tool(tool_id, data)
    if not tool:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Custom tool not found"
        )
    return tool


@router.delete("/custom/{tool_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_custom_tool(
    tool_id: UUID,
    service: ToolService = Depends(get_tool_service)
):
    """Delete a custom tool"""
    deleted = await service.delete_custom_tool(tool_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Custom tool not found"
        )
    return None
