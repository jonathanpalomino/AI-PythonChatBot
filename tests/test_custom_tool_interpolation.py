
import pytest
from uuid import uuid4
from unittest.mock import MagicMock, AsyncMock
from src.tools.custom_tool import CustomToolExecutor

@pytest.mark.asyncio
async def test_interpolate_value():
    executor = CustomToolExecutor(custom_tool_id=uuid4())
    
    # Test string interpolation
    config = {"url": "https://api.weather.com?q={{ciudad}}", "key": "val"}
    variables = {"ciudad": "Lima"}
    result = executor._interpolate_value(config, variables)
    assert result["url"] == "https://api.weather.com?q=Lima"
    assert result["key"] == "val"
    
    # Test nested interpolation
    config = {
        "params": {
            "query": "{{query}}",
            "count": 10
        },
        "list": ["{{item1}}", "item2"]
    }
    variables = {"query": "test query", "item1": "first"}
    result = executor._interpolate_value(config, variables)
    assert result["params"]["query"] == "test query"
    assert result["list"][0] == "first"
    
    # Test non-string variables
    config = {"threshold": "{{level}}"}
    variables = {"level": 0.5}
    result = executor._interpolate_value(config, variables)
    assert result["threshold"] == "0.5"

@pytest.mark.asyncio
async def test_execute_physical_tool_with_interpolation():
    executor = CustomToolExecutor(custom_tool_id=uuid4())
    
    # Mock physical tool
    mock_physical_tool = MagicMock()
    mock_physical_tool.execute = AsyncMock()
    mock_physical_tool.execute.return_value = MagicMock(success=True, data="weather data")
    
    # Mock registry and load config
    executor._get_physical_tool_from_registry = MagicMock(return_value=mock_physical_tool)
    
    # Configuration with placeholder
    mock_config = MagicMock()
    mock_config.tool_type = "http_request"
    mock_config.configuration = {"url": "https://api.weather.com?q={{ciudad}}"}
    mock_config.name = "clima"
    
    # Execute with variable
    await executor._execute_physical_tool(mock_config, ciudad="Madrid")
    
    # Verify interpolation occurred before sub-tool execution
    mock_physical_tool.execute.assert_called_once()
    args, kwargs = mock_physical_tool.execute.call_args
    assert kwargs["url"] == "https://api.weather.com?q=Madrid"

@pytest.mark.asyncio
async def test_find_template_tags():
    executor = CustomToolExecutor(custom_tool_id=uuid4())
    config = {
        "url": "https://api.weather.com/{{ciudad}}",
        "params": {
            "key": "{{api_key}}",
            "units": "metric"
        },
        "nested": {
            "deep": "{{deep_val}}"
        }
    }
    tags = executor._find_template_tags(config)
    assert tags == {"ciudad", "api_key", "deep_val"}

@pytest.mark.asyncio
async def test_get_parameters_auto_discovery():
    executor = CustomToolExecutor(custom_tool_id=uuid4())
    
    # Mock configuration with tags but no explicit parameters list
    mock_config = MagicMock()
    mock_config.configuration = {
        "url": "https://api.com/{{userId}}",
        "body": {"action": "{{actionName}}"}
    }
    executor._custom_tool_config = mock_config
    
    params = executor.get_parameters()
    
    # Should have discovered 2 parameters
    assert len(params) == 2
    names = {p.name for p in params}
    assert names == {"userId", "actionName"}
    assert all(p.required for p in params)
    assert all("Parameter detected" in p.description for p in params)

@pytest.mark.asyncio
async def test_get_parameters_hybrid():
    # Test that explicit parameters are preserved and tags are added as fallbacks
    executor = CustomToolExecutor(custom_tool_id=uuid4())
    
    executor._custom_tool_config = MagicMock()
    executor._custom_tool_config.configuration = {
        "url": "https://api.com/{{userId}}/{{action}}",
        "parameters": [
            {"name": "userId", "type": "integer", "description": "Custom ID"}
        ]
    }
    
    params = executor.get_parameters()
    
    # Should have 2 parameters: 1 explicit, 1 auto-discovered
    assert len(params) == 2
    
    # check explicit one
    user_id_param = next(p for p in params if p.name == "userId")
    assert user_id_param.type == "integer"
    assert user_id_param.description == "Custom ID"
    
    # check discovered one
    action_param = next(p for p in params if p.name == "action")
    assert action_param.type == "string"
    assert "Parameter detected" in action_param.description
