
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4
from src.services.extraction_service import ParameterExtractionService
from src.providers.manager import ChatResponse
from src.models.models import MessageRole

@pytest.mark.asyncio
async def test_extract_parameters_success():
    # Mock DB session
    mock_db = AsyncMock()
    
    # Mock result for _get_conversation_context
    mock_result = MagicMock()
    mock_result.scalars().all.return_value = [] # No history for this simple test
    mock_db.execute.return_value = mock_result
    
    service = ParameterExtractionService(mock_db)
    
    # Parameters to extract
    parameters = [
        {"name": "ciudad", "type": "string", "description": "Nombre de la ciudad", "required": True}
    ]
    
    # Mock LLM response
    mock_response = ChatResponse(content='{"ciudad": "Barcelona"}')
    
    with patch("src.services.extraction_service.provider_manager.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = mock_response
        
        result = await service.extract_parameters(
            user_message="¿Cómo está el clima en Barcelona?",
            parameters=parameters,
            conversation_id=uuid4()
        )
        
        assert result == {"ciudad": "Barcelona"}
        mock_chat.assert_called_once()

@pytest.mark.asyncio
async def test_extract_parameters_with_markdown_json():
    # Test that it handles JSON wrapped in markdown
    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars().all.return_value = []
    mock_db.execute.return_value = mock_result
    
    service = ParameterExtractionService(mock_db)
    
    mock_response = ChatResponse(content='Aquí está el JSON:\n```json\n{"ciudad": "Madrid"}\n```')
    
    with patch("src.services.extraction_service.provider_manager.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = mock_response
        
        result = await service.extract_parameters(
            user_message="Clima en Madrid",
            parameters=[{"name": "ciudad", "type": "string"}],
            conversation_id=uuid4()
        )
        
        assert result == {"ciudad": "Madrid"}

@pytest.mark.asyncio
async def test_extract_parameters_validation():
    # Test that it filters out unknown parameters
    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars().all.return_value = []
    mock_db.execute.return_value = mock_result
    
    service = ParameterExtractionService(mock_db)
    
    # Only "ciudad" is in schema
    mock_response = ChatResponse(content='{"ciudad": "Lima", "persona": "Juan"}')
    
    with patch("src.services.extraction_service.provider_manager.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = mock_response
        
        result = await service.extract_parameters(
            user_message="Clima en Lima para Juan",
            parameters=[{"name": "ciudad", "type": "string"}],
            conversation_id=uuid4()
        )
        
        assert result == {"ciudad": "Lima"}
        assert "persona" not in result
