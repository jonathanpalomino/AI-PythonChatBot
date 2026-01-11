# =============================================================================
# tests/test_embedding_model_validation.py
# Tests for embedding model validation at conversation level
# =============================================================================
"""
Test suite for embedding model validation functionality.
Tests that embedding models can be set initially but cannot be changed later.
"""
import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, patch

from sqlalchemy.ext.asyncio import AsyncSession
from src.services.embedding_model_validator import EmbeddingModelValidator
from src.models.models import Conversation


@pytest.mark.asyncio
async def test_allow_initial_embedding_model():
    """Test that setting an initial embedding model is allowed"""
    validator = EmbeddingModelValidator()
    
    # Mock database session
    mock_db = AsyncMock(spec=AsyncSession)
    conversation_id = uuid4()
    
    # Test case: No current embedding model, setting a new one
    current_settings = {"provider": "local", "model": "mistral"}  # No embedding_model
    new_embedding_model = "all-minilm"
    
    result = await validator.validate_embedding_model_change(
        db=mock_db,
        conversation_id=conversation_id,
        new_embedding_model=new_embedding_model,
        current_settings=current_settings
    )
    
    # Should return None (no error) when no current model is set
    assert result is None


@pytest.mark.asyncio
async def test_allow_same_embedding_model():
    """Test that keeping the same embedding model is allowed"""
    validator = EmbeddingModelValidator()
    
    # Mock database session
    mock_db = AsyncMock(spec=AsyncSession)
    conversation_id = uuid4()
    
    # Test case: Same embedding model
    current_settings = {
        "provider": "local", 
        "model": "mistral",
        "embedding_model": "mxbai-embed-large"
    }
    new_embedding_model = "mxbai-embed-large"  # Same model
    
    result = await validator.validate_embedding_model_change(
        db=mock_db,
        conversation_id=conversation_id,
        new_embedding_model=new_embedding_model,
        current_settings=current_settings
    )
    
    # Should return None (no error) when model is unchanged
    assert result is None


@pytest.mark.asyncio
async def test_block_embedding_model_change():
    """Test that changing embedding model after initial selection is blocked"""
    validator = EmbeddingModelValidator()
    
    # Mock database session
    mock_db = AsyncMock(spec=AsyncSession)
    conversation_id = uuid4()
    
    # Test case: Different embedding model
    current_settings = {
        "provider": "local", 
        "model": "mistral",
        "embedding_model": "mxbai-embed-large"
    }
    new_embedding_model = "all-minilm"  # Different model
    
    result = await validator.validate_embedding_model_change(
        db=mock_db,
        conversation_id=conversation_id,
        new_embedding_model=new_embedding_model,
        current_settings=current_settings
    )
    
    # Should return an error message
    assert result is not None
    assert "Cannot change embedding model" in result
    assert "mxbai-embed-large" in result
    assert "all-minilm" in result


@pytest.mark.asyncio
async def test_allow_clearing_embedding_model():
    """Test that clearing embedding model (setting to None) is allowed"""
    validator = EmbeddingModelValidator()
    
    # Mock database session
    mock_db = AsyncMock(spec=AsyncSession)
    conversation_id = uuid4()
    
    # Test case: Clearing embedding model
    current_settings = {
        "provider": "local", 
        "model": "mistral",
        "embedding_model": "mxbai-embed-large"
    }
    new_embedding_model = None  # Clearing the model
    
    result = await validator.validate_embedding_model_change(
        db=mock_db,
        conversation_id=conversation_id,
        new_embedding_model=new_embedding_model,
        current_settings=current_settings
    )
    
    # Should return None (no error) when clearing to None
    assert result is None


@pytest.mark.asyncio
async def test_has_existing_embeddings():
    """Test the helper method for checking existing embeddings"""
    validator = EmbeddingModelValidator()
    
    # Mock database session
    mock_db = AsyncMock(spec=AsyncSession)
    conversation_id = uuid4()
    
    # Test case: No existing messages
    mock_db.execute.return_value.scalar_one_or_none.return_value = None
    result = await validator.has_existing_embeddings(mock_db, conversation_id)
    assert result is False
    
    # Test case: Existing messages
    mock_message = AsyncMock()
    mock_db.execute.return_value.scalar_one_or_none.return_value = mock_message
    result = await validator.has_existing_embeddings(mock_db, conversation_id)
    assert result is True


@pytest.mark.asyncio
async def test_conversation_update_integration():
    """Integration test for conversation update with embedding model validation"""
    from src.api.v1.conversations import update_conversation
    from src.schemas.schemas import ConversationUpdate
    from fastapi import HTTPException
    
    # Mock database session
    mock_db = AsyncMock(spec=AsyncSession)
    conversation_id = uuid4()
    
    # Mock conversation
    mock_conversation = AsyncMock(spec=Conversation)
    mock_conversation.id = conversation_id
    mock_conversation.settings = {
        "provider": "local", 
        "model": "mistral",
        "embedding_model": "mxbai-embed-large"
    }
    
    # Mock database query
    mock_db.execute.return_value.scalars.return_value.first.return_value = mock_conversation
    
    # Test case: Try to change embedding model (should fail)
    update_data = ConversationUpdate(
        settings={
            "provider": "local", 
            "model": "mistral",
            "embedding_model": "all-minilm"  # Different model
        }
    )
    
    # This should raise HTTPException
    with pytest.raises(HTTPException) as exc_info:
        await update_conversation(conversation_id, update_data, mock_db)
    
    assert exc_info.value.status_code == 400
    assert "Cannot change embedding model" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_conversation_update_allowed():
    """Test that conversation update is allowed when embedding model is not changed"""
    from src.api.v1.conversations import update_conversation
    from src.schemas.schemas import ConversationUpdate
    
    # Mock database session
    mock_db = AsyncMock(spec=AsyncSession)
    conversation_id = uuid4()
    
    # Mock conversation
    mock_conversation = AsyncMock(spec=Conversation)
    mock_conversation.id = conversation_id
    mock_conversation.settings = {
        "provider": "local", 
        "model": "mistral",
        "embedding_model": "mxbai-embed-large"
    }
    
    # Mock database query
    mock_db.execute.return_value.scalars.return_value.first.return_value = mock_conversation
    
    # Test case: Keep same embedding model (should succeed)
    update_data = ConversationUpdate(
        settings={
            "provider": "local", 
            "model": "mistral",
            "embedding_model": "mxbai-embed-large"  # Same model
        }
    )
    
    # This should succeed
    result = await update_conversation(conversation_id, update_data, mock_db)
    assert result is not None
    assert result.settings["embedding_model"] == "mxbai-embed-large"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])