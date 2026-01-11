# =============================================================================
# tests/test_models.py
# Test database models
# =============================================================================
"""
Unit tests for SQLAlchemy models
"""
from uuid import uuid4

import pytest

from src.models.models import (
    Conversation,
    Message,
    MessageRole,
    PromptTemplate,
    VisibilityType
)


class TestPromptTemplate:
    """Test PromptTemplate model"""

    def test_create_prompt_template(self):
        """Test creating a prompt template"""
        template = PromptTemplate(
            name="Test Template",
            description="A test template",
            category="test",
            visibility=VisibilityType.PUBLIC,
            system_prompt="You are a test assistant",
            variables=[],
            settings={}
        )

        assert template.name == "Test Template"
        assert template.category == "test"
        assert template.visibility == VisibilityType.PUBLIC
        assert template.is_active == True
        assert template.version == 1


class TestConversation:
    """Test Conversation model"""

    def test_create_conversation(self):
        """Test creating a conversation"""
        conversation = Conversation(
            title="Test Conversation",
            settings={
                "provider": "local",
                "model": "mistral"
            },
            extra_metadata={"tags": ["test"]}
        )

        assert conversation.title == "Test Conversation"
        assert conversation.settings["provider"] == "local"
        assert "tags" in conversation.extra_metadata


class TestMessage:
    """Test Message model"""

    def test_create_message(self):
        """Test creating a message"""
        conv_id = uuid4()

        message = Message(
            conversation_id=conv_id,
            role=MessageRole.USER,
            content="Hello, world!",
            extra_metadata={"test": True},
            attachments=[]
        )

        assert message.conversation_id == conv_id
        assert message.role == MessageRole.USER
        assert message.content == "Hello, world!"
        assert message.extra_metadata["test"] == True


@pytest.mark.unit
def test_enum_values():
    """Test enum values"""
    assert MessageRole.USER.value == "user"
    assert MessageRole.ASSISTANT.value == "assistant"
    assert MessageRole.SYSTEM.value == "system"

    assert VisibilityType.PUBLIC.value == "public"
    assert VisibilityType.PRIVATE.value == "private"
    assert VisibilityType.SHARED.value == "shared"
