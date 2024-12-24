import pytest
from datetime import datetime
from app.models.chat_message import ChatMessage

def test_chat_message_creation():
    message = ChatMessage(
        content="test message",
        timestamp=datetime.now(),
        role='user'
    )
    assert message.content == "test message"
    assert message.role == 'user'
    assert message.timestamp is not None

def test_chat_message_optional_fields():
    message = ChatMessage(
        content="test message",
        timestamp=datetime.now(),
        role='assistant',
        contexts=["context1", "context2"],
        requires_rag=True
    )
    assert message.contexts == ["context1", "context2"]
    assert message.requires_rag is True