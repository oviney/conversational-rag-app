import pytest
from unittest.mock import MagicMock
from app.services.chat_service import ChatService
from app.models.chat_message import ChatMessage
from datetime import datetime

def test_process_message_without_rag():
    mock_generation_service = MagicMock()
    mock_generation_service.generate_text.return_value = "The weather is sunny."
    mock_rag_service = MagicMock()
    chat_service = ChatService(mock_generation_service, mock_rag_service)
    message = "What is the weather today?"
    response = chat_service.process_message(message, [])
    
    assert response.role == 'assistant'
    assert response.requires_rag is False
    assert "weather" in response.content.lower()


def test_process_message_with_rag():
    mock_generation_service = MagicMock()
    # The final generated text from GenerationService:
    mock_generation_service.generate_text.return_value = "The main topic is testing."

    mock_rag_service = MagicMock()
    # Ensure the RAG service returns a 2‑element tuple or list matching
    # what ChatService expects to unpack, e.g. (embedding, doc_text).
    mock_rag_service.process_query.return_value = ("The main topic is testing.", ["fake_embedding", "Document text about testing."])

    chat_service = ChatService(mock_generation_service, mock_rag_service)
    chunks = ["This is a test chunk.", "Another test chunk."]
    message = "What is the main topic of the document?"
    response = chat_service.process_message(message, chunks)
    
    assert response.role == 'assistant'
    assert response.requires_rag is True
    # Verify that the content matches the final string from generate_text:
    assert "main topic" in response.content.lower()