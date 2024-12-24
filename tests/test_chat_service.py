import pytest
from app.services.chat_service import ChatService
from app.models.chat_message import ChatMessage
from datetime import datetime
from unittest.mock import patch

@patch('app.services.generation_service.GenerationService.generate_text', return_value="The weather is sunny.")
def test_process_message_without_rag(mock_generate_text):
    chat_service = ChatService()
    message = "What is the weather today?"
    response = chat_service.process_message(message)
    
    assert response.role == 'assistant'
    assert response.requires_rag is False
    assert "weather" in response.content.lower()

@patch('app.services.generation_service.GenerationService.generate_text', return_value="The main topic is testing.")
def test_process_message_with_rag(mock_generate_text):
    chat_service = ChatService()
    chunks = ["This is a test chunk.", "Another test chunk."]
    chat_service.rag_service.retrieval_service.create_index(chunks)
    
    message = "What is the main topic of the document?"
    response = chat_service.process_message(message, chunks)
    
    assert response.role == 'assistant'
    assert response.requires_rag is True
    assert response.contexts is not None
    assert len(response.contexts) > 0