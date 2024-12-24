import pytest
from app.services.rag_service import RAGService
from unittest.mock import patch

@patch('app.services.generation_service.GenerationService.generate_text', return_value="This is a test response.")
def test_rag_service_query(mock_generate_text):
    rag_service = RAGService()
    chunks = ["This is a test chunk.", "Another test chunk."]
    rag_service.retrieval_service.create_index(chunks)
    
    query = "test"
    response, contexts = rag_service.process_query(query, chunks)
    
    assert "test" in response.lower()
    assert len(contexts) > 0
    assert all("test" in context for context in contexts)