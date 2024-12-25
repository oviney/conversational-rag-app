import pytest
from app.services.rag_service import RAGService
from unittest.mock import patch, MagicMock

@patch('app.services.generation_service.GenerationService.generate_text', return_value="This is a test response.")
def test_rag_service_query(mock_generate_text):
    mock_retrieval_service = MagicMock()
    mock_generation_service = MagicMock()
    mock_generation_service.generate_text = mock_generate_text

    # Ensure retrieval returns non-empty chunks
    mock_retrieval_service.retrieve_relevant_chunks.return_value = [
        "This is a test chunk.",
        "Another test chunk."
    ]

    rag_service = RAGService(mock_retrieval_service, mock_generation_service)
    chunks = ["This is a test chunk.", "Another test chunk."]
    rag_service.retrieval_service.create_index(chunks)
    
    query = "test"
    response, contexts = rag_service.process_query(query, chunks)
    
    # Now 'contexts' will contain the mocked relevant chunks
    assert "test" in response.lower()
    assert len(contexts) > 0
    assert all("test" in context.lower() for context in contexts)