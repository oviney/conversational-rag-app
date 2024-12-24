import pytest
from app.services.retrieval_service import RetrievalService

def test_retrieve_relevant_chunks_empty():
    retrieval_service = RetrievalService()
    retrieval_service.create_index([])
    result = retrieval_service.retrieve_relevant_chunks("test query")
    assert result == []

def test_retrieve_relevant_chunks_no_index():
    retrieval_service = RetrievalService()
    with pytest.raises(ValueError, match="Index has not been created or loaded."):
        retrieval_service.retrieve_relevant_chunks("test", top_k=2)