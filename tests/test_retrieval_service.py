import pytest
from app.services.retrieval_service import RetrievalService

# Tests for retrieval service
# Placeholder test


def test_retrieval():
    assert True

def test_create_index():
    retrieval_service = RetrievalService()
    chunks = ["This is a test chunk.", "Another test chunk."]
    index = retrieval_service.create_index(chunks)
    assert index is not None
    assert retrieval_service.index is not None
    assert retrieval_service.chunks == chunks

def test_retrieve_relevant_chunks():
    retrieval_service = RetrievalService()
    chunks = ["This is a test chunk.", "Another test chunk."]
    retrieval_service.create_index(chunks)
    query = "test"
    relevant_chunks = retrieval_service.retrieve_relevant_chunks(query, top_k=2)
    assert len(relevant_chunks) > 0
    assert all(chunk in chunks for chunk in relevant_chunks)

def test_retrieve_relevant_chunks_no_index():
    retrieval_service = RetrievalService()
    with pytest.raises(ValueError, match="Index has not been created or loaded."):
        retrieval_service.retrieve_relevant_chunks("test", top_k=2)
