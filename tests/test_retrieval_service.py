import pytest
from app.services.retrieval_service import RetrievalService


@pytest.fixture
def retrieval_service():
    return RetrievalService()

@pytest.fixture
def chunks():
    return ["This is a test chunk.", "Another test chunk.", "Yet another chunk for testing."]

def test_retrieval_service_initialization(retrieval_service):
    assert retrieval_service.model is not None
    assert retrieval_service.device.type in ["cpu", "cuda"]

def test_retrieve_relevant_chunks_empty(retrieval_service):
    retrieval_service.create_index([])
    result = retrieval_service.retrieve_relevant_chunks("test query")
    assert result == []

def test_retrieve_relevant_chunks_no_index(retrieval_service):
    with pytest.raises(ValueError, match="Index has not been created or loaded."):
        retrieval_service.retrieve_relevant_chunks("test", top_k=2)

def test_create_index(retrieval_service, chunks):
    retrieval_service.create_index(chunks)
    assert retrieval_service.index is not None
    assert len(retrieval_service.chunks) == len(chunks)

def test_retrieve_relevant_chunks(retrieval_service, chunks):
    retrieval_service.create_index(chunks)
    query = "test chunk"
    relevant_chunks = retrieval_service.retrieve_relevant_chunks(query, top_k=2)
    assert len(relevant_chunks) == 2
    assert "This is a test chunk." in relevant_chunks
    assert "Another test chunk." in relevant_chunks