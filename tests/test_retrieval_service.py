import pytest
from app.services.retrieval_service import RetrievalService
import faiss


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
    """
    Test that a ValueError is raised when attempting to retrieve chunks with an empty index.
    """
    retrieval_service.create_index([])  # Create an empty index
    with pytest.raises(ValueError, match="Index has not been created or loaded."):
        retrieval_service.retrieve_relevant_chunks("test query")


def test_retrieve_relevant_chunks_no_index(retrieval_service):
    """
    Test that a ValueError is raised when the index is not created.
    """
    retrieval_service.index = faiss.IndexFlatL2(384)  # Ensure an empty index
    retrieval_service.chunks = []  # Ensure chunks are empty
    with pytest.raises(ValueError, match="Index has not been created or loaded."):
        retrieval_service.retrieve_relevant_chunks(query="Test query", top_k=3)


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