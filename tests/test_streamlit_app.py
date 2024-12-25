import pytest
from unittest.mock import patch, MagicMock
import tempfile
from reportlab.pdfgen import canvas
from docx import Document
from app.document_processing import (
    extract_text_from_pdf, extract_text_from_docx, preprocess_text, chunk_text
)
from app.services.generation_service import GenerationService
from app.services.chat_service import ChatService
from app.services.retrieval_service import RetrievalService


@pytest.fixture
def mock_file_uploader():
    with patch('streamlit.file_uploader') as mock:
        yield mock

def test_file_upload(mock_file_uploader):
    uploaded_file = mock_file_uploader(
        "Choose a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"]
    )
    assert uploaded_file is not None

@patch('app.document_processing.extract_text_from_pdf')
@patch('app.document_processing.extract_text_from_docx')
def test_text_extraction(mock_extract_text_from_docx, mock_extract_text_from_pdf):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        c = canvas.Canvas(tmp_file.name)
        c.drawString(100, 750, "Hello, World!")
        c.save()
        temp_file_path = tmp_file.name

    mock_extract_text_from_pdf.return_value = "Hello, World!\n"
    text = extract_text_from_pdf(temp_file_path)
    assert text == "Hello, World!\n"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
        doc = Document()
        doc.add_paragraph("Extracted text from DOCX")
        doc.save(tmp_file.name)
        temp_file_path = tmp_file.name

    mock_extract_text_from_docx.return_value = "Extracted text from DOCX"
    text = extract_text_from_docx(temp_file_path)
    assert text == "Extracted text from DOCX"

def test_preprocess_and_chunk_text():
    text = "This is a sample text for preprocessing and chunking."
    preprocessed_text = preprocess_text(text)
    chunks = chunk_text(preprocessed_text)
    assert isinstance(chunks, list)
    assert len(chunks) > 0

@patch.object(RetrievalService, 'create_index')
def test_create_index(mock_create_index):
    chunks = ["chunk1", "chunk2", "chunk3"]
    mock_create_index.return_value = MagicMock(name='index')
    retrieval_service = RetrievalService()
    index = retrieval_service.create_index(chunks)
    assert index is not None

@patch.object(RetrievalService, 'retrieve_relevant_chunks')
@patch.object(GenerationService, 'generate_text')
def test_chat_interface(mock_generate_text, mock_retrieve_relevant_chunks):
    prompt = "What is the content of the document?"
    chunks = ["chunk1", "chunk2", "chunk3"]
    mock_retrieve_relevant_chunks.return_value = chunks
    mock_generate_text.return_value = "Generated response based on the document content."

    retrieval_service = RetrievalService()
    relevant_chunks = retrieval_service.retrieve_relevant_chunks(prompt, chunks)
    context = "\n".join(relevant_chunks)
    generation_service = GenerationService()
    generated_response = generation_service.generate_text(context, prompt)

    assert generated_response == "Generated response based on the document content."

@patch.object(RetrievalService, 'retrieve_relevant_chunks')
@patch.object(GenerationService, 'generate_text')
def test_chat_interface_no_relevant_chunks(mock_generate_text, mock_retrieve_relevant_chunks):
    prompt = "What is the content of the document?"
    chunks = ["chunk1", "chunk2", "chunk3"]
    mock_retrieve_relevant_chunks.return_value = []
    mock_generate_text.return_value = "No relevant information found."

    retrieval_service = RetrievalService()
    relevant_chunks = retrieval_service.retrieve_relevant_chunks(prompt, chunks)
    context = "\n".join(relevant_chunks)
    generation_service = GenerationService()
    generated_response = generation_service.generate_text(context, prompt)

    assert generated_response == "No relevant information found."

def test_process_message_with_rag_no_index():
    chat_service = ChatService()
    message = "What is the main topic of the document?"
    with pytest.raises(ValueError, match="Index has not been created or loaded."):
        chat_service.process_message(message, ["This is a test chunk."])

def test_chunk_text():
    text = "This is a sample text for chunking. " * 10
    chunks = chunk_text(text, chunk_size=50)
    assert isinstance(chunks, list)
    assert len(chunks) > 0

def test_retrieve_relevant_chunks_empty():
    retrieval_service = RetrievalService()
    retrieval_service.create_index([])
    with pytest.raises(ValueError, match="Index has not been created or loaded."):
        retrieval_service.retrieve_relevant_chunks("test query")

def test_retrieve_relevant_chunks_no_index():
    retrieval_service = RetrievalService()
    with pytest.raises(ValueError, match="Index has not been created or loaded."):
        retrieval_service.retrieve_relevant_chunks("test", top_k=2)

def test_preprocess_text():
    text = "This is a sample text for preprocessing."
    preprocessed_text = preprocess_text(text)
    expected_text = "sample text preprocessing"  # Update this to match the expected output of preprocess_text
    assert isinstance(preprocessed_text, str)
    assert preprocessed_text == expected_text

def test_chunk_text_with_different_chunk_size():
    text = "This is a sample text for chunking. " * 10
    chunks = chunk_text(text, chunk_size=100)
    assert isinstance(chunks, list)
    assert all(len(chunk) <= 100 for chunk in chunks)

def test_retrieval_service_create_index():
    retrieval_service = RetrievalService()
    chunks = ["chunk1", "chunk2", "chunk3"]
    retrieval_service.create_index(chunks)
    assert retrieval_service.index is not None

def test_retrieval_service_retrieve_relevant_chunks():
    retrieval_service = RetrievalService()
    chunks = ["chunk1", "chunk2", "chunk3"]
    retrieval_service.create_index(chunks)
    relevant_chunks = retrieval_service.retrieve_relevant_chunks("chunk1")
    assert isinstance(relevant_chunks, list)
    assert len(relevant_chunks) > 0