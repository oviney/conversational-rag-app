import pytest
from unittest.mock import patch, MagicMock
import tempfile
import os
import warnings
from reportlab.pdfgen import canvas
from docx import Document
from app.document_processing import extract_text_from_pdf, extract_text_from_docx, preprocess_text, chunk_text
from app.services.retrieval_service import RetrievalService
from app.services.generation_service import GenerationService
import streamlit as st

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="reportlab")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="faiss")

@pytest.fixture
def mock_file_uploader():
    with patch('streamlit.file_uploader') as mock:
        yield mock

def test_file_upload(mock_file_uploader):
    mock_file_uploader.return_value = MagicMock(name='uploaded_file')
    uploaded_file = mock_file_uploader("Choose a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])
    assert uploaded_file is not None

@patch('app.document_processing.extract_text_from_pdf')
@patch('app.document_processing.extract_text_from_docx')
def test_text_extraction(mock_extract_text_from_docx, mock_extract_text_from_pdf):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        c = canvas.Canvas(tmp_file.name)
        c.drawString(100, 750, "Hello, World!")
        c.save()
        temp_file_path = tmp_file.name

    mock_extract_text_from_pdf.return_value = "Hello, World!\n\n"
    text = extract_text_from_pdf(temp_file_path)
    assert text == "Hello, World!\n\n"

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
    index = MagicMock(name='index')
    model = MagicMock(name='model')
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