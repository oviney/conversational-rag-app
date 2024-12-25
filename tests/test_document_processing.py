# Tests for document processing
import pytest
import tempfile
from app.document_processing import extract_text_from_pdf, extract_text_from_docx, preprocess_text, chunk_text
from reportlab.pdfgen import canvas
from docx import Document


def test_preprocess_text():
    text = "This is a sample text, with punctuation! And stopwords."
    preprocessed_text = preprocess_text(text)
    assert preprocessed_text == "sample text with punctuation and stopwords"


def test_extract_text_from_pdf():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        c = canvas.Canvas(tmp_file.name)
        c.drawString(100, 750, "Hello, World!")
        c.save()
        temp_file_path = tmp_file.name

    text = extract_text_from_pdf(temp_file_path)
    assert text.strip() == "Hello, World!"

def test_extract_text_from_docx():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
        doc = Document()
        doc.add_paragraph("Extracted text from DOCX")
        doc.save(tmp_file.name)
        temp_file_path = tmp_file.name

    text = extract_text_from_docx(temp_file_path)
    assert text.strip() == "Extracted text from DOCX"

def test_chunk_text():
    text = "This is a sample text for chunking. " * 10
    chunks = chunk_text(text, chunk_size=50)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(len(chunk) <= 50 for chunk in chunks)
