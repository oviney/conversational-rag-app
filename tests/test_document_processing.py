# Tests for document processing
import pytest
import tempfile
from app.document_processing import extract_text_from_pdf, extract_text_from_docx, preprocess_text, chunk_text
from reportlab.pdfgen import canvas
from docx import Document

@pytest.mark.unit
def test_preprocess_text():
    text = "This is a sample text for preprocessing."
    preprocessed_text = preprocess_text(text)
    assert preprocessed_text == "sample text preprocessing"

@pytest.mark.unit
def test_extract_text_from_pdf():
    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n4 0 obj\n<< /Length 44 >>\nstream\nBT /F1 24 Tf 100 700 Td (Hello, World!) Tj ET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000053 00000 n \n0000000100 00000 n \n0000000173 00000 n \ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n231\n%%EOF"
    text = extract_text_from_pdf(pdf_bytes)
    assert "Hello, World!" in text

@pytest.mark.unit
def test_extract_text_from_docx():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
        doc = Document()
        doc.add_paragraph("Extracted text from DOCX")
        doc.save(tmp_file.name)
        temp_file_path = tmp_file.name

    text = extract_text_from_docx(temp_file_path)
    assert text.strip() == "Extracted text from DOCX"

@pytest.mark.unit
def test_chunk_text():
    text = "This is a sample text for chunking. " * 10
    chunks = chunk_text(text, chunk_size=50)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(len(chunk) <= 50 for chunk in chunks)
