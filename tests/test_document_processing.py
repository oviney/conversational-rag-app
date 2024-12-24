# Tests for document processing
import pytest
from app.document_processing import extract_text_from_pdf, extract_text_from_docx, preprocess_text


def test_preprocess_text():
    assert preprocess_text("Text") == "text"
