# Document processing utilities
import PyPDF2
from docx import Document

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        return ''.join(page.extract_text() for page in reader.pages)

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    return '\n'.join(p.text for p in doc.paragraphs)

def preprocess_text(text):
    return text.lower()  # Simplified example