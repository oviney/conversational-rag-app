import logging
from typing import List
from PyPDF2 import PdfReader  # Updated import
from docx import Document
import re
import spacy

logging.basicConfig(level=logging.DEBUG)

# Load Spacy English model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path: str) -> str:
    logging.debug(f"Extracting text from PDF: {pdf_path}")
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)  # Updated to PdfReader
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_docx(docx_path: str) -> str:
    logging.debug(f"Extracting text from DOCX: {docx_path}")
    doc = Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def preprocess_text(text: str) -> str:
    logging.debug(f"Preprocessing text: {text[:100]}")  # Log the first 100 characters
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stop words (optional)
    stop_words = set(['this', 'is', 'a', 'for'])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    # Join words back into a single string
    preprocessed_text = ' '.join(words)
    return preprocessed_text

def chunk_text(text: str, chunk_size: int = 512) -> List[str]:
    logging.debug(f"Chunking text into size: {chunk_size}")
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks
