import logging
from typing import List
from PyPDF2 import PdfReader  # Updated import
from docx import Document
import re
import spacy
from io import BytesIO
import streamlit as st


# Load Spacy English model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_bytes):
    """
    Extract text from PDF content with robust error handling.
    
    Args:
        pdf_bytes: PDF content as bytes
    
    Returns:
        str: Extracted text from the PDF
    
    Raises:
        ValueError: If PDF processing fails
    """
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        text = []
        
        for page in reader.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
            except Exception as page_error:
                logging.warning(f"Could not extract text from page: {str(page_error)}")
                continue
        
        return "\n\n".join(text)
        
    except Exception as e:
        logging.error(f"PDF extraction error: {str(e)}")
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")


def extract_text_from_docx(docx_path: str) -> str:
    logging.debug(f"Extracting text from DOCX: {docx_path}")
    try:
        doc = Document(docx_path)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]  # Skip empty paragraphs
        logging.debug(f"Number of paragraphs extracted: {len(paragraphs)}")
        return "\n".join(paragraphs)
    except Exception as e:
        logging.error(f"Error extracting text from DOCX: {e}")
        raise


def preprocess_text(text: str, stop_words: set = None) -> str:
    logging.debug(f"Preprocessing text: {text[:100]}")  # Log the first 100 characters
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Use a predefined set of stopwords if not provided
    if stop_words is None:
        stop_words = set(['this', 'is', 'a', 'for'])  # Default stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    # Join words back into a single string
    preprocessed_text = ' '.join(words)
    logging.debug(f"Preprocessed text: {preprocessed_text}")
    return preprocessed_text


def chunk_text(text: str, chunk_size: int = 512) -> List[str]:
    """
    Splits text into chunks of approximately `chunk_size` characters, 
    ensuring no words are split across chunks.

    Args:
        text (str): The input text to chunk.
        chunk_size (int): Maximum size of each chunk in characters.

    Returns:
        List[str]: List of text chunks.
    """

    logging.debug(f"Chunking text into size: {chunk_size}")
    chunks = []
    words = text.split()
    current_chunk = []

    for word in words:
        # Check if adding the next word would exceed chunk size
        if len(' '.join(current_chunk) + ' ' + word) > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]  # Start a new chunk
        else:
            current_chunk.append(word)

    # Append the last chunk if it contains any words
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    logging.debug(f"Number of chunks created: {len(chunks)}")
    return chunks

