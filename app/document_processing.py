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
    try:
        with open(pdf_path, "rb") as file:
            reader = PdfReader(file)  # Updated to PdfReader
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:  # Check if text was extracted
                    text += page_text
                else:
                    logging.warning(f"No text found on page {i + 1}")
        logging.debug(f"Extracted text length: {len(text)}")
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        raise


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

