from pypdf import PdfReader
from docx import Document
import re
import spacy

# Load Spacy English model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    print(f"Original Text: {text}")
    # Process text with Spacy
    doc = nlp(text)
    # Tokenize text, remove stop words, and perform stemming
    words = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    print(f"Processed Words: {words}")
    # Join words back into a single string
    return ' '.join(words)

def chunk_text(text, chunk_size=100):
    # Split text into chunks of specified size
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
