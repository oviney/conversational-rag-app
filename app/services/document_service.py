# Handles document operations
from app.document_processing import extract_text_from_pdf, extract_text_from_docx, preprocess_text


class DocumentService:
    @staticmethod
    def load_document(file_path):
        extension = file_path.split('.')[-1]
        if extension == 'pdf':
            return extract_text_from_pdf(file_path)
        elif extension == 'docx':
            return extract_text_from_docx(file_path)
        else:
            with open(file_path, 'r') as f:
                return f.read()

    @staticmethod
    def preprocess_and_chunk(text):
        preprocessed = preprocess_text(text)
        return chunk_text(preprocessed)
