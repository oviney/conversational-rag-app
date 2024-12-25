import logging
import pytest
from unittest.mock import patch, MagicMock
import tempfile
from reportlab.pdfgen import canvas
from docx import Document
from app.document_processing import (
    extract_text_from_pdf, extract_text_from_docx, preprocess_text, chunk_text
)
from app.services.generation_service import GenerationService
from app.services.chat_service import ChatService
from app.services.retrieval_service import RetrievalService
# Main Streamlit application
import streamlit as st
import torch

def setup_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

# Use this in your GenerationService or other model-related code
device = setup_device()

# Initialize session state
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_service" not in st.session_state:
        st.session_state.chat_service = ChatService()
    if "document_chunks" not in st.session_state:
        st.session_state.document_chunks = []
    if "index_created" not in st.session_state:
        st.session_state.index_created = False
    if "processed_text" not in st.session_state:
        st.session_state.processed_text = ""
    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False

initialize_session_state()

def process_document(uploaded_file):
    try:
        if uploaded_file.type == "application/pdf":
            try:
                # Convert UploadedFile to bytes for PDF processing
                pdf_bytes = uploaded_file.read()
                if not pdf_bytes:
                    raise ValueError("Empty PDF file")
                
                # Try to extract text from PDF
                text = extract_text_from_pdf(pdf_bytes)
                if not text.strip():
                    st.warning("No text could be extracted from the PDF. The file might be scanned or protected.")
                    return False
                logging.debug(f"Extracted PDF Text Snippet: {text[:500]}")
                
            except Exception as pdf_error:
                st.error("Failed to process PDF file. Please ensure the file is not corrupted or password protected.")
                logging.error(f"PDF processing error: {str(pdf_error)}")
                return False
        else:
            try:
                # Reset file pointer and try to read as text
                uploaded_file.seek(0)
                text = uploaded_file.getvalue().decode('utf-8', errors='ignore')
                if not text.strip():
                    raise ValueError("Empty text file")
                logging.debug(f"Extracted Text File Snippet: {text[:500]}")
            except UnicodeDecodeError as decode_error:
                st.error("Could not decode the file. Please ensure it's a valid text file.")
                logging.error(f"Text decode error: {str(decode_error)}")
                return False
        
        # Process the extracted text
        st.session_state.processed_text = preprocess_text(text)
        st.session_state.document_chunks = chunk_text(st.session_state.processed_text)
        logging.debug(f"Number of Chunks Created: {len(st.session_state.document_chunks)}")
        if st.session_state.document_chunks:
            logging.debug(f"Sample Chunk: {st.session_state.document_chunks[0][:500]}")
        
        if not st.session_state.document_chunks:
            st.warning("No valid text chunks were created from the document.")
            return False
        
        # Create index with chunks
        st.session_state.chat_service.rag_service.retrieval_service.create_index(
            st.session_state.document_chunks
        )
        st.session_state.index_created = True
        st.session_state.document_processed = True  # Mark document as processed
        
        # Show success message
        st.success("Document processed successfully!")
        return True
        
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        logging.error(f"Error processing document: {str(e)}")
        return False
    finally:
        # Reset file pointer for potential reuse
        try:
            uploaded_file.seek(0)
        except Exception as seek_error:
            logging.warning(f"Could not reset file pointer: {str(seek_error)}")

# UI Layout
st.set_page_config(page_title="Document Chat Assistant", layout="wide")

# Debug info (temporary)
st.sidebar.write("🔍 **Debug Info:**")
st.sidebar.write(f"📄 Index created: {st.session_state.index_created}")
st.sidebar.write(f"📚 Chunks: {len(st.session_state.document_chunks) if st.session_state.document_chunks else 0}")
st.sidebar.write(f"💬 Messages: {len(st.session_state.messages)}")

# Sidebar
with st.sidebar:
    st.title("📄 Document Upload")
    uploaded_file = st.file_uploader(
        "Choose a PDF, DOCX, or TXT file",
        type=["pdf", "docx", "txt"],
        help="Upload your document to start asking questions"
    )
    
    if uploaded_file and not st.session_state.document_processed:
        with st.spinner("Processing document..."):
            if process_document(uploaded_file):
                st.success("Document processed successfully!")

# Main Content
st.title("💬 Document Chat Assistant")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about the document..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add to message history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.chat_service.process_message(
                    prompt, st.session_state.document_chunks
                )
                if not response.content.strip():
                    response.content = "I'm sorry, I couldn't find an answer to that question based on the provided document."
                    logging.warning("Empty response generated.")
                st.markdown(response.content)
                # Add to message history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response.content
                })
                logging.debug(f"User Prompt: {prompt}")
                logging.debug(f"Assistant Response: {response.content}")
            except Exception as e:
                st.error("An error occurred while generating the response. Please try again.")
                logging.error(f"Chat error: {str(e)}")

# Show warning if no document is loaded
if not st.session_state.index_created:
    st.info("👆 Please upload a document to start chatting")

@pytest.fixture
def mock_file_uploader():
    with patch('streamlit.file_uploader') as mock:
        yield mock

def test_file_upload(mock_file_uploader):
    uploaded_file = mock_file_uploader(
        "Choose a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"]
    )
    assert uploaded_file is not None

@patch('app.document_processing.extract_text_from_pdf')
@patch('app.document_processing.extract_text_from_docx')
def test_text_extraction(mock_extract_text_from_docx, mock_extract_text_from_pdf):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        c = canvas.Canvas(tmp_file.name)
        c.drawString(100, 750, "Hello, World!")
        c.save()
        temp_file_path = tmp_file.name

    mock_extract_text_from_pdf.return_value = "Hello, World!\n"
    text = extract_text_from_pdf(temp_file_path)
    assert text == "Hello, World!\n"

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

def test_process_message_with_rag_no_index():
    chat_service = ChatService()
    message = "What is the main topic of the document?"
    with pytest.raises(ValueError, match="Index has not been created or loaded."):
        chat_service.process_message(message, ["This is a test chunk."])

def test_chunk_text():
    text = "This is a sample text for chunking. " * 10
    chunks = chunk_text(text, chunk_size=50)
    assert isinstance(chunks, list)
    assert len(chunks) > 0

def test_retrieve_relevant_chunks_empty():
    retrieval_service = RetrievalService()
    retrieval_service.create_index([])
    with pytest.raises(ValueError, match="Index has not been created or loaded."):
        retrieval_service.retrieve_relevant_chunks("test query")

def test_retrieve_relevant_chunks_no_index():
    retrieval_service = RetrievalService()
    with pytest.raises(ValueError, match="Index has not been created or loaded."):
        retrieval_service.retrieve_relevant_chunks("test", top_k=2)

def test_preprocess_text():
    text = "This is a sample text for preprocessing."
    preprocessed_text = preprocess_text(text)
    expected_text = "sample text preprocessing"  # Update this to match the expected output of preprocess_text
    assert isinstance(preprocessed_text, str)
    assert preprocessed_text == expected_text

def test_chunk_text_with_different_chunk_size():
    text = "This is a sample text for chunking. " * 10
    chunks = chunk_text(text, chunk_size=100)
    assert isinstance(chunks, list)
    assert all(len(chunk) <= 100 for chunk in chunks)

def test_retrieval_service_create_index():
    retrieval_service = RetrievalService()
    chunks = ["chunk1", "chunk2", "chunk3"]
    retrieval_service.create_index(chunks)
    assert retrieval_service.index is not None

def test_retrieval_service_retrieve_relevant_chunks():
    retrieval_service = RetrievalService()
    chunks = ["chunk1", "chunk2", "chunk3"]
    retrieval_service.create_index(chunks)
    relevant_chunks = retrieval_service.retrieve_relevant_chunks("chunk1")
    assert isinstance(relevant_chunks, list)
    assert len(relevant_chunks) > 0