# Main Streamlit application
import streamlit as st
import os
from app.document_processing import (
    extract_text_from_pdf, extract_text_from_docx, preprocess_text, chunk_text
)
from app.services.retrieval_service import RetrievalService
from app.services.generation_service import GenerationService
from app.services.chat_service import ChatService
from app.models.chat_message import ChatMessage
from datetime import datetime
import tempfile
import torch
import logging

# Configure the root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# Set the logging level for all loggers to WARNING
for logger_name in logging.root.manager.loggerDict:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Example log message to verify logging is working
logging.debug("Logging is configured correctly.")

def setup_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

# Use this in your GenerationService or other model-related code
device = setup_device()

# Initialize session state
if 'processed_text' not in st.session_state:
    st.session_state.processed_text = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = None
if 'retrieval_service' not in st.session_state:
    st.session_state.retrieval_service = RetrievalService()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chat_service' not in st.session_state:
    st.session_state.chat_service = ChatService()

# Cache the document processing
@st.cache_data(show_spinner=True)
def process_document(file_content, file_extension):
    try:
        if file_extension == '.txt':
            text = file_content.decode("utf-8")
        elif file_extension == '.pdf':
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file_content)
                text = extract_text_from_pdf(tmp_file.name)
        elif file_extension == '.docx':
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                tmp_file.write(file_content)
                text = extract_text_from_docx(tmp_file.name)
        
        return text
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return None

# UI Layout
st.set_page_config(page_title="Document Chat Assistant", layout="wide")

# Sidebar
with st.sidebar:
    st.title("📄 Document Upload")
    uploaded_file = st.file_uploader(
        "Choose a PDF, DOCX, or TXT file",
        type=["pdf", "docx", "txt"],
        help="Upload your document to start asking questions"
    )
    
    if uploaded_file:
        with st.spinner("Processing document..."):
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            text = process_document(uploaded_file.getvalue(), file_extension)
            
            if text:
                st.session_state.processed_text = preprocess_text(text)
                st.session_state.chunks = chunk_text(st.session_state.processed_text)
                st.session_state.retrieval_service.create_index(st.session_state.chunks)
                st.success("Document processed successfully!")

# Main Content
st.title("💬 Document Chat Assistant")

# Display chat messages
for message in st.session_state.messages:
    if message.role == 'user':
        st.chat_message(message.content, is_user=True)
    else:
        st.chat_message(message.content)

# Chat input
if prompt := st.chat_input("Ask a question or chat with your document assistant..."):
    logging.debug(f"User prompt: {prompt}")
    # Add user message
    user_message = ChatMessage(
        content=prompt,
        timestamp=datetime.now(),
        role='user'
    )
    st.session_state.messages.append(user_message)
    
    # Process response
    with st.spinner("Thinking..."):
        response = st.session_state.chat_service.process_message(
            prompt,
            st.session_state.chunks if 'chunks' in st.session_state else None
        )
        logging.debug(f"Assistant response: {response.content}")
        st.session_state.messages.append(response)

    # Display response
    st.chat_message(response.content)
