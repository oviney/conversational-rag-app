# Main Streamlit application
import streamlit as st
import os
from app.document_processing import (
    extract_text_from_pdf, extract_text_from_docx, preprocess_text, chunk_text
)
from app.services.retrieval_service import RetrievalService
from app.services.generation_service import GenerationService
import chardet
import tempfile
import torch
from app.services.chat_service import ChatService
from app.models.chat_message import ChatMessage
from datetime import datetime

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
st.set_page_config(page_title="Document Q&A", layout="wide")

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
st.title("🤖 Document Q&A Assistant")

if 'processed_text' in st.session_state and st.session_state.processed_text:
    query = st.text_input("Ask a question about your document:", 
                         placeholder="What is the main topic of the document?")
    
    if query:
        with st.spinner("Thinking..."):
            try:
                relevant_chunks = st.session_state.retrieval_service.retrieve_relevant_chunks(
                    query, top_k=3
                )
                
                if relevant_chunks:
                    generation_service = GenerationService()
                    response = generation_service.generate_text(
                        "\n".join(relevant_chunks), query
                    )
                    
                    st.write("### Response:")
                    st.markdown(response)
                    
                    with st.expander("View relevant context"):
                        for i, chunk in enumerate(relevant_chunks, 1):
                            st.markdown(f"**Chunk {i}:**\n{chunk}")
                else:
                    st.warning("No relevant information found in the document.")
            
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
else:
    st.info("👈 Please upload a document to get started")

# Chat interface
st.title("💬 Document Chat")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message.role):
        st.write(message.content)
        if message.contexts:
            with st.expander("View source context"):
                for i, context in enumerate(message.contexts, 1):
                    st.markdown(f"**Context {i}:**\n{context}")

# Chat input
if prompt := st.chat_input("What would you like to know?"):
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
        st.session_state.messages.append(response)
