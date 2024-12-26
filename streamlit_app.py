# Main Streamlit application
import streamlit as st
import os
from app.document_processing import (
    extract_text_from_pdf, preprocess_text, chunk_text
)
from app.services.retrieval_service import RetrievalService
from app.services.generation_service import GenerationService, load_model_and_tokenizer
from app.services.chat_service import ChatService
from app.services.rag_service import RAGService
from app.models.chat_message import ChatMessage
from datetime import datetime
import tempfile
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure the root logger
# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Set the logging level for specific loggers to ERROR to reduce log output
logging.getLogger('watchdog.observers.inotify_buffer').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('faiss.loader').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('pdfminer').setLevel(logging.ERROR)  # Control all 'pdfminer' related logs

# Example log message to verify logging is working
logging.debug("Logging is configured correctly.")

def setup_device():
    device = torch.device("cpu")
    return device

def load_model_and_tokenizer(model_name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to("cpu")  # Explicitly move the model to CPU
    return model, tokenizer

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer()

# Initialize services
retrieval_service = RetrievalService()
generation_service = GenerationService(model, tokenizer)
rag_service = RAGService(retrieval_service, generation_service)
chat_service = ChatService(generation_service, rag_service)

# Initialize session state
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_service" not in st.session_state:
        st.session_state.chat_service = chat_service
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
        
        # Join chunks into a single string before creating the index
        document_text = ' '.join(st.session_state.document_chunks)
        
        # Create index with chunks
        st.session_state.chat_service.rag_service.retrieval_service.create_index(document_text)
        st.session_state.index_created = True
        
        # Mark document as processed
        st.session_state.document_processed = True
        
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
