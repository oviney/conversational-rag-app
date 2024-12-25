# Main Streamlit application
import streamlit as st
from app.document_processing import (
    extract_text_from_pdf, preprocess_text, chunk_text
)
from app.services.chat_service import ChatService
import torch
import logging

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,  # Adjust to INFO or WARNING to reduce unwanted logs
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("./logs/app.log"),  # Log to a file
        logging.StreamHandler()  # Log to console
    ]
)

# Suppress unwanted loggers
unwanted_loggers = [
    "watchdog.observers",
    "torch",
    "transformers",
    "faiss.loader",
    "sentence_transformers"
]

for logger_name in unwanted_loggers:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# Debug logs for verification
logging.debug("Logging configuration updated to suppress unwanted entries.")

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
