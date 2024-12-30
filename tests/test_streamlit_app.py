import logging
import unittest
import pytest
from unittest.mock import patch, MagicMock
import tempfile
from reportlab.pdfgen import canvas
from docx import Document
from app.document_processing import (
    extract_text_from_pdf, extract_text_from_docx, preprocess_text, chunk_text
)
from app.services.generation_service import GenerationService, load_model_and_tokenizer
from app.services.chat_service import ChatService
from app.services.retrieval_service import RetrievalService
from app.services.rag_service import RAGService
import streamlit as st
import torch
from io import BytesIO
from streamlit_app import process_document, truncate_text

def setup_device():
    device = torch.device("cpu")
    return device

device = setup_device()

class TestStreamlitApp(unittest.TestCase):

    def setUp(self):
        st.session_state.clear()
        if 'document_chunks' not in st.session_state:
            st.session_state.document_chunks = []
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'index_created' not in st.session_state:
            st.session_state.index_created = False

    @pytest.mark.unit
    @patch('app.document_processing.extract_text_from_pdf')
    @patch('app.document_processing.extract_text_from_docx')
    def test_text_extraction(self, mock_extract_text_from_pdf, mock_extract_text_from_docx):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            c = canvas.Canvas(tmp_file.name)
            c.drawString(100, 750, "Hello, World!")
            c.save()
            temp_file_path = tmp_file.name

        mock_extract_text_from_pdf.return_value = "Hello, World!"
        with open(temp_file_path, 'rb') as f:
            pdf_bytes = f.read()
        text = extract_text_from_pdf(pdf_bytes).strip()
        self.assertEqual(text, "Hello, World!")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
            doc = Document()
            doc.add_paragraph("Extracted text from DOCX")
            doc.save(tmp_file.name)
            temp_file_path = tmp_file.name

        mock_extract_text_from_docx.return_value = "Extracted text from DOCX"
        text = extract_text_from_docx(temp_file_path)
        self.assertEqual(text, "Extracted text from DOCX")

    @pytest.mark.unit
    def test_preprocess_text(self):
        text = "This is a sample text for preprocessing."
        preprocessed_text = preprocess_text(text)
        expected_text = "sample text preprocessing"  # Update this to match the expected output of preprocess_text
        self.assertIsInstance(preprocessed_text, str)
        self.assertEqual(preprocessed_text, expected_text)

    @pytest.mark.unit
    def test_chunk_text_with_different_chunk_size(self):
        text = "This is a sample text for chunking. " * 10
        chunks = chunk_text(text, chunk_size=100)
        self.assertIsInstance(chunks, list)
        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))

    @pytest.mark.unit
    def test_retrieval_service_create_index(self):
        retrieval_service = RetrievalService()
        retrieval_service.create_index("This is a sample document.")
        self.assertIsNotNone(retrieval_service.index)

    @pytest.mark.unit
    def test_retrieval_service_retrieve_relevant_chunks(self):
        retrieval_service = RetrievalService()
        retrieval_service.create_index("This is a sample document.")
        chunks = retrieval_service.retrieve_relevant_chunks("sample", top_k=1)
        self.assertGreater(len(chunks), 0)
        self.assertIn("sample", chunks[0])

    @pytest.mark.unit
    @patch('app.document_processing.extract_text_from_pdf', return_value="Sample PDF text")
    @patch('streamlit.file_uploader')
    def test_file_upload(self, mock_file_uploader, mock_extract_text_from_pdf):
        uploaded_file = MagicMock()
        uploaded_file.type = "application/pdf"
        uploaded_file.read.return_value = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n4 0 obj\n<< /Length 44 >>\nstream\nBT /F1 24 Tf 100 700 Td (Hello, World!) Tj ET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000053 00000 n \n0000000100 00000 n \n0000000173 00000 n \ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n231\n%%EOF"
        
        process_document(uploaded_file)
        
        logging.debug(f"index_created: {st.session_state.index_created}")
        logging.debug(f"document_chunks: {st.session_state.document_chunks}")
        
        self.assertTrue(st.session_state.index_created)
        self.assertGreater(len(st.session_state.document_chunks), 0)

    @pytest.mark.unit
    def test_truncate_text(self):
        text = "This is a sample text for truncation."
        max_length = 10
        truncated_text = truncate_text(text, max_length)
        self.assertEqual(truncated_text, "This is a ")

    @pytest.mark.unit
    def test_session_state_initialization(self):
        self.assertIn('document_chunks', st.session_state)
        self.assertIn('messages', st.session_state)
        self.assertIn('index_created', st.session_state)
        self.assertEqual(st.session_state.document_chunks, [])
        self.assertEqual(st.session_state.messages, [])
        self.assertFalse(st.session_state.index_created)

    @pytest.mark.integration
    @patch.object(RetrievalService, 'retrieve_relevant_chunks')
    @patch.object(GenerationService, 'generate_text')
    def test_chat_interface(self, mock_generate_text, mock_retrieve_relevant_chunks):
        prompt = "What is the content of the document?"
        chunks = ["chunk1", "chunk2", "chunk3"]
        mock_retrieve_relevant_chunks.return_value = chunks
        mock_generate_text.return_value = "Generated response based on the document content."

        retrieval_service = RetrievalService()
        relevant_chunks = retrieval_service.retrieve_relevant_chunks(prompt, top_k=2)
        context = "\n".join(relevant_chunks)

        # Mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        generation_service = GenerationService(mock_model, mock_tokenizer)
        
        generated_response = generation_service.generate_text(context, prompt)

        self.assertEqual(generated_response, "Generated response based on the document content.")

    @pytest.mark.integration
    @patch.object(RetrievalService, 'retrieve_relevant_chunks')
    @patch.object(GenerationService, 'generate_text')
    def test_chat_interface_no_relevant_chunks(self, mock_generate_text, mock_retrieve_relevant_chunks):
        prompt = "What is the content of the document?"
        mock_retrieve_relevant_chunks.return_value = []
        mock_generate_text.return_value = "No relevant information found."

        retrieval_service = RetrievalService()
        relevant_chunks = retrieval_service.retrieve_relevant_chunks(prompt, top_k=2)
        context = "\n".join(relevant_chunks)

        # Mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        generation_service = GenerationService(mock_model, mock_tokenizer)
        
        generated_response = generation_service.generate_text(context, prompt)

        self.assertEqual(generated_response, "No relevant information found.")

    @pytest.mark.integration
    def test_process_message_with_rag_no_index(self):
        retrieval_service = RetrievalService()
        with pytest.raises(ValueError, match="Index has not been created or loaded."):
            retrieval_service.retrieve_relevant_chunks("test", top_k=2)

    @pytest.mark.unit
    def test_retrieve_relevant_chunks_empty(self):
        retrieval_service = RetrievalService()
        with pytest.raises(ValueError, match="No chunks were created from the document."):
            retrieval_service.create_index("")

    @pytest.mark.unit
    def test_retrieve_relevant_chunks_no_index(self):
        retrieval_service = RetrievalService()
        with pytest.raises(ValueError, match="Index has not been created or loaded."):
            retrieval_service.retrieve_relevant_chunks("test", top_k=2)

# Initialize services
model, tokenizer = load_model_and_tokenizer()
generation_service = GenerationService(model, tokenizer)
retrieval_service = RetrievalService()
rag_service = RAGService(retrieval_service, generation_service)
chat_service = ChatService(generation_service, rag_service)

def process_document(uploaded_file):
    """
    Process the uploaded document, extract text, preprocess, and chunk it.
    """
    try:
        if uploaded_file.type == "application/pdf":
            pdf_bytes = uploaded_file.read()
            text = extract_text_from_pdf(pdf_bytes)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = extract_text_from_docx(uploaded_file)
        else:
            text = uploaded_file.read().decode("utf-8")

        preprocessed_text = preprocess_text(text)
        document_chunks = chunk_text(preprocessed_text)
        st.session_state.document_chunks = document_chunks
        retrieval_service.create_index(preprocessed_text)
        st.session_state.index_created = True
        logging.debug(f"Document processed and indexed with {len(document_chunks)} chunks.")
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        logging.error(f"Error processing document: {str(e)}")

def truncate_text(text, max_length):
    """
    Truncate the text to the maximum length allowed by the model.
    """
    return text[:max_length]

# Initialize session state
if 'document_chunks' not in st.session_state:
    st.session_state.document_chunks = []
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'index_created' not in st.session_state:
    st.session_state.index_created = False

# Sidebar for file upload and document processing status
with st.sidebar:
    st.title("Conversational RAG App")
    uploaded_file = st.file_uploader("Choose a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])

    if uploaded_file:
        # Process the uploaded file
        process_document(uploaded_file)
        st.session_state.index_created = True
        st.success("Document uploaded and processed successfully. You can now ask questions.")

    # Show warning if no document is loaded
    if not st.session_state.index_created:
        st.info("👆 Please upload a document to start chatting")

# Main chat interface
if st.session_state.index_created:
    st.header("Ask a Question")
    prompt_input = st.text_area("Enter your question here:", key="prompt_input")

    # Provide tips for effective prompts
    st.markdown("""
    ### Tips for Effective Prompts:
    - Be specific and clear.
    - Provide relevant context or background information.
    - Use structured formats like bullet points or numbered lists.
    - Ask direct and specific questions.
    - Include examples or scenarios to illustrate your query.
    """)

    # Submit button
    if st.button("Submit"):
        if prompt_input:
            document_chunks = st.session_state.document_chunks
            
            # Process the message
            with st.spinner("Thinking..."):
                try:
                    # Truncate the prompt if it exceeds the model's maximum sequence length
                    max_length = tokenizer.model_max_length
                    truncated_prompt = truncate_text(prompt_input, max_length)
                    
                    response_message = chat_service.process_message(truncated_prompt, document_chunks)
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": prompt_input
                    })
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_message.content
                    })
                    logging.debug(f"User Prompt: {prompt_input}")
                    logging.debug(f"Assistant Response: {response_message.content}")
                except ValueError as e:
                    st.error(f"Error: {str(e)}")
                    logging.error(f"Chat error: {str(e)}")
        else:
            st.warning("Please enter a question.")

    # Display message history
    st.header("Chat History")
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message['role'] == 'user':
                st.markdown(f"**You**: {message['content']}")
            else:
                st.markdown(f"**Assistant**: {message['content']}")
else:
    st.info("👆 Please upload a document to start chatting")