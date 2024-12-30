import streamlit as st
import logging
from app.services.chat_service import ChatService
from app.services.generation_service import GenerationService, load_model_and_tokenizer
from app.services.retrieval_service import RetrievalService
from app.services.rag_service import RAGService
from app.document_processing import extract_text_from_pdf, extract_text_from_docx, preprocess_text, chunk_text

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
    st.session_state.index_created = False
    try:
        if uploaded_file.type == "application/pdf":
            pdf_bytes = uploaded_file.read()
            text = extract_text_from_pdf(pdf_bytes)
            logging.debug(f"Extracted text from PDF: {text}")
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = extract_text_from_docx(uploaded_file)
            logging.debug(f"Extracted text from DOCX: {text}")
        else:
            text = uploaded_file.read().decode("utf-8")
            logging.debug(f"Extracted text from TXT: {text}")

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