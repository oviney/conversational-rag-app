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

# --- Sidebar ---
st.sidebar.title("Upload Document")
uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"]
)

# --- Main UI ---
st.title("Conversational RAG Application")
st.write("Upload a document to get started.")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    if file_extension == '.txt':
        text = uploaded_file.getvalue().decode("utf-8")
    else:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=file_extension
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name

        if file_extension == '.pdf':
            text = extract_text_from_pdf(temp_file_path)
        elif file_extension == '.docx':
            text = extract_text_from_docx(temp_file_path)
        else:
            raw_text = uploaded_file.getvalue()
            result = chardet.detect(raw_text)
            encoding = result['encoding']
            text = raw_text.decode(encoding)

    # Preprocess and chunk text
    text = preprocess_text(text)
    chunks = chunk_text(text)

    # Create FAISS index with caching
    @st.cache
    def cached_create_index(chunks):
        retrieval_service = RetrievalService()
        return retrieval_service.create_index(chunks)
    
    index = cached_create_index(chunks)

    # Chat interface
    st.write(
        "Document uploaded successfully! You can now ask questions about the document."
    )
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if index is not None:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                retrieval_service = RetrievalService()
                retrieval_service.index = index  # Ensure the index is set
                relevant_chunks = retrieval_service.retrieve_relevant_chunks(
                    prompt, top_k=5
                )
                context = "\n".join(relevant_chunks)
                generation_service = GenerationService()
                generated_response = generation_service.generate_text(context, prompt)

                for chunk in generated_response.split():
                    full_response += chunk + " "
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
