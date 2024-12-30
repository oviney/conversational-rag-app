import warnings
from sentence_transformers import SentenceTransformer, util
import faiss
import torch
import logging
from typing import List, Dict
import numpy as np
import spacy
import streamlit as st

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="reportlab")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="faiss")

# Load Spacy English model
nlp = spacy.load("en_core_web_sm")

class RetrievalService:
    """
    RetrievalService is a class that provides methods to create an index of text chunks and retrieve relevant chunks
    based on a query using a pre-trained SentenceTransformer model and FAISS for efficient similarity search.

    Attributes:
        model (SentenceTransformer): The pre-trained SentenceTransformer model used for encoding text.
        index (faiss.IndexFlatL2): The FAISS index used for similarity search.
        document_chunks (list): A list of text chunks to be indexed and searched.
        metadata (list): A list of metadata associated with each chunk.
        device (torch.device): The device (CPU or GPU) used for computation.
    """
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the RetrievalService with a pre-trained SentenceTransformer model.

        Args:
            model_name (str): The name of the pre-trained SentenceTransformer model to use.
        """
        self.model = SentenceTransformer(model_name)  # High-quality embedding model
        self.index = None
        self.document_chunks = []
        self.metadata = []
        self.device = torch.device("cpu")
        logging.debug(f"RetrievalService initialized with model {model_name}.")

    def chunk_document(self, document: str, chunk_size: int = 100) -> List[Dict[str, str]]:
        """
        Chunk the document into smaller pieces with metadata.

        Args:
            document (str): The document to be chunked.
            chunk_size (int): The number of words per chunk.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing document chunks and their metadata.
        """
        # Use Spacy to split the document into sentences
        doc = nlp(document)
        sentences = [sent.text for sent in doc.sents]

        # Combine sentences into chunks of approximately chunk_size words
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > chunk_size:
                chunk_text = ' '.join(current_chunk)
                chunks.append({"text": chunk_text, "metadata": {"length": len(chunk_text.split())}})
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # Add the last chunk if it contains any sentences
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({"text": chunk_text, "metadata": {"length": len(chunk_text.split())}})

        self.document_chunks = [chunk["text"] for chunk in chunks]
        self.metadata = [chunk["metadata"] for chunk in chunks]
        logging.debug(f"Document chunked into {len(chunks)} chunks.")
        return chunks

    def create_index(self, document: str):
        """
        Create an index for the document.

        Args:
            document (str): The document to be indexed.
        """
        chunks = self.chunk_document(document)
        if not chunks:
            raise ValueError("No chunks were created from the document.")
        embeddings = self.model.encode([chunk["text"] for chunk in chunks], convert_to_tensor=True)
        embeddings = embeddings.cpu()  # Move embeddings to CPU
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        faiss.normalize_L2(embeddings.numpy())
        self.index.add(embeddings.numpy())
        logging.debug("Index created successfully.")

    def retrieve_relevant_chunks(self, query: str, top_k: int = 2) -> List[str]:
        """
        Retrieve the most relevant chunks for the given query.

        Args:
            query (str): The query to search for.
            top_k (int): The number of top relevant chunks to retrieve.

        Returns:
            List[str]: The list of relevant chunks.
        """
        if self.index is None:
            raise ValueError("Index has not been created or loaded.")
        
        query_embedding = self.model.encode(query, convert_to_tensor=True).cpu()
        query_embedding_np = query_embedding.numpy()
        if query_embedding_np.ndim == 1:
            query_embedding_np = query_embedding_np.reshape(1, -1)
        faiss.normalize_L2(query_embedding_np)
        scores, indices = self.index.search(query_embedding_np, top_k)
        logging.debug(f"Top {top_k} scores: {scores}")
        logging.debug(f"Top {top_k} indices: {indices}")

        retrieved_chunks = [self.document_chunks[idx] for idx in indices[0]]
        logging.debug(f"Retrieved Chunks: {retrieved_chunks}")

        return retrieved_chunks

    def is_relevant_chunk(self, chunk: str, query: str) -> bool:
        """
        Determine if a chunk is relevant to the query.
    
        Args:
            chunk (str): The chunk to check.
            query (str): The query to compare against.
    
        Returns:
            bool: True if the chunk is relevant, False otherwise.
        """
        chunk_embedding = self.model.encode(chunk, convert_to_tensor=True).cpu()
        query_embedding = self.model.encode(query, convert_to_tensor=True).cpu()
        similarity = util.pytorch_cos_sim(chunk_embedding, query_embedding).item()
        logging.debug(f"Chunk: {chunk}")
        logging.debug(f"Query: {query}")
        logging.debug(f"Similarity: {similarity}")
        return similarity > 0.5  # Adjust the threshold as needed