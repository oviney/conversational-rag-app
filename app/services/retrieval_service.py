import warnings
from sentence_transformers import SentenceTransformer
import faiss
import torch
import logging
from typing import List

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="reportlab")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="faiss")


class RetrievalService:
    """
    RetrievalService is a class that provides methods to create an index of text chunks and retrieve relevant chunks
    based on a query using a pre-trained SentenceTransformer model and FAISS for efficient similarity search.
    Attributes:
        model (SentenceTransformer): The pre-trained SentenceTransformer model used for encoding text.
        chunks (list): A list of text chunks to be indexed and searched.
        index (faiss.IndexFlatL2): The FAISS index used for similarity search.
    """
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Use CPU
        self.chunks = []
        self.index = faiss.IndexFlatL2(384)  # Assuming the embedding size is 384
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def create_index(self, chunks: List[str]):
        logging.debug("Creating index for document chunks.")
        self.chunks = chunks
        if not chunks:
            self.chunks = []  # Ensure chunks is set to an empty list
            self.index = faiss.IndexFlatL2(384)  # Reinitialize to an empty index
            return
        embeddings = self.model.encode(chunks, convert_to_tensor=True)  # Use CPU
        embeddings = embeddings.cpu().detach().numpy()  # Move to CPU and convert to NumPy array
        self.index.add(embeddings)
        logging.debug("Index created successfully.")

    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[str]:
        if self.index.ntotal == 0 or not self.chunks:  # Check if index or chunks are empty
            raise ValueError("Index has not been created or loaded.")
        query_embedding = self.model.encode([query], convert_to_tensor=True)  # Use CPU
        query_embedding = query_embedding.cpu().numpy()  # Move to CPU and convert to NumPy array
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding, top_k)
        relevant_chunks = [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
        return relevant_chunks
