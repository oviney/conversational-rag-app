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
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # High-quality embedding model
        self.index = None
        self.chunks = []
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def create_index(self, chunks: List[str]):
        logging.debug("Creating index for document chunks.")
        try:
            self.chunks = chunks
            if not chunks:
                self.chunks = []  # Ensure chunks is set to an empty list
                self.index = faiss.IndexFlatL2(384)  # Reinitialize to an empty index
                return
            embeddings = self.model.encode(chunks, convert_to_tensor=True)  # Use CPU
            embeddings = embeddings.cpu().detach().numpy()  # Move to CPU and convert to NumPy array
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
            logging.debug("Index created successfully with embeddings.")
        except Exception as e:
            logging.error(f"Index creation error: {str(e)}")
            raise

    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[str]:
        if self.index is None or not self.chunks:
            raise ValueError("Index has not been created or loaded.")
        query_embedding = self.model.encode([query], convert_to_tensor=True)
        query_embedding = query_embedding.cpu().numpy()
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding, top_k)
        relevant_chunks = [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
        logging.debug(f"Retrieved Chunks for Query '{query}': {relevant_chunks}")
        return relevant_chunks
