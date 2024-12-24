from app.config import Config
# Handles text retrieval using FAISS
from sentence_transformers import SentenceTransformer
import faiss


class RetrievalService:
    """
    RetrievalService is a class that provides methods to create an index of text chunks and retrieve relevant chunks based on a query using a pre-trained SentenceTransformer model and FAISS for efficient similarity search.
    Attributes:
        model (SentenceTransformer): The pre-trained SentenceTransformer model used for encoding text.
        chunks (list): A list of text chunks to be indexed and searched.
        index (faiss.IndexFlatL2): The FAISS index used for similarity search.
    Methods:
        __init__():
            Initializes the RetrievalService with a pre-trained SentenceTransformer model and empty chunks and index.
        create_index(chunks):
            Creates a FAISS index from the provided text chunks.
            Args:
                chunks (list): A list of text chunks to be indexed.
            Returns:
                faiss.IndexFlatL2: The created FAISS index.
        retrieve_relevant_chunks(query, top_k=5):
            Retrieves the most relevant text chunks based on the provided query.
            Args:
                query (str): The query string to search for relevant chunks.
                top_k (int, optional): The number of top relevant chunks to retrieve. Defaults to 5.
            Returns:
                list: A list of the most relevant text chunks.
            Raises:
                ValueError: If the index has not been created or loaded.
    """
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunks = []
        self.index = None

    def create_index(self, chunks):
        if not chunks:
            self.index = faiss.IndexFlatL2(1)  # Create an empty index with dimension 1
            return self.index
        self.chunks = chunks
        embeddings = self.model.encode(chunks)
        dimension = embeddings.shape[1]
        
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        return self.index

    def retrieve_relevant_chunks(self, query, top_k=5):
        if not self.index:
            raise ValueError("Index has not been created or loaded.")
        if not self.chunks:
            return []
            
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))
        
        return [self.chunks[int(idx)] for idx in indices[0] if 0 <= idx < len(self.chunks)]