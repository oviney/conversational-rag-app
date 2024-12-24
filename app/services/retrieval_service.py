from app.config import Config
# Handles text retrieval using FAISS
from sentence_transformers import SentenceTransformer
import faiss


class RetrievalService:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunks = []
        self.index = None

    def create_index(self, chunks):
        if not chunks:
            self.index = None
            return None
        self.chunks = chunks
        embeddings = self.model.encode(chunks)
        dimension = embeddings.shape[1]
        
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        return self.index

    def save_index(self):
        if self.index:
            faiss.write_index(self.index, Config.FAISS_INDEX_FILE)

    def load_index(self):
        if Config.FAISS_INDEX_FILE:
            self.index = faiss.read_index(Config.FAISS_INDEX_FILE)

    def retrieve_relevant_chunks(self, query, top_k=5):
        if not self.index:
            raise ValueError("Index has not been created or loaded.")
        if not self.chunks:
            return []
            
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))
        
        return [self.chunks[int(idx)] for idx in indices[0] if 0 <= idx < len(self.chunks)]
