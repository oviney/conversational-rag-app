from app.config import Config
# Handles text retrieval using FAISS
from sentence_transformers import SentenceTransformer
import faiss


class RetrievalService:
    def __init__(self):
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL, device='cpu')
        self.index = None
        self.chunks = []

    def create_index(self, chunks):
        self.chunks = chunks  # Store the chunks for later retrieval
        embeddings = self.model.encode(chunks)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        return self.index

    def save_index(self):
        if self.index:
            faiss.write_index(self.index, Config.FAISS_INDEX_FILE)

    def load_index(self):
        if Config.FAISS_INDEX_FILE:
            self.index = faiss.read_index(Config.FAISS_INDEX_FILE)

    def retrieve_relevant_chunks(self, query, top_k=5):
        if self.index is None:
            raise ValueError("Index has not been created or loaded.")
        query_embedding = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.chunks[int(idx)] for idx in indices[0]]
