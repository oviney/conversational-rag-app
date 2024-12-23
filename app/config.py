# Centralized configuration
import os

class Config:
    CHUNK_SIZE = 512
    OVERLAP = 50
    EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'
    GENERATION_MODEL = 'gpt2'
    FAISS_INDEX_FILE = os.getenv("FAISS_INDEX_FILE", "faiss_index.bin")
    CACHE_DIR = os.getenv("CACHE_DIR", "/tmp/cache")