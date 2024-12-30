# Centralized configuration
import os
import logging

class Config:
    CHUNK_SIZE = 512
    OVERLAP = 50
    EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'
    GENERATION_MODEL = 'gpt2'
    FAISS_INDEX_FILE = os.getenv("FAISS_INDEX_FILE", "faiss_index.bin")
    CACHE_DIR = os.getenv("CACHE_DIR", "/tmp/cache")

# Ensure the logs directory exists
os.makedirs("./logs", exist_ok=True)

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("./logs/app.log")
    ]
)

# Example usage
logger = logging.getLogger(__name__)
logger.debug("Logging is configured and debug logs are enabled.")
