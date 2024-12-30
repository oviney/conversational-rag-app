import unittest
import logging
from app.services.embedding_service import generate_embedding, cosine_similarity

# Configure logging for detailed debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestEmbeddingFunctions(unittest.TestCase):
    def test_generate_embedding(self):
        """
        Test that generate_embedding produces a non-empty embedding for valid text input.
        """
        text = "Quality is defined as value to some person."
        embedding = generate_embedding(text)
        logger.debug(f"Embedding for '{text}': {embedding}")
        
        # Ensure the embedding is not None or empty
        self.assertIsNotNone(embedding, "Embedding should not be None")
        self.assertTrue(len(embedding) > 0, "Embedding should not be empty")
    
    def test_cosine_similarity_identical_embeddings(self):
        """
        Test that cosine similarity between identical embeddings is close to 1.
        """
        text = "Quality is defined as value to some person."
        embedding1 = generate_embedding(text)
        embedding2 = generate_embedding(text)
        similarity = cosine_similarity(embedding1, embedding2)
        logger.debug(f"Similarity between identical embeddings: {similarity}")
        
        # Ensure similarity is close to 1 for identical embeddings
        self.assertGreater(similarity, 0.99, "Expected similarity for identical embeddings to be close to 1")
    
    def test_cosine_similarity_different_embeddings(self):
        """
        Test that cosine similarity between embeddings of different texts is below a threshold.
        """
        text1 = "Quality is defined as value to some person."
        text2 = "ISBN 12345 book details."
        embedding1 = generate_embedding(text1)
        embedding2 = generate_embedding(text2)
        similarity = cosine_similarity(embedding1, embedding2)
        logger.debug(f"Similarity between different embeddings: {similarity}")
        
        # Ensure similarity is below a reasonable threshold for different texts
        self.assertLess(similarity, 0.8, "Expected similarity for different embeddings to be below 0.8")


if __name__ == "__main__":
    unittest.main()
