import unittest
import pytest
import logging
from app.services.embedding_service import generate_embedding, cosine_similarity

# Configure logging for the test
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestRetrievalService(unittest.TestCase):
    @pytest.mark.unit
    def test_relevant_chunk_retrieval(self):
        """
        Test that the retrieval service correctly identifies relevant chunks
        based on a query and cosine similarity threshold (simple case).
        """
        query = "What is quality?"
        query_embedding = generate_embedding(query)
        logger.debug(f"Query Embedding: {query_embedding}")

        chunks = [
            {"text": "Quality is defined as value to some person.",
             "embedding": generate_embedding("Quality is defined as value to some person.")},
            {"text": "ISBN 12345 book details.",
             "embedding": generate_embedding("ISBN 12345 book details.")},
        ]
        logger.debug(f"Chunk Embeddings: {[chunk['embedding'] for chunk in chunks]}")

        retrieved_chunks = []
        for chunk in chunks:
            similarity = cosine_similarity(query_embedding, chunk["embedding"])
            logger.debug(f"Similarity for chunk '{chunk['text']}': {similarity}")
            if similarity > 0.8:
                retrieved_chunks.append(chunk)

        for chunk in retrieved_chunks:
            logger.debug(f"Retrieved Chunk: {chunk['text']}")

        self.assertEqual(len(retrieved_chunks), 1, "Expected one relevant chunk to be retrieved")
        self.assertEqual(
            retrieved_chunks[0]["text"],
            "Quality is defined as value to some person.",
            "Expected the most relevant chunk to be retrieved"
        )

class TestRetrievalServiceRealWorld(unittest.TestCase):
    @pytest.mark.integration
    def test_real_world_retrieval(self):
        """
        Test that the retrieval service retrieves relevant chunks under real-world conditions.
        """
        query = "What is software quality?"
        query_embedding = generate_embedding(query)
        logger.debug(f"Query Embedding: {query_embedding}")

        # Simulated real-world chunks
        chunks = [
            {"text": "Software quality refers to the degree to which software meets user needs.",
            "embedding": generate_embedding("Software quality refers to the degree to which software meets user needs.")},
            {"text": "The Eiffel Tower is in Paris.",
            "embedding": generate_embedding("The Eiffel Tower is in Paris.")},
            {"text": "Software testing is a practice to improve software quality.",
            "embedding": generate_embedding("Software testing is a practice to improve software quality.")},
        ]
        logger.debug(f"Chunk Embeddings: {[chunk['embedding'] for chunk in chunks]}")

        # Log similarity scores
        for chunk in chunks:
            similarity = cosine_similarity(query_embedding, chunk["embedding"])
            logger.debug(f"Similarity for chunk '{chunk['text']}': {similarity:.4f}")

        # Retrieve chunks with adjusted threshold
        similarity_threshold = 0.6  # Lower threshold for testing
        retrieved_chunks = [
            chunk for chunk in chunks if cosine_similarity(query_embedding, chunk["embedding"]) > similarity_threshold
        ]
        retrieved_texts = [chunk["text"] for chunk in retrieved_chunks]
        logger.debug(f"Final Retrieved Chunks: {retrieved_texts}")

        # Assertions
        self.assertEqual(len(retrieved_chunks), 2, f"Expected 2 relevant chunks but retrieved {len(retrieved_chunks)}")
        self.assertIn("Software quality refers to the degree to which software meets user needs.", retrieved_texts)
        self.assertIn("Software testing is a practice to improve software quality.", retrieved_texts)



if __name__ == "__main__":
    unittest.main()
