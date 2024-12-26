import unittest
import pytest
from app.services.retrieval_service import RetrievalService

class TestRetrievalService(unittest.TestCase):
    def setUp(self):
        self.retrieval_service = RetrievalService()
        self.document = (
            "In the software world, we often talk about quality. "
            "Quality can mean different things to different people. "
            "For some, it means fewer bugs. For others, it means a great user experience. "
            "In this document, we will explore the various aspects of quality in software development."
        )
        self.query = "What is quality?"

    @pytest.mark.unit
    def test_chunk_document(self):
        chunks = self.retrieval_service.chunk_document(self.document, chunk_size=10)
        self.assertEqual(len(chunks), 5)
        self.assertIn("In the software world, we often talk about quality.", chunks[0]["text"])

    @pytest.mark.unit
    def test_create_index(self):
        self.retrieval_service.create_index(self.document)
        self.assertIsNotNone(self.retrieval_service.index)

    @pytest.mark.unit
    def test_retrieve_relevant_chunks(self):
        self.retrieval_service.create_index(self.document)
        relevant_chunks = self.retrieval_service.retrieve_relevant_chunks(self.query, top_k=2)
        self.assertTrue(any("quality" in chunk for chunk in relevant_chunks))
        self.assertTrue(any("In the software world, we often talk about quality." in chunk for chunk in relevant_chunks))

    @pytest.mark.unit
    def test_is_relevant_chunk(self):
        chunk = "In the software world, we often talk about quality."
        self.assertTrue(self.retrieval_service.is_relevant_chunk(chunk, self.query))
        self.assertFalse(self.retrieval_service.is_relevant_chunk("This is irrelevant.", self.query))

if __name__ == "__main__":
    unittest.main()