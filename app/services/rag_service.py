from sentence_transformers import SentenceTransformer
import logging
from typing import List
from app.services.generation_service import GenerationService


class RAGService:
    def __init__(self):
        from app.services.retrieval_service import RetrievalService  # Local import to avoid circular dependency
        self.retrieval_service = RetrievalService()
        self.generation_service = GenerationService()
        
    def process_query(self, query, chunks, top_k=3):
        logging.debug(f"Processing query: {query}")
        if not chunks:
            raise ValueError("No document chunks available")
            
        relevant_chunks = self.retrieval_service.retrieve_relevant_chunks(query, top_k)
        if not relevant_chunks:
            return "No relevant information found.", []
            
        context = "\n".join(relevant_chunks)
        response = self.generation_service.generate_text(context, query)
        
        logging.debug(f"Generated response: {response}")
        return response, relevant_chunks
