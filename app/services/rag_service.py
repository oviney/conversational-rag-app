import logging
from app.services.generation_service import GenerationService
from app.services.retrieval_service import RetrievalService

class RAGService:
    def __init__(self, retrieval_service: RetrievalService, generation_service: GenerationService):
        self.retrieval_service = retrieval_service
        self.generation_service = generation_service
        
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
