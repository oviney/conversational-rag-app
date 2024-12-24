from app.services.retrieval_service import RetrievalService
from app.services.generation_service import GenerationService

class RAGService:
    def __init__(self):
        self.retrieval_service = RetrievalService()
        self.generation_service = GenerationService()
        
    def process_query(self, query, chunks, top_k=3):
        if not chunks:
            raise ValueError("No document chunks available")
            
        relevant_chunks = self.retrieval_service.retrieve_relevant_chunks(query, top_k)
        if not relevant_chunks:
            return "No relevant information found.", []
            
        context = "\n".join(relevant_chunks)
        response = self.generation_service.generate_text(context, query)
        
        return response, relevant_chunks