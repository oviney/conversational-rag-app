import logging
from app.services.generation_service import GenerationService
from app.services.retrieval_service import RetrievalService

class RAGService:
    def __init__(self, retrieval_service, generation_service):
        self.retrieval_service = retrieval_service
        self.generation_service = generation_service
        
    def process_query(self, query, chunks, top_k=3):
        logging.debug(f"Processing query: {query}")
        if not chunks:
            raise ValueError("No document chunks available")
        
        try:
            relevant_chunks = self.retrieval_service.retrieve_relevant_chunks(query, top_k) or []
            if not relevant_chunks:
                return "No relevant information found.", []
            
            # Summarize the retrieved chunks
            summarized_chunks = self.summarize_chunks(relevant_chunks)
            response = self.generation_service.generate_text(summarized_chunks, query)
            
            logging.debug(f"Generated response: {response}")
            return response, relevant_chunks
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            raise ValueError(str(e))

    def summarize_chunks(self, chunks):
        """
        Summarize the retrieved chunks into a cohesive context.

        Args:
            chunks (List[str]): The list of retrieved chunks.

        Returns:
            str: The summarized context.
        """
        prompt = "Summarize the following retrieved information into a cohesive answer:\n" + "\n".join(chunks)
        response = self.generation_service.generate_text("", prompt)
        return response