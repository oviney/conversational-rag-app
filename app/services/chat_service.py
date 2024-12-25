import logging
from datetime import datetime
from app.services.rag_service import RAGService
from app.services.retrieval_service import RetrievalService
from app.services.generation_service import GenerationService
from app.models.chat_message import ChatMessage


class ChatService:
    def __init__(self, generation_service, rag_service):
        self.generation_service = generation_service
        self.rag_service = rag_service
        
    def process_message(self, prompt, document_chunks):
        if document_chunks:
            try:
                response, contexts = self.rag_service.process_query(prompt, document_chunks)
            except ValueError as e:
                logging.error(f"Error processing query with RAG service: {str(e)}")
                raise ValueError("Index has not been created or loaded.")
            logging.debug(f"RAG response: {response}")
            return ChatMessage(
                content=response,
                timestamp=datetime.now(),
                role='assistant',
                contexts=contexts,
                requires_rag=True
            )
        else:
            response = self.generation_service.generate_text("", prompt)
            logging.debug(f"Generated response: {response}")
            return ChatMessage(
                content=response,
                timestamp=datetime.now(),
                role='assistant',
                contexts=[],
                requires_rag=False
            )