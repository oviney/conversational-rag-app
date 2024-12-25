import logging
from datetime import datetime
from app.services.rag_service import RAGService
from app.services.retrieval_service import RetrievalService
from app.services.generation_service import GenerationService
from app.models.chat_message import ChatMessage


class ChatService:
    def __init__(self):
        retrieval_service = RetrievalService()
        generation_service = GenerationService()
        self.rag_service = RAGService(retrieval_service, generation_service)
        
    def process_message(self, message, chunks=None):
        if chunks:
            try:
                response, contexts = self.rag_service.process_query(message, chunks)
            except ValueError as e:
                response = str(e)
                contexts = []
            logging.debug(f"RAG response: {response}")
            return ChatMessage(
                content=response,
                timestamp=datetime.now(),
                role='assistant',
                contexts=contexts,
                requires_rag=True
            )
        else:
            response = self.rag_service.generation_service.generate_text("", message)
            logging.debug(f"Generated response: {response}")
            return ChatMessage(
                content=response,
                timestamp=datetime.now(),
                role='assistant',
                contexts=[],
                requires_rag=False
            )