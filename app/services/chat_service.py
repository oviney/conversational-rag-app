import logging
from app.services.rag_service import RAGService
from app.services.generation_service import GenerationService
from app.models.chat_message import ChatMessage
from datetime import datetime
from typing import List
from app.services.retrieval_service import RetrievalService

logging.basicConfig(level=logging.DEBUG)

class ChatService:
    def __init__(self):
        self.rag_service = RAGService()
        self.generation_service = GenerationService()
        
    def process_message(self, message: str, chunks: List[str] = None) -> ChatMessage:
        logging.debug(f"Processing message: {message}")
        requires_rag = chunks is not None and self._requires_document_context(message)
        logging.debug(f"Requires RAG: {requires_rag}")
        
        if requires_rag:
            if not self.rag_service.retrieval_service.index or not self.rag_service.retrieval_service.chunks:
                raise ValueError("Index has not been created or loaded.")
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
            response = self.generation_service.generate_text("", message)
            logging.debug(f"Generated response: {response}")
            return ChatMessage(
                content=response,
                timestamp=datetime.now(),
                role='assistant',
                requires_rag=False
            )
            
    def _requires_document_context(self, message: str) -> bool:
        document_related_keywords = ['document', 'text', 'content', 'written']
        return any(keyword in message.lower() for keyword in document_related_keywords)