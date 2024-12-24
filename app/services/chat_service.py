from app.services.rag_service import RAGService
from app.services.retrieval_service import RetrievalService
from app.services.generation_service import GenerationService
from app.models.chat_message import ChatMessage
from datetime import datetime
from typing import List

class ChatService:
    def __init__(self):
        self.rag_service = RAGService()
        self.generation_service = GenerationService()
        
    def process_message(self, message: str, chunks: List[str] = None) -> ChatMessage:
        # Determine if RAG is needed
        requires_rag = chunks is not None and self._requires_document_context(message)
        
        if requires_rag:
            response, contexts = self.rag_service.process_query(message, chunks)
            return ChatMessage(
                content=response,
                timestamp=datetime.now(),
                role='assistant',
                contexts=contexts,
                requires_rag=True
            )
        else:
            response = self.generation_service.generate_text("", message)
            return ChatMessage(
                content=response,
                timestamp=datetime.now(),
                role='assistant',
                requires_rag=False
            )
            
    def _requires_document_context(self, message: str) -> bool:
        # Add logic to determine if message needs document context
        document_related_keywords = ['document', 'text', 'content', 'written']
        return any(keyword in message.lower() for keyword in document_related_keywords)