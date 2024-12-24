from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List

@dataclass
class ChatMessage:
    content: str
    timestamp: datetime
    role: str  # 'user' or 'assistant'
    contexts: Optional[List[str]] = None
    requires_rag: bool = False