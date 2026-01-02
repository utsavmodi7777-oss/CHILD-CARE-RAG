"""
Client Manager
Singleton pattern for managing LLM client instances (Cohere) to prevent cleanup issues
"""

import os
import atexit
from typing import Optional
from langchain_cohere import ChatCohere, CohereEmbeddings


class OpenAIClientManager:
    """Singleton manager for LLM clients to prevent garbage collection issues"""
    
    _instance: Optional['OpenAIClientManager'] = None
    _chat_client: Optional[ChatCohere] = None
    _embedding_client: Optional[CohereEmbeddings] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialized = True
            # Register cleanup on exit
            atexit.register(self.cleanup)
    
    def get_chat_client(self, model: str = "command-r", temperature: float = 0.1) -> ChatCohere:
        """Get or create a ChatCohere client"""
        if self._chat_client is None:
            self._chat_client = ChatCohere(
                model=model,
                cohere_api_key=os.getenv("COHERE_API_KEY"),
                temperature=temperature
            )
        return self._chat_client
    
    def get_embedding_client(self, model: str = "embed-english-v3.0") -> CohereEmbeddings:
        """Get or create a Cohere embeddings client"""
        if self._embedding_client is None:
            self._embedding_client = CohereEmbeddings(
                model=model,
                cohere_api_key=os.getenv("COHERE_API_KEY")
            )
        return self._embedding_client
    
    def cleanup(self):
        """Cleanup all clients properly"""
        if self._chat_client:
            try:
                # Properly close the client if it has a close method
                if hasattr(self._chat_client, 'client') and hasattr(self._chat_client.client, 'close'):
                    self._chat_client.client.close()
            except Exception:
                pass  # Ignore cleanup errors
            self._chat_client = None
        
        if self._embedding_client:
            try:
                # Properly close the client if it has a close method
                if hasattr(self._embedding_client, 'client') and hasattr(self._embedding_client.client, 'close'):
                    self._embedding_client.client.close()
            except Exception:
                pass  # Ignore cleanup errors
            self._embedding_client = None


# Global instance
client_manager = OpenAIClientManager()
