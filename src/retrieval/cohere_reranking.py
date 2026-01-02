"""
Cohere Re-ranking Component
Second-pass re-ranking using Cohere's re-ranking model via LangChain
"""

from typing import List
from langchain_core.documents import Document
from langchain_cohere import CohereRerank
import os
from dotenv import load_dotenv

load_dotenv()


class CohereReranking:
    """Re-ranks documents using Cohere's re-ranking model"""
    
    def __init__(self, top_k: int = 5, model: str = "rerank-v3.5"):
        """
        Initialize Cohere re-ranker
        
        Args:
            top_k: Number of top documents to return after re-ranking
            model: Cohere re-ranking model to use
        """
        self.top_k = top_k
        self.cohere_reranker = CohereRerank(
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            model=model
        )
    
    def rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Re-rank documents using Cohere's re-ranking model
        
        Args:
            query: Original search query
            documents: List of documents to re-rank
            
        Returns:
            List of re-ranked documents (top_k)
        """
        try:
            if not documents:
                return []
            
            # If we have fewer documents than top_k, return all
            if len(documents) <= self.top_k:
                return documents
            
            # Apply Cohere re-ranking  
            compressed_docs = self.cohere_reranker.compress_documents(
                documents=documents,
                query=query
            )
            
            # Take only top_k results
            reranked_docs = compressed_docs[:self.top_k]
            
            return reranked_docs
            
        except Exception as e:
            # Fallback: return top documents without re-ranking
            return documents[:self.top_k]
    
    def rerank_with_scores(self, query: str, documents: List[Document]) -> List[tuple]:
        """
        Re-rank documents and return with relevance scores
        
        Args:
            query: Original search query
            documents: List of documents to re-rank
            
        Returns:
            List of (document, relevance_score) tuples
        """
        try:
            if not documents:
                return []
            
            # Get re-ranked documents
            reranked_docs = self.rerank_documents(query, documents)
            
            # Extract relevance scores from metadata if available
            results = []
            for doc in reranked_docs:
                # Cohere adds relevance score to metadata
                relevance_score = doc.metadata.get('relevance_score', 0.0)
                results.append((doc, relevance_score))
            
            return results
            
        except Exception as e:
            # Fallback: return top documents with default scores
            return [(doc, 0.5) for doc in documents[:self.top_k]]
    
    def get_reranking_stats(self, original_docs: List[Document], reranked_docs: List[Document]) -> dict:
        """
        Get statistics about the re-ranking process
        
        Args:
            original_docs: Documents before re-ranking
            reranked_docs: Documents after re-ranking
            
        Returns:
            Dictionary with re-ranking statistics
        """
        return {
            "original_count": len(original_docs),
            "reranked_count": len(reranked_docs),
            "reduction_ratio": len(reranked_docs) / len(original_docs) if original_docs else 0,
            "avg_relevance_score": self._calculate_avg_relevance(reranked_docs)
        }
    
    def _calculate_avg_relevance(self, documents: List[Document]) -> float:
        """Calculate average relevance score from documents"""
        if not documents:
            return 0.0
        
        scores = [
            doc.metadata.get('relevance_score', 0.0) 
            for doc in documents
        ]
        
        return sum(scores) / len(scores) if scores else 0.0
