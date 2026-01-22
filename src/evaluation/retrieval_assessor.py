"""
Enhanced Retrieval Quality Assessor
Multi-tier confidence calculation for intelligent document routing
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from pymilvus import MilvusClient

logger = logging.getLogger(__name__)


class RetrievalAssessor:
    """Enhanced retrieval quality assessment with multi-tier confidence scoring"""
    
    def __init__(self, milvus_client: MilvusClient, collection_name: str, embedding_model):
        """
        Initialize retrieval assessor
        
        Args:
            milvus_client: Zilliz/Milvus client for accessing precomputed embeddings
            collection_name: Collection name in vector database
            embedding_model: OpenAI embedding model for query processing
        """
        self.client = milvus_client
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self._cached_query_embedding = None
        self._cached_domain_relevance = None
        
    def calculate_enhanced_confidence(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        """
        Calculate enhanced confidence score using multiple quality indicators
        
        Args:
            query: Original search query
            documents: Cohere reranked documents with relevance scores
            
        Returns:
            Dictionary with confidence score and component breakdown
        """
        if not documents:
            return {
                "confidence_score": 0.0,
                "tier": "low",
                "strategy": "web_priority",
                "components": {}
            }
        
        # Cache query embedding for efficiency
        self._cache_query_embedding(query)
        
        # Calculate individual components
        top_doc_score = self._get_top_document_score(documents)
        consistency_score = self._calculate_consistency(documents)
        semantic_score = self._calculate_semantic_alignment(query, documents)
        domain_score = self._get_domain_relevance_score(query)
        diversity_score = self._calculate_result_diversity(documents)
        
        # Weighted confidence calculation
        confidence = (
            top_doc_score * 0.35 +
            consistency_score * 0.25 +
            semantic_score * 0.20 +
            domain_score * 0.15 +
            diversity_score * 0.05
        )
        
        confidence = min(confidence, 1.0)
        
        # Determine tier and strategy
        tier, strategy = self._determine_tier_and_strategy(confidence)
        
        return {
            "confidence_score": confidence,
            "tier": tier,
            "strategy": strategy,
            "components": {
                "top_document_score": top_doc_score,
                "consistency_score": consistency_score,
                "semantic_alignment": semantic_score,
                "domain_relevance": domain_score,
                "result_diversity": diversity_score
            }
        }
    
    def _cache_query_embedding(self, query: str) -> None:
        """Cache query embedding to avoid recomputation"""
        if self._cached_query_embedding is None:
            self._cached_query_embedding = self.embedding_model.embed_query(query)
    
    def _get_top_document_score(self, documents: List[Document]) -> float:
        """Extract top document relevance score from Cohere reranking"""
        if not documents:
            return 0.0
        return documents[0].metadata.get('relevance_score', 0.0)
    
    def _calculate_consistency(self, documents: List[Document]) -> float:
        """Calculate consistency across top 3 documents"""
        if not documents:
            return 0.0
        
        top_3_scores = [
            doc.metadata.get('relevance_score', 0.0) 
            for doc in documents[:3]
        ]
        
        if not top_3_scores:
            return 0.0
            
        return sum(top_3_scores) / len(top_3_scores)
    
    def _calculate_semantic_alignment(self, query: str, documents: List[Document]) -> float:
        """
        Calculate semantic alignment using precomputed embeddings
        Optimized version using dot product on L2 normalized embeddings
        """
        if not documents or self._cached_query_embedding is None:
            return 0.0
        
        similarities = []
        
        for doc in documents[:3]:
            doc_id = doc.metadata.get('id')
            if doc_id is None:
                continue
                
            doc_embedding = self._get_document_embedding(doc_id)
            if doc_embedding is not None:
                # Direct dot product since embeddings are L2 normalized
                similarity = float(np.dot(self._cached_query_embedding, doc_embedding))
                similarities.append(max(0.0, similarity))  # Ensure non-negative
        
        if not similarities:
            return 0.0
        
        # Weighted average emphasizing top documents
        weights = [0.5, 0.3, 0.2][:len(similarities)]
        weighted_sim = sum(sim * weight for sim, weight in zip(similarities, weights))
        
        return min(weighted_sim, 1.0)
    
    def _get_document_embedding(self, doc_id: str) -> Optional[np.ndarray]:
        """Fetch precomputed embedding from Zilliz"""
        try:
            # Properly escape document ID for Milvus query
            # Use double quotes to handle special characters in document IDs
            escaped_id = doc_id.replace('"', '\\"')  # Escape any existing quotes
            result = self.client.query(
                collection_name=self.collection_name,
                filter=f'id == "{escaped_id}"',
                output_fields=["vector"]
            )
            
            if result and len(result) > 0:
                vector = result[0].get("vector")
                return np.array(vector) if vector else None
            return None
            
        except Exception as e:
            logger.warning(f"Could not fetch embedding for document {doc_id}: {e}")
            return None
    
    def _get_domain_relevance_score(self, query: str) -> float:
        """
        Get domain relevance score using cached result from query relevance checker
        This should be called after the relevance check in the main pipeline
        """
        if self._cached_domain_relevance is not None:
            return self._cached_domain_relevance
        
        # Fallback: assume high domain relevance if query passed initial check
        return 0.9
    
    def set_domain_relevance_score(self, score: float) -> None:
        """Cache domain relevance score from query relevance checker"""
        self._cached_domain_relevance = score
    
    def _calculate_result_diversity(self, documents: List[Document]) -> float:
        """
        Calculate diversity using precomputed embeddings
        Measures how different the retrieved documents are from each other
        """
        if len(documents) < 2:
            return 0.0
        
        doc_embeddings = []
        for doc in documents[:5]:
            doc_id = doc.metadata.get('id')
            if doc_id is None:
                continue
                
            embedding = self._get_document_embedding(doc_id)
            if embedding is not None:
                doc_embeddings.append(embedding)
        
        if len(doc_embeddings) < 2:
            return 0.0
        
        # Calculate pairwise similarities using dot product
        similarities = []
        for i in range(len(doc_embeddings)):
            for j in range(i + 1, len(doc_embeddings)):
                sim = float(np.dot(doc_embeddings[i], doc_embeddings[j]))
                similarities.append(max(0.0, sim))
        
        # Diversity = 1 - average similarity
        avg_similarity = sum(similarities) / len(similarities)
        diversity = 1.0 - avg_similarity
        
        return max(0.0, min(diversity, 1.0))
    
    def _determine_tier_and_strategy(self, confidence: float) -> tuple:
        """Determine confidence tier and routing strategy"""
        if confidence >= 0.85:
            return "high", "local_only"
        elif confidence >= 0.65:
            return "medium", "hybrid"
        else:
            return "low", "web_priority"
    
    def reset_cache(self) -> None:
        """Reset cached values for new query processing"""
        self._cached_query_embedding = None
        self._cached_domain_relevance = None
