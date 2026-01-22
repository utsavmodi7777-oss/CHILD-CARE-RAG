"""
Reciprocal Rank Fusion (RRF) Component
Combines multiple retrieval results using RRF scoring
"""

from typing import List, Tuple, Dict, Any
from langchain_core.documents import Document


class ReciprocalRankFusion:
    """Implements Reciprocal Rank Fusion for combining multiple retrieval results"""
    
    def __init__(self, k: int = 60):
        """
        Initialize RRF with parameter k
        
        Args:
            k: RRF parameter for 1/(rank + k) scoring (default: 60)
        """
        self.k = k
    
    def fuse_results(self, retrieval_results: List[List[Document]], max_results: int = 40) -> List[Tuple[Document, float]]:
        """
        Apply Reciprocal Rank Fusion to combine multiple retrieval results
        
        Args:
            retrieval_results: List of document lists from different queries
            max_results: Maximum number of results to return
            
        Returns:
            List of (document, fused_score) tuples sorted by score
        """
        fused_scores = {}
        
        for result_list in retrieval_results:
            for rank, doc in enumerate(result_list):
                # Create unique document identifier
                doc_id = self._get_document_id(doc)
                
                # Initialize if first time seeing this document
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = {
                        'doc': doc,
                        'score': 0.0,
                        'appearances': 0
                    }
                
                # Add RRF score: 1 / (rank + k)
                fused_scores[doc_id]['score'] += 1.0 / (rank + self.k)
                fused_scores[doc_id]['appearances'] += 1
        
        # Sort by fused score (descending) and take top results
        sorted_results = sorted(
            fused_scores.values(), 
            key=lambda x: x['score'], 
            reverse=True
        )
        
        # Return top max_results as (document, score) tuples
        return [
            (item['doc'], item['score']) 
            for item in sorted_results[:max_results]
        ]
    
    def _get_document_id(self, doc: Document) -> str:
        """
        Generate unique identifier for document
        
        Args:
            doc: Document to identify
            
        Returns:
            Unique string identifier
        """
        # Try to use existing ID from metadata
        if 'id' in doc.metadata:
            return str(doc.metadata['id'])
        
        # Try to use binary_hash if available
        if 'binary_hash' in doc.metadata:
            return str(doc.metadata['binary_hash'])
        
        # Fallback: use content hash
        return str(hash(doc.page_content[:100]))  # Use first 100 chars for uniqueness
    
    def get_fusion_stats(self, fused_results: List[Tuple[Document, float]]) -> Dict[str, Any]:
        """
        Get statistics about the fusion process
        
        Args:
            fused_results: Results from fuse_results method
            
        Returns:
            Dictionary with fusion statistics
        """
        if not fused_results:
            return {"total_docs": 0, "avg_score": 0.0, "max_score": 0.0, "min_score": 0.0}
        
        scores = [score for _, score in fused_results]
        
        return {
            "total_docs": len(fused_results),
            "avg_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "score_range": max(scores) - min(scores)
        }
