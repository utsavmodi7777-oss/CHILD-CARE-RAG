"""
Enhanced CRAG (Corrective Retrieval Augmented Generation) Component
Multi-tier routing with intelligent web search fallback capabilities
"""

import logging
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)


class CRAGFallback:
    """Enhanced CRAG implementation with multi-tier confidence routing"""
    
    def __init__(
        self, 
        high_confidence_threshold: float = 0.85,
        low_confidence_threshold: float = 0.65,
        max_web_results: int = 5
    ):
        """
        Initialize enhanced CRAG fallback system
        
        Args:
            high_confidence_threshold: Threshold for local-only strategy
            low_confidence_threshold: Threshold for web-priority strategy
            max_web_results: Maximum number of web search results
        """
        self.high_threshold = high_confidence_threshold
        self.low_threshold = low_confidence_threshold
        self.max_web_results = max_web_results
        
        # Initialize Tavily search tool
        try:
            from langchain_community.tools.tavily_search.tool import TavilySearchResults
            self.web_search = TavilySearchResults(
                max_results=max_web_results,
                api_key=os.getenv("TAVILY_API_KEY")
            )
            logger.info(f"Enhanced CRAG initialized with multi-tier thresholds (high: {high_confidence_threshold}, low: {low_confidence_threshold})")
        except Exception as e:
            logger.error(f"Failed to initialize Tavily: {e}")
            self.web_search = None
    
    def evaluate_documents(self, query: str, documents: List[Document], assessment: Dict[str, Any]) -> tuple[List[Document], str]:
        """
        Enhanced document evaluation with multi-tier routing
        
        Args:
            query: Original search query
            documents: Retrieved and re-ranked documents
            assessment: Enhanced confidence assessment from RetrievalAssessor
            
        Returns:
            Tuple of (final_documents, strategy_used)
        """
        if not documents:
            return self._web_search_fallback(query), "web_fallback_no_docs"
        
        confidence = assessment.get("confidence_score", 0.0)
        strategy = assessment.get("strategy", "web_priority")
        
        logger.info(f"CRAG confidence score: {confidence:.3f} (strategy: {strategy})")
        
        if strategy == "local_only":
            return self._use_local_documents(documents), "local_only"
        elif strategy == "hybrid":
            return self._hybrid_approach(query, documents), "hybrid"
        else:
            return self._web_priority_approach(query, documents), "web_priority"
    
    def _use_local_documents(self, documents: List[Document]) -> List[Document]:
        """Use local documents only - high confidence scenario"""
        logger.info("High confidence - using local documents only")
        return documents[:self.max_web_results]
    
    def _hybrid_approach(self, query: str, documents: List[Document]) -> List[Document]:
        """Hybrid approach - combine local docs with targeted web search"""
        logger.info("Medium confidence - using hybrid approach")
        
        # Use top 2-3 local documents as foundation
        local_docs = documents[:3]
        
        # Perform targeted web search
        web_docs = self._targeted_web_search(query, max_results=2)
        
        # Combine with web results first for freshness
        combined_docs = web_docs + local_docs
        
        logger.info(f"Hybrid search: combined {len(web_docs)} web results with {len(local_docs)} local results")
        
        return combined_docs[:self.max_web_results]
    
    def _web_priority_approach(self, query: str, documents: List[Document]) -> List[Document]:
        """Web priority approach - comprehensive web search with minimal local context"""
        logger.info("Low confidence - using web priority approach")
        
        # Comprehensive web search
        web_docs = self._comprehensive_web_search(query, max_results=4)
        
        # Add minimal local context if available
        local_context = documents[:1] if documents else []
        
        combined_docs = web_docs + local_context
        
        logger.info(f"Web priority: {len(web_docs)} web results with {len(local_context)} local context")
        
        return combined_docs[:self.max_web_results]
    
    def _targeted_web_search(self, query: str, max_results: int = 2) -> List[Document]:
        """Perform focused web search for hybrid approach"""
        if not self.web_search:
            return []
        
        try:
            # Modify query for more current/specific information
            enhanced_query = f"{query} latest guidelines 2024 2025"
            
            web_results = self.web_search.run(enhanced_query)
            web_docs = []
            
            for result in web_results[:max_results]:
                web_docs.append(Document(
                    page_content=result.get('content', ''),
                    metadata={
                        'source': f"Web Search: {result.get('title', 'Unknown')}",
                        'url': result.get('url', ''),
                        'page': 'Web Result',
                        'relevance_score': 0.85
                    }
                ))
            
            return web_docs
            
        except Exception as e:
            logger.error(f"Targeted web search failed: {e}")
            return []
    
    def _comprehensive_web_search(self, query: str, max_results: int = 4) -> List[Document]:
        """Perform comprehensive web search for web priority approach"""
        if not self.web_search:
            return []
        
        try:
            web_results = self.web_search.run(query)
            web_docs = []
            
            for result in web_results[:max_results]:
                web_docs.append(Document(
                    page_content=result.get('content', ''),
                    metadata={
                        'source': f"Web Search: {result.get('title', 'Unknown')}",
                        'url': result.get('url', ''),
                        'page': 'Web Result',
                        'relevance_score': 0.80
                    }
                ))
            
            return web_docs
            
        except Exception as e:
            logger.error(f"Comprehensive web search failed: {e}")
            return []
    
    def _web_search_fallback(self, query: str) -> List[Document]:
        """Fallback web search when no local documents available"""
        logger.warning("No local documents available - performing fallback web search")
        return self._comprehensive_web_search(query, self.max_web_results)
    
    def get_strategy_metadata(self, strategy: str, web_results_count: int = 0, local_results_count: int = 0) -> Dict[str, Any]:
        """
        Get metadata about the strategy used for frontend display
        
        Args:
            strategy: Strategy used (local_only, hybrid, web_priority)
            web_results_count: Number of web search results used
            local_results_count: Number of local documents used
            
        Returns:
            Dictionary with strategy metadata for frontend
        """
        return {
            "strategy": strategy,
            "web_search_used": web_results_count > 0,
            "web_results_count": web_results_count,
            "local_results_count": local_results_count,
            "confidence_tiers": {
                "high_threshold": self.high_threshold,
                "low_threshold": self.low_threshold
            }
        }
