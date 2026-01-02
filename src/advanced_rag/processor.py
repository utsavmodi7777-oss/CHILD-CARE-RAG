"""
Production-Ready Advanced RAG Processor
Main processing engine for the Childcare RAG System

This module integrates the Advanced RAG Pipeline with Zilliz Cloud vector database
and provides a clean interface for query processing that can be easily used by
the future Streamlit application.
"""

import os
import sys
import time
import asyncio
import logging
from typing import Dict, Any, List
from pymilvus import MilvusClient
from langchain_core.documents import Document
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.client_manager import client_manager

from advanced_rag.pipeline import AdvancedRAGPipeline
from retrieval.query_relevance_checker import QueryRelevanceChecker
from evaluation.retrieval_assessor import RetrievalAssessor
from evaluation.answer_evaluator import AnswerEvaluator
from evaluation.answer_enhancer import AnswerEnhancer

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ZillizRetriever:
    """Custom retriever for Zilliz Cloud integration with async support"""
    
    def __init__(self, client: MilvusClient, collection_name: str, embedding_model):
        self.client = client
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
    async def similarity_search_async(self, query: str, k: int = 20) -> list[Document]:
        """
        Perform similarity search in Zilliz Cloud asynchronously for better performance
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of Document objects
        """
        try:
            query_embedding = await asyncio.to_thread(self.embedding_model.embed_query, query)
            
            search_results = await asyncio.to_thread(
                self.client.search,
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=k,
                output_fields=["*"]
            )
            
            documents = []
            for hits in search_results:
                for hit in hits:
                    content = hit.get('entity', {}).get('content', '')
                    metadata = {
                        'id': hit.get('id'),
                        'distance': hit.get('distance'),
                        'source_file': hit.get('entity', {}).get('source_file', ''),
                        'page_numbers': hit.get('entity', {}).get('page_numbers', []),
                        'headings': hit.get('entity', {}).get('headings', []),
                        'binary_hash': hit.get('entity', {}).get('binary_hash', ''),
                        'relevance_score': 1.0 - hit.get('distance', 1.0)
                    }
                    
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)
            
            logger.info(f"Retrieved {len(documents)} documents for query: {query[:50]}...")
            return documents
            
        except Exception as e:
            logger.error(f"Error in async similarity search: {e}")
            return []
        
class AdvancedRAGProcessor:
    """
    Production-ready Advanced RAG processor
    Main interface for processing queries through the complete Advanced RAG pipeline
    """
    
    def __init__(self):
        """Initialize the Advanced RAG processor"""
        self.client = None
        self.retriever = None
        self.pipeline = None
        self.embedding_model = None
        self.query_checker = None
        self.retrieval_assessor = None
        self.answer_evaluator = None
        self.answer_enhancer = None
        self.is_ready = False
        
        logger.info("Initializing Advanced RAG Processor...")
        
    def initialize(self) -> bool:
        """
        Initialize all components and connections
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Loading OpenAI embedding model...")
            self.embedding_model = client_manager.get_embedding_client()
            logger.info("OpenAI embedding model loaded")
            
            logger.info("Connecting to Zilliz Cloud...")
            self.client = MilvusClient(
                uri=os.getenv("ZILLIZ_URI"),
                token=os.getenv("ZILLIZ_TOKEN")
            )
            
            collections = self.client.list_collections()
            logger.info(f"Connected to Zilliz Cloud. Available collections: {collections}")
            
            collection_name = "childcare_knowledge_base"
            if collection_name not in collections:
                logger.error(f"Collection '{collection_name}' not found!")
                return False
                
            self.retriever = ZillizRetriever(
                client=self.client,
                collection_name=collection_name,
                embedding_model=self.embedding_model
            )
            logger.info("Zilliz retriever initialized")
            
            # Step 4: Initialize Advanced RAG Pipeline
            logger.info("Initializing Advanced RAG Pipeline...")
            self.pipeline = AdvancedRAGPipeline()
            self.pipeline.set_vector_retriever(self.retriever)
            logger.info("Advanced RAG Pipeline initialized")
            
            logger.info("Initializing Query Relevance Checker...")
            self.query_checker = QueryRelevanceChecker()
            logger.info("Query Relevance Checker initialized")
            
            logger.info("Initializing Enhanced Retrieval Assessor...")
            self.retrieval_assessor = RetrievalAssessor(
                milvus_client=self.client,
                collection_name=collection_name,
                embedding_model=self.embedding_model
            )
            self.pipeline.set_retrieval_assessor(self.retrieval_assessor)
            logger.info("Enhanced Retrieval Assessor initialized")
            
            logger.info("Initializing LLM Answer Evaluator...")
            self.answer_evaluator = AnswerEvaluator()
            logger.info("LLM Answer Evaluator initialized")
            
            logger.info("Initializing Answer Enhancer...")
            self.answer_enhancer = AnswerEnhancer()
            logger.info("Answer Enhancer initialized")
            
            required_keys = ['OPENAI_API_KEY', 'COHERE_API_KEY', 'TAVILY_API_KEY']
            missing_keys = [key for key in required_keys if not os.getenv(key)]
            
            if missing_keys:
                logger.error(f"Missing API keys: {missing_keys}")
                return False
            
            logger.info("All API keys verified")
            
            self.is_ready = True
            logger.info("Advanced RAG Processor fully initialized and ready!")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    

    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            status = {
                'system_ready': self.is_ready,
                'components': {
                    'zilliz_connection': self.client is not None,
                    'embedding_model': self.embedding_model is not None,
                    'retriever': self.retriever is not None,
                    'pipeline': self.pipeline is not None
                },
                'api_keys': {
                    'openai': bool(os.getenv('OPENAI_API_KEY')),
                    'cohere': bool(os.getenv('COHERE_API_KEY')),
                    'tavily': bool(os.getenv('TAVILY_API_KEY')),
                    'zilliz': bool(os.getenv('ZILLIZ_TOKEN'))
                }
            }
            
            if self.client:
                try:
                    collections = self.client.list_collections()
                    status['database'] = {
                        'collections': collections,
                        'target_collection': 'childcare_knowledge_base' in collections
                    }
                except Exception as e:
                    status['database'] = {'error': str(e)}
            
            return status
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
        
    async def multi_retrieval_async(self, queries: List[str], hyde_doc: str, retrieval_k: int = 20) -> List[Document]:
        """
        Perform multi-retrieval for a list of queries and HyDE document
        
        Args:
            queries: List of alternative queries
            hyde_doc: HyDE generated document
            retrieval_k: Number of documents to retrieve per query
            
        Returns:
            List of retrieval results for each query
        """
        if not self.is_ready:
            raise Exception("System not initialized. Please call initialize() first.")
            
        all_queries = queries + [hyde_doc]
        
        if self.retriever and hasattr(self.retriever, 'similarity_search_async'):
            retrieval_tasks = [
                self.retriever.similarity_search_async(q, k=retrieval_k)
                for q in all_queries
            ]
            retrieval_results = await asyncio.gather(*retrieval_tasks)
        else:
            retrieval_results = []
            if self.retriever:
                for q in all_queries:
                    docs = self.retriever.similarity_search(q, k=retrieval_k)
                    retrieval_results.append(docs)
            else:
                raise Exception("Vector retriever not available")
                
        return retrieval_results
    
    def apply_rrf_fusion_async(self, retrieval_results: List[List[Document]]) -> List[Document]:
        """
        Apply Reciprocal Rank Fusion to combine retrieval results
        
        Args:
            retrieval_results: List of retrieval result lists
            
        Returns:
            Fused and ranked documents (documents only, scores removed)
        """
        if not self.is_ready:
            raise Exception("System not initialized. Please call initialize() first.")
            
        fused_with_scores = self.pipeline.rrf_fusion.fuse_results(retrieval_results)
        
        return [doc for doc, score in fused_with_scores]
    
    def apply_cohere_reranking_async(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Re-rank documents using Cohere reranker with scores
        
        Args:
            query: Original query
            documents: Documents to rerank
            
        Returns:
            Re-ranked documents with relevance scores in metadata
        """
        if not self.is_ready:
            raise Exception("System not initialized. Please call initialize() first.")
            
        try:
            if hasattr(self.pipeline.cohere_reranker, 'rerank_with_scores'):
                doc_score_pairs = self.pipeline.cohere_reranker.rerank_with_scores(query, documents)
                
                # Add scores to document metadata
                reranked_docs = []
                for doc, score in doc_score_pairs:
                    doc.metadata['cohere_score'] = score
                    doc.metadata['score'] = score  # For backward compatibility
                    reranked_docs.append(doc)
                
                return reranked_docs
            else:
                return self.pipeline.cohere_reranker.rerank_documents(query, documents)
                
        except Exception as e:
            logger.warning(f"Cohere reranking failed: {e}")
            return documents[:5]
    
    async def assess_confidence_and_generate_async(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        """
        Use the complete pipeline process instead of replicating logic
        Since we already have reranked documents, we need to use the final steps only
        
        Args:
            query: Original query
            documents: Already reranked documents from Cohere
            
        Returns:
            Final result with proper confidence assessment and routing
        """
        if not self.is_ready:
            raise Exception("System not initialized. Please call initialize() first.")
        
        try:
            assessment = None
            if hasattr(self, 'retrieval_assessor') and self.retrieval_assessor:
                assessment = self.retrieval_assessor.calculate_enhanced_confidence(query, documents)
            
            if assessment and hasattr(self.pipeline, 'crag_fallback') and self.pipeline.crag_fallback:
                final_docs, strategy_used = self.pipeline.crag_fallback.evaluate_documents(query, documents, assessment)
                confidence_score = assessment.get('confidence_score', 0.5)
                confidence_tier = assessment.get('tier', 'medium')
            else:
                final_docs, strategy_used = self.pipeline._simple_crag_fallback(query, documents)
                confidence_score = 0.5
                confidence_tier = 'fallback'
                assessment = {
                    'confidence_score': confidence_score,
                    'tier': confidence_tier,
                    'strategy': strategy_used
                }
            
            # Step 8: Final Context Assembly
            context = self.pipeline._assemble_final_context(final_docs)
            
            answer = self.pipeline._synthesize_answer(query, context)
            
            evaluation_result = None
            if hasattr(self, 'answer_evaluator') and self.answer_evaluator:
                try:
                    source_documents = final_docs
                    
                    evaluation_result = self.answer_evaluator.evaluate_answer(
                        query=query,
                        answer=answer,
                        source_documents=source_documents,
                        strategy_used=strategy_used,
                        retrieval_confidence=confidence_score
                    )
                    
                    logger.info(f"LLM-as-a-Judge evaluation: Quality: {evaluation_result.quality_grade}, "
                               f"Composite Score: {evaluation_result.composite_score:.2f}")
                               
                except Exception as e:
                    logger.warning(f"LLM evaluation failed: {e}")
                    evaluation_result = None
            
            action_mapping = {
                'local_only': 'local_documents_used',
                'local_only_simple': 'local_documents_used',
                'local_fallback_simple': 'local_documents_used',
                'hybrid': 'hybrid_search_used', 
                'web_priority': 'web_search_used',
                'web_fallback_no_docs': 'web_search_used',
                'fallback': 'local_documents_used',
                'no_docs_simple': 'web_search_used',
                'error_fallback': 'local_documents_used'
            }
            
            action_taken = action_mapping.get(strategy_used, strategy_used)
            
            result = {
                'answer': answer,
                'confidence': confidence_score,
                'action': action_taken,
                'strategy': strategy_used,
                'documents_used': len(final_docs),
                'confidence_tier': confidence_tier,
                'context_metadata': context.get('metadata', {})
            }
            
            if evaluation_result:
                result['evaluation'] = {
                    'llm_quality_score': evaluation_result.composite_score,
                    'quality_grade': evaluation_result.quality_grade,
                    'metrics': {
                        'faithfulness': evaluation_result.faithfulness,
                        'relevance': evaluation_result.relevance,
                        'coherence': evaluation_result.coherence,
                        'conciseness': evaluation_result.conciseness
                    },
                    'safety_level': evaluation_result.safety_level,
                    'reasoning': evaluation_result.reasoning
                }
            
            if assessment:
                components = assessment.get('components', {})
                result['assessment_details'] = {
                    'relevance_score': components.get('top_document_score', 'N/A'),
                    'coverage_score': components.get('consistency_score', 'N/A'),
                    'clarity_score': components.get('semantic_alignment', 'N/A'),
                    'specificity_score': components.get('domain_relevance', 'N/A'),
                    'confidence_reasoning': assessment.get('reasoning', f"Multi-factor assessment: {confidence_score:.3f}"),
                    'routing_decision': f"Score {confidence_score:.3f} → {strategy_used}",
                    'tier_thresholds': f"High: ≥0.85, Medium: 0.65-0.84, Low: <0.65"
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in CRAG assessment: {e}")
            return {
                'answer': "I encountered an error while processing your query. Please try again.",
                'confidence': 0.0,
                'action': 'error',
                'error': str(e)
            }
