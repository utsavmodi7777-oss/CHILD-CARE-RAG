"""
Advanced RAG Pipeline
Main orchestrator for the complete advanced RAG system with async support
"""

import asyncio
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

from src.retrieval.query_expansion import QueryExpansion
from src.retrieval.hyde_generation import HyDEGeneration
from src.retrieval.reciprocal_rank_fusion import ReciprocalRankFusion
from src.retrieval.cohere_reranking import CohereReranking
from src.generation.enhanced_crag import CRAGFallback
from src.utils.client_manager import client_manager

load_dotenv()


class AdvancedRAGPipeline:
    """Complete Advanced RAG Pipeline implementation"""
    
    def __init__(self, vector_retriever=None):
        """
        Initialize the Advanced RAG Pipeline
        
        Args:
            vector_retriever: Vector database retriever (to be injected)
        """
        self.vector_retriever = vector_retriever
        
        self.query_expansion = QueryExpansion()
        self.hyde_generation = HyDEGeneration()
        self.rrf_fusion = ReciprocalRankFusion(k=60)
        self.cohere_reranker = CohereReranking(top_k=5)
        self.crag_fallback = CRAGFallback(
            high_confidence_threshold=0.85,
            low_confidence_threshold=0.65,
            max_web_results=5
        )
        
        self.llm = client_manager.get_chat_client(
            model="gpt-4o-mini",
            temperature=0.1
        )
        self.synthesis_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""Answer the question based on the provided context. Be faithful to the retrieved information.

Question: {query}

Context:
{context}

Answer:"""
        )
    
    async def process_async(self, query: str, retrieval_k: int = 20) -> Dict[str, Any]:
        """
        Process a query through the complete Advanced RAG pipeline with async optimization
        
        Args:
            query: User query
            retrieval_k: Number of documents to retrieve per query
            
        Returns:
            Dictionary with answer and pipeline metadata
        """
        pipeline_metadata = {}
        
        try:
            query_expansion_task = self.query_expansion.generate_alternatives_async(query)
            hyde_generation_task = self.hyde_generation.generate_hypothetical_document_async(query)
            
            alternative_queries, hyde_document = await asyncio.gather(
                query_expansion_task,
                hyde_generation_task
            )
            
            pipeline_metadata['alternative_queries'] = alternative_queries
            pipeline_metadata['hyde_document'] = hyde_document
            
            all_queries = alternative_queries + [hyde_document]
            pipeline_metadata['retrieval_queries_count'] = len(all_queries)
            
            if self.vector_retriever and hasattr(self.vector_retriever, 'similarity_search_async'):
                retrieval_tasks = [
                    self.vector_retriever.similarity_search_async(q, k=retrieval_k)
                    for q in all_queries
                ]
                retrieval_results = await asyncio.gather(*retrieval_tasks)
            else:
                retrieval_results = []
                if self.vector_retriever:
                    for q in all_queries:
                        docs = self.vector_retriever.similarity_search(q, k=retrieval_k)
                        retrieval_results.append(docs)
                else:
                    retrieval_results = [[] for _ in all_queries]
            
            pipeline_metadata['total_retrieved_docs'] = sum(len(docs) for docs in retrieval_results)
            
            fused_results = self.rrf_fusion.fuse_results(retrieval_results, max_results=40)
            fused_documents = [doc for doc, score in fused_results]
            pipeline_metadata['rrf_results_count'] = len(fused_results)
            
            reranked_docs = self.cohere_reranker.rerank_documents(query, fused_documents)
            pipeline_metadata['reranked_docs_count'] = len(reranked_docs)
            
            if hasattr(self, 'retrieval_assessor') and self.retrieval_assessor:
                assessment = self.retrieval_assessor.calculate_enhanced_confidence(query, reranked_docs)
                pipeline_metadata['confidence_assessment'] = assessment
                
                final_docs, strategy_used = self.crag_fallback.evaluate_documents(query, reranked_docs, assessment)
                pipeline_metadata['retrieval_strategy'] = strategy_used
                pipeline_metadata['confidence_score'] = assessment.get('confidence_score', 0.0)
                pipeline_metadata['confidence_tier'] = assessment.get('tier', 'unknown')
            else:
                final_docs, strategy_used = self._simple_crag_fallback(query, reranked_docs)
                pipeline_metadata['retrieval_strategy'] = strategy_used
                pipeline_metadata['confidence_score'] = 0.5
                pipeline_metadata['confidence_tier'] = 'fallback'
            
            pipeline_metadata['final_docs_count'] = len(final_docs)
            
            context = self._assemble_final_context(final_docs)
            pipeline_metadata['context_metadata'] = context['metadata']
            
            answer = self._synthesize_answer(query, context)
            
            return {
                'answer': answer,
                'context': context,
                'pipeline_metadata': pipeline_metadata,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'answer': f"I apologize, but I encountered an error processing your query: {str(e)}",
                'context': {'documents': [], 'metadata': {}},
                'pipeline_metadata': pipeline_metadata,
                'status': 'error',
                'error': str(e)
            }
    
    def _assemble_final_context(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Combine top documents with metadata
        
        Args:
            documents: Final filtered documents
            
        Returns:
            Dictionary with documents and metadata
        """
        top_docs = documents[:5]
        
        source_files = set()
        for doc in top_docs:
            source = doc.metadata.get('source_file', doc.metadata.get('source', 'unknown'))
            source_files.add(source)
        
        relevance_scores = [
            doc.metadata.get('relevance_score', 0.0) 
            for doc in top_docs
        ]
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        
        return {
            'documents': top_docs,
            'metadata': {
                'source_count': len(source_files),
                'avg_relevance': avg_relevance,
                'doc_ids': [doc.metadata.get('id', f'doc_{i}') for i, doc in enumerate(top_docs)],
                'source_files': list(source_files)
            }
        }
    
    def _synthesize_answer(self, query: str, context: Dict[str, Any]) -> str:
        """
        Generate final answer with provenance
        
        Args:
            query: Original query
            context: Final context with documents
            
        Returns:
            Generated answer
        """
        try:
            context_text = ""
            for i, doc in enumerate(context['documents']):
                source = doc.metadata.get('source_file', doc.metadata.get('source', 'Unknown'))
                context_text += f"Document {i+1} (Source: {source}):\n{doc.page_content}\n\n"
            
            prompt = self.synthesis_prompt.format(query=query, context=context_text)
            
            if len(context['documents']) < 3:
                prompt += "\n\nNote: Answer based on limited available evidence."
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def _simple_crag_fallback(self, query: str, documents: List[Document]) -> tuple[List[Document], str]:
        """Simple CRAG fallback for backward compatibility"""
        try:
            if documents:
                avg_score = sum(doc.metadata.get('relevance_score', 0.0) for doc in documents) / len(documents)
                if avg_score >= 0.8:
                    return documents[:5], "local_only_simple"
                else:
                    return documents[:5], "local_fallback_simple"
            else:
                return [], "no_docs_simple"
        except Exception:
            return documents[:5] if documents else [], "error_fallback"
    
    def set_vector_retriever(self, retriever):
        """Set the vector database retriever"""
        self.vector_retriever = retriever
    
    def set_retrieval_assessor(self, assessor):
        """Set the enhanced retrieval assessor"""
        self.retrieval_assessor = assessor
    
    def get_pipeline_summary(self) -> Dict[str, str]:
        """Get summary of pipeline components"""
        return {
            "query_expansion": "Generates 4 alternative search queries",
            "hyde_generation": "Creates hypothetical document for better retrieval",
            "retrieval": "Searches vector database with 5 queries (4 alternatives + 1 HyDE)",
            "rrf_fusion": "Combines results using Reciprocal Rank Fusion",
            "cohere_reranking": "Re-ranks top results using Cohere model",
            "crag_fallback": "Triggers web search if confidence is low",
            "answer_synthesis": "Generates final answer from top-ranked context"
        }
