"""
RAG Chainlit Integration
Wrapper for Advanced RAG Pipeline with real-time step visualization
"""

import asyncio
import sys
import os
from typing import Dict, Any, List
from datetime import datetime

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import chainlit as cl
from src.advanced_rag.processor import AdvancedRAGProcessor
from src.retrieval.query_relevance_checker import QueryRelevanceChecker
from src.config import settings


class RAGChainlitIntegration:
    """Integration wrapper for Advanced RAG Pipeline with Chainlit"""
    
    def __init__(self):
        """Initialize RAG components"""
        self.rag_processor = None
        self.query_checker = QueryRelevanceChecker()
        self.processing_stats = {}
    
    def _build_retrieval_queries(
        self,
        original_query: str,
        alternative_queries: List[str],
        hyde_doc: str
    ) -> List[str]:
        max_queries = max(1, settings.max_retrieval_queries)
        queries = []
        
        if original_query:
            queries.append(original_query)
        
        if settings.use_hyde and hyde_doc and len(queries) < max_queries:
            queries.append(hyde_doc)
        
        for alternative in alternative_queries:
            if len(queries) >= max_queries:
                break
            if alternative:
                queries.append(alternative)
        
        deduped_queries = []
        seen = set()
        for query in queries:
            normalized = " ".join(query.split()).lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped_queries.append(query)
        
        return deduped_queries
    
    def _get_retrieval_limits(self) -> tuple[int, bool, int]:
        max_queries = max(1, settings.max_retrieval_queries)
        use_hyde = settings.use_hyde and max_queries > 1
        expansion_budget = max_queries - 1 - (1 if use_hyde else 0)
        expansion_count = min(settings.query_expansion_count, max(0, expansion_budget))
        return expansion_count, use_hyde, max_queries
        
    async def initialize_rag(self):
        """Initialize RAG processor (call this on first use)"""
        if self.rag_processor is None:
            self.rag_processor = AdvancedRAGProcessor()
            initialization_success = self.rag_processor.initialize()
            
            if initialization_success:
                await cl.Message(
                    content="RAG system initialized successfully!",
                    author="System"
                ).send()
            else:
                await cl.Message(
                    content="RAG system initialization failed!",
                    author="System"
                ).send()
                raise Exception("RAG system initialization failed")
    
    @cl.step(name="Query Relevance Check", type="tool", show_input=True)
    async def check_query_relevance(self, query: str) -> Dict[str, Any]:
        """Step 1: Check if query is relevant to childcare"""
        current_step = cl.context.current_step
        current_step.input = f"Query: {query}"
        
        start_time = datetime.now()
        
        relevance_result = self.query_checker.check_relevance(query)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        current_step.output = f"""
**Relevance Assessment:**
- **Is Relevant:** {relevance_result['is_relevant']}
- **Confidence:** {relevance_result.get('confidence', 'N/A')}
- **Reasoning:** {relevance_result.get('reasoning', 'N/A')}
- **Processing Time:** {processing_time:.2f}s
        """
        
        return relevance_result
    
    @cl.step(name="Query Expansion", type="llm", show_input=True)
    async def expand_query(self, query: str) -> List[str]:
        """Step 2: Generate alternative query formulations"""
        current_step = cl.context.current_step
        current_step.input = f"Original Query: {query}"
        
        start_time = datetime.now()
        
        if self.rag_processor is None:
            await self.initialize_rag()
        
        expansion_count, _, _ = self._get_retrieval_limits()
        alternatives = await self.rag_processor.pipeline.query_expansion.generate_alternatives_async(
            query, count=expansion_count
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        if alternatives:
            alternatives_formatted = "\n".join([f"• {alt}" for alt in alternatives])
        else:
            alternatives_formatted = "• None"
        current_step.output = f"""
**Generated Alternatives:**
{alternatives_formatted}

**Processing Time:** {processing_time:.2f}s
**Total Queries Generated:** {len(alternatives)}
        """
        
        return alternatives
    
    @cl.step(name="HyDE Document Generation", type="llm", show_input=True)
    async def generate_hyde_document(self, query: str) -> str:
        """Step 3: Generate hypothetical document"""
        current_step = cl.context.current_step
        current_step.input = f"Query: {query}"
        
        start_time = datetime.now()
        
        if self.rag_processor is None:
            await self.initialize_rag()
        _, use_hyde, _ = self._get_retrieval_limits()
        if not use_hyde:
            processing_time = (datetime.now() - start_time).total_seconds()
            current_step.output = f"""
**Generated Hypothetical Document:**
HyDE generation disabled by configuration.

**Processing Time:** {processing_time:.2f}s
**Document Length:** 0 characters
            """
            return ""
        
        hyde_doc = await self.rag_processor.pipeline.hyde_generation.generate_hypothetical_document_async(query)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        current_step.output = f"""
**Generated Hypothetical Document:**
{hyde_doc[:300]}{'...' if len(hyde_doc) > 300 else ''}

**Processing Time:** {processing_time:.2f}s
**Document Length:** {len(hyde_doc)} characters
        """
        
        return hyde_doc
    
    @cl.step(name="Multi-Retrieval Process", type="retrieval", show_input=True)
    async def perform_retrieval(
        self,
        original_query: str,
        queries: List[str],
        hyde_doc: str,
        retrieval_k: int | None = None
    ) -> List[Any]:
        """Step 4: Perform retrieval for all queries"""
        current_step = cl.context.current_step
        if retrieval_k is None:
            retrieval_k = settings.top_k_retrieval
        
        _, use_hyde, _ = self._get_retrieval_limits()
        retrieval_queries = self._build_retrieval_queries(original_query, queries, hyde_doc)
        current_step.input = (
            f"Queries: {len(retrieval_queries)}, "
            f"HyDE: {'yes' if (use_hyde and hyde_doc) else 'no'}, "
            f"K: {retrieval_k}"
        )
        
        start_time = datetime.now()
        
        if self.rag_processor is None:
            await self.initialize_rag()
        
        retrieval_results = await self.rag_processor.multi_retrieval_async(
            retrieval_queries, None, retrieval_k
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        total_documents = sum(len(results) for results in retrieval_results)
        
        current_step.output = f"""
**Retrieval Results:**
- **Queries Processed:** {len(retrieval_results)}
- **Total Documents Retrieved:** {total_documents}
- **Average Documents per Query:** {total_documents / max(len(retrieval_results), 1):.1f}
- **Processing Time:** {processing_time:.2f}s

**Retrieval Breakdown:**
{chr(10).join([f"• Query {i+1}: {len(results)} documents" for i, results in enumerate(retrieval_results)])}
        """
        
        return retrieval_results
    
    @cl.step(name="Reciprocal Rank Fusion", type="tool", show_input=True)
    async def apply_rrf_fusion(self, retrieval_results: List[Any]) -> List[Any]:
        """Step 5: Apply RRF to fuse retrieval results"""
        current_step = cl.context.current_step
        current_step.input = f"Result sets: {len(retrieval_results)}"
        
        start_time = datetime.now()
        
        if self.rag_processor is None:
            await self.initialize_rag()
        
        fused_results = await self.rag_processor.apply_rrf_fusion_async(retrieval_results)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        current_step.output = f"""
**RRF Fusion Results:**
- **Input Result Sets:** {len(retrieval_results)}
- **Fused Documents:** {len(fused_results)}
- **Processing Time:** {processing_time:.2f}s
- **Fusion Method:** Reciprocal Rank Fusion (k=60)
        """
        
        return fused_results
    
    @cl.step(name="Cohere Re-ranking", type="llm", show_input=True)
    async def apply_cohere_reranking(self, query: str, documents: List[Any]) -> List[Any]:
        """Step 6: Re-rank documents using Cohere with detailed scoring"""
        current_step = cl.context.current_step
        current_step.input = f"Query: {query}, Documents: {len(documents)}"
        
        start_time = datetime.now()
        
        if self.rag_processor is None:
            await self.initialize_rag()
        
        reranked_docs = await self.rag_processor.apply_cohere_reranking_async(query, documents)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Extract scoring information
        top_score = 'N/A'
        score_details = ''
        
        if reranked_docs:
            top_doc = reranked_docs[0]
            top_score = top_doc.metadata.get('cohere_score', top_doc.metadata.get('score', 'N/A'))
            
            score_details = '\n**📊 Top Document Scores:**'
            for i, doc in enumerate(reranked_docs[:3], 1):
                score = doc.metadata.get('cohere_score', doc.metadata.get('score', 'N/A'))
                content_preview = doc.page_content[:100] + '...' if len(doc.page_content) > 100 else doc.page_content
                score_details += f'\n• **Doc {i}:** Score {score} - "{content_preview}"'
        
        current_step.output = f"""**Cohere Re-ranking Results:**

**Input Documents:** {len(documents)}
**Re-ranked Documents:** {len(reranked_docs)}
**Processing Time:** {processing_time:.2f}s
**Top Document Score:** {top_score}
{score_details}"""
        
        return reranked_docs
    
    @cl.step(name="CRAG Confidence Assessment", type="tool", show_input=True)
    async def assess_confidence_and_generate(self, query: str, documents: List[Any]) -> Dict[str, Any]:
        """Step 7: Assess confidence and generate final answer with detailed CRAG assessment"""
        current_step = cl.context.current_step
        current_step.input = f"Query: {query}, Documents: {len(documents)}"
        
        start_time = datetime.now()
        
        if self.rag_processor is None:
            await self.initialize_rag()
        
        final_result = await self.rag_processor.assess_confidence_and_generate_async(query, documents)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        confidence = final_result.get('confidence', 0.0)
        action = final_result.get('action', 'unknown')
        strategy = final_result.get('strategy', 'unknown')
        confidence_tier = final_result.get('confidence_tier', 'unknown')
        assessment_details = final_result.get('assessment_details', {})
        
        output_content = f"""**Confidence Score Calculation:**"""
        
        if assessment_details:
            relevance_score = assessment_details.get('relevance_score', 'N/A')
            coverage_score = assessment_details.get('coverage_score', 'N/A')
            clarity_score = assessment_details.get('clarity_score', 'N/A')
            specificity_score = assessment_details.get('specificity_score', 'N/A')
            
            if (relevance_score != 'N/A' and coverage_score != 'N/A' and 
                clarity_score != 'N/A' and specificity_score != 'N/A'):
                try:
                    rel_val = float(relevance_score) if relevance_score != 'N/A' else 0.0
                    cov_val = float(coverage_score) if coverage_score != 'N/A' else 0.0
                    cla_val = float(clarity_score) if clarity_score != 'N/A' else 0.0
                    spe_val = float(specificity_score) if specificity_score != 'N/A' else 0.0
                    
                    output_content += f"""
• **Relevance Component:** {rel_val:.3f} × 0.40 = {rel_val * 0.40:.3f}
• **Coverage Component:** {cov_val:.3f} × 0.25 = {cov_val * 0.25:.3f}
• **Clarity Component:** {cla_val:.3f} × 0.20 = {cla_val * 0.20:.3f}
• **Specificity Component:** {spe_val:.3f} × 0.15 = {spe_val * 0.15:.3f}
**Final Confidence Score: {confidence:.3f}**"""
                except (ValueError, TypeError):
                    output_content += f"""
• **Relevance Score:** {relevance_score}
• **Coverage Score:** {coverage_score}
• **Clarity Score:** {clarity_score}
• **Specificity Score:** {specificity_score}
**Final Confidence Score: {confidence:.3f}** (Multi-factor assessment)"""
            else:
                output_content += f"""
**Final Confidence Score: {confidence:.3f}** (Multi-factor assessment)"""
        else:
            output_content += f"""
**Final Confidence Score: {confidence:.3f}** (Enhanced assessment)"""

        output_content += f"""

**Confidence-Based Routing:**
• High Threshold: ≥0.85 → Local documents
• Medium Threshold: 0.65-0.84 → Hybrid search
• Low Threshold: <0.65 → Web search"""
            
        output_content += f"""

**📊 CRAG Assessment Results:**
• **Confidence Score:** {confidence:.3f}
• **Confidence Tier:** {confidence_tier}
• **Action Taken:** {action}
• **Strategy Used:** {strategy}
• **Processing Time:** {processing_time:.2f}s
• **Answer Length:** {len(final_result.get('answer', ''))} characters"""

        # Add LLM-as-a-Judge results from final_result
        llm_evaluation = final_result.get('evaluation', {})
        if llm_evaluation:
            metrics = llm_evaluation.get('metrics', {})
            output_content += f"""

**📊 LLM-as-a-Judge Assessment:**
• **Faithfulness Score:** {metrics.get('faithfulness', 'N/A')}
• **Relevance Score:** {metrics.get('relevance', 'N/A')}
• **Coherence Score:** {metrics.get('coherence', 'N/A')}
• **Conciseness Score:** {metrics.get('conciseness', 'N/A')}
• **Quality Grade:** {llm_evaluation.get('quality_grade', 'N/A')}
• **Safety Level:** {llm_evaluation.get('safety_level', 'N/A')}"""
        
        current_step.output = output_content
        
        return final_result
    
    async def process_complete_pipeline(self, query: str) -> Dict[str, Any]:
        """Execute the complete RAG pipeline with step visualization"""
        
        await self.initialize_rag()
        
        try:
            relevance = await self.check_query_relevance(query)
            
            if not relevance.get('is_relevant', False):
                return {
                    'answer': "I apologize, but this query doesn't appear to be related to childcare. Please ask questions about child development, parenting strategies, or childcare practices.",
                    'confidence': 0.0,
                    'action': 'query_rejected'
                }
            
            alternative_queries = await self.expand_query(query)
            
            hyde_document = await self.generate_hyde_document(query)
            
            retrieval_results = await self.perform_retrieval(
                query, alternative_queries, hyde_document
            )
            
            fused_results = await self.apply_rrf_fusion(retrieval_results)
            
            reranked_docs = await self.apply_cohere_reranking(query, fused_results)
            
            final_result = await self.assess_confidence_and_generate(query, reranked_docs)
            
            return final_result
            
        except Exception as e:
            error_msg = f"Error in RAG pipeline: {str(e)}"
            await cl.Message(
                f"❌ **Error:** {error_msg}",
                author="System"
            ).send()
            
            return {
                'answer': "I encountered an error while processing your query. Please try again.",
                'confidence': 0.0,
                'action': 'error'
            }


# Global RAG integration instance
rag_integration = RAGChainlitIntegration()
