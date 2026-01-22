"""
LLM-as-a-Judge Answer Evaluator for comprehensive answer quality assessment
"""

import os
import json
import time
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document

# Import the client manager
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)
from src.utils.client_manager import client_manager

from .evaluation_prompts import EvaluationPrompts

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Complete evaluation result with all metrics and metadata"""
    faithfulness: float
    relevance: int  
    coherence: int
    conciseness: int
    composite_score: float
    quality_grade: str
    safety_level: str
    reasoning: Dict[str, str]
    evaluation_time: float
    
    def to_dict(self) -> Dict:
        """Convert evaluation result to dictionary for API response"""
        return {
            "faithfulness": self.faithfulness,
            "relevance": self.relevance,
            "coherence": self.coherence, 
            "conciseness": self.conciseness,
            "composite_score": self.composite_score,
            "quality_grade": self.quality_grade,
            "safety_level": self.safety_level,
            "evaluation_reasoning": self.reasoning,
            "evaluation_time": self.evaluation_time
        }

class AnswerEvaluator:
    """LLM-as-a-Judge evaluator for comprehensive answer quality assessment"""
    
    def __init__(self, openai_api_key: Optional[str] = None, model_name: str = "gpt-4o-mini"):
        """Initialize the answer evaluator with OpenAI client"""
        self.llm = client_manager.get_chat_client()
        self.prompts = EvaluationPrompts()
        
        # Evaluation weights for composite scoring
        self.weights = {
            "faithfulness": 0.40,  # Most critical for preventing misinformation
            "relevance": 0.30,     # Core requirement for useful answers
            "coherence": 0.20,     # Important for user experience
            "conciseness": 0.10    # Efficiency in communication
        }
        
        logger.info("Answer evaluator initialized with LLM-as-a-Judge")
    
    def evaluate_answer(
        self,
        query: str,
        answer: str,
        source_documents: List[Document],
        strategy_used: str,
        retrieval_confidence: float
    ) -> EvaluationResult:
        """
        Comprehensive evaluation of generated answer using LLM-as-a-Judge
        
        Args:
            query: Original user query
            answer: Generated answer to evaluate
            source_documents: Documents used for generation
            strategy_used: Generation strategy (local/hybrid/web_priority)
            retrieval_confidence: Confidence score from retrieval assessment
            
        Returns:
            Complete evaluation result with all metrics
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting LLM-as-a-Judge evaluation for query: {query[:50]}...")
            
            # Prepare context from source documents
            context_text = self._format_context_documents(source_documents)
            
            # Run evaluations in parallel for efficiency
            evaluation_results = self._run_parallel_evaluations(
                query, answer, context_text
            )
            
            # Calculate composite score and quality grade
            composite_score = self._calculate_composite_score(evaluation_results)
            quality_grade = self._determine_quality_grade(composite_score)
            
            # Extract safety assessment
            safety_level = evaluation_results.get("safety", {}).get("safety_level", "SAFE")
            
            # Compile reasoning from all evaluations
            reasoning = self._compile_reasoning(evaluation_results)
            
            evaluation_time = time.time() - start_time
            
            result = EvaluationResult(
                faithfulness=evaluation_results["faithfulness"]["faithfulness_score"],
                relevance=evaluation_results["relevance"]["relevance_score"],
                coherence=evaluation_results["coherence"]["coherence_score"],
                conciseness=evaluation_results["conciseness"]["conciseness_score"],
                composite_score=composite_score,
                quality_grade=quality_grade,
                safety_level=safety_level,
                reasoning=reasoning,
                evaluation_time=evaluation_time
            )
            
            logger.info(f"Evaluation completed: Grade {quality_grade}, Score {composite_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Answer evaluation failed: {e}")
            # Return fallback evaluation result
            return self._create_fallback_evaluation(time.time() - start_time)
    
    def _format_context_documents(self, documents: List[Document]) -> str:
        """Format source documents into readable context for evaluation"""
        if not documents:
            return "No context documents provided."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            # Extract meaningful content and metadata
            content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
            source = doc.metadata.get('source', f'Document {i}')
            context_parts.append(f"Source {i} ({source}):\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _run_parallel_evaluations(self, query: str, answer: str, context: str) -> Dict:
        """Run all evaluation metrics in parallel for efficiency"""
        
        evaluation_tasks = {
            "faithfulness": (self.prompts.get_faithfulness_prompt, (query, answer, context)),
            "relevance": (self.prompts.get_relevance_prompt, (query, answer)),
            "coherence": (self.prompts.get_coherence_prompt, (answer,)),
            "conciseness": (self.prompts.get_conciseness_prompt, (query, answer)),
            "safety": (self.prompts.get_safety_prompt, (answer,))
        }
        
        results = {}
        
        # Use ThreadPoolExecutor for parallel LLM calls
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_metric = {
                executor.submit(self._evaluate_single_metric, prompt_func, args): metric
                for metric, (prompt_func, args) in evaluation_tasks.items()
            }
            
            for future in as_completed(future_to_metric):
                metric = future_to_metric[future]
                try:
                    results[metric] = future.result()
                except Exception as e:
                    logger.warning(f"Evaluation failed for {metric}: {e}")
                    results[metric] = self._get_fallback_metric_result(metric)
        
        return results
    
    def _evaluate_single_metric(self, prompt_func, args) -> Dict:
        """Evaluate a single metric using LLM"""
        try:
            prompt = prompt_func(*args)
            response = self.llm.invoke(prompt)
            
            # Parse JSON response from LLM
            result = json.loads(response.content.strip())
            return result
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM evaluation response: {e}")
            raise
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            raise
    
    def _calculate_composite_score(self, results: Dict) -> float:
        """Calculate weighted composite score from individual metrics"""
        try:
            # Normalize relevance, coherence, conciseness to 0-1 scale
            normalized_relevance = (results["relevance"]["relevance_score"] - 1) / 4
            normalized_coherence = (results["coherence"]["coherence_score"] - 1) / 4
            normalized_conciseness = (results["conciseness"]["conciseness_score"] - 1) / 4
            
            composite = (
                results["faithfulness"]["faithfulness_score"] * self.weights["faithfulness"] +
                normalized_relevance * self.weights["relevance"] +
                normalized_coherence * self.weights["coherence"] +
                normalized_conciseness * self.weights["conciseness"]
            )
            
            return min(max(composite, 0.0), 1.0)  # Ensure 0-1 range
            
        except (KeyError, TypeError) as e:
            logger.warning(f"Error calculating composite score: {e}")
            return 0.5  # Fallback to neutral score
    
    def _determine_quality_grade(self, composite_score: float) -> str:
        """Determine quality grade based on composite score"""
        if composite_score >= 0.90:
            return "A+"
        elif composite_score >= 0.85:
            return "A"
        elif composite_score >= 0.80:
            return "B+"
        elif composite_score >= 0.75:
            return "B"
        elif composite_score >= 0.65:
            return "C"
        else:
            return "F"
    
    def _compile_reasoning(self, results: Dict) -> Dict[str, str]:
        """Compile reasoning from all evaluation metrics"""
        reasoning = {}
        
        for metric in ["faithfulness", "relevance", "coherence", "conciseness", "safety"]:
            if metric in results and "reasoning" in results[metric]:
                reasoning[f"{metric}_note"] = results[metric]["reasoning"]
        
        return reasoning
    
    def _get_fallback_metric_result(self, metric: str) -> Dict:
        """Provide fallback results for failed metric evaluations"""
        fallback_results = {
            "faithfulness": {"faithfulness_score": 0.7, "reasoning": "Evaluation unavailable"},
            "relevance": {"relevance_score": 3, "reasoning": "Evaluation unavailable"},
            "coherence": {"coherence_score": 3, "reasoning": "Evaluation unavailable"},
            "conciseness": {"conciseness_score": 3, "reasoning": "Evaluation unavailable"},
            "safety": {"safety_level": "CAUTION", "reasoning": "Safety evaluation unavailable"}
        }
        return fallback_results.get(metric, {})
    
    def _create_fallback_evaluation(self, evaluation_time: float) -> EvaluationResult:
        """Create fallback evaluation result when evaluation fails"""
        return EvaluationResult(
            faithfulness=0.7,
            relevance=3,
            coherence=3,
            conciseness=3,
            composite_score=0.65,
            quality_grade="C",
            safety_level="CAUTION",
            reasoning={"evaluation_note": "Evaluation system temporarily unavailable"},
            evaluation_time=evaluation_time
        )
