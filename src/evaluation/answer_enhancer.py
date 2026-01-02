"""
Answer enhancement system for applying appropriate prefixes and suffixes based on evaluation results
"""

from typing import Dict, Tuple
from .answer_evaluator import EvaluationResult

class AnswerEnhancer:
    """Applies appropriate prefixes and suffixes to answers based on evaluation metrics"""
    
    def __init__(self):
        """Initialize answer enhancer with enhancement rules"""
        self.enhancement_rules = self._define_enhancement_rules()
    
    def enhance_answer(
        self,
        original_answer: str,
        evaluation_result: EvaluationResult,
        strategy_used: str,
        retrieval_confidence: float
    ) -> Tuple[str, str]:
        """
        Apply appropriate enhancement to answer based on evaluation and strategy
        
        Args:
            original_answer: The generated answer to enhance
            evaluation_result: LLM evaluation results
            strategy_used: Generation strategy (local/hybrid/web_priority)
            retrieval_confidence: Confidence from retrieval assessment
            
        Returns:
            Tuple of (enhanced_answer, enhancement_type)
        """
        
        # Determine enhancement type based on multiple factors
        enhancement_type = self._determine_enhancement_type(
            evaluation_result, strategy_used, retrieval_confidence
        )
        
        # Get appropriate prefix and suffix
        prefix, suffix = self._get_enhancement_content(enhancement_type)
        
        # Apply enhancement to answer
        enhanced_answer = f"{prefix}{original_answer}{suffix}"
        
        return enhanced_answer, enhancement_type
    
    def _determine_enhancement_type(
        self,
        evaluation: EvaluationResult,
        strategy: str,
        retrieval_confidence: float
    ) -> str:
        """Determine the appropriate enhancement type based on evaluation results"""
        
        # Safety check first - override everything if unsafe
        if evaluation.safety_level == "UNSAFE":
            return "unsafe_content"
        
        # Quality-based routing
        if evaluation.composite_score >= 0.85:
            if strategy == "local_only":
                return "high_quality_local"
            elif strategy == "hybrid":
                return "high_quality_hybrid"
            else:
                return "high_quality_web"
        
        elif evaluation.composite_score >= 0.75:
            if strategy == "local_only":
                return "medium_quality_local"
            elif strategy == "hybrid":
                return "medium_quality_hybrid"
            else:
                return "medium_quality_web"
        
        elif evaluation.composite_score >= 0.65:
            if evaluation.safety_level == "CAUTION":
                return "low_quality_caution"
            else:
                return "low_quality_standard"
        
        else:
            return "poor_quality"
    
    def _get_enhancement_content(self, enhancement_type: str) -> Tuple[str, str]:
        """Get prefix and suffix for the given enhancement type"""
        return self.enhancement_rules.get(
            enhancement_type,
            ("", "\n\nConsider consulting with a childcare professional for personalized guidance.")
        )
    
    def _define_enhancement_rules(self) -> Dict[str, Tuple[str, str]]:
        """Define all enhancement rules with prefixes and suffixes"""
        return {
            # High quality responses
            "high_quality_local": (
                "Based on comprehensive childcare research: ",
                "\n\nThis guidance is well-supported by authoritative childcare sources."
            ),
            
            "high_quality_hybrid": (
                "Combining expert knowledge with current information: ",
                "\n\nThis combines established childcare principles with recent insights."
            ),
            
            "high_quality_web": (
                "Based on current expert guidance: ",
                "\n\nThis reflects up-to-date childcare recommendations from reliable sources."
            ),
            
            # Medium quality responses
            "medium_quality_local": (
                "Based on available childcare guidance: ",
                "\n\nThis information is drawn from established childcare resources."
            ),
            
            "medium_quality_hybrid": (
                "Combining available knowledge with current information: ",
                "\n\nConsider this alongside advice from your healthcare provider."
            ),
            
            "medium_quality_web": (
                "Based on current available information: ",
                "\n\nFor personalized advice, consult with your pediatrician or childcare professional."
            ),
            
            # Lower quality responses
            "low_quality_caution": (
                "Based on available information with caution: ",
                "\n\nIMPORTANT: Please consult with a qualified healthcare provider for personalized advice specific to your situation."
            ),
            
            "low_quality_standard": (
                "Based on available information: ",
                "\n\nFor the most reliable guidance, consider consulting with a pediatrician or childcare professional."
            ),
            
            # Poor quality or unsafe content
            "poor_quality": (
                "I apologize, but I cannot provide sufficiently reliable guidance on this topic. ",
                "\n\nFor important childcare decisions, I strongly recommend consulting with a qualified pediatrician or childcare professional who can provide personalized, expert advice based on your specific situation."
            ),
            
            "unsafe_content": (
                "I cannot provide guidance on this topic as it may involve safety concerns. ",
                "\n\nPlease consult with a qualified healthcare provider or childcare professional for safe, appropriate guidance specific to your situation."
            )
        }
    
    def get_enhancement_metadata(self, enhancement_type: str) -> Dict[str, str]:
        """Get metadata about the applied enhancement for response tracking"""
        enhancement_descriptions = {
            "high_quality_local": "High confidence local knowledge with strong evaluation",
            "high_quality_hybrid": "High quality hybrid approach with excellent assessment",
            "high_quality_web": "High quality web-enhanced response",
            "medium_quality_local": "Moderate confidence local knowledge",
            "medium_quality_hybrid": "Medium quality hybrid approach",
            "medium_quality_web": "Medium quality web-enhanced response",
            "low_quality_caution": "Lower quality response requiring caution",
            "low_quality_standard": "Lower quality response with standard disclaimer",
            "poor_quality": "Insufficient quality for reliable guidance",
            "unsafe_content": "Content flagged as potentially unsafe"
        }
        
        return {
            "enhancement_type": enhancement_type,
            "enhancement_reason": enhancement_descriptions.get(
                enhancement_type, 
                "Standard enhancement applied"
            )
        }
