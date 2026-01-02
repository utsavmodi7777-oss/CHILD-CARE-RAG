"""
Query Relevance Checker
Determines if a query is related to childcare/parenting before processing
"""

import os
import logging
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Import the client manager
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)
from src.utils.client_manager import client_manager

load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)


class QueryRelevanceScore(BaseModel):
    """Relevance assessment for childcare/parenting queries"""
    
    is_relevant: bool = Field(
        description="Whether the query is related to childcare, parenting, child development, or education"
    )
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0 for the relevance assessment"
    )
    reasoning: str = Field(
        description="Brief explanation of why the query is or isn't relevant to childcare"
    )


class QueryRelevanceChecker:
    """Checks if queries are relevant to childcare/parenting domain"""
    
    def __init__(self):
        """Initialize the relevance checker"""
        self.llm = client_manager.get_chat_client()
        self.structured_llm = self.llm.with_structured_output(QueryRelevanceScore)
        
        # Define system prompt for relevance checking
        self.system_prompt = """You are a domain relevance checker for a childcare and parenting knowledge base.

Your task is to determine if a user's query is related to:
- Childcare and parenting
- Child development (physical, cognitive, social, emotional)
- Early childhood education
- Pregnancy and child health
- Family relationships and dynamics
- Educational activities for children
- Child behavior management
- Safety and nutrition for children
- Common parenting challenges

IMPORTANT GUIDELINES:
- Be strict but reasonable in your assessment
- Queries about specific children's products, toys, or brands are relevant
- General health questions are relevant if they could apply to children
- Educational questions are relevant if they relate to child learning
- Completely unrelated topics (politics, adult technology, sports scores, etc.) should be marked as irrelevant
- Current events are irrelevant unless they directly impact child welfare

Return:
- is_relevant: true/false
- confidence: 0.0-1.0 (how sure you are)
- reasoning: brief explanation"""
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Query: {query}\n\nAssess if this query is relevant to childcare/parenting.")
        ])
        
        self.chain = self.prompt | self.structured_llm
    
    def check_relevance(self, query: str) -> Dict[str, Any]:
        """
        Check if query is relevant to childcare/parenting
        
        Args:
            query: User's query to check
            
        Returns:
            Dictionary with relevance assessment
        """
        try:
            result = self.chain.invoke({"query": query})
            
            return {
                "is_relevant": result.is_relevant,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "should_proceed": result.is_relevant and result.confidence >= 0.7
            }
            
        except Exception as e:
            logger.error(f"Error in relevance check: {e}")
            # If check fails, be conservative and reject for safety
            return {
                "is_relevant": False,
                "confidence": 0.8,
                "reasoning": f"Relevance check failed due to technical error: {str(e)}. Defaulting to non-relevant for safety.",
                "should_proceed": False
            }
    
    def generate_rejection_response(self, query: str, reasoning: str) -> str:
        """
        Generate a polite rejection response for irrelevant queries
        
        Args:
            query: The irrelevant query
            reasoning: Why it was deemed irrelevant
            
        Returns:
            Polite rejection message
        """
        return f"""I'm a specialized assistant focused on childcare, parenting, and child development topics. 

Your question about "{query}" appears to be outside my area of expertise. 

I'm designed to help with topics like:
- Child development and behavior
- Parenting strategies and challenges  
- Early childhood education
- Child health and safety
- Family relationships
- Educational activities for children

If you have any questions related to childcare or parenting, I'd be happy to help! Otherwise, you might want to consult a general-purpose AI assistant for your current question."""
