"""
Query Expansion Component
Generates 4 alternative search queries using ChatOpenAI with async support
"""

import asyncio
from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

# Import the client manager
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)
from src.utils.client_manager import client_manager

load_dotenv()


class QueryExpansion:
    """Generates alternative queries for improved retrieval coverage"""
    
    def __init__(self):
        self.llm = client_manager.get_chat_client(model="gpt-4o-mini", temperature=0.3)
        self.prompt_template = PromptTemplate(
            input_variables=["query"],
            template="""Generate 4 alternative search queries for: {query}

Create:
1. A paraphrase with different words
2. A keyword expansion version  
3. A shorter, more specific version
4. A longer, more detailed version

Return only the 4 queries, one per line, without numbering or formatting."""
        )
    
    async def generate_alternatives_async(self, query: str) -> List[str]:
        """Generate 4 alternative queries asynchronously for improved performance"""
        try:
            prompt = self.prompt_template.format(query=query)
            
            # Use async LLM call for concurrent processing
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            
            # Parse response into list of queries
            alternative_queries = [
                line.strip() 
                for line in response.content.strip().split('\n') 
                if line.strip()
            ]
            
            # Ensure we have exactly 4 alternatives
            if len(alternative_queries) >= 4:
                return alternative_queries[:4]
            else:
                # Fill with variations if needed
                while len(alternative_queries) < 4:
                    alternative_queries.append(f"Information about {query}")
                return alternative_queries[:4]
                
        except Exception as e:
            # Fallback: generate simple alternatives
            return [
                f"How to {query}",
                f"Guide for {query}",
                f"Tips about {query}",
                f"Best practices for {query}"
            ]
    
    def generate_alternatives(self, query: str) -> List[str]:
        """Generate 4 alternative queries (sync wrapper for backward compatibility)"""
        try:
            prompt = self.prompt_template.format(query=query)
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Parse response into list of queries
            alternative_queries = [
                line.strip() 
                for line in response.content.strip().split('\n') 
                if line.strip()
            ]
            
            # Ensure we have exactly 4 alternatives
            if len(alternative_queries) >= 4:
                return alternative_queries[:4]
            else:
                # If we don't get 4, pad with original query variants
                while len(alternative_queries) < 4:
                    alternative_queries.append(query)
                return alternative_queries[:4]
                
        except Exception as e:
            # Fallback: return simple variants of the original query
            return [
                query,
                f"How to {query.lower()}",
                f"{query} methods",
                f"{query} techniques"
            ]
