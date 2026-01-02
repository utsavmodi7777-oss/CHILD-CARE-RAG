"""
HyDE (Hypothetical Document Embeddings) Generation Component
Generates hypothetical document passages for improved retrieval with async support
"""

import asyncio
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


class HyDEGeneration:
    """Generates hypothetical documents that would answer the query"""
    
    def __init__(self):
        self.llm = client_manager.get_chat_client()
        self.prompt_template = PromptTemplate(
            input_variables=["query"],
            template="""Write a detailed passage that would perfectly answer this question: {query}

Write as if from an authoritative source. Include specific details and examples.
The passage should be informative and comprehensive, as if it came from a professional document.

Passage:"""
        )
    
    async def generate_hypothetical_document_async(self, query: str) -> str:
        """Generate a hypothetical document passage asynchronously for concurrent processing"""
        try:
            prompt = self.prompt_template.format(query=query)
            
            # Use async LLM call for parallel execution with query expansion
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            
            # Return the generated hypothetical document
            return response.content.strip()
                
        except Exception as e:
            # Fallback: return a simple hypothetical answer
            return f"This document discusses {query} and provides detailed information about related concepts, methods, and best practices in the field."
    
    def generate_hypothetical_document(self, query: str) -> str:
        """Generate a hypothetical document passage (sync wrapper for backward compatibility)"""
        try:
            prompt = self.prompt_template.format(query=query)
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Return the generated hypothetical document
            return response.content.strip()
                
        except Exception as e:
            # Fallback: return a simple hypothetical answer
            return f"This document discusses {query} and provides detailed information about related concepts, methods, and best practices in the field."
