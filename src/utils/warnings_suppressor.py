"""
Warnings Suppressor
Suppresses known warnings and error messages that don't affect functionality
"""

import warnings
import logging
import os
import sys


def suppress_warnings():
    """Suppress known non-critical warnings"""
    
    # Suppress ALL warnings for a cleaner experience
    warnings.filterwarnings("ignore")
    
    # Specifically suppress Pydantic warnings about model namespace conflicts
    warnings.filterwarnings("ignore", message="Field.*has conflict with protected namespace.*model_.*")
    
    # Suppress pkg_resources deprecation warnings
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")
    
    # Suppress LangChain deprecation warnings for TavilySearchResults
    warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*TavilySearchResults.*")
    
    # Monkey-patch OpenAI SyncHttpxClientWrapper to prevent __del__ errors
    try:
        from openai._base_client import SyncHttpxClientWrapper
        original_del = SyncHttpxClientWrapper.__del__
        
        def safe_del(self):
            try:
                original_del(self)
            except (AttributeError, Exception):
                # Ignore all cleanup errors silently
                pass
        
        SyncHttpxClientWrapper.__del__ = safe_del
    except ImportError:
        pass  # OpenAI module not available


def setup_clean_logging():
    """Set up logging to reduce noise"""
    
    # Configure root logger to be less verbose
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    # Reduce httpx logging level to ERROR to hide HTTP request logs completely
    logging.getLogger("httpx").setLevel(logging.ERROR)
    
    # Reduce openai logging level to ERROR
    logging.getLogger("openai").setLevel(logging.ERROR)
    
    # Reduce pymilvus logging level to WARNING  
    logging.getLogger("pymilvus").setLevel(logging.WARNING)
    
    # Reduce pydantic logging
    logging.getLogger("pydantic").setLevel(logging.ERROR)


# Auto-apply when imported
suppress_warnings()
setup_clean_logging()
