# Main src module init - Clean implementation
from .config import settings, logger

# Make docling imports optional to avoid blocking on model downloads
try:
    from .data_processing import (
        DoclingProcessor, 
        DoclingChunk,
        DocumentProcessor
    )
    __all__ = [
        "settings", "logger",
        "DoclingProcessor", 
        "DoclingChunk",
        "DocumentProcessor"
    ]
except Exception as e:
    # Docling not available - app will work but document processing features disabled
    logger.warning(f"Docling imports failed: {e}. Document processing features will be limited.")
    DoclingProcessor = None
    DoclingChunk = None
    DocumentProcessor = None
    __all__ = ["settings", "logger"]
