# Data processing module init - Clean implementation
# Lazy imports to avoid loading heavy docling dependencies at app startup
__all__ = [
    "DoclingProcessor", 
    "DoclingChunk",
    "DocumentProcessor"
]

def __getattr__(name):
    """Lazy load docling components only when actually used"""
    if name == "DoclingProcessor":
        from .docling_processor import DoclingProcessor
        return DoclingProcessor
    elif name == "DoclingChunk":
        from .docling_processor import DoclingChunk
        return DoclingChunk
    elif name == "DocumentProcessor":
        from .document_processor import DocumentProcessor
        return DocumentProcessor
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
