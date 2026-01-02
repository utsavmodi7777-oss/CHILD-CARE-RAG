"""
Document processing pipeline using PROPER Docling implementation.
Replaced old trash code with proper Docling usage.
"""

from pathlib import Path
from typing import List, Dict, Any

from .docling_processor import DoclingProcessor, DoclingChunk
from ..config import settings, logger


class DocumentProcessor:
    """PROPER Document processor using Docling."""
    
    def __init__(self):
        """Initialize with OpenAI-aligned Docling processor."""
        self.docling_processor = DoclingProcessor()
        self.output_dir = Path(settings.new_processed_data_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("DocumentProcessor initialized with OpenAI-aligned Docling")
    
    def process_documents(self, input_path: str, force_reprocess: bool = False) -> List[DoclingChunk]:
        """
        Process documents using PROPER Docling workflow.
        
        Args:
            input_path: Path to PDF file or directory
            force_reprocess: Whether to force reprocessing
            
        Returns:
            List of DoclingChunk objects
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input path not found: {input_path}")
        
        logger.info(f"Processing documents from: {input_path}")
        
        all_chunks = []
        
        if input_path.is_file() and input_path.suffix.lower() == '.pdf':
            # Process single PDF
            chunks = self.docling_processor.process_pdf(str(input_path))
            all_chunks.extend(chunks)
            
        elif input_path.is_dir():
            # Process directory of PDFs
            chunks = self.docling_processor.process_directory(str(input_path))
            all_chunks.extend(chunks)
            
        else:
            raise ValueError(f"Unsupported input: {input_path}")
        
        logger.info(f"Total chunks processed: {len(all_chunks)}")
        return all_chunks
    
    def get_processing_stats(self, chunks: List[DoclingChunk]) -> Dict[str, Any]:
        """Get processing statistics."""
        if not chunks:
            return {
                'total_chunks': 0,
                'total_content_length': 0,
                'avg_chunk_size': 0,
                'avg_token_count': 0,
                'files_processed': 0
            }
        
        # Calculate stats
        total_content = sum(len(chunk.content) for chunk in chunks)
        total_tokens = sum(chunk.metadata.get('token_count', 0) for chunk in chunks)
        files = set(chunk.metadata.get('filename', '') for chunk in chunks)
        
        return {
            'total_chunks': len(chunks),
            'total_content_length': total_content,
            'avg_chunk_size': total_content // len(chunks) if chunks else 0,
            'avg_token_count': total_tokens // len(chunks) if chunks else 0,
            'files_processed': len(files)
        }
