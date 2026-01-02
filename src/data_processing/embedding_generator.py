"""
Embedding Generator
Generate embeddings for processed chunks using OpenAI text-embedding-3-small
Uses context-enriched content from Docling processing
"""

import os
import json
import time
import logging
import math
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
import tiktoken

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the client manager
from src.utils.client_manager import client_manager

from src.config import settings

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Handles embedding generation with batch processing and error handling"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize embedding generator
        
        Args:
            api_key: OpenAI API key (if None, will try to get from environment)
        """
        self.model = settings.embedding_model
        self.embedding_dimension = 1536
        self.max_tokens_per_request = 8191
        self.max_batch_size = 100
        self.rate_limit_delay = 1.0
        
        # Set paths from settings
        self.processed_data_path = Path(settings.new_processed_data_path)
        self.embeddings_path = Path(settings.new_embeddings_path)
        
        # Ensure directories exist
        self.embeddings_path.mkdir(parents=True, exist_ok=True)
        
        # Use managed OpenAI embedding client
        self.client = client_manager.get_embedding_client()
        logger.info("Using managed OpenAI embedding client")
        
        # Stats tracking
        self.stats = {
            "total_chunks": 0,
            "successful_embeddings": 0,
            "failed_embeddings": 0,
            "total_tokens": 0,
            "total_api_calls": 0,
            "start_time": None,
            "end_time": None
        }
    
    def truncate_to_max_tokens(self, text: str, max_tokens: int = 8000) -> str:
        """Truncate text to max tokens for embedding model"""
        try:
            encoder = tiktoken.encoding_for_model("text-embedding-3-small")
            tokens = encoder.encode(text)
            if len(tokens) > max_tokens:
                truncated = encoder.decode(tokens[:max_tokens])
                logger.warning(f"Text truncated from {len(tokens)} to {max_tokens} tokens")
                return truncated
            return text
        except Exception as e:
            logger.warning(f"Token truncation failed: {e}, using original text")
            return text
    
    def load_processed_data(self) -> List[Dict[str, Any]]:
        """Load processed chunks from Docling processing"""
        logger.info("Loading processed data from Docling...")
        
        if not self.processed_data_path.exists():
            logger.error(f"Processed data directory not found: {self.processed_data_path}")
            return []
        
        # Load all JSON files
        json_files = list(self.processed_data_path.glob("*_docling_chunks.json"))
        
        if not json_files:
            logger.error("No Docling chunk files found")
            return []
        
        logger.info(f"🔄 Processing all {len(json_files)} PDF files for complete embedding generation")
        
        all_chunks = []
        
        for json_file in json_files:
            try:
                logger.info(f"Loading chunks from: {json_file.name}")
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                file_metadata = data.get('metadata', {})
                chunks = data.get('chunks', [])
                
                logger.info(f"Found {len(chunks)} chunks in {json_file.name}")
                
                for idx, chunk in enumerate(chunks):
                    if chunk.get('content', '').strip():
                        # Extract metadata from docling processing
                        docling_meta = chunk.get('docling_meta', {})
                        
                        # Extract page numbers from doc_items
                        page_numbers = []
                        headings = docling_meta.get('headings', [])
                        doc_items = docling_meta.get('doc_items', [])
                        bbox_coordinates = []
                        
                        for item in doc_items:
                            if 'page_no' in item:
                                page_no = item['page_no']
                                if page_no not in page_numbers:
                                    page_numbers.append(page_no)
                            
                            if 'bbox' in item and item['bbox']:
                                bbox_coordinates.append(item['bbox'])
                        
                        chunk_data = {
                            "id": chunk.get('id', f"{json_file.stem}_{idx}"),
                            "content": chunk['content'],
                            "context_enriched_content": chunk.get('context_enriched_content', chunk['content']),
                            "source_file": file_metadata.get('source_file', json_file.name),
                            "chunk_index": chunk.get('chunk_index', idx),
                            "processing_timestamp": file_metadata.get('processing_timestamp', ''),
                            "page_numbers": sorted(page_numbers) if page_numbers else [1],
                            "headings": headings,
                            "content_length": len(chunk['content']),
                            "docling_meta": docling_meta,
                            "bbox_coordinates": bbox_coordinates,
                            "token_count": chunk.get('token_count', 0)
                        }
                        
                        all_chunks.append(chunk_data)
                
            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(all_chunks)} chunks from {len(json_files)} files")
        return all_chunks

    def extract_metadata(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and format metadata for vector database storage"""
        docling_meta = chunk.get('docling_meta', {})
        origin = docling_meta.get('origin', {})
        
        # Extract detailed docling metadata
        metadata = {
            "source_file": chunk.get('source_file', ''),
            "chunk_index": chunk.get('chunk_index', 0),
            "page_numbers": chunk.get('page_numbers', []),
            "headings": chunk.get('headings', []),
            "content_length": chunk.get('content_length', 0),
            "processing_timestamp": chunk.get('processing_timestamp', ''),
            "token_count": chunk.get('token_count', 0),
            "binary_hash": origin.get('binary_hash', ''),
            
            # Include full docling metadata
            "doc_items": docling_meta.get('doc_items', [])
        }
        
        return metadata

    def prepare_content_for_embedding(self, chunk: Dict[str, Any]) -> str:
        """
        Prepare chunk content for embedding generation.
        Uses context-enriched content as recommended by Docling documentation.
        """
        # Use context-enriched content as recommended by Docling
        content = chunk.get('context_enriched_content', chunk['content']).strip()
        
        # Ensure content is within token limits using proper tokenization
        content = self.truncate_to_max_tokens(content, max_tokens=8000)
        
        return content
    
    def generate_batch_embeddings(self, contents: List[str]) -> Tuple[bool, List[List[float]], str]:
        """Generate embeddings for a batch of contents"""
        if not self.client:
            return False, [], "OpenAI client not initialized"
        
        try:
            logger.debug(f"Generating embeddings for batch of {len(contents)} items")
            
            response = self.client.embeddings.create(
                input=contents,
                model=self.model
            )
            
            embeddings = [data.embedding for data in response.data]
            self.stats["total_api_calls"] += 1
            self.stats["total_tokens"] += response.usage.total_tokens
            
            return True, embeddings, ""
            
        except Exception as e:
            error_msg = f"Embedding generation failed: {e}"
            logger.error(error_msg)
            return False, [], error_msg
    
    def process_chunks_in_batches(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Process all chunks in batches with error handling.
        Returns True if successful, False otherwise.
        """
        logger.info(f"Starting batch embedding generation for {len(chunks)} chunks")
        
        if not chunks:
            logger.error("No chunks to process")
            return False
        
        if not self.client:
            logger.error("OpenAI client not initialized")
            return False
        
        # Ensure output directory exists
        self.embeddings_path.mkdir(parents=True, exist_ok=True)
        
        batch_size = 100  # Optimized batch size for better performance
        total_batches = math.ceil(len(chunks) / batch_size)
        processed_embeddings = []
        
        self.stats["start_time"] = time.time()
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(chunks))
            batch_chunks = chunks[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches} "
                       f"(chunks {start_idx + 1}-{end_idx})")
            
            # Prepare contents for embedding
            batch_contents = []
            for chunk in batch_chunks:
                content = self.prepare_content_for_embedding(chunk)
                batch_contents.append(content)
            
            # Generate embeddings for batch
            success, embeddings, error = self.generate_batch_embeddings(batch_contents)
            
            if not success:
                logger.error(f"Failed to generate embeddings for batch {batch_idx + 1}: {error}")
                return False
            
            # Combine chunks with embeddings and metadata
            for chunk, embedding in zip(batch_chunks, embeddings):
                embedding_record = {
                    "id": chunk["id"],
                    "vector": embedding,
                    "content": chunk['content'],  # Original content
                    "context_enriched_content": chunk.get('context_enriched_content', chunk['content']),  # Enriched content
                    "metadata": self.extract_metadata(chunk)
                }
                processed_embeddings.append(embedding_record)
                self.stats["successful_embeddings"] += 1
            
            # Progress update
            logger.info(f"Completed batch {batch_idx + 1}/{total_batches} - "
                       f"Total embeddings: {len(processed_embeddings)}")
            
            # Rate limiting
            if batch_idx < total_batches - 1:
                time.sleep(self.rate_limit_delay)
        
        # Save embeddings to file
        output_file = self.embeddings_path / "embeddings.json"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_embeddings, f, indent=2, ensure_ascii=False)
            
            self.stats["end_time"] = time.time()
            self.stats["total_chunks"] = len(chunks)
            
            logger.info(f"Successfully saved {len(processed_embeddings)} embeddings to {output_file}")
            logger.info(f"Processing completed in {self.stats['end_time'] - self.stats['start_time']:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
            return False

    def run_complete_pipeline(self) -> bool:
        """
        Execute complete embedding generation pipeline.
        Returns True if successful, False otherwise.
        """
        logger.info("Starting complete embedding generation pipeline")
        
        # Check OpenAI client
        if not self.client:
            logger.error("OpenAI client not initialized")
            return False
        
        # Load processed data
        chunks = self.load_processed_data()
        if not chunks:
            logger.error("No chunks loaded, cannot proceed")
            return False
        
        # Process all chunks in batches
        success = self.process_chunks_in_batches(chunks)
        
        if success:
            logger.info("Embedding generation pipeline completed successfully")
            logger.info(f"Final statistics: {self.stats}")
        else:
            logger.error("Embedding generation pipeline failed")
        
        return success

def main():
    """Main entry point for embedding generation"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('embedding_generation.log'),
            logging.StreamHandler()
        ]
    )
    
    generator = EmbeddingGenerator()
    
    logger.info("=== STARTING COMPLETE EMBEDDING PIPELINE ===")
    
    # Run complete pipeline for all PDFs
    success = generator.run_complete_pipeline()
    
    if success:
        logger.info("🎉 Embedding generation completed successfully!")
        logger.info(f"📊 Final statistics: {generator.stats}")
    else:
        logger.error("❌ Embedding generation failed")
    
    return success

if __name__ == "__main__":
    main()
