"""
Enhanced PDF processing with batch conversion and optimized settings
Based on Docling batch conversion patterns for maximum performance
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from docling_core.types.doc import ImageRefMode
import tiktoken
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OptimizedDoclingProcessor:
    """Optimized Docling processor using batch conversion for maximum speed"""
    
    def __init__(self):
        """Initialize with optimized configuration"""
        # Optimized accelerator configuration
        if torch.cuda.is_available():
            accelerator_options = AcceleratorOptions(
                num_threads=8, device=AcceleratorDevice.CUDA
            )
            logger.info("Using CUDA acceleration")
        elif torch.backends.mps.is_available():
            accelerator_options = AcceleratorOptions(
                num_threads=8, device=AcceleratorDevice.MPS
            )
            logger.info("Using MPS acceleration")
        else:
            accelerator_options = AcceleratorOptions(
                num_threads=8, device=AcceleratorDevice.CPU
            )
            logger.info("Using CPU acceleration")
        
        # Optimized pipeline configuration
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.generate_page_images = False  # Disable to save time
        
        # Initialize converter with optimized settings
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )
        
        # Setup chunker with updated token sizes
        self.tokenizer = OpenAITokenizer(
            tokenizer=tiktoken.encoding_for_model("text-embedding-3-small"),
            max_tokens=settings.chunk_size,  # Now 1024
        )
        
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            merge_peers=True,
        )
        # NOTE: HybridChunker handles chunking internally without explicit overlap
        
        # Ensure output directory exists
        self.output_dir = Path(settings.new_processed_data_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"OptimizedDoclingProcessor initialized with chunk_size={settings.chunk_size}")
    
    def process_batch(self, pdf_paths: List[Path]) -> tuple:
        """Process multiple PDFs in batch for maximum efficiency"""
        logger.info(f"Starting batch processing of {len(pdf_paths)} PDFs")
        
        start_time = time.time()
        
        # Batch convert all PDFs
        conv_results = self.converter.convert_all(
            pdf_paths,
            raises_on_error=False  # Continue processing even if some fail
        )
        
        conversion_time = time.time() - start_time
        logger.info(f"Batch conversion completed in {conversion_time:.2f} seconds")
        
        # Process results and chunk
        success_count = 0
        failure_count = 0
        partial_success_count = 0
        total_chunks = 0
        
        chunking_start = time.time()
        
        for conv_res in conv_results:
            pdf_name = conv_res.input.file.stem
            
            if conv_res.status == ConversionStatus.SUCCESS:
                try:
                    # Chunk the document
                    chunks = list(self.chunker.chunk(conv_res.document))
                    
                    # Save chunks
                    self._save_chunks(chunks, conv_res.input.file, conv_res.document)
                    
                    success_count += 1
                    total_chunks += len(chunks)
                    logger.info(f"SUCCESS: {pdf_name} - {len(chunks)} chunks")
                    
                except Exception as e:
                    logger.error(f"CHUNKING FAILED: {pdf_name} - {e}")
                    failure_count += 1
                    
            elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
                logger.warning(f"PARTIAL SUCCESS: {pdf_name}")
                partial_success_count += 1
            else:
                logger.error(f"CONVERSION FAILED: {pdf_name}")
                failure_count += 1
        
        chunking_time = time.time() - chunking_start
        total_time = time.time() - start_time
        
        logger.info(f"Chunking completed in {chunking_time:.2f} seconds")
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        
        return success_count, partial_success_count, failure_count, total_chunks
    
    def _extract_docling_metadata(self, chunk) -> Dict[str, Any]:
        """
        Extract rich metadata from Docling chunk - PROPER JSON serializable format.
        Based on the original docling_processor.py implementation.
        """
        try:
            meta = {}
            
            # Extract dl_meta structure properly
            if hasattr(chunk, 'meta') and chunk.meta:
                if hasattr(chunk.meta, 'doc_items') and chunk.meta.doc_items:
                    doc_items = []
                    for item in chunk.meta.doc_items:
                        doc_item = {
                            'self_ref': getattr(item, 'self_ref', ''),
                            'label': getattr(item, 'label', 'unknown'),
                        }
                        
                        # Extract bbox if available  
                        if hasattr(item, 'prov') and item.prov:
                            prov = item.prov[0] if item.prov else None
                            if prov:
                                doc_item['page_no'] = getattr(prov, 'page_no', 0)
                                # Convert BoundingBox to serializable dict
                                bbox = getattr(prov, 'bbox', None)
                                if bbox:
                                    doc_item['bbox'] = {
                                        'l': getattr(bbox, 'l', 0),
                                        't': getattr(bbox, 't', 0), 
                                        'r': getattr(bbox, 'r', 0),
                                        'b': getattr(bbox, 'b', 0)
                                    }
                                else:
                                    doc_item['bbox'] = {}
                        
                        doc_items.append(doc_item)
                    
                    meta['doc_items'] = doc_items
                
                # Extract headings
                if hasattr(chunk.meta, 'headings') and chunk.meta.headings:
                    meta['headings'] = list(chunk.meta.headings)
                
                # Extract origin
                if hasattr(chunk.meta, 'origin'):
                    origin = chunk.meta.origin
                    meta['origin'] = {
                        'mimetype': getattr(origin, 'mimetype', ''),
                        'filename': getattr(origin, 'filename', ''),
                        'binary_hash': getattr(origin, 'binary_hash', None)
                    }
            
            return meta
            
        except Exception as e:
            logger.warning(f"Could not extract Docling metadata: {e}")
            return {'doc_items': [], 'headings': []}
    
    def _save_chunks(self, chunks, pdf_path: Path, document):
        """Save chunks to JSON and TXT files"""
        pdf_name = pdf_path.stem
        
        # Prepare chunk data
        chunk_data = {
            'metadata': {
                'source_file': pdf_path.name,
                'processing_timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'chunk_count': len(chunks),
                'chunking_method': 'docling_hybrid',
                'tokenizer': 'openai_tiktoken',
                'chunk_size': settings.chunk_size,
                'docling_version': 'latest'
            },
            'chunks': []
        }
        
        # Process chunks with enhanced metadata
        for idx, chunk in enumerate(chunks):
            # PROPER context enrichment using chunker.contextualize()
            enriched_content = self.chunker.contextualize(chunk=chunk)
            
            # Extract metadata properly to avoid JSON serialization errors
            docling_meta = self._extract_docling_metadata(chunk)
            
            chunk_info = {
                'content': chunk.text,
                'context_enriched_content': enriched_content,
                'docling_meta': docling_meta
            }
            
            # Extract metadata if available
            if hasattr(chunk, 'meta') and chunk.meta:
                chunk_info['docling_meta'] = chunk.meta
            
            chunk_data['chunks'].append(chunk_info)
        
        # Save JSON file
        json_path = self.output_dir / f"{pdf_name}_docling_chunks.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, indent=2, ensure_ascii=False)
        
        # Save TXT file for quick reading
        txt_path = self.output_dir / f"{pdf_name}_docling_chunks.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Source: {pdf_path.name}\n")
            f.write(f"Chunks: {len(chunks)}\n")
            f.write(f"Processed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            
            for idx, chunk in enumerate(chunks):
                f.write(f"CHUNK {idx + 1}:\n")
                f.write(chunk.text)
                f.write("\n" + "-" * 30 + "\n\n")
        
        logger.info(f"Saved chunks to {json_path} and {txt_path}")

def process_all_pdfs_optimized():
    """Process all PDFs with optimized batch processing"""
    
    # Initialize processor
    processor = OptimizedDoclingProcessor()
    
    # PDF source directory
    pdfs_dir = Path(r"d:\Projects\Parenting\pdfs")
    
    if not pdfs_dir.exists():
        logger.error(f"PDFs directory not found: {pdfs_dir}")
        return False
    
    # Get all PDF files
    pdf_files = list(pdfs_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.error("No PDF files found")
        return False
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Process in batches for optimal performance
    batch_size = 4  # Process 4 PDFs at a time for local processing (balanced for memory)
    total_success = 0
    total_partial = 0
    total_failure = 0
    total_chunks = 0
    
    overall_start = time.time()
    
    for i in range(0, len(pdf_files), batch_size):
        batch = pdf_files[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(pdf_files) + batch_size - 1) // batch_size
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} files)")
        
        success, partial, failure, chunks = processor.process_batch(batch)
        
        total_success += success
        total_partial += partial
        total_failure += failure
        total_chunks += chunks
        
        logger.info(f"Batch {batch_num} completed: {success} success, {partial} partial, {failure} failed")
    
    total_time = time.time() - overall_start
    
    # Final summary
    logger.info("=" * 60)
    logger.info("FINAL PROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total files processed: {len(pdf_files)}")
    logger.info(f"Successful: {total_success}")
    logger.info(f"Partial success: {total_partial}")
    logger.info(f"Failed: {total_failure}")
    logger.info(f"Total chunks generated: {total_chunks}")
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    logger.info(f"Average time per PDF: {total_time/len(pdf_files):.2f} seconds")
    logger.info(f"Output directory: {processor.output_dir}")
    logger.info("=" * 60)
    
    return total_failure == 0

if __name__ == "__main__":
    success = process_all_pdfs_optimized()
    if success:
        logger.info("ALL PDFs processed successfully!")
        logger.info("Ready for embedding generation...")
    else:
        logger.warning("Some PDFs failed processing")
