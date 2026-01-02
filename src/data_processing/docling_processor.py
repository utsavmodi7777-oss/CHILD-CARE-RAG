"""
PROPER Docling Implementation - Clean, Industry Standard.
Based on COMPLETE analysis of all Docling documentation.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

import torch
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
import tiktoken

from src.config import settings, logger

@dataclass
class DoclingChunk:
    """Properly structured Docling chunk."""
    content: str
    context_enriched_content: str
    metadata: Dict[str, Any]
    chunk_index: int
    docling_meta: Dict[str, Any]


class DoclingProcessor:
    """
    PROPER Docling implementation following EXACT documentation patterns.
    Uses Docling's HybridChunker as intended, not LangChain workarounds.
    """
    
    def __init__(self):
        """Initialize with OpenAI-aligned Docling configuration."""
        self.logger = logger
        
        # PROPER accelerator configuration (from Accelerator options - Docling.txt)
        if torch.cuda.is_available() and settings.use_gpu:
            accelerator_options = AcceleratorOptions(
                num_threads=8, device=AcceleratorDevice.CUDA
            )
            self.logger.info("Using CUDA acceleration for Docling")
        elif torch.backends.mps.is_available() and settings.use_gpu:
            accelerator_options = AcceleratorOptions(
                num_threads=8, device=AcceleratorDevice.MPS
            )
            self.logger.info("Using MPS acceleration for Docling")
        else:
            accelerator_options = AcceleratorOptions(
                num_threads=8, device=AcceleratorDevice.CPU
            )
            self.logger.info("Using CPU acceleration for Docling")
        
        # PROPER pipeline configuration (from Accelerator options - Docling.txt)
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        
        # PROPER DocumentConverter initialization
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )
        
        # HybridChunker setup with OpenAI tokenizer for embedding alignment
        
        self.tokenizer = OpenAITokenizer(
            tokenizer=tiktoken.encoding_for_model("text-embedding-3-small"),
            max_tokens=settings.chunk_size,
        )
        
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            merge_peers=True,
        )
        
        # Ensure output directory exists
        self.output_dir = Path(settings.new_processed_data_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("DoclingProcessor initialized with OpenAI tokenizer alignment")
    
    def process_pdf(self, pdf_path: str) -> List[DoclingChunk]:
        """
        Process PDF using PROPER Docling workflow.
        Following EXACT patterns from documentation.
        """
        try:
            pdf_path = Path(pdf_path)
            
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF not found: {pdf_path}")
            
            self.logger.info(f"Converting PDF with Docling: {pdf_path.name}")
            
            # PROPER conversion (from Hybrid chunking - Docling.txt)
            conversion_result = self.converter.convert(source=str(pdf_path))
            docling_document = conversion_result.document
            
            self.logger.info(f"Chunking with HybridChunker: {pdf_path.name}")
            
            # PROPER chunking (from Hybrid chunking - Docling.txt)
            chunk_iter = self.chunker.chunk(dl_doc=docling_document)
            docling_chunks = list(chunk_iter)
            
            # Process chunks
            processed_chunks = []
            
            for i, chunk in enumerate(docling_chunks):
                # PROPER context enrichment (from Hybrid chunking - Docling.txt)
                enriched_content = self.chunker.contextualize(chunk=chunk)
                
                # Extract Docling metadata (from RAG with LangChain - Docling.txt)
                docling_meta = self._extract_docling_metadata(chunk)
                
                processed_chunk = DoclingChunk(
                    content=chunk.text,
                    context_enriched_content=enriched_content,
                    metadata={},  # Remove our custom metadata completely
                    chunk_index=i,
                    docling_meta=docling_meta
                )
                
                processed_chunks.append(processed_chunk)
            
            # Save chunks (following Batch conversion - Docling.txt patterns)
            self._save_chunks(processed_chunks, pdf_path)
            
            self.logger.info(f"Processed {len(processed_chunks)} chunks from {pdf_path.name}")
            return processed_chunks
            
        except Exception as e:
            self.logger.error(f"Error processing {pdf_path}: {str(e)}")
            raise
    
    def _extract_docling_metadata(self, chunk) -> Dict[str, Any]:
        """
        Extract rich metadata from Docling chunk.
        Based on RAG with LangChain - Docling.txt patterns.
        """
        try:
            meta = {}
            
            # Extract dl_meta structure (from RAG with LangChain - Docling.txt)
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
                
                # Extract headings (from RAG with LangChain - Docling.txt)
                if hasattr(chunk.meta, 'headings') and chunk.meta.headings:
                    meta['headings'] = list(chunk.meta.headings)
                
                # Extract origin (from RAG with LangChain - Docling.txt)
                if hasattr(chunk.meta, 'origin'):
                    origin = chunk.meta.origin
                    meta['origin'] = {
                        'mimetype': getattr(origin, 'mimetype', ''),
                        'filename': getattr(origin, 'filename', ''),
                        'binary_hash': getattr(origin, 'binary_hash', None)
                    }
            
            return meta
            
        except Exception as e:
            self.logger.warning(f"Could not extract Docling metadata: {e}")
            return {}
    
    def _save_chunks(self, chunks: List[DoclingChunk], pdf_path: Path):
        """
        Save chunks following Batch conversion patterns.
        Multiple export formats as shown in Batch conversion - Docling.txt.
        """
        try:
            base_name = pdf_path.stem
            
            # Main chunks file (JSON format)
            chunks_file = self.output_dir / f"{base_name}_docling_chunks.json"
            
            chunks_data = {
                'metadata': {
                    'source_file': pdf_path.name,
                    'processing_timestamp': datetime.now().isoformat(),
                    'chunk_count': len(chunks),
                    'chunking_method': 'docling_hybrid',
                    'tokenizer': 'openai_tiktoken',
                    'docling_version': 'latest'
                },
                'chunks': []
            }
            
            for chunk in chunks:
                chunk_data = {
                    'content': chunk.content,
                    'context_enriched_content': chunk.context_enriched_content,
                    'docling_meta': chunk.docling_meta,
                    'chunk_index': chunk.chunk_index
                }
                chunks_data['chunks'].append(chunk_data)
            
            # Save JSON
            with open(chunks_file, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)
            
            # Save text format for easy reading (following Batch conversion patterns)
            txt_file = self.output_dir / f"{base_name}_docling_chunks.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                for i, chunk in enumerate(chunks):
                    f.write(f"=== CHUNK {i} ===\n")
                    f.write(f"Content: {chunk.content}\n")
                    f.write(f"Context Enriched: {chunk.context_enriched_content}\n")
                    f.write(f"Headings: {chunk.docling_meta.get('headings', [])}\n")
                    f.write(f"Doc Items Count: {len(chunk.docling_meta.get('doc_items', []))}\n")
                    f.write("-" * 80 + "\n\n")
            
            self.logger.info(f"Saved chunks to {chunks_file} and {txt_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving chunks: {str(e)}")
            raise
    
    def process_directory(self, directory_path: str) -> List[DoclingChunk]:
        """Process all PDFs in directory using PROPER Docling workflow."""
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        pdf_files = list(directory.glob("*.pdf"))
        
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {directory}")
            return []
        
        self.logger.info(f"Processing {len(pdf_files)} PDF files with Docling")
        
        all_chunks = []
        
        for pdf_file in pdf_files:
            try:
                chunks = self.process_pdf(str(pdf_file))
                all_chunks.extend(chunks)
            except Exception as e:
                self.logger.error(f"Failed to process {pdf_file}: {e}")
                continue
        
        self.logger.info(f"Total chunks processed: {len(all_chunks)}")
        return all_chunks
