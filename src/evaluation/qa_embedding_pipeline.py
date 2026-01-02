#!/usr/bin/env python3
"""
QA Embedding Pipeline: embedding → pool → expand → final-goldset
Processes QA pairs to create embeddings, retrieve candidates, and expand chunk sets.
"""

import json
import csv
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import time
from tqdm import tqdm
import sys

# Import the client manager
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)
from src.utils.client_manager import client_manager

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from openai import OpenAI
from pymilvus import MilvusClient
from src.config.settings import Settings

class QAEmbeddingPipeline:
    """Complete QA embedding and expansion pipeline"""
    
    def __init__(self):
        self.settings = Settings()
        self.setup_clients()
        self.setup_paths()
        self.embedding_cost = 0.0
        
    def setup_clients(self):
        """Initialize OpenAI and Milvus clients"""
        # Use managed embedding client instead of direct OpenAI client
        self.embedding_client = client_manager.get_embedding_client()
        
        if not self.settings.zilliz_uri or not self.settings.zilliz_token:
            raise ValueError("Zilliz credentials not found in environment")
            
        self.milvus_client = MilvusClient(
            uri=self.settings.zilliz_uri,
            token=self.settings.zilliz_token
        )
        
    def setup_paths(self):
        """Setup file paths for inputs and outputs"""
        self.qa_input_path = Path("new_data/qa_pairs/qa_pairs.json")
        self.qa_embeddings_path = Path("new_data/embeddings/qa_pairs_embeddings.json")
        self.qa_top50_path = Path("new_data/qa_pairs/qa_pairs_top_50.json")
        self.qa_final_path = Path("new_data/qa_pairs/qa_pairs_final.json")
        self.sample_csv_path = Path("src/evaluation/sample_data.csv")
        
        # Ensure output directories exist
        self.qa_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        self.qa_top50_path.parent.mkdir(parents=True, exist_ok=True)
        self.sample_csv_path.parent.mkdir(parents=True, exist_ok=True)
        
    def load_qa_pairs(self) -> List[Dict[str, Any]]:
        """Load QA pairs from JSON file"""
        print(f"Loading QA pairs from {self.qa_input_path}")
        with open(self.qa_input_path, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)
        print(f"Loaded {len(qa_pairs)} QA pairs")
        return qa_pairs
    
    def create_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Create embeddings for texts in batches"""
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
            batch_texts = texts[i:i + batch_size]
            
            try:
                response = self.openai_client.embeddings.create(
                    model=self.settings.embedding_model,
                    input=batch_texts
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Calculate cost (text-embedding-3-small: $0.00002 per 1K tokens)
                total_tokens = response.usage.total_tokens
                batch_cost = (total_tokens / 1000) * 0.00002
                self.embedding_cost += batch_cost
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"Error creating embeddings for batch {i//batch_size + 1}: {e}")
                raise
                
        return all_embeddings
    
    def step1_generate_qa_embeddings(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Step 1: Generate embeddings for queries and ground truth answers"""
        if self.qa_embeddings_path.exists():
            print(f"Found existing embeddings file: {self.qa_embeddings_path}")
            with open(self.qa_embeddings_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        print("Step 1: Generating QA embeddings")
        
        # Prepare texts for embedding
        queries = [qa['query'] for qa in qa_pairs]
        answers = [qa['ground_truth_answer'] for qa in qa_pairs]
        
        print(f"Generating embeddings for {len(queries)} queries")
        query_embeddings = self.create_embeddings_batch(queries)
        
        print(f"Generating embeddings for {len(answers)} answers")
        answer_embeddings = self.create_embeddings_batch(answers)
        
        # Add embeddings to QA pairs
        enriched_qa_pairs = []
        for i, qa in enumerate(qa_pairs):
            enriched_qa = qa.copy()
            enriched_qa['query_embedding'] = query_embeddings[i]
            enriched_qa['ground_truth_answer_embedding'] = answer_embeddings[i]
            enriched_qa_pairs.append(enriched_qa)
        
        # Save embeddings
        print(f"Saving embeddings to {self.qa_embeddings_path}")
        with open(self.qa_embeddings_path, 'w', encoding='utf-8') as f:
            json.dump(enriched_qa_pairs, f, ensure_ascii=False, indent=2)
        
        print(f"Generated embeddings for {len(enriched_qa_pairs)} QA pairs")
        return enriched_qa_pairs
    
    def step2_retrieve_top50_candidates(self, qa_pairs_with_embeddings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Step 2: Retrieve top-50 candidates from Milvus for each query"""
        if self.qa_top50_path.exists():
            print(f"Found existing top-50 file: {self.qa_top50_path}")
            with open(self.qa_top50_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        print("Step 2: Retrieving top-50 candidates from Milvus")
        
        collection_name = "childcare_knowledge_base"
        
        # Load collection
        self.milvus_client.load_collection(collection_name)
        
        qa_pairs_with_candidates = []
        
        for i, qa in enumerate(tqdm(qa_pairs_with_embeddings, desc="Retrieving candidates")):
            try:
                # Search with query embedding
                search_results = self.milvus_client.search(
                    collection_name=collection_name,
                    data=[qa['query_embedding']],
                    limit=50,
                    output_fields=["id"],
                    search_params={"metric_type": "IP"}
                )
                
                # Process search results
                relative_chunks = []
                if search_results and len(search_results) > 0:
                    for rank, result in enumerate(search_results[0]):
                        relative_chunks.append({
                            "id": result["entity"]["id"],
                            "similarity": float(result["distance"]),
                            "rank": rank + 1
                        })
                
                # Add to QA pair
                qa_with_candidates = qa.copy()
                qa_with_candidates['relative_chunks'] = relative_chunks
                qa_pairs_with_candidates.append(qa_with_candidates)
                
            except Exception as e:
                print(f"Error retrieving candidates for QA {i}: {e}")
                raise
        
        # Save results
        print(f"Saving top-50 candidates to {self.qa_top50_path}")
        with open(self.qa_top50_path, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs_with_candidates, f, ensure_ascii=False, indent=2)
        
        print(f"Retrieved candidates for {len(qa_pairs_with_candidates)} QA pairs")
        return qa_pairs_with_candidates
    
    def get_chunk_embeddings(self, chunk_ids: List[str]) -> Dict[str, List[float]]:
        """Retrieve chunk embeddings from Milvus"""
        collection_name = "childcare_knowledge_base"
        
        # Query for embeddings in batches to avoid query limits
        embeddings_dict = {}
        batch_size = 100
        
        for i in range(0, len(chunk_ids), batch_size):
            batch_ids = chunk_ids[i:i + batch_size]
            
            try:
                # Create filter expression for batch
                id_list_str = "[" + ",".join([f'"{chunk_id}"' for chunk_id in batch_ids]) + "]"
                filter_expr = f"id in {id_list_str}"
                
                results = self.milvus_client.query(
                    collection_name=collection_name,
                    filter=filter_expr,
                    output_fields=["id", "vector"]
                )
                
                for result in results:
                    embeddings_dict[result["id"]] = result["vector"]
                    
            except Exception as e:
                print(f"Error retrieving chunk embeddings for batch {i//batch_size + 1}: {e}")
                continue
                
        return embeddings_dict
    
    def step3_expand_chunk_sets(self, qa_pairs_with_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Step 3: Expand chunk sets based on similarity thresholds"""
        if self.qa_final_path.exists():
            print(f"Found existing final file: {self.qa_final_path}")
            with open(self.qa_final_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        print("Step 3: Expanding chunk sets with similarity thresholds")
        
        qa_pairs_final = []
        total_expanded = 0
        
        for qa in tqdm(qa_pairs_with_candidates, desc="Expanding chunk sets"):
            # Get candidate chunk IDs
            candidate_ids = [chunk["id"] for chunk in qa["relative_chunks"]]
            
            if not candidate_ids:
                # No candidates found, keep original
                final_qa = qa.copy()
                final_qa.pop('query_embedding', None)
                final_qa.pop('ground_truth_answer_embedding', None)
                final_qa.pop('relative_chunks', None)
                qa_pairs_final.append(final_qa)
                continue
            
            # Get chunk embeddings
            chunk_embeddings = self.get_chunk_embeddings(candidate_ids)
            
            # Convert QA embeddings to numpy arrays
            query_embedding = np.array(qa['query_embedding'])
            answer_embedding = np.array(qa['ground_truth_answer_embedding'])
            
            # Start with original chunk_ids
            expanded_chunk_ids = set(qa['chunk_ids'])
            original_count = len(expanded_chunk_ids)
            
            # Check similarity for each candidate
            for chunk in qa["relative_chunks"]:
                chunk_id = chunk["id"]
                
                if chunk_id in chunk_embeddings:
                    chunk_embedding = np.array(chunk_embeddings[chunk_id])
                    
                    # Compute similarities using dot product (L2 normalized vectors)
                    sim_q = float(np.dot(query_embedding, chunk_embedding))
                    sim_a = float(np.dot(answer_embedding, chunk_embedding))
                    
                    # Add if similarity meets threshold
                    if sim_q >= 0.85 or sim_a >= 0.85:
                        expanded_chunk_ids.add(chunk_id)
            
            # Track expansion
            if len(expanded_chunk_ids) > original_count:
                total_expanded += 1
            
            # Create final QA entry
            final_qa = qa.copy()
            final_qa['chunk_ids'] = list(expanded_chunk_ids)
            
            # Remove embeddings and candidates from final output
            final_qa.pop('query_embedding', None)
            final_qa.pop('ground_truth_answer_embedding', None)
            final_qa.pop('relative_chunks', None)
            
            qa_pairs_final.append(final_qa)
        
        # Save final results
        print(f"Saving final QA pairs to {self.qa_final_path}")
        with open(self.qa_final_path, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs_final, f, ensure_ascii=False, indent=2)
        
        print(f"Expanded chunk sets for {total_expanded} QA pairs")
        return qa_pairs_final
    
    def step4_create_validation_sample(self, qa_pairs_final: List[Dict[str, Any]], 
                                     qa_pairs_with_candidates: List[Dict[str, Any]], 
                                     sample_size: int = 200):
        """Step 4: Create human validation CSV sample"""
        if self.sample_csv_path.exists():
            print(f"Found existing sample file: {self.sample_csv_path}")
            return
        
        print(f"Step 4: Creating validation sample (N={sample_size})")
        
        # Select sample
        total_qa = len(qa_pairs_final)
        if total_qa <= sample_size:
            sample_indices = list(range(total_qa))
        else:
            # Stratified sampling by confidence_hint if available
            import random
            random.seed(42)
            sample_indices = random.sample(range(total_qa), sample_size)
        
        # Create CSV
        with open(self.sample_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'qa_index', 'query', 'ground_truth_answer', 'answer_type', 
                'confidence_hint', 'original_chunk_ids', 'final_chunk_ids',
                'top_5_candidates', 'expansion_count'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for idx in sample_indices:
                qa_final = qa_pairs_final[idx]
                qa_with_candidates = qa_pairs_with_candidates[idx]
                
                # Get top 5 candidates
                top_5_candidates = []
                if 'relative_chunks' in qa_with_candidates:
                    top_5_candidates = [
                        f"{chunk['id']} (sim: {chunk['similarity']:.3f})"
                        for chunk in qa_with_candidates['relative_chunks'][:5]
                    ]
                
                # Calculate expansion
                original_ids = set(qa_pairs_final[0].get('chunk_ids', []))  # Use first QA's structure
                final_ids = set(qa_final.get('chunk_ids', []))
                expansion_count = len(final_ids) - len(original_ids)
                
                writer.writerow({
                    'qa_index': idx,
                    'query': qa_final['query'],
                    'ground_truth_answer': qa_final['ground_truth_answer'],
                    'answer_type': qa_final.get('answer_type', ''),
                    'confidence_hint': qa_final.get('confidence_hint', ''),
                    'original_chunk_ids': '; '.join(original_ids),
                    'final_chunk_ids': '; '.join(final_ids),
                    'top_5_candidates': '; '.join(top_5_candidates),
                    'expansion_count': expansion_count
                })
        
        print(f"Created validation sample with {len(sample_indices)} entries")
    
    def print_summary(self, qa_pairs: List[Dict[str, Any]], qa_pairs_final: List[Dict[str, Any]]):
        """Print pipeline summary"""
        print("\n" + "="*60)
        print("QA EMBEDDING PIPELINE SUMMARY")
        print("="*60)
        
        total_qa = len(qa_pairs)
        total_embeddings = total_qa * 2  # query + answer embeddings
        
        # Count expansions
        expanded_count = 0
        total_original_chunks = 0
        total_final_chunks = 0
        
        for i, qa_final in enumerate(qa_pairs_final):
            original_chunks = len(qa_pairs[i]['chunk_ids'])
            final_chunks = len(qa_final['chunk_ids'])
            
            total_original_chunks += original_chunks
            total_final_chunks += final_chunks
            
            if final_chunks > original_chunks:
                expanded_count += 1
        
        print(f"QA pairs processed: {total_qa}")
        print(f"Embeddings created: {total_embeddings}")
        print(f"QAs with expanded chunks: {expanded_count}")
        print(f"Original chunks total: {total_original_chunks}")
        print(f"Final chunks total: {total_final_chunks}")
        print(f"Chunk expansion: +{total_final_chunks - total_original_chunks}")
        print(f"Estimated embedding cost: ${self.embedding_cost:.4f}")
        print("="*60)
    
    def run_pipeline(self):
        """Run the complete QA embedding pipeline"""
        print("Starting QA Embedding Pipeline")
        print("="*60)
        
        try:
            # Load input data
            qa_pairs = self.load_qa_pairs()
            
            # Step 1: Generate embeddings
            qa_pairs_with_embeddings = self.step1_generate_qa_embeddings(qa_pairs)
            
            # Step 2: Retrieve candidates
            qa_pairs_with_candidates = self.step2_retrieve_top50_candidates(qa_pairs_with_embeddings)
            
            # Step 3: Expand chunk sets
            qa_pairs_final = self.step3_expand_chunk_sets(qa_pairs_with_candidates)
            
            # Step 4: Create validation sample
            self.step4_create_validation_sample(qa_pairs_final, qa_pairs_with_candidates)
            
            # Print summary
            self.print_summary(qa_pairs, qa_pairs_final)
            
            print("Pipeline completed successfully!")
            return True
            
        except Exception as e:
            print(f"Pipeline failed: {e}")
            return False

def main():
    """Main execution function"""
    pipeline = QAEmbeddingPipeline()
    success = pipeline.run_pipeline()
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
