"""
Generate Embeddings using HuggingFace Inference API (FREE) and Populate Zilliz
Much faster than Cohere free tier - better rate limits!
"""
import json
import os
import sys
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from tqdm import tqdm
import time
import requests

sys.path.insert(0, str(Path(__file__).parent / "src"))
load_dotenv()

from pymilvus import MilvusClient
from config.vector_config import COLLECTION_CONFIG, ZILLIZ_CONFIG, SCHEMA_CONFIG

# HuggingFace Inference API endpoint (FREE - no API key needed for public models)
HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

def load_qa_chunks() -> List[Dict]:
    """Load all QA chunks from new_data/qa_chunks/"""
    qa_chunks_dir = Path("new_data/qa_chunks")
    all_chunks = []
    
    json_files = list(qa_chunks_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for chunk in data:
                    all_chunks.append({
                        'content': chunk.get('context_enriched_content', ''),
                        'context_enriched_content': chunk.get('context_enriched_content', ''),
                        'metadata': {
                            'source_file': json_file.name,
                            'chunk_id': chunk.get('id', ''),
                            'page_numbers': [],
                            'token_count': len(chunk.get('context_enriched_content', '').split()),
                            'binary_hash': ''
                        },
                        'chunk_index': len(all_chunks)
                    })
            print(f"Loaded {len(data)} chunks from {json_file.name}")
        except Exception as e:
            print(f"Error loading {json_file.name}: {e}")
    
    print(f"\nTotal chunks loaded: {len(all_chunks)}")
    return all_chunks

def generate_embeddings_hf(chunks: List[Dict]) -> List[Dict]:
    """Generate embeddings using HuggingFace Inference API (FREE)"""
    print("\nGenerating embeddings with HuggingFace API (FREE)...")
    
    embeddings_data = []
    batch_size = 5  # Conservative for free API
    
    headers = {"Content-Type": "application/json"}
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="Generating embeddings"):
        batch = chunks[i:i + batch_size]
        texts = [chunk['context_enriched_content'] or chunk['content'] for chunk in batch]
        
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Call HuggingFace API
                response = requests.post(
                    HF_API_URL,
                    headers=headers,
                    json={"inputs": texts, "options": {"wait_for_model": True}},
                    timeout=60
                )
                
                if response.status_code == 200:
                    batch_embeddings = response.json()
                    
                    # Combine with metadata
                    for j, chunk in enumerate(batch):
                        metadata = chunk.get('metadata', {})
                        embeddings_data.append({
                            'id': i + j,
                            'vector': batch_embeddings[j],
                            'content': chunk['content'],
                            'context_enriched_content': chunk['context_enriched_content'] or chunk['content'],
                            'source_file': metadata.get('source_file', ''),
                            'chunk_index': chunk.get('chunk_index', 0),
                            'page_numbers': ','.join(map(str, metadata.get('page_numbers', []))),
                            'token_count': metadata.get('token_count', 0),
                            'binary_hash': str(metadata.get('binary_hash', ''))
                        })
                    break  # Success
                    
                elif response.status_code == 503:
                    # Model loading
                    retry_count += 1
                    wait_time = 10 if retry_count == 1 else 5
                    print(f"\nModel loading, waiting {wait_time}s (retry {retry_count}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"\nError {response.status_code}, retry {retry_count}/{max_retries}...")
                        time.sleep(3)
                    else:
                        print(f"\nFailed batch {i//batch_size + 1}: {response.text}")
                        break
                        
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"\nException, retry {retry_count}/{max_retries}: {e}")
                    time.sleep(3)
                else:
                    print(f"\nFailed batch {i//batch_size + 1}: {e}")
                    break
        
        # Small delay between batches
        time.sleep(0.5)
        
        # Save progress every 100 batches
        if (i // batch_size + 1) % 100 == 0:
            print(f"\nProgress: {len(embeddings_data)} embeddings generated")
    
    print(f"\nGenerated {len(embeddings_data)} embeddings")
    
    # Save embeddings
    embeddings_path = Path("new_data/embeddings")
    embeddings_path.mkdir(parents=True, exist_ok=True)
    
    print("Saving embeddings to JSON...")
    with open(embeddings_path / "embeddings.json", 'w', encoding='utf-8') as f:
        json.dump(embeddings_data, f, indent=2)
    
    print(f"Saved to {embeddings_path / 'embeddings.json'}")
    return embeddings_data

def create_and_populate_database(embeddings_data: List[Dict]):
    """Create collection and insert embeddings into Zilliz"""
    print("\nConnecting to Zilliz Cloud...")
    
    client = MilvusClient(
        uri=ZILLIZ_CONFIG["uri"],
        token=ZILLIZ_CONFIG["token"]
    )
    
    collection_name = COLLECTION_CONFIG["name"]
    
    # Drop existing collection
    collections = client.list_collections()
    if collection_name in collections:
        print(f"Dropping existing collection: {collection_name}")
        client.drop_collection(collection_name)
    
    # Create collection with 384 dimensions (all-MiniLM-L6-v2)
    print(f"Creating collection: {collection_name}")
    from pymilvus import DataType, CollectionSchema, FieldSchema
    
    fields = []
    for field_config in SCHEMA_CONFIG["fields"]:
        if field_config["dtype"] == "varchar":
            field = FieldSchema(
                name=field_config["name"],
                dtype=DataType.VARCHAR,
                max_length=field_config["max_length"],
                is_primary=field_config.get("is_primary", False)
            )
        elif field_config["dtype"] == "float_vector":
            field = FieldSchema(
                name=field_config["name"],
                dtype=DataType.FLOAT_VECTOR,
                dim=384  # all-MiniLM-L6-v2
            )
        elif field_config["dtype"] == "int64":
            field = FieldSchema(
                name=field_config["name"],
                dtype=DataType.INT64
            )
        fields.append(field)
    
    schema = CollectionSchema(
        fields=fields,
        description="Childcare knowledge base with HF embeddings"
    )
    
    client.create_collection(
        collection_name=collection_name,
        schema=schema
    )
    
    # Create index
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type=COLLECTION_CONFIG["index_type"],
        metric_type=COLLECTION_CONFIG["metric_type"]
    )
    
    client.create_index(
        collection_name=collection_name,
        index_params=index_params
    )
    
    print("Collection created successfully")
    
    # Insert data
    print("\nInserting embeddings...")
    batch_size = 1000
    total_inserted = 0
    
    for i in tqdm(range(0, len(embeddings_data), batch_size), desc="Inserting data"):
        batch = embeddings_data[i:i + batch_size]
        client.insert(collection_name=collection_name, data=batch)
        total_inserted += len(batch)
    
    print(f"\nSuccessfully inserted {total_inserted} embeddings")
    print("Database setup complete!")

def main():
    print("=" * 60)
    print("HuggingFace FREE Embeddings Generation")
    print("Using: all-MiniLM-L6-v2 (384 dimensions)")
    print("=" * 60)
    
    chunks = load_qa_chunks()
    if not chunks:
        print("No chunks found. Exiting.")
        return
    
    embeddings_data = generate_embeddings_hf(chunks)
    if not embeddings_data:
        print("No embeddings generated. Exiting.")
        return
    
    create_and_populate_database(embeddings_data)
    
    print("\n" + "=" * 60)
    print("✅ Setup Complete!")
    print("=" * 60)
    print("\nNow start the Chainlit app:")
    print("cd chainlit-app")
    print("chainlit run app.py")

if __name__ == "__main__":
    main()
