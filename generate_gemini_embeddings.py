"""
Generate Embeddings using Google Gemini API (100% FREE - No credit card!)
Gemini has generous free tier: 15 requests/min
"""
import json
import os
import sys
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from tqdm import tqdm
import time

sys.path.insert(0, str(Path(__file__).parent / "src"))
load_dotenv()

import google.generativeai as genai
from pymilvus import MilvusClient
from config.vector_config import COLLECTION_CONFIG, ZILLIZ_CONFIG, SCHEMA_CONFIG

def load_qa_chunks() -> List[Dict]:
    """Load all QA chunks"""
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

def generate_embeddings_gemini(chunks: List[Dict], api_key: str) -> List[Dict]:
    """Generate embeddings using Google Gemini (FREE)"""
    print("\nGenerating embeddings with Google Gemini API (FREE)...")
    
    genai.configure(api_key=api_key)
    
    embeddings_data = []
    batch_size = 10  # Conservative for free tier (15 req/min limit)
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="Generating embeddings"):
        batch = chunks[i:i + batch_size]
        texts = [chunk['context_enriched_content'] or chunk['content'] for chunk in batch]
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Gemini embedding model (768 dimensions)
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=texts,
                    task_type="retrieval_document"
                )
                
                batch_embeddings = result['embedding'] if isinstance(result['embedding'][0], list) else [result['embedding']]
                
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
                
            except Exception as e:
                retry_count += 1
                error_msg = str(e)
                
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    # Rate limit - wait longer
                    wait_time = 60 if retry_count == 1 else 30
                    print(f"\nRate limit, waiting {wait_time}s (retry {retry_count}/{max_retries})...")
                    time.sleep(wait_time)
                elif retry_count < max_retries:
                    print(f"\nError, retry {retry_count}/{max_retries}: {error_msg[:100]}")
                    time.sleep(5)
                else:
                    print(f"\nFailed batch {i//batch_size + 1}: {error_msg[:100]}")
                    break
        
        # Rate limiting: 15 req/min = 4 sec between requests
        time.sleep(4)
        
        # Save progress every 50 batches
        if (i // batch_size + 1) % 50 == 0:
            print(f"\nProgress: {len(embeddings_data)}/{len(chunks)} embeddings")
    
    print(f"\nGenerated {len(embeddings_data)} embeddings")
    
    # Save embeddings
    embeddings_path = Path("new_data/embeddings")
    embeddings_path.mkdir(parents=True, exist_ok=True)
    
    print("Saving embeddings...")
    with open(embeddings_path / "embeddings.json", 'w', encoding='utf-8') as f:
        json.dump(embeddings_data, f, indent=2)
    
    print(f"Saved to {embeddings_path / 'embeddings.json'}")
    return embeddings_data

def create_and_populate_database(embeddings_data: List[Dict]):
    """Create collection and insert embeddings"""
    print("\nConnecting to Zilliz...")
    
    client = MilvusClient(
        uri=ZILLIZ_CONFIG["uri"],
        token=ZILLIZ_CONFIG["token"]
    )
    
    collection_name = COLLECTION_CONFIG["name"]
    
    # Drop existing
    collections = client.list_collections()
    if collection_name in collections:
        print(f"Dropping existing collection: {collection_name}")
        client.drop_collection(collection_name)
    
    # Create with 768 dimensions (Gemini text-embedding-004)
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
                dim=768  # Gemini embedding dimension
            )
        elif field_config["dtype"] == "int64":
            field = FieldSchema(
                name=field_config["name"],
                dtype=DataType.INT64
            )
        fields.append(field)
    
    schema = CollectionSchema(
        fields=fields,
        description="Childcare knowledge base with Gemini embeddings"
    )
    
    client.create_collection(collection_name=collection_name, schema=schema)
    
    # Create index
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type=COLLECTION_CONFIG["index_type"],
        metric_type=COLLECTION_CONFIG["metric_type"]
    )
    client.create_index(collection_name=collection_name, index_params=index_params)
    
    print("Collection created")
    
    # Insert data
    print("\nInserting embeddings...")
    batch_size = 1000
    
    for i in tqdm(range(0, len(embeddings_data), batch_size), desc="Inserting"):
        batch = embeddings_data[i:i + batch_size]
        client.insert(collection_name=collection_name, data=batch)
    
    print(f"\n✅ Inserted {len(embeddings_data)} embeddings!")

def main():
    print("=" * 60)
    print("Google Gemini FREE Embeddings Generation")
    print("NO CREDIT CARD - 100% FREE")
    print("=" * 60)
    
    # Get API key
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("\n❌ GEMINI_API_KEY not found in .env file")
        print("\n🔑 Get your FREE API key:")
        print("1. Go to: https://aistudio.google.com/app/apikey")
        print("2. Click 'Create API key'")
        print("3. Add to .env file: GEMINI_API_KEY=your_key_here")
        return
    
    chunks = load_qa_chunks()
    if not chunks:
        return
    
    embeddings_data = generate_embeddings_gemini(chunks, gemini_key)
    if not embeddings_data:
        return
    
    create_and_populate_database(embeddings_data)
    
    print("\n" + "=" * 60)
    print("✅ ✅ ✅ COMPLETE! ✅ ✅ ✅")
    print("=" * 60)
    print("\nNow start the app:")
    print("cd chainlit-app")
    print("chainlit run app.py")

if __name__ == "__main__":
    main()
