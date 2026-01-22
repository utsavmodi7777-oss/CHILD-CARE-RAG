"""
Generate Cohere Embeddings and Populate Zilliz Database
"""
import json
import os
import sys
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

load_dotenv()

from utils.client_manager import client_manager
from pymilvus import MilvusClient
from config.vector_config import COLLECTION_CONFIG, ZILLIZ_CONFIG, SCHEMA_CONFIG

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
            
            # Handle array format directly
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
            # Original format with 'chunks' key
            elif 'chunks' in data:
                for chunk in data['chunks']:
                    all_chunks.append({
                        'content': chunk.get('content', ''),
                        'context_enriched_content': chunk.get('context_enriched_content', ''),
                        'metadata': chunk.get('metadata', {}),
                        'chunk_index': chunk.get('chunk_index', 0)
                    })
                print(f"Loaded {len(data.get('chunks', []))} chunks from {json_file.name}")
        except Exception as e:
            print(f"Error loading {json_file.name}: {e}")
    
    print(f"\nTotal chunks loaded: {len(all_chunks)}")
    return all_chunks

def generate_embeddings(chunks: List[Dict]) -> List[Dict]:
    """Generate embeddings using Cohere"""
    print("\nGenerating Cohere embeddings...")
    
    embedding_client = client_manager.get_embedding_client()
    
    embeddings_data = []
    batch_size = 10  # Very small batch for free tier
    
    import time
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="Generating embeddings"):
        batch = chunks[i:i + batch_size]
        texts = [chunk['context_enriched_content'] or chunk['content'] for chunk in batch]
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Generate embeddings
                batch_embeddings = embedding_client.embed_documents(texts)
                
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
                
                # Success - break retry loop
                break
                
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"\nRetry {retry_count}/{max_retries} for batch {i//batch_size + 1}: {e}")
                    time.sleep(10)  # Wait longer on error
                else:
                    print(f"\nFailed batch {i//batch_size + 1} after {max_retries} retries: {e}")
        
        # Rate limiting: wait 3 seconds between batches
        time.sleep(3)
    
    print(f"\nGenerated {len(embeddings_data)} embeddings")
    
    # Save embeddings
    embeddings_path = Path("new_data/embeddings")
    embeddings_path.mkdir(parents=True, exist_ok=True)
    
    with open(embeddings_path / "embeddings.json", 'w', encoding='utf-8') as f:
        json.dump(embeddings_data, f, indent=2)
    
    print(f"Saved embeddings to {embeddings_path / 'embeddings.json'}")
    
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
    
    # Create new collection with schema
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
                dim=field_config["dim"]
            )
        elif field_config["dtype"] == "int64":
            field = FieldSchema(
                name=field_config["name"],
                dtype=DataType.INT64
            )
        fields.append(field)
    
    schema = CollectionSchema(
        fields=fields,
        description="Childcare knowledge base with Cohere embeddings"
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
    
    print(f"Collection created successfully")
    
    # Insert data in batches
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
    print("Cohere Embeddings Generation & Database Population")
    print("=" * 60)
    
    # Step 1: Load chunks
    chunks = load_qa_chunks()
    if not chunks:
        print("No chunks found. Exiting.")
        return
    
    # Step 2: Generate embeddings
    embeddings_data = generate_embeddings(chunks)
    if not embeddings_data:
        print("No embeddings generated. Exiting.")
        return
    
    # Step 3: Create and populate database
    create_and_populate_database(embeddings_data)
    
    print("\n" + "=" * 60)
    print("✅ Setup Complete!")
    print("=" * 60)
    print("\nYou can now start the Chainlit app:")
    print("cd chainlit-app")
    print("chainlit run app.py")

if __name__ == "__main__":
    main()
