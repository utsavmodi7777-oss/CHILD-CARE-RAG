"""
Generate Embeddings using Local SentenceTransformer Model
Much faster than API calls, no rate limits
"""
import json
import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

load_dotenv()

from pymilvus import MilvusClient
from src.config.vector_config import COLLECTION_CONFIG, ZILLIZ_CONFIG, SCHEMA_CONFIG

# Use a lightweight but effective model
MODEL_NAME = "all-MiniLM-L6-v2"  # 384 dimensions, very fast
# Note: We'll need to update vector config for 384 dimensions

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
        except Exception as e:
            print(f"Error loading {json_file.name}: {e}")
    
    print(f"\nTotal chunks loaded: {len(all_chunks)}")
    return all_chunks

def generate_embeddings(chunks: List[Dict]) -> List[Dict]:
    """Generate embeddings using local SentenceTransformer model"""
    print(f"\nLoading local model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"Model loaded! Embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    embeddings_data = []
    batch_size = 100  # Can process much larger batches locally
    
    print("\nGenerating embeddings (local, no rate limits)...")
    for i in tqdm(range(0, len(chunks), batch_size), desc="Generating embeddings"):
        batch = chunks[i:i + batch_size]
        texts = [chunk['context_enriched_content'] or chunk['content'] for chunk in batch]
        
        # Generate embeddings - very fast locally
        batch_embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        
        # Combine with metadata
        for j, chunk in enumerate(batch):
            metadata = chunk.get('metadata', {})
            embeddings_data.append({
                'id': i + j,
                'vector': batch_embeddings[j].tolist(),
                'content': chunk['content'],
                'context_enriched_content': chunk['context_enriched_content'] or chunk['content'],
                'source_file': metadata.get('source_file', ''),
                'chunk_index': chunk.get('chunk_index', 0),
                'page_numbers': ','.join(map(str, metadata.get('page_numbers', []))),
                'token_count': metadata.get('token_count', 0),
                'binary_hash': str(metadata.get('binary_hash', ''))
            })
    
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
    
    # Get actual embedding dimension from first embedding
    actual_dim = len(embeddings_data[0]['vector'])
    print(f"Using embedding dimension: {actual_dim}")
    
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
            # Use actual dimension from embeddings
            field = FieldSchema(
                name=field_config["name"],
                dtype=DataType.FLOAT_VECTOR,
                dim=actual_dim
            )
        elif field_config["dtype"] == "int64":
            field = FieldSchema(
                name=field_config["name"],
                dtype=DataType.INT64
            )
        fields.append(field)
    
    schema = CollectionSchema(
        fields=fields,
        description=f"Childcare knowledge base with {MODEL_NAME} embeddings"
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
    print("Local Embeddings Generation & Database Population")
    print(f"Model: {MODEL_NAME} (fast, no rate limits)")
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
