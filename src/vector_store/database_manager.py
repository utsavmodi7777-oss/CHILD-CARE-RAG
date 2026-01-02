"""
Vector database manager for Childcare RAG System
Handles Zilliz Cloud database operations including creation, deletion, and data population
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.vector_config import ZILLIZ_CONFIG, COLLECTION_CONFIG, SCHEMA_CONFIG
from src.config.settings import settings

logger = logging.getLogger(__name__)

class VectorDatabaseManager:
    """Manages Zilliz Cloud vector database operations"""
    
    def __init__(self):
        self.client = None
        self.collection_name = COLLECTION_CONFIG["name"]
        self.embeddings_path = Path(settings.new_embeddings_path) / "embeddings.json"
        
    def connect(self) -> bool:
        """Establish connection to Zilliz Cloud"""
        try:
            self.client = MilvusClient(
                uri=ZILLIZ_CONFIG["uri"],
                token=ZILLIZ_CONFIG["token"]
            )
            logger.info("Successfully connected to Zilliz Cloud")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Zilliz Cloud: {e}")
            return False
    
    def delete_collection(self) -> bool:
        """Delete existing collection"""
        try:
            if not self.client:
                logger.error("Not connected to database")
                return False
                
            collections = self.client.list_collections()
            if self.collection_name in collections:
                self.client.drop_collection(collection_name=self.collection_name)
                logger.info(f"Successfully deleted collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} does not exist")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False
    
    def create_collection(self) -> bool:
        """Create new collection with proper schema"""
        try:
            if not self.client:
                logger.error("Not connected to database")
                return False
            
            # Create schema fields
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
            
            # Create schema
            schema = CollectionSchema(
                fields=fields,
                description="Childcare knowledge base embeddings"
            )
            
            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema
            )
            
            # Create index
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type=COLLECTION_CONFIG["index_type"],
                metric_type=COLLECTION_CONFIG["metric_type"]
            )
            
            self.client.create_index(
                collection_name=self.collection_name,
                index_params=index_params
            )
            
            logger.info(f"Successfully created collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False
    
    def load_embeddings_data(self) -> List[Dict]:
        """Load embeddings data from JSON file"""
        try:
            if not self.embeddings_path.exists():
                logger.error(f"Embeddings file not found: {self.embeddings_path}")
                return []
            
            with open(self.embeddings_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Successfully loaded {len(data)} embeddings")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load embeddings data: {e}")
            return []
    
    def transform_data_for_milvus(self, embeddings_data: List[Dict]) -> List[Dict]:
        """Transform embeddings data for Milvus insertion"""
        transformed_data = []
        
        for item in embeddings_data:
            metadata = item.get("metadata", {})
            
            # Convert page_numbers list to string
            page_numbers = metadata.get("page_numbers", [])
            page_numbers_str = ",".join(map(str, page_numbers)) if page_numbers else ""
            
            # Convert binary_hash to string
            binary_hash = metadata.get("binary_hash", "")
            binary_hash_str = str(binary_hash) if binary_hash else ""
            
            transformed_item = {
                "id": item["id"],
                "vector": item["vector"],
                "content": item["content"],
                "context_enriched_content": item["context_enriched_content"],
                "source_file": metadata.get("source_file", ""),
                "chunk_index": metadata.get("chunk_index", 0),
                "page_numbers": page_numbers_str,
                "token_count": metadata.get("token_count", 0),
                "binary_hash": binary_hash_str
            }
            
            transformed_data.append(transformed_item)
        
        return transformed_data
    
    def insert_data(self, batch_size: int = 1000) -> bool:
        """Insert embeddings data into collection"""
        try:
            if not self.client:
                logger.error("Not connected to database")
                return False
            
            # Load embeddings data
            embeddings_data = self.load_embeddings_data()
            if not embeddings_data:
                return False
            
            # Transform data for Milvus
            milvus_data = self.transform_data_for_milvus(embeddings_data)
            
            # Insert data in batches
            total_inserted = 0
            for i in range(0, len(milvus_data), batch_size):
                batch = milvus_data[i:i + batch_size]
                
                # Insert batch
                result = self.client.insert(
                    collection_name=self.collection_name,
                    data=batch
                )
                
                total_inserted += len(batch)
                logger.info(f"Inserted batch {i//batch_size + 1}: {len(batch)} records (Total: {total_inserted})")
            
            logger.info(f"Successfully inserted {total_inserted} embeddings into collection")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert data: {e}")
            return False
    
    def setup_complete_database(self) -> bool:
        """Complete database setup: delete old, create new, insert data"""
        try:
            logger.info("Starting complete database setup")
            
            # Connect to database
            if not self.connect():
                return False
            
            # Delete existing collection
            if not self.delete_collection():
                return False
            
            # Create new collection
            if not self.create_collection():
                return False
            
            # Insert data
            if not self.insert_data():
                return False
            
            logger.info("Complete database setup finished successfully")
            return True
            
        except Exception as e:
            logger.error(f"Complete database setup failed: {e}")
            return False

def main():
    """Main entry point for database setup"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    manager = VectorDatabaseManager()
    success = manager.setup_complete_database()
    
    if success:
        logger.info("Database setup completed successfully")
    else:
        logger.error("Database setup failed")
    
    return success

if __name__ == "__main__":
    main()
