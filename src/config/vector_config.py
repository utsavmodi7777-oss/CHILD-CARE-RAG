"""
Vector store configuration for Childcare RAG System
"""

import os
from typing import Dict, Any

# Zilliz Cloud Configuration
ZILLIZ_CONFIG = {
    "uri": os.getenv("ZILLIZ_URI"),
    "token": os.getenv("ZILLIZ_TOKEN"),
    "free_tier_limits": {
        "cu": 1,
        "vectors": 1000000,  # 1M vectors (sufficient for our 3,593 chunks)
        "storage": "10GB"
    }
}

# Local Milvus Configuration (Alternative)
LOCAL_MILVUS_CONFIG = {
    "uri": "./data/vectorstore/milvus_childcare.db",
    "token": None
}

# Collection Configuration
COLLECTION_CONFIG = {
    "name": "childcare_knowledge_base",
    "embedding_dimension": 1024,  # Cohere embed-english-v3.0
    "metric_type": "IP",  # Inner Product - optimal for L2 normalized embeddings (13x faster)
    "index_type": "AUTOINDEX"  # Zilliz Cloud optimized index
}

# Schema Configuration
SCHEMA_CONFIG = {
    "fields": [
        {
            "name": "id",
            "dtype": "varchar",
            "max_length": 512,
            "is_primary": True
        },
        {
            "name": "vector",
            "dtype": "float_vector",
            "dim": 1536
        },
        {
            "name": "content",
            "dtype": "varchar",
            "max_length": 32768
        },
        {
            "name": "context_enriched_content",
            "dtype": "varchar",
            "max_length": 32768
        },
        {
            "name": "source_file",
            "dtype": "varchar",
            "max_length": 512
        },
        {
            "name": "chunk_index",
            "dtype": "int64"
        },
        {
            "name": "page_numbers",
            "dtype": "varchar",
            "max_length": 1024
        },
        {
            "name": "token_count",
            "dtype": "int64"
        },
        {
            "name": "binary_hash",
            "dtype": "varchar",
            "max_length": 64
        }
    ]
}
