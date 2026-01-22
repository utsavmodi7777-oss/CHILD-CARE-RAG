"""
Configuration module for the Childcare RAG System.
Contains environment variables, model settings, and system parameters.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings and configuration."""
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    tavily_api_key: Optional[str] = Field(default=None, env="TAVILY_API_KEY")
    cohere_api_key: Optional[str] = Field(default=None, env="COHERE_API_KEY")
    cohere_api_key_2: Optional[str] = Field(default=None, env="COHERE_API_KEY_2")
    gemini_api_key: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    
    # Zilliz Cloud Configuration
    zilliz_token: Optional[str] = Field(default=None, env="ZILLIZ_TOKEN")
    zilliz_uri: Optional[str] = Field(default=None, env="ZILLIZ_URI")
    
    # LLM Configuration
    openai_model: str = Field(default="gpt-4o-mini", env="OPENAI_MODEL")
    default_model: str = Field(default="gpt-4o-mini", env="DEFAULT_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")
    
    # GPU Configuration
    use_gpu: bool = Field(default=True, env="USE_GPU")
    gpu_device: int = Field(default=0, env="GPU_DEVICE")
    gpu_memory_limit: Optional[float] = Field(default=None, env="GPU_MEMORY_LIMIT")  # GB
    
    # Vector Database Configuration
    milvus_host: str = Field(default="localhost", env="MILVUS_HOST")
    milvus_port: int = Field(default=19530, env="MILVUS_PORT")
    milvus_collection_name: str = Field(default="childcare_documents", env="MILVUS_COLLECTION_NAME")
    
    # Document Processing
    chunk_size: int = Field(default=1024, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=128, env="CHUNK_OVERLAP")  # Used for non-Docling chunkers only
    max_document_size: int = Field(default=41943040, env="MAX_DOCUMENT_SIZE")  # 40MB
    
    # Retrieval Configuration
    top_k_retrieval: int = Field(default=5, env="TOP_K_RETRIEVAL")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    adaptive_retrieval_enabled: bool = Field(default=True, env="ADAPTIVE_RETRIEVAL_ENABLED")
    
    # Web Search Configuration
    web_search_enabled: bool = Field(default=True, env="WEB_SEARCH_ENABLED")
    max_web_results: int = Field(default=3, env="MAX_WEB_RESULTS")
    
    # Generation Configuration
    max_tokens: int = Field(default=2048, env="MAX_TOKENS")
    max_tokens_per_request: int = Field(default=8192, env="MAX_TOKENS_PER_REQUEST")
    temperature: float = Field(default=0.1, env="TEMPERATURE")
    
    # Application Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    debug_mode: bool = Field(default=False, env="DEBUG_MODE")
    
    # Chainlit Configuration
    chainlit_auth_secret: str = Field(default="", env="CHAINLIT_AUTH_SECRET")
    auth_user_1: Optional[str] = Field(default=None, env="AUTH_USER_1")
    auth_user_2: Optional[str] = Field(default=None, env="AUTH_USER_2")
    auth_user_3: Optional[str] = Field(default=None, env="AUTH_USER_3")
    auth_user_4: Optional[str] = Field(default=None, env="AUTH_USER_4")
    
    # Data Paths
    pdf_data_path: str = Field(default="../pdfs", env="PDF_DATA_PATH")
    processed_data_path: str = Field(default="./data/processed", env="PROCESSED_DATA_PATH")
    new_processed_data_path: str = Field(default="./new_data/processed", env="NEW_PROCESSED_DATA_PATH")
    new_embeddings_path: str = Field(default="./new_data/embeddings", env="NEW_EMBEDDINGS_PATH")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
