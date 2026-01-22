
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    max_retrieval_queries: int = 3
    use_hyde: bool = True
    query_expansion_count: int = 3
    top_k_retrieval: int = 5
    embedding_provider: str = 'local'
    local_embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
settings = Settings()
