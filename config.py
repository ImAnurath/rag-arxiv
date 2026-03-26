from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # arXiv fetch settings
    ARXIV_MAX_RESULTS: int = 10 # how many papers to fetch
    ARXIV_SEARCH_QUERY: str = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL" # arXiv search query (see https://arxiv.org/help/api/user-manual#search_query for syntax)
    ARXIV_DOWNLOAD_DIR: Path = Path("data/pdfs") # where to save downloaded PDFs

    # Chunking
    CHUNK_SIZE: int = 512 # target chunk size in characters (not exact, just a guideline for the chunker)
    CHUNK_OVERLAP: int = 64 # how many characters to overlap between chunks (for better context in retrieval)

    # Embedding
    # Swap to "text-embedding-3-small" if you want OpenAI instead
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2" # Hugging Face model name for embedding (see https://huggingface.co/models?filter=sentence-transformers for options)
    EMBEDDING_BATCH_SIZE: int = 64   # chunks per embedding batch
    EMBEDDING_DIMENSION: int = 384   # MiniLM output size (1536 for OpenAI)

    # Qdrant
    QDRANT_HOST: str = "localhost" # Qdrant host URL
    QDRANT_PORT: int = 6333 # Qdrant port
    QDRANT_COLLECTION: str = "arxiv_papers" # name of the Qdrant collection to store vectors

    # PostgreSQL
    POSTGRES_URL: str = "postgresql://raguser:ragpass@localhost:5432/ragdb" # SQLAlchemy-style connection URL for PostgreSQL

    class Config:
        env_file = ".env"


settings = Settings()