from pydantic_settings import BaseSettings
from pathlib import Path
'''
Centralized configurations
'''

class Settings(BaseSettings):
    # arXiv fetch settings
    ARXIV_MAX_RESULTS: int = 25 # how many papers to fetch
    ARXIV_SEARCH_QUERY: str = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL" # arXiv search query (https://arxiv.org/help/api/user-manual#search_query for syntax)
    ARXIV_DOWNLOAD_DIR: Path = Path("data/pdfs") # where to save downloaded PDFs TODO: still need to update it for current dir structure just in case

    # Chunking
    CHUNK_SIZE: int = 512 # target chunk size in characters (not exact, just a guideline for the chunker)
    CHUNK_OVERLAP: int = 64 # how many characters to overlap between chunks (for better context in retrieval)

    # Embedding
    # (https://huggingface.co/models?filter=sentence-transformers for options)
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2" # Hugging Face model name for embedding TODO: should get a better model down the line later on
    EMBEDDING_BATCH_SIZE: int = 64   # chunks per embedding batch
    EMBEDDING_DIMENSION: int = 384   # MiniLM output size

    # Qdrant
    QDRANT_HOST: str = "localhost" # Qdrant host URL
    QDRANT_PORT: int = 6333 # Qdrant port
    QDRANT_COLLECTION: str = "arxiv_papers" # name of the Qdrant collection to store vectors

    # PostgreSQL
    POSTGRES_URL: str = "postgresql://raguser:ragpass@localhost:5432/ragdb" # SQLAlchemy-style connection URL for PostgreSQL

    # Retrieval
    RETRIEVAL_TOP_K: int = 20       # candidates fetched from each retriever
    RERANK_TOP_K: int = 5           # final chunks passed to LLM after reranking
    RRF_K: int = 60                 # RRF constant — 60 is standard default
    
    # LLM provider selection
    # Options: "anthropic", "ollama", "openai"
    LLM_PROVIDER: str = "ollama"
    
    # Shared
    LLM_API_KEY: str = ""        # not needed for ollama
    LLM_MODEL: str = "llama3.2"  # override per provider in .env
    LLM_MAX_TOKENS: int = 1024
    
    # Ollama specific
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    # OpenAI-compatible specific (also works for Gemini via their OpenAI endpoint)
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    
    class Config:
        env_file = ".env"


settings = Settings()