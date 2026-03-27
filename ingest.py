from loguru import logger

from ingestion.arxiv_loader import ArxivLoader
from ingestion.chunker import RecursiveChunker
from ingestion.deduplicator import Deduplicator
from embedding.embedder import Embedder
from storage.qdrant_store import QdrantStore
from storage.postgres_store import PostgresStore


def run_ingestion():
    logger.info("=== Starting ingestion pipeline ===")

    # Phase 1 — load, chunk, deduplicate
    loader = ArxivLoader()
    documents = loader.load()
    logger.info(f"Documents loaded: {len(documents)}")

    chunker = RecursiveChunker()
    chunks = chunker.chunk(documents)

    deduplicator = Deduplicator()
    fresh_chunks = deduplicator.filter(chunks)

    if not fresh_chunks:
        logger.info("No new chunks to process — pipeline complete")
        return

    # Phase 2 — embed + store
    embedder = Embedder()
    embedded = embedder.embed_chunks(fresh_chunks)

    if embedder.dimension is None:
        raise ValueError("Embedder dimension is not initialized")
    
    qdrant = QdrantStore(dimension=embedder.dimension)
    qdrant.upsert(embedded)

    postgres = PostgresStore()
    postgres.insert_chunks(fresh_chunks)

    # Summary
    logger.info("=== Pipeline complete ===")
    logger.info(f"Vectors in Qdrant: {qdrant.count()}")


if __name__ == "__main__":
    run_ingestion()