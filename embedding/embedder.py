from sentence_transformers import SentenceTransformer
from loguru import logger
from typing import Generator
import numpy as np

from ingestion.chunker import Chunk
from config import settings


def _batched(items: list, size: int) -> Generator:
    for i in range(0, len(items), size):
        yield items[i : i + size]


class Embedder:
    def __init__(
        self,
        model_name: str = settings.EMBEDDING_MODEL,
        batch_size: int = settings.EMBEDDING_BATCH_SIZE,
    ):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.success(f"Model loaded — embedding dimension: {self.dimension}")

    def embed_chunks(self, chunks: list[Chunk]) -> list[tuple[Chunk, list[float]]]:
        """
        Returns a list of (chunk, embedding_vector) pairs.
        Processes in batches to avoid OOM on large inputs.
        """
        results: list[tuple[Chunk, list[float]]] = []
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

        for i, batch in enumerate(_batched(chunks, self.batch_size)):
            logger.debug(f"Embedding batch {i + 1}/{total_batches} ({len(batch)} chunks)")
            texts = [c.content for c in batch]
            vectors: np.ndarray = self.model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,  # cosine similarity works better normalised
            )
            for chunk, vector in zip(batch, vectors):
                results.append((chunk, vector.tolist()))

        logger.success(f"Embedded {len(results)} chunks")
        return results