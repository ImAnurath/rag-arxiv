from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from loguru import logger
from typing import Optional

from ingestion.chunker import Chunk
from config import settings


class QdrantStore:
    def __init__(
        self,
        host: str = settings.QDRANT_HOST,
        port: int = settings.QDRANT_PORT,
        collection: str = settings.QDRANT_COLLECTION,
        dimension: int = settings.EMBEDDING_DIMENSION,
    ):
        self.client = QdrantClient(host=host, port=port)
        self.collection = collection
        self.dimension = dimension
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            logger.info(f"Creating Qdrant collection: {self.collection}")
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE,
                ),
            )
        else:
            logger.info(f"Collection already exists: {self.collection}")

    def upsert(self, embedded_chunks: list[tuple[Chunk, list[float]]]) -> None:
        """
        Write chunk vectors + metadata to Qdrant.
        Uses chunk_id as the point ID (hashed to int for Qdrant's uint64 requirement).
        """
        points = []
        for chunk, vector in embedded_chunks:
            point_id = abs(hash(chunk.chunk_id)) % (2**63)
            payload = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "source": chunk.source,
                "source_type": chunk.source_type,
                "title": chunk.title,
                "authors": chunk.authors,
                **chunk.extra_metadata,
            }
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        # Upsert in batches of 100 to stay within Qdrant's recommended limits
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=self.collection, points=batch)
            logger.debug(f"Upserted batch {i // batch_size + 1} ({len(batch)} points)")

        logger.success(f"Stored {len(points)} vectors in Qdrant collection '{self.collection}'")

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        source_type_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Basic similarity search. Optional filter by source_type.
        This will be extended in Phase 3 with hybrid retrieval.
        """
        query_filter = None
        if source_type_filter:
            query_filter = Filter(
                must=[FieldCondition(
                    key="source_type",
                    match=MatchValue(value=source_type_filter),
                )]
            )

        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        return [
            {"score": r.score, **r.payload}
            for r in results
        ]

    def count(self) -> int:
        return self.client.count(collection_name=self.collection).count