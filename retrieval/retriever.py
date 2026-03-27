from loguru import logger
from collections import defaultdict

from embedding.embedder import Embedder
from storage.qdrant_store import QdrantStore
from storage.postgres_store import PostgresStore
from config import settings

from rank_bm25 import BM25Okapi


class HybridRetriever:
    def __init__(self):
        self.embedder = Embedder()
        self.qdrant = QdrantStore(dimension=self.embedder.dimension)
        self.postgres = PostgresStore()

        # BM25 index is built at startup from all chunks in Postgres
        self._bm25: BM25Okapi | None = None
        self._bm25_chunks: list[dict] = []
        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        logger.info("Building BM25 index from Postgres")
        chunks = self.postgres.get_all_chunks()
        if not chunks:
            logger.warning("No chunks found in Postgres — BM25 index is empty")
            return

        self._bm25_chunks = chunks
        tokenized = [c["content"].lower().split() for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.success(f"BM25 index built over {len(chunks)} chunks")

    def retrieve(self, query: str, top_k: int = settings.RETRIEVAL_TOP_K) -> list[dict]:
        # 1. Dense retrieval via Qdrant
        query_vector = self.embedder.embed_chunks([
            _QueryChunk(content=query)
        ])[0][1]

        dense_results = self.qdrant.search(query_vector, top_k=top_k)

        # 2. Sparse retrieval via BM25
        sparse_results = self._bm25_search(query, top_k=top_k)

        # 3. Fuse with RRF
        fused = self._reciprocal_rank_fusion(dense_results, sparse_results)

        logger.debug(
            f"Retrieval: {len(dense_results)} dense + {len(sparse_results)} sparse "
            f"→ {len(fused)} fused results"
        )
        return fused

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        if not self._bm25 or not self._bm25_chunks:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        # Pair chunks with scores and sort
        scored = sorted(
            zip(self._bm25_chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [chunk for chunk, score in scored[:top_k]]

    def _reciprocal_rank_fusion(
        self,
        dense: list[dict],
        sparse: list[dict],
        k: int = settings.RRF_K,
    ) -> list[dict]:
        # Map chunk_id → payload for deduplication
        all_chunks: dict[str, dict] = {}
        scores: dict[str, float] = defaultdict(float)

        for rank, chunk in enumerate(dense):
            cid = chunk["chunk_id"]
            all_chunks[cid] = chunk
            scores[cid] += 1.0 / (k + rank + 1)

        for rank, chunk in enumerate(sparse):
            cid = chunk["chunk_id"]
            all_chunks[cid] = chunk
            scores[cid] += 1.0 / (k + rank + 1)

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        results = []
        for cid in sorted_ids:
            chunk = all_chunks[cid].copy()
            chunk["rrf_score"] = round(scores[cid], 6)
            results.append(chunk)

        return results


# Minimal stub so we can embed a plain query string using the existing Embedder
class _QueryChunk:
    def __init__(self, content: str):
        self.content = content