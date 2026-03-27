from sentence_transformers import CrossEncoder
from loguru import logger
from config import settings


class Reranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = settings.RERANK_TOP_K,
    ):
        logger.info(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)
        self.top_k = top_k
        logger.success("Reranker ready")

    def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        if not chunks:
            return []

        pairs = [(query, c["content"]) for c in chunks]
        scores = self.model.predict(pairs)

        scored = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for chunk, score in scored[: self.top_k]:
            chunk = chunk.copy()
            chunk["rerank_score"] = round(float(score), 4)
            results.append(chunk)

        logger.debug(
            f"Reranker: {len(chunks)} candidates → top {len(results)} kept"
        )
        return results