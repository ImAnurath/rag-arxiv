from config import settings
from loguru import logger
from generation.providers import get_provider
from generation.providers.base import BaseLLMProvider

# I will update this and move it to config.py or .env later
# for now it is easier to keep it packed here
SYSTEM_PROMPT = """You are a research assistant with access to a curated set of arXiv papers.
Answer the user's question using ONLY the provided context chunks.
For every claim you make, cite the source paper using [Title, arXiv: ID] format.
If the context does not contain enough information to answer, say so clearly — do not hallucinate."""


class Generator:
    def __init__(self, provider: BaseLLMProvider | None = None):
        self.provider = provider or get_provider()

    def generate(self, query: str, chunks: list[dict]) -> dict:
        context = self._build_context(chunks)
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

        logger.debug(f"Generating answer for: {query[:60]}...")
        answer, usage = self.provider.complete(SYSTEM_PROMPT, user_prompt)

        return {
            "answer": answer,
            "sources": self._extract_sources(chunks),
            "usage": usage,
        }
    def _extract_sources(self, chunks: list[dict]) -> list[dict]:
        seen = set()
        sources = []
        for chunk in chunks:
            source = chunk.get("source", "")
            if source not in seen:
                seen.add(source)
                sources.append({
                    "arxiv_id": source,
                    "title":    chunk.get("title", ""),
                    "authors":  chunk.get("authors", []),
                })
        return sources
    
    def _build_context(self, chunks: list[dict]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            title   = chunk.get("title", "Unknown")
            source  = chunk.get("source", "")
            content = chunk.get("content", "")
            parts.append(f"[{i}] {title} (arXiv: {source})\n{content}")
        return "\n\n---\n\n".join(parts)

    def _mock_generate(self, query: str, chunks: list[dict]) -> dict:
        """
        Skips the LLM entirely. Returns the raw retrieved chunks so you can
        verify that retrieval + reranking are working correctly.
        """
        sources = self._extract_sources(chunks)
        preview_lines = []
        for i, chunk in enumerate(chunks, 1):
            preview_lines.append(
                f"[{i}] {chunk.get('title', 'Unknown')} (arXiv: {chunk.get('source', '')})\n"
                f"    Score: {chunk.get('rerank_score', 'n/a')}\n"
                f"    Preview: {chunk.get('content', '')[:150]}..."
            )
        answer = (
            f"[MOCK — no LLM call made]\n\n"
            f"Query: {query}\n\n"
            f"Top {len(chunks)} chunks after hybrid retrieval + reranking:\n\n"
            + "\n\n".join(preview_lines)
        )
        return {
            "answer": answer,
            "sources": sources,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }


