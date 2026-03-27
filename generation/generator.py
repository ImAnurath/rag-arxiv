import anthropic
from loguru import logger
from config import settings

# I will update this and move it to config.py or .env later
# for now it is easier to keep it packed here
SYSTEM_PROMPT = """You are a research assistant with access to a curated set of arXiv papers.
Answer the user's question using ONLY the provided context chunks.
For every claim you make, cite the source paper using [Title, arXiv: ID] format.
If the context does not contain enough information to answer, say so clearly — do not hallucinate."""


class Generator:
    def __init__(self):
        # TODO: add support for other LLM providers later on, but for now I'm hardcoding Anthropic since that's what I have the most experience with
        # also at some point, I will implement local model support using something like huggingface transformers
        key = settings.LLM_API_KEY
        self.use_mock = not key or not key.startswith("sk-ant") # crude check for valid Anthropic API key, since they all start with "sk-ant"
        if self.use_mock:
            logger.warning("No valid ANTHROPIC_API_KEY set — using mock generator")
        else:
            self.client = anthropic.Anthropic(api_key=key)
            self.model = settings.LLM_MODEL

    def generate(self, query: str, chunks: list[dict]) -> dict:
        if self.use_mock:
            return self._mock_generate(query, chunks)

        context = self._build_context(chunks)
        prompt = f"Context:\n{context}\n\nQuestion: {query}"
        logger.debug(f"Calling {self.model} with {len(chunks)} context chunks")

        message = self.client.messages.create(
            model=self.model,
            max_tokens=settings.LLM_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        return {
            "answer": message.content[0].text,
            "sources": self._extract_sources(chunks),
            "usage": {
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            },
        }

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

    def _build_context(self, chunks: list[dict]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            title = chunk.get("title", "Unknown")
            source = chunk.get("source", "")
            content = chunk.get("content", "")
            parts.append(f"[{i}] {title} (arXiv: {source})\n{content}")
        return "\n\n---\n\n".join(parts)

    def _extract_sources(self, chunks: list[dict]) -> list[dict]:
        seen = set()
        sources = []
        for chunk in chunks:
            source = chunk.get("source", "")
            if source not in seen:
                seen.add(source)
                sources.append({
                    "arxiv_id": source,
                    "title": chunk.get("title", ""),
                    "authors": chunk.get("authors", []),
                })
        return sources