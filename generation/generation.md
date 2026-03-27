## Generation Layer

### `generation/generator.py`
LLM-based answer generation with source citations.

**System prompt:**
> "You are a research assistant with access to a curated set of arXiv papers. Answer using ONLY the provided context. Cite sources as [Title, arXiv: ID]. Do not hallucinate."

**Key features:**
- Uses Anthropic Claude API
- Falls back to mock mode if no `LLM_API_KEY` is set
- Mock mode returns raw chunks for debugging retrieval

**Response structure:**
```python
{
    "answer": "...",           # LLM-generated response
    "sources": [...],          # Unique papers cited
    "usage": {                 # Token counts
        "input_tokens": 1234,
        "output_tokens": 567,
    }
}
```

**Methods:**
- `generate(query, chunks)` — Main entry point
- `_build_context(chunks)` — Formats chunks with citations
- `_extract_sources(chunks)` — Deduplicates source papers
- `_mock_generate(query, chunks)` — Debug mode, skips LLM