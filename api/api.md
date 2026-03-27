## API Layer

### `api/schemas.py`
Pydantic models for API request/response validation.

**Models:**

| Model | Purpose | Fields |
|---|---|---|
| `QueryRequest` | POST /query body | `query: str`, `top_k: int = 5` |
| `SourceInfo` | Paper provenance | `arxiv_id`, `title`, `authors` |
| `QueryResponse` | POST /query response | `answer`, `sources`, `retrieved_chunks`, `usage` |

---

### `api/main.py`
FastAPI application exposing the RAG pipeline as an HTTP API.

**Endpoints:**

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Returns `{"status": "ok", "collection": "arxiv_papers"}` |
| `/query` | POST | Main RAG query endpoint |

**Startup lifecycle:**
- Heavy components (retriever, reranker, generator) load once at startup via `lifespan` context manager
- Not instantiated per-request — avoids model reload latency

**Query flow:**
1. Validate request (non-empty query)
2. Hybrid retrieval → 20 candidates
3. Reranking → top 5 chunks
4. LLM generation → answer with citations
5. Return `QueryResponse`