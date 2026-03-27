## Core Configuration

### `config.py`
Central settings file using Pydantic's `BaseSettings`. Every tunable parameter lives here and can be overridden via a `.env` file.

**Key settings groups:**

| Category | Setting | Default | Description |
|---|---|---|---|
| **arXiv** | `ARXIV_MAX_RESULTS` | 10 | Papers to fetch per run |
| | `ARXIV_SEARCH_QUERY` | `cat:cs.AI OR cat:cs.LG OR cat:cs.CL` | arXiv category filter |
| | `ARXIV_DOWNLOAD_DIR` | `data/pdfs` | PDF cache directory |
| **Chunking** | `CHUNK_SIZE` | 512 | Target characters per chunk |
| | `CHUNK_OVERLAP` | 64 | Character overlap between chunks |
| **Embedding** | `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model |
| | `EMBEDDING_BATCH_SIZE` | 64 | Chunks per embedding batch |
| | `EMBEDDING_DIMENSION` | 384 | Output vector size |
| **Qdrant** | `QDRANT_HOST` | `localhost` | Vector DB host |
| | `QDRANT_PORT` | 6333 | Vector DB port |
| | `QDRANT_COLLECTION` | `arxiv_papers` | Collection name |
| **PostgreSQL** | `POSTGRES_URL` | `postgresql://raguser:ragpass@localhost:5432/ragdb` | Metadata store URL |
| **Retrieval** | `RETRIEVAL_TOP_K` | 20 | Candidates from each retriever |
| | `RERANK_TOP_K` | 5 | Final chunks to LLM |
| | `RRF_K` | 60 | Reciprocal Rank Fusion constant |
| **LLM** | `LLM_API_KEY` | (placeholder) | Anthropic API key |
| | `LLM_MODEL` | (placeholder) | Claude model name |
| | `LLM_MAX_TOKENS` | 1024 | Max generation tokens |

---

### `ingest.py`
The main entry point. Wires all components together.

**Pipeline flow:**
```
ArxivLoader → RecursiveChunker → Deduplicator → Embedder → Qdrant + PostgreSQL
```

**What happens:**
1. Loads papers from arXiv via `ArxivLoader`
2. Splits documents into chunks via `RecursiveChunker`
3. Removes duplicates via `Deduplicator`
4. Embeds fresh chunks via `Embedder`
5. Stores vectors in Qdrant via `QdrantStore`
6. Stores metadata in PostgreSQL via `PostgresStore`

**Logs summary:**
- Documents loaded
- Total chunks produced
- Duplicates removed
- Vectors stored in Qdrant

## Data Flow Diagram

```
arXiv API
    │  (metadata: title, authors, abstract, categories)
    ▼
ArxivLoader.load()
    │  downloads PDFs to data/pdfs/
    │  extracts + cleans full text
    ▼
list[Document]
    │  one Document per paper
    │  fields: doc_id, content, source, title, authors, published_at, ...
    ▼
RecursiveChunker.chunk()
    │  splits each Document into N passages
    ▼
list[Chunk]
    │  one Chunk per passage
    │  fields: chunk_id, doc_id, content, chunk_index, source, title, ...
    ▼
Deduplicator.filter()
    │  drops chunks already in data/seen_hashes.json
    ▼
list[Chunk] (fresh only)
    │
    ├─────────────────────────────────────┐
    ▼                                     ▼
Embedder.embed_chunks()           PostgresStore.insert_chunks()
    │                                     │
    ▼                                     ▼
list[(Chunk, vector)]              PostgreSQL chunks table
    │
    ▼
QdrantStore.upsert()
    │
    ▼
Qdrant collection (vectors + payload)


=== Query Time ===

User query
    │
    ├──────────────────────┐
    ▼                      ▼
Qdrant (dense)        Postgres (BM25)
cosine similarity      keyword search
    │                      │
    └──────────┬───────────┘
               ▼
    Reciprocal Rank Fusion
               │
               ▼
    Reranker (cross-encoder)
               │
               ▼
    Generator (Claude)
               │
               ▼
    Answer + citations
```

---

## Design Decisions

**Deterministic IDs:** `doc_id` and `chunk_id` are SHA-256 hashes of content. The same paper/chunk always gets the same ID, making deduplication and debugging trivial.

**Metadata denormalization:** Every `Chunk` carries `title`, `authors`, and `source`. When retrieval returns a chunk, you immediately know its provenance without a second database lookup.

**Recursive chunking:** Splits on paragraph → sentence → word boundaries in that order. Never cuts mid-sentence if a larger boundary fits within `CHUNK_SIZE`.

**Hybrid retrieval:** Dense (vector) search captures semantic similarity. Sparse (BM25) captures exact keyword matches. RRF fuses both for better coverage.

**Reranking:** Bi-encoders are fast but approximate. Cross-encoders are slower but more accurate. The hybrid retriever fetches 20 candidates; the reranker narrows to 5 best for the LLM.

**Mock generator:** If no `LLM_API_KEY` is set, the generator returns raw chunks instead of calling the LLM. Useful for debugging retrieval without burning tokens.
