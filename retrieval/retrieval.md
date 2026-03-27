## Retrieval Layer

### `retrieval/retriever.py`
Hybrid retrieval combining dense (Qdrant) and sparse (BM25) search.

**Architecture:**
```
                    Query
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
    Dense (Qdrant)           Sparse (BM25)
    vector search              keyword search
         │                         │
         └────────────┬────────────┘
                      ▼
              Reciprocal Rank Fusion
                      │
                      ▼
              Ranked results
```

**How it works:**

1. **Dense retrieval** — Embeds query, searches Qdrant for top-20 similar vectors
2. **Sparse retrieval** — BM25 keyword search over all chunks in PostgreSQL
3. **Fusion** — Reciprocal Rank Fusion (RRF) combines rankings:
   ```
   score(chunk) = 1/(k + rank_dense) + 1/(k + rank_sparse)
   ```
   Default `k=60` is standard.

**BM25 index:**
- Built at startup from all chunks in PostgreSQL
- Uses `rank_bm25.BM25Okapi`
- Tokenizes by lowercase split on whitespace

---

### `retrieval/reranker.py`
Cross-encoder reranking for final selection.

**What it does:**
- Takes top-20 candidates from hybrid retriever
- Scores each `(query, chunk_content)` pair using cross-encoder
- Returns top-5 highest-scoring chunks

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Lightweight, fast
- Trained on MS MARCO dataset
- Outputs relevance score 0–1

**Why reranking matters:**
- Bi-encoders (like Qdrant's vectors) are fast but approximate
- Cross-encoders are slower but more accurate — they see query + document together
- Reranking narrows 20 candidates → 5 best for LLM context