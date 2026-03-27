## Storage Layer

### `storage/qdrant_store.py`
Vector storage and similarity search using Qdrant.

**What it does:**
- Creates collection on first run (if not exists)
- Stores chunk vectors with full metadata as payload
- Uses `chunk_id` hash as point ID (converted to uint64)
- Upserts in batches of 100 (Qdrant recommendation)

**Key methods:**
- `upsert(embedded_chunks)` — Write vectors + metadata
- `search(query_vector, top_k, source_type_filter)` — Cosine similarity search
- `count()` — Total vectors in collection

**Payload structure:**
```python
{
    "chunk_id": "...",
    "doc_id": "...",
    "content": "...",
    "chunk_index": 0,
    "source": "2401.00001",
    "source_type": "arxiv_pdf",
    "title": "...",
    "authors": [...],
    "abstract": "...",
    "categories": [...],
    "doi": "...",
}
```

---

### `storage/postgres_store.py`
SQLAlchemy-based metadata store for chunks.

**Schema (`chunks` table):**

| Column | Type | Notes |
|---|---|---|
| `id` | Integer | Auto-increment primary key |
| `chunk_id` | String(64) | Unique constraint |
| `doc_id` | String(64) | Parent document |
| `source` | String(256) | arXiv ID |
| `source_type` | String(64) | Source type enum |
| `title` | Text | Paper title |
| `authors` | JSON | Author list |
| `chunk_index` | Integer | Position in doc |
| `content_preview` | Text | First 300 chars |
| `ingested_at` | DateTime | Timestamp |
| `extra_metadata` | JSON | Abstract, categories, DOI |

**Key methods:**
- `insert_chunks(chunks)` — Insert with duplicate skipping
- `get_ingestion_stats()` — Total chunks + breakdown by source
- `get_all_chunks()` — Returns all chunks as dicts (for BM25 indexing)
- `_sanitize(value)` — Strips NUL bytes from strings