## Embedding Layer

### `embedding/embedder.py`
Batch embeds chunks using SentenceTransformers.

**Key features:**
- Loads `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- Processes chunks in batches of 64 to avoid OOM
- Normalizes embeddings for cosine similarity
- Returns `list[tuple[Chunk, list[float]]]`

**Usage:**
```python
embedder = Embedder()
embedded = embedder.embed_chunks(chunks)
# embedded = [(chunk1, [0.1, 0.2, ...]), (chunk2, [...]), ...]
```