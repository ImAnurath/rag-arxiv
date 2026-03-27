## File by file

### `config.py`
Central settings file. Every tuneable parameter lives here, the arXiv search query, how many papers to fetch, chunk size, and database URLs for later phases. Values can be overridden by creating a `.env` file in the project root without touching source code.

Key settings:
- `ARXIV_SEARCH_QUERY` — the arXiv category filter. Default is `cat:cs.AI OR cat:cs.LG OR cat:cs.CL`, which covers AI, machine learning, and NLP papers.
- `ARXIV_MAX_RESULTS` — how many papers to fetch per run. Keep this low (10–20) while developing.
- `CHUNK_SIZE` — max characters per chunk. 512 is a good default for most embedding models.
- `CHUNK_OVERLAP` — how many characters bleed between adjacent chunks. 64 is intentionally small to avoid bloating the vector store.

---

### `ingestion/base.py`
Defines the two core abstractions the whole pipeline is built on.

**`Document`** is the normalised object that every loader produces. No matter whether the source is a PDF, a webpage, or a CSV, it always becomes a `Document` with the same fields:

| Field | What it holds |
|---|---|
| `doc_id` | SHA-256 hash of content + source. Stable and deterministic. |
| `content` | The full extracted text of the paper. |
| `source` | The arXiv short ID, e.g. `2401.00001`. |
| `source_type` | An enum: `arxiv_pdf`, `web`, `csv`, `markdown`. |
| `title` | Paper title from the arXiv API. |
| `authors` | List of author names. |
| `published_at` | Original publication date. |
| `extra_metadata` | Catch-all dict for abstract, categories, DOI, PDF URL. |

**`BaseLoader`** is the interface all loaders inherit from. It enforces one method: `load() -> list[Document]`. Adding a new source (e.g. a web scraper) in a future phase means writing one new class that inherits `BaseLoader` — nothing else changes.

---

### `ingestion/arxiv_loader.py`
The main workhorse of Phase 1. Does three things:

**1. Fetches paper metadata from the arXiv API**
Uses the `arxiv` Python library to search for papers matching your query, sorted by submission date (newest first). For each result it gets the title, authors, abstract, categories, DOI, and PDF URL.

**2. Downloads PDFs**
Downloads each paper's PDF into `data/pdfs/` using `httpx`. PDFs are cached — if a file already exists on disk it is not downloaded again. This means re-running the pipeline is fast and doesn't hammer arXiv's servers.

**3. Extracts and cleans text**
Opens each PDF with `PyMuPDF` (`fitz`) and extracts the raw text page by page. Then applies basic cleaning:
- Collapses runs of 3+ blank lines down to 2
- Removes repeated whitespace
- Strips common arXiv header artifacts (version strings like `arXiv:2401.00001v2`)

The result is the `content` field of a `Document` — the full paper text as a single cleaned string.

> **Why only PDFs are downloaded when you run it:** The loader does everything — fetch, download, extract, clean — in `load()`. The output is a list of `Document` objects printed to the logger. In Phase 2, these documents will be passed directly to the embedder. Right now, `ingest.py` calls `load()` and then passes documents to the chunker, but since there is no storage yet, nothing is visibly persisted beyond the PDFs and the hash file.

---

### `ingestion/chunker.py`
Takes a list of `Document` objects and splits each one into smaller pieces called `Chunk` objects.

**Why chunking is necessary:** Embedding models have a token limit (typically 512–8192 tokens). A full arXiv paper is 6,000–15,000 words — far too long to embed as one unit. More importantly, you want retrieval to return a *specific relevant passage*, not an entire paper.

**How `RecursiveChunker` works:**
It tries to split on the largest meaningful boundary first, then falls back progressively:

```
paragraph break (\n\n)  →  line break (\n)  →  sentence end (". ")  →  space  →  character
```

This means it will almost never cut in the middle of a sentence. If a paragraph fits within `CHUNK_SIZE` characters, it stays whole. Only if it is too long does it split further.

Each `Chunk` carries:

| Field | What it holds |
|---|---|
| `chunk_id` | 16-char hash of `doc_id + chunk_index`. Unique per chunk. |
| `doc_id` | Links back to the parent `Document`. |
| `content` | The text of this specific chunk. |
| `chunk_index` | Position of this chunk within the document (0, 1, 2, …). |
| `source` | Inherited from the parent document (arXiv ID). |
| `title`, `authors` | Inherited — kept on every chunk so retrieval results are self-contained. |
| `extra_metadata` | Inherited — includes abstract, categories, DOI. |

---

### `ingestion/deduplicator.py`
Prevents the same chunk from being processed twice across multiple pipeline runs.

**How it works:** Every `chunk_id` that passes through is written to `data/seen_hashes.json`. On the next run, any chunk whose ID is already in that file is silently dropped before it reaches the embedder.

This matters because:
- You will run the ingestion pipeline many times during development
- arXiv search results overlap between runs (a paper from yesterday appears in today's results too)
- Without dedup, your vector store would fill up with duplicate vectors, degrading retrieval quality

> **Note:** This is a file-based deduplicator for Phase 1 only. In Phase 2 it will be replaced with a database lookup against the Qdrant collection, which is more reliable and scalable.

---

### `ingest.py`
The entry point. Wires all the components together in order:

```python
loader      → documents        # Fetch + extract PDFs
chunker     → chunks           # Split into passages
deduplicator → fresh_chunks    # Remove already-seen chunks
# (Phase 2) embedder → store   # Embed + write to Qdrant
```

At the end of a run the logger tells you exactly how many documents were loaded, how many total chunks were produced, and how many were new after deduplication.

---

## Data flow diagram

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
list[Chunk]  ← ready for Phase 2 (embedding + Qdrant)
```

---

## Running Phase 1

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: create a .env to override defaults
echo "ARXIV_MAX_RESULTS=10" > .env
echo "ARXIV_SEARCH_QUERY=cat:cs.AI" >> .env

# Run
python ingest.py
```

Expected output:
```
INFO  | Fetching up to 10 papers for: cat:cs.AI
INFO  | Documents loaded: 10
INFO  | Total chunks produced: 412
INFO  | Deduplication: removed 0 already-seen chunks  (first run)
INFO  | Fresh chunks ready for embedding: 412
```

On a second run with the same query:
```
INFO  | Deduplication: removed 412 already-seen chunks
INFO  | Fresh chunks ready for embedding: 0
```

---

## What is produced on disk

```
data/
├── pdfs/
│   ├── 2401.00001v2.pdf
│   ├── 2401.00002v1.pdf
│   └── ...
└── seen_hashes.json       ← chunk IDs that have already been processed
```

The `Document` and `Chunk` objects exist only in memory during this phase. Persistent storage (Qdrant for vectors, PostgreSQL for metadata) is introduced in Phase 2.

---

## Design decisions worth knowing

**Why `doc_id` is a hash, not a UUID:** A UUID changes every run. A hash of content + source is deterministic — the same paper always gets the same ID. This makes deduplication and debugging much easier.

**Why `chunk_overlap=64` is small:** With 512-character chunks, a 128-character overlap (25%) would mean every fourth chunk is pure repeat. 64 characters (12.5%) gives just enough context continuity without significantly inflating the vector store.

**Why metadata is copied onto every chunk:** When retrieval returns a chunk, you immediately know its title, authors, and source without doing a second database lookup. This is a deliberate denormalization for read performance.

**Why `BaseLoader` exists:** Every future loader (web, CSV, Markdown) inherits it and outputs the same `Document` format. The chunker, deduplicator, and embedder never need to know what source the data came from.