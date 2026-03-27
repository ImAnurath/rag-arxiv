## Ingestion Pipeline

### `ingestion/base.py`
Defines the two core abstractions the whole pipeline is built on.

**`SourceType`** — Enum for tracking where data came from:
- `ARXIV_PDF`, `WEB`, `CSV`, `MARKDOWN`

**`Document`** — The normalized object every loader produces:

| Field | Type | Description |
|---|---|---|
| `doc_id` | `str` | SHA-256 hash of `source::content[:500]` — deterministic |
| `content` | `str` | Full extracted text |
| `source` | `str` | arXiv ID, URL, or file path |
| `source_type` | `SourceType` | Enum value |
| `title` | `str | None` | Paper title |
| `authors` | `list[str] | None` | Author list |
| `published_at` | `datetime | None` | Publication date |
| `ingested_at` | `datetime` | Auto-set to `utcnow()` |
| `extra_metadata` | `dict` | Abstract, categories, DOI, PDF URL |

**`BaseLoader`** — Abstract interface all loaders implement:
```python
def load(self) -> list[Document]:
    raise NotImplementedError
```

---

### `ingestion/arxiv_loader.py`
The main workhorse of Phase 1. Inherits from `BaseLoader`.

**What it does:**

1. **Fetches paper metadata from arXiv API**
   - Uses the `arxiv` Python library
   - Search sorted by `SubmittedDate` (newest first)
   - Extracts: title, authors, abstract, categories, DOI, PDF URL

2. **Downloads PDFs to disk**
   - Saves to `data/pdfs/{arxiv_id}.pdf`
   - Uses `httpx` with 60s timeout and redirect following
   - Caches PDFs — skips download if file exists

3. **Extracts and cleans text**
   - Opens PDFs with `PyMuPDF` (`fitz`)
   - Extracts text page-by-page
   - Applies cleanup:
     - Strips NUL bytes (PostgreSQL rejects them)
     - Collapses 3+ blank lines → 2
     - Removes repeated whitespace
     - Strips arXiv header artifacts (`arXiv:2401.00001v2...`)

**Key methods:**
- `load()` — Main entry point, returns `list[Document]`
- `_process_paper()` — Handles single paper download + extraction
- `_download_pdf()` — HTTP download with `httpx`
- `_extract_text()` — PyMuPDF text extraction
- `_clean_text()` — Regex-based cleanup

---

### `ingestion/chunker.py`
Splits `Document` objects into smaller `Chunk` objects.

**Why chunking is necessary:**
- Embedding models have token limits (512–8192 tokens)
- Full arXiv papers are 6,000–15,000 words — too long to embed
- Retrieval should return *specific passages*, not entire papers

**`Chunk` model:**

| Field | Description |
|---|---|
| `chunk_id` | 16-char SHA-256 hash of `doc_id::chunk_index` |
| `doc_id` | Parent document ID |
| `content` | The text of this chunk |
| `chunk_index` | Position in document (0, 1, 2, …) |
| `source` | Inherited from parent (arXiv ID) |
| `source_type` | Inherited from parent |
| `title`, `authors` | Inherited — kept for self-contained retrieval |
| `extra_metadata` | Inherited — abstract, categories, DOI |

**How `RecursiveChunker` works:**
Uses LangChain's `RecursiveCharacterTextSplitter` with fallback separators:
```
paragraph break (\n\n) → line break (\n) → sentence end (". ") → space → character
```
This ensures it never cuts mid-sentence if a larger boundary fits.

---

### `ingestion/deduplicator.py`
Prevents the same chunk from being processed twice across pipeline runs.

**How it works:**
- Maintains a JSON file at `data/seen_hashes.json`
- On each run:
  1. Loads existing chunk IDs into a set
  2. Filters out any chunk whose `chunk_id` is already seen
  3. Saves new chunk IDs back to the file

**Why this matters:**
- arXiv search results overlap between runs
- Without dedup, vector store fills with duplicate vectors
- Degrades retrieval quality and wastes storage

> **Note:** File-based dedup is for Phase 1. Phase 2 uses database lookup against Qdrant/PostgreSQL for scalability.