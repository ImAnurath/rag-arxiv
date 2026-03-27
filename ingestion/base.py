from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
from typing import Optional
import hashlib
'''
Base classes and data models for document loading, chunking, and deduplication.

Document: standard format for any ingested text, with metadata
BaseLoader: abstract class for all loaders (arXiv, web, etc.)
Deduplicator: simple in-memory set of seen chunk hashes to avoid duplicates'''

class SourceType(str, Enum):
    ARXIV_PDF = "arxiv_pdf"
    WEB = "web"
    CSV = "csv"
    MARKDOWN = "markdown"


class Document(BaseModel):
    doc_id: str                          # Stable hash of content
    content: str                         # Raw extracted text
    source: str                          # URL, file path, or arXiv ID
    source_type: SourceType
    title: Optional[str] = None
    authors: Optional[list[str]] = None
    published_at: Optional[datetime] = None
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    extra_metadata: dict = Field(default_factory=dict)

    @classmethod
    def make_id(cls, content: str, source: str) -> str:
        """Deterministic ID — same content + source always gives same ID."""
        raw = f"{source}::{content[:500]}"
        return hashlib.sha256(raw.encode()).hexdigest()


class BaseLoader:
    """All loaders inherit from this and implement `load()`."""

    def load(self) -> list[Document]:
        raise NotImplementedError