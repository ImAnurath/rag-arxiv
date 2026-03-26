from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from loguru import logger

from .base import Document
from config import settings


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int
    source: str
    source_type: str
    title: str | None = None
    authors: list[str] = []
    extra_metadata: dict = {}


class RecursiveChunker:
    """
    Splits on paragraph → sentence → word boundaries, in that order.
    Never cuts in the middle of a sentence if it can help it.
    """

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        all_chunks = []
        for doc in documents:
            chunks = self._chunk_document(doc)
            all_chunks.extend(chunks)
            logger.debug(f"  {doc.source}: {len(chunks)} chunks")

        logger.info(f"Total chunks produced: {len(all_chunks)}")
        return all_chunks

    def _chunk_document(self, doc: Document) -> list[Chunk]:
        texts = self.splitter.split_text(doc.content)
        chunks = []
        for i, text in enumerate(texts):
            import hashlib
            chunk_id = hashlib.sha256(f"{doc.doc_id}::{i}".encode()).hexdigest()[:16]
            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                content=text,
                chunk_index=i,
                source=doc.source,
                source_type=doc.source_type,
                title=doc.title,
                authors=doc.authors or [],
                extra_metadata=doc.extra_metadata,
            ))
        return chunks