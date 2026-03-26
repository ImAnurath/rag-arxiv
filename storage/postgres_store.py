from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    DateTime,
    JSON,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Session
from loguru import logger
from datetime import datetime
from sqlalchemy import func as sqlalchemy_func_count
from ingestion.chunker import Chunk
from config import settings


class Base(DeclarativeBase):
    pass


class ChunkRecord(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id = Column(String(64), nullable=False)
    doc_id = Column(String(64), nullable=False)
    source = Column(String(256), nullable=False)
    source_type = Column(String(64), nullable=False)
    title = Column(Text, nullable=True)
    authors = Column(JSON, nullable=True)
    chunk_index = Column(Integer, nullable=False)
    content_preview = Column(Text, nullable=True)  # first 300 chars for debugging
    ingested_at = Column(DateTime, default=datetime.utcnow)
    extra_metadata = Column(JSON, nullable=True)

    __table_args__ = (UniqueConstraint("chunk_id", name="uq_chunk_id"),)


class PostgresStore:
    def __init__(self, db_url: str = settings.POSTGRES_URL):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        logger.info("PostgreSQL store ready")

    def insert_chunks(self, chunks: list[Chunk]) -> int:
        """
        Insert chunks into Postgres. Skips duplicates via ON CONFLICT DO NOTHING.
        Returns number of rows actually inserted.
        """
        records = [
            ChunkRecord(
                chunk_id=self._sanitize(c.chunk_id),
                doc_id=self._sanitize(c.doc_id),
                source=self._sanitize(c.source),
                source_type=self._sanitize(c.source_type),
                title=self._sanitize(c.title),
                authors=c.authors,
                chunk_index=c.chunk_index,
                content_preview=self._sanitize(c.content[:300]),
                extra_metadata=c.extra_metadata,
            )
            for c in chunks
        ]

        inserted = 0
        with Session(self.engine) as session:
            for record in records:
                existing = (
                    session.query(ChunkRecord)
                    .filter_by(chunk_id=record.chunk_id)
                    .first()
                )
                if not existing:
                    session.add(record)
                    inserted += 1
            session.commit()

        logger.success(f"PostgreSQL: inserted {inserted} new chunk records")
        return inserted

    def get_ingestion_stats(self) -> dict:
        with Session(self.engine) as session:
            total = session.query(ChunkRecord).count()
            by_source_type = (
                session.query(ChunkRecord.source_type,
                               sqlalchemy_func_count(ChunkRecord.id))
                .group_by(ChunkRecord.source_type)
                .all()
            )
        return {
            "total_chunks": total,
            "by_source_type": dict(by_source_type),
        }
    def _sanitize(self, value: str | None) -> str | None:
        if value is None:
            return None
        return value.replace('\x00', '')