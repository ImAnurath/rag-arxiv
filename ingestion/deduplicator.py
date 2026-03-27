import hashlib
from pathlib import Path
import json
from loguru import logger

from .chunker import Chunk


class Deduplicator:
    """
    Maintains a local set of seen chunk hashes.
    """

    def __init__(self, state_file: Path = Path("data/seen_hashes.json")):
        self.state_file = state_file
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._seen: set[str] = self._load()

    def filter(self, chunks: list[Chunk]) -> list[Chunk]:
        fresh = [c for c in chunks if c.chunk_id not in self._seen]
        duplicates = len(chunks) - len(fresh)
        if duplicates:
            logger.info(f"Deduplication: removed {duplicates} already-seen chunks")
        self._seen.update(c.chunk_id for c in fresh)
        self._save()
        return fresh

    def _load(self) -> set[str]:
        if self.state_file.exists():
            return set(json.loads(self.state_file.read_text()))
        return set()

    def _save(self) -> None:
        self.state_file.write_text(json.dumps(list(self._seen)))