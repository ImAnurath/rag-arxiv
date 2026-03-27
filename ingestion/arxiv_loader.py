import arxiv
import fitz  # PyMuPDF
import httpx
from pathlib import Path
from loguru import logger
from typing import Optional

from .base import BaseLoader, Document, SourceType
from config import settings


class ArxivLoader(BaseLoader):
    def __init__(
        self,
        query: Optional[str] = None,
        max_results: Optional[int] = None,
        download_dir: Optional[Path] = None,
    ):
        self.query = query or settings.ARXIV_SEARCH_QUERY
        self.max_results = max_results or settings.ARXIV_MAX_RESULTS
        self.download_dir = download_dir or settings.ARXIV_DOWNLOAD_DIR
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[Document]: # 
        logger.info(f"Fetching up to {self.max_results} papers for: {self.query}")
        client = arxiv.Client()
        search = arxiv.Search(
            query=self.query,
            max_results=self.max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        documents = []
        for paper in client.results(search):
            doc = self._process_paper(paper)
            if doc:
                documents.append(doc)

        logger.success(f"Loaded {len(documents)} documents from arXiv")
        return documents

    def _process_paper(self, paper: arxiv.Result) -> Optional[Document]:
        arxiv_id = paper.get_short_id()
        pdf_path = self.download_dir / f"{arxiv_id.replace('/', '_')}.pdf"

        try:
            # Download PDF if not already cached
            if not pdf_path.exists():
                logger.debug(f"Downloading {arxiv_id}")
                self._download_pdf(paper.pdf_url, pdf_path)

            text = self._extract_text(pdf_path)
            if not text.strip():
                logger.warning(f"Empty text extracted from {arxiv_id}, skipping")
                return None

            return Document(
                doc_id=Document.make_id(text, arxiv_id),
                content=text,
                source=arxiv_id,
                source_type=SourceType.ARXIV_PDF,
                title=paper.title,
                authors=[str(a) for a in paper.authors],
                published_at=paper.published,
                extra_metadata={
                    "abstract": paper.summary,
                    "categories": paper.categories,
                    "doi": paper.doi,
                    "pdf_url": paper.pdf_url,
                },
            )
        except Exception as e:
            logger.error(f"Failed to process {arxiv_id}: {e}")
            return None

    def _download_pdf(self, url: str, dest: Path) -> None:
        with httpx.Client(follow_redirects=True, timeout=60) as client:
            response = client.get(url)
            response.raise_for_status()
            dest.write_bytes(response.content)

    def _extract_text(self, pdf_path: Path) -> str:
        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            text = page.get_text("text")
            pages.append(text)
        doc.close()

        full_text = "\n\n".join(pages)

        # Basic cleanup — arXiv PDFs often have header/footer noise
        full_text = self._clean_text(full_text)
        return full_text

    def _clean_text(self, text: str) -> str:
        import re
        # Strip NUL bytes — PostgreSQL rejects them
        text = text.replace('\x00', '')
        # Remove excessive whitespace and blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        # Remove common arXiv header artifacts
        text = re.sub(r'arXiv:\d{4}\.\d{4,5}v\d+.*?\n', '', text)
        return text.strip()