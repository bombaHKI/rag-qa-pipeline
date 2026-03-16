"""
Document storage management for ingested Wikipedia articles.
Fetches, chunks, and stores article content as embeddings in ChromaDB via EmbeddingStore.
"""

import logging
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, unquote, quote

import requests
import wikipediaapi
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import (
    DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DEFAULT_URLS,
    WIKI_LANGUAGE,
    WIKI_USER_AGENT,
    SKIP_SECTIONS,
)
from src.storage.embedding_store import EmbeddingStore

logger = logging.getLogger(__name__)


class DocumentStore:
    """Manages document ingestion and embeddings (ChromaDB)."""

    WIKI_API_URL = "https://en.wikipedia.org/w/api.php"

    def __init__(self, data_dir: str | Path | None = None):
        self._data_dir = Path(data_dir) if data_dir else DATA_DIR
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_store = EmbeddingStore(persist_dir=self._data_dir / "chroma")

        self._wiki = wikipediaapi.Wikipedia(
            user_agent=WIKI_USER_AGENT,
            language=WIKI_LANGUAGE,
        )

        # HTTP session with retry + proper user-agent
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": WIKI_USER_AGENT})

        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        self._session.mount("https://", adapter)

        # Initialize ID counter once
        docs = self.embedding_store.get_all_doc_metadata()
        existing_ids = [int(d["id"]) for d in docs if str(d.get("id", "")).isdigit()]
        self._doc_id_counter = max(existing_ids, default=0)

    # -- private helpers --

    @staticmethod
    def _validate_wikipedia_url(url: str) -> tuple[bool, Optional[str]]:
        """Validate a Wikipedia URL. Returns (is_valid, extracted_title)."""
        try:
            parsed = urlparse(url.strip())
            if parsed.scheme not in ("http", "https"):
                return False, None

            domain = parsed.netloc.lower()
            if "wikipedia.org" not in domain:
                return False, None

            path = parsed.path.strip("/")
            if path.startswith("wiki/"):
                title = unquote(path[5:])
                if title:
                    return True, title

            return False, None

        except Exception as e:
            logger.error(f"Error validating Wikipedia URL: {e}")
            return False, None

    @staticmethod
    def _extract_section_text(section: wikipediaapi.WikipediaPageSection) -> str:
        """Recursively extract text from a section and its subsections."""
        if section.title in SKIP_SECTIONS:
            return ""

        parts = [section.text] if section.text else []

        for sub in section.sections:
            parts.append(DocumentStore._extract_section_text(sub))

        return "\n".join(parts)

    def _get_page_text(self, page: wikipediaapi.WikipediaPage) -> str:
        """Extract clean prose text from a Wikipedia page."""
        parts = [page.summary] if page.summary else []

        for section in page.sections:
            text = self._extract_section_text(section)
            if text:
                parts.append(text)

        return "\n".join(parts)

    @staticmethod
    def _chunk_text(
        text: str,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
    ) -> list[str]:
        """Split text into overlapping chunks by word count."""
        if not text:
            return []

        words = text.split()
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for word in words:
            word_len = len(word) + (1 if current else 0)

            if current_len + word_len > chunk_size and current:
                chunks.append(" ".join(current))

                overlap_words: list[str] = []
                overlap_len = 0

                for w in reversed(current):
                    if overlap_len + len(w) + 1 > overlap:
                        break
                    overlap_words.append(w)
                    overlap_len += len(w) + 1

                overlap_words.reverse()
                current = overlap_words
                current_len = sum(len(w) for w in current) + max(0, len(current) - 1)

            current.append(word)
            current_len += word_len

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _next_id(self) -> str:
        """Return next document ID."""
        self._doc_id_counter += 1
        return str(self._doc_id_counter)

    # -- public API --

    def list_documents(self) -> list[dict]:
        return self.embedding_store.get_all_doc_metadata()

    def add_document(self, url: str) -> tuple[bool, str, Optional[str]]:
        """Fetch, chunk, embed, and store a Wikipedia article."""
        is_valid, title = self._validate_wikipedia_url(url)

        if not is_valid:
            return False, "Invalid Wikipedia URL", None

        if self.embedding_store.doc_exists(url):
            return False, f"Document already ingested: {title}", None

        page = self._wiki.page(title)

        if not page.exists():
            return False, f"Page not found: {title}", None

        content = self._get_page_text(page)

        if not content.strip():
            return False, f"No text content found for: {title}", None

        chunks = self._chunk_text(content)
        doc_id = self._next_id()

        self.embedding_store.add_chunks(
            doc_id=doc_id,
            chunks=chunks,
            title=title,
            url=url,
        )

        logger.info(
            f"Document added: {title} (ID={doc_id}, chunks={len(chunks)})"
        )

        return True, f"Successfully added: {title}", doc_id

    def add_documents(self, urls: list[str]) -> tuple[int, int, list[dict]]:
        success_count = 0
        failure_count = 0
        results: list[dict] = []

        for url in urls:
            ok, message, doc_id = self.add_document(url)

            _, title = self._validate_wikipedia_url(url)

            results.append(
                {
                    "url": url,
                    "success": ok,
                    "message": message,
                    "title": title or "Unknown",
                }
            )

            if ok:
                success_count += 1
            else:
                failure_count += 1

        return success_count, failure_count, results

    def delete_document(self, doc_id: str) -> tuple[bool, str]:
        docs = {d["id"]: d for d in self.embedding_store.get_all_doc_metadata()}

        if doc_id not in docs:
            return False, f"Document not found: {doc_id}"

        title = docs[doc_id]["title"]

        self.embedding_store.delete_by_doc_id(doc_id)

        logger.info(f"Document deleted: {title} (ID={doc_id})")

        return True, f"Deleted: {title}"

    def delete_documents(self, doc_ids: list[str]) -> tuple[int, int]:
        success_count = 0
        failure_count = 0

        for doc_id in doc_ids:
            ok, _ = self.delete_document(doc_id)

            if ok:
                success_count += 1
            else:
                failure_count += 1

        return success_count, failure_count

    def delete_all_documents(self) -> tuple[int, str]:
        count = len(self.embedding_store.get_all_doc_metadata())

        if count == 0:
            return 0, "No documents to delete"

        self.embedding_store.delete_all()
        self._doc_id_counter = 0

        logger.info(f"All documents deleted ({count})")

        return count, f"Deleted all {count} document(s)"

    def reset_to_default_documents(self) -> tuple[int, int, list[dict]]:
        logger.info("Resetting collection to default articles...")
        self.delete_all_documents()
        return self.add_documents(DEFAULT_URLS)

    def get_random_articles(self, count: int = 5) -> list[str]:
        """Fetch random Wikipedia article URLs not yet ingested."""

        ingested_urls = {d["url"] for d in self.embedding_store.get_all_doc_metadata()}
        articles: list[str] = []

        try:
            resp = self._session.get(
                self.WIKI_API_URL,
                params={
                    "action": "query",
                    "list": "random",
                    "rnnamespace": 0,
                    "rnlimit": 2 * count,
                    "format": "json",
                },
                timeout=10,
            )

            resp.raise_for_status()

            random_pages = resp.json().get("query", {}).get("random", [])

        except Exception as e:
            logger.error(f"Error fetching random articles: {e}")
            return articles

        for item in random_pages:

            if len(articles) >= count:
                break

            title = item["title"]
            encoded_title = quote(title.replace(" ", "_"))

            url = f"https://en.wikipedia.org/wiki/{encoded_title}"

            if url not in ingested_urls:
                articles.append(url)
                ingested_urls.add(url)

        logger.info(f"Retrieved {len(articles)} random articles (requested {count})")

        time.sleep(0.2)  # polite rate limiting

        return articles