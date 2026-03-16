"""
Embedding storage handler using ChromaDB.
Manages a persistent vector database for storing and querying document chunk embeddings.
"""

import logging
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from src.config import CHROMA_COLLECTION_NAME, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


class EmbeddingStore:
    def __init__(
        self,
        persist_dir: str | Path,
        collection_name: str = CHROMA_COLLECTION_NAME,
        model_name: str = EMBEDDING_MODEL_NAME,
    ):
        self.persist_dir = str(persist_dir)
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name,
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
        )

    def add_chunks(
        self,
        doc_id: str,
        chunks: list[str],
        title: str,
        url: str,
    ) -> int:
        """Store text chunks with embeddings. Returns the number of chunks added."""
        if not chunks:
            return 0

        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {"doc_id": doc_id, "title": title, "url": url, "chunk_index": i}
            for i in range(len(chunks))
        ]
        self.collection.add(documents=chunks, metadatas=metadatas, ids=ids)
        logger.info(f"Added {len(chunks)} chunks for doc_id={doc_id}")
        return len(chunks)

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Remove all chunks belonging to a document."""
        self.collection.delete(where={"doc_id": doc_id})
        logger.info(f"Deleted all chunks for doc_id={doc_id}")

    def delete_all(self) -> None:
        """Wipe the entire collection and recreate it."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
        )
        logger.info("Deleted all embeddings")

    def query(self, query_text: str, n_results: int = 5) -> dict:
        """Return the top-n most similar chunks for a query string."""
        return self.collection.query(query_texts=[query_text], n_results=n_results)

    def count(self) -> int:
        """Total number of chunks stored."""
        return self.collection.count()

    def get_all_doc_metadata(self) -> list[dict]:
        """Return a list of unique documents with their metadata (doc_id, title, url)."""
        result = self.collection.get(include=["metadatas"])
        seen: dict[str, dict] = {}
        for meta in result["metadatas"] or []:
            doc_id = meta["doc_id"]
            if doc_id not in seen:
                seen[doc_id] = {"id": doc_id, "title": meta["title"], "url": meta["url"]}
        return list(seen.values())

    def doc_exists(self, url: str) -> bool:
        """Check whether a document with this URL is already stored."""
        result = self.collection.get(where={"url": url}, limit=1, include=[])
        return bool(result["ids"])
