"""
Document storage management for ingested Wikipedia articles.
Stores metadata about ingested documents in a local storage file.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, unquote
from datetime import datetime
import wikipedia

logger = logging.getLogger(__name__)

STORAGE_DIR = Path(__file__).parent.parent.parent / "data"
STORAGE_FILE = STORAGE_DIR / "ingested_documents.json"


def _ensure_storage_dir():
    """Ensure storage directory exists."""
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def _load_documents() -> dict:
    """Load documents from storage file."""
    _ensure_storage_dir()
    if not STORAGE_FILE.exists():
        return {}
    try:
        with open(STORAGE_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        return {}


def _save_documents(documents: dict) -> None:
    """Save documents to storage file."""
    _ensure_storage_dir()
    try:
        with open(STORAGE_FILE, "w") as f:
            json.dump(documents, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving documents: {e}")
        raise


def validate_wikipedia_url(url: str) -> tuple[bool, Optional[str]]:
    """
    Validate if a URL is a valid Wikipedia article link.
    Returns (is_valid, extracted_title).
    """
    try:
        parsed = urlparse(url.strip())
        
        # Check if it's a Wikipedia domain
        domain = parsed.netloc.lower()
        if not any(x in domain for x in ["wikipedia.org", "en.wikipedia.org"]):
            return False, None
        
        # Extract the article title from the path
        path = parsed.path.strip('/')
        if path.startswith('wiki/'):
            title = unquote(path[5:])  # Remove 'wiki/' prefix and decode URL encoding
            if title:
                return True, title
        
        return False, None
    except Exception as e:
        logger.error(f"Error validating Wikipedia URL: {e}")
        return False, None


def add_document(url: str) -> tuple[bool, str, Optional[str]]:
    """
    Add a document URL to the ingestion store.
    Returns (success, message, doc_id).
    """
    is_valid, title = validate_wikipedia_url(url)
    if not is_valid:
        return False, "Invalid Wikipedia URL format. Expected: https://en.wikipedia.org/wiki/ArticleTitle", None
    
    documents = _load_documents()
    
    # Check if already exists
    for doc_id, doc in documents.items():
        if doc["url"] == url:
            return False, f"Document already ingested: {title}", doc_id
    
    # Create new document entry
    doc_id = str(len(documents) + 1)
    documents[doc_id] = {
        "id": doc_id,
        "title": title,
        "url": url,
        "added_at": datetime.now().isoformat()
    }
    
    _save_documents(documents)
    logger.info(f"Document added: {title} (ID: {doc_id})")
    return True, f"Successfully added: {title}", doc_id


def add_documents(urls: list[str]) -> tuple[int, int, list[dict]]:
    """
    Add multiple document URLs.
    Returns (success_count, failure_count, results).
    results is list of {"url": str, "success": bool, "message": str, "title": str}
    """
    success_count = 0
    failure_count = 0
    results = []
    
    for url in urls:
        success, message, doc_id = add_document(url)
        is_valid, title = validate_wikipedia_url(url)
        results.append({
            "url": url,
            "success": success,
            "message": message,
            "title": title or "Unknown"
        })
        
        if success:
            success_count += 1
        else:
            failure_count += 1
    
    return success_count, failure_count, results


def list_documents() -> list[dict]:
    """List all ingested documents."""
    documents = _load_documents()
    return [doc for doc in documents.values()]


def delete_document(doc_id: str) -> tuple[bool, str]:
    """
    Delete a single document.
    Returns (success, message).
    """
    documents = _load_documents()
    
    if doc_id not in documents:
        return False, f"Document not found: {doc_id}"
    
    title = documents[doc_id]["title"]
    del documents[doc_id]
    _save_documents(documents)
    logger.info(f"Document deleted: {title} (ID: {doc_id})")
    return True, f"Deleted: {title}"


def delete_documents(doc_ids: list[str]) -> tuple[int, int]:
    """
    Delete multiple documents.
    Returns (success_count, failure_count).
    """
    success_count = 0
    failure_count = 0
    
    for doc_id in doc_ids:
        success, _ = delete_document(doc_id)
        if success:
            success_count += 1
        else:
            failure_count += 1
    
    return success_count, failure_count


def delete_all_documents() -> tuple[int, str]:
    """
    Delete all documents.
    Returns (count_deleted, message).
    """
    documents = _load_documents()
    count = len(documents)
    
    if count == 0:
        return 0, "No documents to delete"
    
    _ensure_storage_dir()
    STORAGE_FILE.unlink(missing_ok=True)
    logger.info(f"All documents deleted ({count} total)")
    return count, f"Deleted all {count} document(s)"


def reset_to_default_documents() -> tuple[int, int, list[dict]]:
    """
    Reset the collection to default articles by deleting all and reseeding.
    Returns (success_count, failure_count, results).
    """
    logger.info("Resetting collection to default articles...")
    delete_all_documents()
    
    # Now seed the defaults
    default_urls = [
        "https://en.wikipedia.org/wiki/Hungary",
        "https://en.wikipedia.org/wiki/Central_Europe",
        "https://en.wikipedia.org/wiki/Buda",
        "https://en.wikipedia.org/wiki/Buda_Castle",
        "https://en.wikipedia.org/wiki/Budapest",
        "https://en.wikipedia.org/wiki/Hungarian_Parliament_Building",
        "https://en.wikipedia.org/wiki/Sz%C3%A9chenyi_thermal_bath",
    ]
    
    success_count, failure_count, results = add_documents(default_urls)
    logger.info(f"Reset complete: {success_count} added, {failure_count} failed")
    return success_count, failure_count, results


def get_random_articles(count: int = 5) -> list[str]:
    """
    Get a list of random Wikipedia article URLs.
    Uses Wikipedia's random article feature to fetch real random articles.
    Filters out disambiguation pages and already ingested documents.
    """
    documents = _load_documents()
    ingested_urls = {doc["url"] for doc in documents.values()}
    
    articles = []
    attempts = 0
    max_attempts = count * 4  # Try up to 4x the requested count to account for failures
    
    while len(articles) < count and attempts < max_attempts:
        try:
            # Get a random article title
            title = wikipedia.random()
            attempts += 1
            
            # Try to fetch the page to validate it's not a disambiguation page
            try:
                page = wikipedia.page(title, auto_suggest=False)
                
                # Construct the URL from the actual page title (handles redirects)
                url = f"https://en.wikipedia.org/wiki/{page.title.replace(' ', '_')}"
                
                # Check if not already ingested
                if url not in ingested_urls:
                    articles.append(url)
                    ingested_urls.add(url)  # Mark as ingested to avoid duplicates
                    
            except wikipedia.exceptions.DisambiguationError:
                # Skip disambiguation pages
                logger.debug(f"Skipping disambiguation page: {title}")
                continue
            except wikipedia.exceptions.PageError:
                # Skip if page not found
                logger.debug(f"Skipping non-existent page: {title}")
                continue
                
        except Exception as e:
            logger.error(f"Error fetching random article: {e}")
            attempts += 1
            if attempts >= max_attempts:
                break
    
    logger.info(f"Retrieved {len(articles)} random articles (out of {count} requested)")
    return articles

