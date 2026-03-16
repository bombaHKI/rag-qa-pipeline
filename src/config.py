"""
Central configuration for the RAG Q&A pipeline.
"""

from pathlib import Path

# --- Paths ---
DATA_DIR = Path(__file__).parent.parent / "data"

# --- Chunking ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Wikipedia ---
WIKI_LANGUAGE = "en"
WIKI_USER_AGENT = "RAG-QA-Pipeline/1.0"
SKIP_SECTIONS = {
    "See also", "References", "External links",
    "Further reading", "Notes", "Bibliography", "Sources",
}

# --- Embedding / ChromaDB ---
CHROMA_COLLECTION_NAME = "wiki_chunks"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Default Wikipedia articles ---
DEFAULT_URLS = [
    "https://en.wikipedia.org/wiki/Hungary",
    "https://en.wikipedia.org/wiki/Central_Europe",
    "https://en.wikipedia.org/wiki/Buda",
    "https://en.wikipedia.org/wiki/Buda_Castle",
    "https://en.wikipedia.org/wiki/Budapest",
    "https://en.wikipedia.org/wiki/Hungarian_Parliament_Building",
    "https://en.wikipedia.org/wiki/Sz%C3%A9chenyi_thermal_bath",
]
