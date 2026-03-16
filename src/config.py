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

# --- Retrieval ---
RETRIEVAL_TOP_K = 10

# --- Reranker ---
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_K = 5

# --- Generator ---
GENERATOR_MODEL_NAME = "google/flan-t5-base"
GENERATOR_MAX_NEW_TOKENS = 256

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

PROMPT_TEMPLATE = """Extract the answer to the question from the context.
If the context does not contain the answer, say "I don't have enough information to answer this question."

Question: {question}

Context:
{context}

Answer:"""

