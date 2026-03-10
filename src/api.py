"""
FastAPI backend for the RAG Q&A pipeline.
Exposes endpoints for asking questions, ingesting data, and evaluating answers.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    global pipeline, evaluator
    logger.info("Starting up RAG API...")
    yield
    logger.info("Shutting down RAG API.")

app = FastAPI(
    title="RAG Q&A API",
    description="Retrieval-Augmented Generation pipeline for answering questions from Wikipedia.",
    version="1.0.0",
    lifespan=lifespan,
)

# --- Endpoints ---

@app.get("/health")
async def health_check():
    """Health check and collection stats."""
    return {'status': 'healthy'}


if __name__ == "__main__":
    import uvicorn
    from src.config import API_HOST, API_PORT
    uvicorn.run("app.api:app", host=API_HOST, port=API_PORT, reload=False)

