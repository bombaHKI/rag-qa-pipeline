"""
FastAPI backend for the RAG Q&A pipeline.
Exposes endpoints for asking questions, ingesting data, and evaluating answers.
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.storage import document_store


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Pydantic models ---

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="The question to answer")

class SourceInfo(BaseModel):
    title: str
    text: str
    score: float
    url: str

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceInfo]

class EvalRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    reference_answer: str = Field(..., min_length=1, max_length=2000)

class EvalResponse(BaseModel):
    question: str
    generated_answer: str
    reference_answer: str
    metrics: dict
    sources: list[SourceInfo]


# --- Ingestion models ---

class IngestRequest(BaseModel):
    """Request to ingest one or more Wikipedia URLs."""
    urls: list[str] = Field(..., min_items=1, max_items=100, description="List of Wikipedia URLs to ingest")

class IngestResponse(BaseModel):
    """Response from ingestion request."""
    success_count: int
    failure_count: int
    results: list[dict]

class DocumentInfo(BaseModel):
    """Information about an ingested document."""
    id: str
    title: str
    url: str
    added_at: str

class DocumentListResponse(BaseModel):
    """List of all ingested documents."""
    documents: list[DocumentInfo]
    count: int

class DeleteRequest(BaseModel):
    """Request to delete documents."""
    doc_ids: Optional[list[str]] = Field(None, description="List of document IDs to delete")
    delete_all: bool = Field(False, description="If true, delete all documents")

class DeleteResponse(BaseModel):
    """Response from delete request."""
    success: bool
    message: str
    deleted_count: int


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
    """Health check endpoint."""
    return {'status': 'healthy'}

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(req: QuestionRequest):
    return AnswerResponse(
        question=req.question,
        answer="This is a good question, I don't have the response... yet :)",
        sources=[
            {
                "title": "source title placeholder",
                "score": 13.0,
                "text": "Sadly I don't have the answer.",
                "url": "url"
            }
        ]
    )

@app.post("/evaluate", response_model=EvalResponse)
async def evaluate_answer(req: EvalRequest):
    """Dummy evaluation response."""
    
    # Dummy sources
    dummy_sources = [
        SourceInfo(
            title="Example source 1",
            text="This is a placeholder passage.",
            score=0.95,
            url="https://example.com/source1"
        ),
        SourceInfo(
            title="Example source 2",
            text="Another placeholder passage for testing.",
            score=0.87,
            url="https://example.com/source2"
        ),
    ]
    
    # Dummy metrics
    dummy_metrics = {
        "rouge1": 0.8,
        "rouge2": 0.6,
        "rougeL": 0.75,
        "semantic_similarity": 0.82,
        "faithfulness_score": 0.9
    }
    
    return EvalResponse(
        question=req.question,
        generated_answer="This is a dummy generated answer.",
        reference_answer=req.reference_answer,
        metrics=dummy_metrics,
        sources=dummy_sources
    )


# --- Document Ingestion Endpoints ---

@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(req: IngestRequest):
    """Ingest one or more Wikipedia URLs."""
    logger.info(f"Ingesting {len(req.urls)} document(s)")
    success_count, failure_count, results = document_store.add_documents(req.urls)
    return IngestResponse(
        success_count=success_count,
        failure_count=failure_count,
        results=results
    )


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """List all ingested documents."""
    documents = document_store.list_documents()
    logger.info(f"Listing {len(documents)} document(s)")
    return DocumentListResponse(
        documents=documents,
        count=len(documents)
    )


@app.delete("/documents", response_model=DeleteResponse)
async def delete_documents(
    doc_ids: Optional[list[str]] = Query(None, description="List of document IDs to delete"),
    delete_all: bool = Query(False, description="If true, delete all documents")
):
    """Delete one or more documents or all documents."""
    
    if delete_all:
        deleted_count, message = document_store.delete_all_documents()
        logger.info(f"Deleted all documents: {message}")
        return DeleteResponse(
            success=deleted_count > 0 or not doc_ids,
            message=message,
            deleted_count=deleted_count
        )
    
    if not doc_ids:
        raise HTTPException(
            status_code=400,
            detail="Provide either doc_ids or set delete_all=true"
        )
    
    success_count, failure_count = document_store.delete_documents(doc_ids)
    message = f"Deleted {success_count} document(s)" + (f", {failure_count} failed" if failure_count > 0 else "")
    logger.info(message)
    
    return DeleteResponse(
        success=success_count > 0,
        message=message,
        deleted_count=success_count
    )


class RandomArticlesResponse(BaseModel):
    """List of random Wikipedia article URLs."""
    articles: list[str]
    count: int


@app.get("/random-articles", response_model=RandomArticlesResponse)
async def get_random_articles(count: int = Query(5, ge=1, le=20, description="Number of random articles to fetch")):
    """Get random Wikipedia article URLs that are not yet ingested."""
    articles = document_store.get_random_articles(count=count)
    logger.info(f"Generated {len(articles)} random article suggestions")
    return RandomArticlesResponse(
        articles=articles,
        count=len(articles)
    )


@app.post("/reset-to-defaults", response_model=IngestResponse)
async def reset_to_defaults():
    """Reset the collection to default articles."""
    logger.info("Resetting to default articles")
    success_count, failure_count, results = document_store.reset_to_default_documents()
    return IngestResponse(
        success_count=success_count,
        failure_count=failure_count,
        results=results
    )


if __name__ == "__main__":
    import uvicorn
    from src.config import API_HOST, API_PORT
    uvicorn.run("app.api:app", host=API_HOST, port=API_PORT, reload=False)

