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

class SourceInfo(BaseModel):
    title: str
    score: float
    text: str
    url: str
class EvalRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    reference_answer: str = Field(..., min_length=1, max_length=2000)

class EvalResponse(BaseModel):
    question: str
    generated_answer: str
    reference_answer: str
    metrics: dict
    sources: list[SourceInfo]


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


if __name__ == "__main__":
    import uvicorn
    from src.config import API_HOST, API_PORT
    uvicorn.run("app.api:app", host=API_HOST, port=API_PORT, reload=False)

