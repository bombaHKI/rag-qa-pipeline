FROM python:3.11-slim

WORKDIR /app

# System dependencies (needed for torch / tokenizers)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Pre-download models — these layers only invalidate when the ARG defaults change
# important to keep up to date with src/config.py
ARG EMBEDDING_MODEL=all-MiniLM-L6-v2
ARG RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
ARG GENERATOR_MODEL=google/flan-t5-base

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('${RERANKER_MODEL}')"
RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; AutoTokenizer.from_pretrained('${GENERATOR_MODEL}'); AutoModelForSeq2SeqLM.from_pretrained('${GENERATOR_MODEL}')"

# Copy source
COPY src /app/src

ENV PYTHONPATH=/app

CMD ["streamlit", "run", "src/ui.py", "--server.port=8500", "--server.address=0.0.0.0"]
