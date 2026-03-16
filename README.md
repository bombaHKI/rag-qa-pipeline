# RAG Q&A Pipeline

A Retrieval-Augmented Generation (RAG) application that answers questions using **Simple English Wikipedia** as its knowledge base.  
The app runs entirely **locally** with no paid APIs and allows you to ask questions, ingest data, and evaluate answer quality.

---

## 📌 What the App Does

This app helps you explore and query knowledge from Wikipedia in a structured way.  
Key functionalities:

- **Ingest data**: Import Wikipedia articles into the local vector store.  
- **Ask questions**: Retrieve relevant passages and generate answers.  
- **Evaluate answer quality**: Compare generated answers against a reference with dummy metrics for now.

---

## 🖼 Screenshot

![RAG Q&A UI](screenshots/image.png)

---

## 🏗 Structure & Libraries

- **Frontend**: Streamlit (calls the RAG orchestrator directly)
- **Vector Search**: ChromaDB + Sentence Transformers  
- **Reranker**: Cross-encoder (MS MARCO MiniLM)  
- **Answer Generation**: Flan-T5 seq2seq model  

The internal pipeline retrieves, re-ranks, and generates answers, but details are **in progress**.

---

## ⚡ Docker Setup

The easiest way to run the app is with Docker. The repository includes a single Dockerfile managed with `docker-compose`.

### Service

- **app**: Streamlit application
  - Exposes port `8500`
  - Uses volumes for model cache and data

### Volumes

- `model_cache`: stores Hugging Face model cache

### Build & Run

**Run with docker**

From the `docker` folder, run:

```bash
docker compose up --build
```

**Run locally**

From root:
```bash
streamlit run src/ui.py --server.port=8500 --server.address=0.0.0.0
```