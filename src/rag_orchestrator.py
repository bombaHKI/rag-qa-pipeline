from src.config import RETRIEVAL_TOP_K, RERANK_TOP_K
from src.services.evaluator import Evaluator
from src.services.generator import Generator
from src.services.reranker import Reranker
from src.storage import DocumentStore


class RAG_Orchestrator:
    def __init__(self):
        self.document_store = DocumentStore()
        self.reranker = Reranker()
        self.generator = Generator()
        self.evaluator = Evaluator()

    def answer(self, question: str) -> dict:
        # 1. Retrieve top-k candidates from vector store
        results = self.document_store.embedding_store.query(question, n_results=RETRIEVAL_TOP_K)
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        if not documents:
            return {
                "question": question,
                "answer": "No documents found. Please ingest some articles first.",
                "sources": [],
            }

        # 2. Rerank candidates
        ranked = self.reranker.rerank(question, documents)
        ranked = ranked[:RERANK_TOP_K]

        # 3. Build reranked sources
        sources = []
        for orig_idx, rerank_score in ranked:
            text = documents[orig_idx]
            meta = metadatas[orig_idx]
            sources.append({
                "title": meta.get("title", ""),
                "text": text,
                "score": rerank_score,
                "url": meta.get("url", ""),
            })

        # 4. Generate answer from top passages
        context_passages = [s["text"] for s in sources]
        generated_answer = self.generator.generate(question, context_passages)

        return {
            "question": question,
            "answer": generated_answer,
            "sources": sources,
            "context_passages": context_passages,
        }

    def evaluate(
        self,
        answer: str,
        context_passages: list[str],
        question: str,
        reference_answer: str | None = None,
    ) -> dict:
        return self.evaluator.evaluate(
            question=question,
            answer=answer,
            context_passages=context_passages,
            reference_answer=reference_answer,
        )

    def list_documents(self) -> list[dict]:
        return self.document_store.list_documents()

    def add_documents(self, urls: list[str]) -> tuple[int, int, list[dict]]:
        return self.document_store.add_documents(urls)

    def delete_document(self, doc_id: str) -> tuple[bool, str]:
        return self.document_store.delete_document(doc_id)

    def delete_documents(self, doc_ids: list[str]) -> tuple[int, int]:
        return self.document_store.delete_documents(doc_ids)

    def delete_all_documents(self) -> tuple[int, str]:
        return self.document_store.delete_all_documents()

    def get_random_articles(self, count: int = 5) -> list[str]:
        return self.document_store.get_random_articles(count=count)

    def reset_to_default_documents(self) -> tuple[int, int, list[dict]]:
        return self.document_store.reset_to_default_documents()


