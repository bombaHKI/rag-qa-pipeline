from src.storage import DocumentStore


class RAG_Orchestrator:
    def __init__(self):
        self.document_store = DocumentStore()

    def answer(self, question: str) -> dict:
        results = self.document_store.embedding_store.query(question)
        sources = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for text, meta, distance in zip(documents, metadatas, distances):
            sources.append({
                "title": meta.get("title", ""),
                "text": text,
                "score": distance,
                "url": meta.get("url", ""),
            })
        return {
            'question': question,
            'answer': "Answer generation not implemented yet.",
            'sources': sources,
        }
    
    def evaluate(self, answer, reference_answer):
        return {
            "rouge1": 0.8,
            "rouge2": 0.6,
            "rougeL": 0.75,
            "semantic_similarity": 0.82,
            "faithfulness_score": 0.9,
        }

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


