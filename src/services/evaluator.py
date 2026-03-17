"""
Evaluator for RAG pipeline answers.

Computes metrics that assess retrieval and generation quality:
- Faithfulness: Is the answer grounded in the retrieved context?
- Context Precision: Are the top-ranked passages actually relevant?
- Answer Relevance: Does the answer address the question?
- Context Recall: Does the context cover the reference answer?
- Answer Correctness: Factual overlap between generated and reference answers.
"""

import logging

import numpy as np
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        logger.info(f"Loading evaluator embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    def _cosine_similarity(self, text_a: str, text_b: str) -> float:
        embeddings = self.model.encode([text_a, text_b])
        a, b = embeddings[0], embeddings[1]
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def faithfulness(self, answer: str, context_passages: list[str]) -> float:
        """How well the answer is grounded in the retrieved context.

        Computes max cosine similarity between the answer and each passage,
        then averages the top-k scores. A high score means the answer draws
        from retrieved evidence rather than hallucinating.
        """
        if not context_passages or not answer.strip():
            return 0.0

        texts = [answer] + context_passages
        embeddings = self.model.encode(texts)
        answer_emb = embeddings[0]
        passage_embs = embeddings[1:]

        similarities = [
            float(np.dot(answer_emb, p) / (np.linalg.norm(answer_emb) * np.linalg.norm(p) + 1e-10))
            for p in passage_embs
        ]
        # Use the top-3 passage similarities (or fewer if less available)
        top_k = sorted(similarities, reverse=True)[: min(3, len(similarities))]
        return float(np.mean(top_k))

    def context_precision(self, question: str, context_passages: list[str]) -> float:
        """Are the top-ranked passages relevant to the question?

        Measures average precision: passages early in the ranking that are
        relevant (similarity > threshold) contribute more to the score.
        """
        if not context_passages or not question.strip():
            return 0.0

        texts = [question] + context_passages
        embeddings = self.model.encode(texts)
        q_emb = embeddings[0]
        passage_embs = embeddings[1:]

        threshold = 0.35
        relevant_count = 0
        precision_sum = 0.0

        for rank, p_emb in enumerate(passage_embs, start=1):
            sim = float(np.dot(q_emb, p_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(p_emb) + 1e-10))
            if sim >= threshold:
                relevant_count += 1
                precision_sum += relevant_count / rank

        if relevant_count == 0:
            return 0.0
        return precision_sum / relevant_count

    def answer_relevance(self, question: str, answer: str) -> float:
        """Does the answer actually address the question?

        Cosine similarity between question and answer embeddings.
        """
        if not question.strip() or not answer.strip():
            return 0.0
        return max(0.0, self._cosine_similarity(question, answer))

    def context_recall(self, reference_answer: str, context_passages: list[str]) -> float:
        """Does the retrieved context contain the information in the reference answer?

        Max cosine similarity between the reference answer and any passage.
        """
        if not context_passages or not reference_answer.strip():
            return 0.0

        texts = [reference_answer] + context_passages
        embeddings = self.model.encode(texts)
        ref_emb = embeddings[0]
        passage_embs = embeddings[1:]

        similarities = [
            float(np.dot(ref_emb, p) / (np.linalg.norm(ref_emb) * np.linalg.norm(p) + 1e-10))
            for p in passage_embs
        ]
        return float(max(similarities))

    def answer_correctness(self, answer: str, reference_answer: str) -> float:
        """Factual overlap between generated and reference answers.

        Combines ROUGE-L F1 (lexical overlap) and semantic similarity
        in a 0.4 / 0.6 weighted blend.
        """
        if not answer.strip() or not reference_answer.strip():
            return 0.0

        rouge_scores = self.rouge.score(reference_answer, answer)
        rouge_l_f1 = rouge_scores["rougeL"].fmeasure

        semantic_sim = max(0.0, self._cosine_similarity(answer, reference_answer))

        return 0.4 * rouge_l_f1 + 0.6 * semantic_sim

    def evaluate(
        self,
        question: str,
        answer: str,
        context_passages: list[str],
        reference_answer: str | None = None,
    ) -> dict:
        """Run all evaluation metrics and return a results dict."""
        results = {
            "faithfulness": self.faithfulness(answer, context_passages),
            "context_precision": self.context_precision(question, context_passages),
            "answer_relevance": self.answer_relevance(question, answer),
        }

        if reference_answer and reference_answer.strip():
            results["context_recall"] = self.context_recall(reference_answer, context_passages)
            results["answer_correctness"] = self.answer_correctness(answer, reference_answer)

        return results
