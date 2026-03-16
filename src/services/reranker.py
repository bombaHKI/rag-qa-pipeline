"""
Cross-encoder reranker for reranking retrieved chunks against a query.
"""

import logging

from sentence_transformers import CrossEncoder

from src.config import RERANKER_MODEL_NAME

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        logger.info(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, passages: list[str]) -> list[tuple[int, float]]:
        """Rerank passages against a query.
        Returns list of (original_index, score) sorted by score descending"""
        pairs = [[query, p] for p in passages]
        scores = self.model.predict(pairs)
        indexed_scores = list(enumerate(scores.tolist()))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return indexed_scores
