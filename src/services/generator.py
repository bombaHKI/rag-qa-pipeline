"""
Seq2seq answer generator using a pretrained language model.
"""

import logging

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.config import GENERATOR_MODEL_NAME, GENERATOR_MAX_NEW_TOKENS, PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class Generator:
    def __init__(self, model_name: str = GENERATOR_MODEL_NAME):
        logger.info(f"Loading generator model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, question: str, context_passages: list[str]) -> str:
        """Generate an answer given a question and context passages."""
        context = "\n- ".join(context_passages)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        logger.debug(f"Trying to answer prompt:\n{prompt}")
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model.generate(**inputs, max_new_tokens=GENERATOR_MAX_NEW_TOKENS)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
