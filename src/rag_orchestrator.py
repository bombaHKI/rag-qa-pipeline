class RAG_Orchestrator:
    def __init__(slef, ):
        pass

    def answer(self, question: str) -> dict:
        return {
            'question': question,
            'answer': "This is a good question, I don't have the response... yet :)",
            'sources': [
                {
                    "title": "source title placeholder",
                    "score": 13.0,
                    "text": "Sadly I don't have the answer.",
                    "url": "url"
                }
            ]
        }


