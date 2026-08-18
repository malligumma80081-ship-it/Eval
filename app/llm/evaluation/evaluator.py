from app.llm.ollama_client import OllamaClient


class Evaluator:

    def __init__(self, llm=None, model=None):
        self.llm = llm or OllamaClient(model=model)

    def evaluate(
        self,
        question,
        expected_answer,
        actual_answer
    ):

        prompt = f"""
You are an AI answer evaluator.

Question:
{question}

Expected Answer:
{expected_answer}

Actual Answer:
{actual_answer}

Evaluate the actual answer against the expected answer.

Give a score from 1 to 5.

1 = Completely incorrect
2 = Mostly incorrect
3 = Partially correct
4 = Mostly correct
5 = Completely correct

Return:
Score: <number>
Reason: <short explanation>
"""

        return self.llm.generate(prompt)