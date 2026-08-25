from typing import Any, cast

from .llm_judge import LLMJudge


class RAGEvaluator:

    def __init__(
        self,
        model="llama3.2"
    ):

        self.judge = LLMJudge(
            model=model
        )

    def _evaluate_prompt(self, prompt: str):
        judge = cast(Any, self.judge)
        if hasattr(judge, "evaluate_prompt"):
            return judge.evaluate_prompt(prompt)
        if hasattr(judge, "evaluate"):
            return judge.evaluate(prompt)
        raise AttributeError("LLMJudge does not expose a prompt evaluation method.")

    def evaluate_context_relevance(
        self,
        question,
        context
    ):

        context_text = "\n".join(
            context
        )

        prompt = f"""
You are evaluating a RAG retriever.

Question:
{question}

Retrieved Context:
{context_text}

Determine whether the retrieved context
contains information relevant to answering
the question.

Score from 1 to 5.

Return ONLY valid JSON:

{{
    "context_relevance": 1,
    "reason": "short explanation"
}}
"""

        return self._evaluate_prompt(prompt)

    def evaluate_faithfulness(
        self,
        question,
        context,
        answer
    ):

        context_text = "\n".join(
            context
        )

        prompt = f"""
You are evaluating a RAG answer.

Question:
{question}

Context:
{context_text}

Answer:
{answer}

Determine whether the answer is supported
by the provided context.

Score from 1 to 5.

Return ONLY valid JSON:

{{
    "faithfulness": 1,
    "reason": "short explanation"
}}
"""

        return self._evaluate_prompt(prompt)

    def evaluate_answer_relevance(
        self,
        question,
        answer
    ):

        prompt = f"""
You are evaluating answer relevance.

Question:
{question}

Answer:
{answer}

Determine whether the answer directly addresses
the user's question.

Score from 1 to 5.

Return ONLY valid JSON:

{{
    "answer_relevance": 1,
    "reason": "short explanation"
}}
"""

        return self._evaluate_prompt(prompt)