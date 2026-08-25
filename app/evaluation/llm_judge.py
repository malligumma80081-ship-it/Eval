import json
import ollama


class LLMJudge:

    def __init__(
        self,
        model="llama3.2"
    ):
        self.model = model

    def build_prompt(
        self,
        question,
        answer,
        context=None
    ):

        context_text = context or "No context provided."

        prompt = f"""
You are an expert evaluator.

Evaluate the answer to the question below.

QUESTION:
{question}

ANSWER:
{answer}

CONTEXT:
{context_text}

Evaluate the answer using these criteria:

1. correctness
2. relevance
3. completeness

Give each score from 1 to 5.

Scoring:
1 = Very Poor
2 = Poor
3 = Average
4 = Good
5 = Excellent

Return ONLY valid JSON.

Required JSON format:

{{
    "correctness": 1,
    "relevance": 1,
    "completeness": 1,
    "overall_score": 1,
    "reason": "short explanation"
}}
"""

        return prompt

    def calculate_overall(
        self,
        result
    ):

        correctness = result["correctness"]
        relevance = result["relevance"]
        completeness = result["completeness"]

        overall = (
            correctness
            + relevance
            + completeness
        ) / 3

        return round(
            overall,
            2
        )

    def judge(
        self,
        question,
        answer,
        context=None
    ):

        prompt = self.build_prompt(
            question=question,
            answer=answer,
            context=context
        )

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        content = response["message"]["content"]

        result = self.parse_response(
            content
        )

        if "error" not in result:

            result["calculated_overall"] = (
                self.calculate_overall(result)
            )

        return result

    def evaluate_prompt(
        self,
        prompt
    ):

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        content = response["message"]["content"]

        return self.parse_response(
            content
        )

    def parse_response(
        self,
        response
    ):

        try:

            result = json.loads(
                response
            )

            return result

        except json.JSONDecodeError:

            return {
                "error": "Judge did not return valid JSON",
                "raw_response": response
            }
    