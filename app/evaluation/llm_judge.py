# import json
# import ollama


# class LLMJudge:

#     def __init__(
#         self,
#         model="llama3.2"
#     ):
#         self.model = model

#     def build_prompt(
#         self,
#         question,
#         answer,
#         context=None
#     ):

#         context_text = context or "No context provided."

#         prompt = f"""
# You are an expert evaluator.

# Evaluate the answer to the question below.

# QUESTION:
# {question}

# ANSWER:
# {answer}

# CONTEXT:
# {context_text}

# Evaluate the answer using these criteria:

# 1. correctness
# 2. relevance
# 3. completeness

# Give each score from 1 to 5.

# Scoring:
# 1 = Very Poor
# 2 = Poor
# 3 = Average
# 4 = Good
# 5 = Excellent

# Return ONLY valid JSON.

# Required JSON format:

# {{
#     "correctness": 1,
#     "relevance": 1,
#     "completeness": 1,
#     "overall_score": 1,
#     "reason": "short explanation"
# }}
# """

#         return prompt

#     def calculate_overall(
#         self,
#         result
#     ):

#         correctness = result["correctness"]
#         relevance = result["relevance"]
#         completeness = result["completeness"]

#         overall = (
#             correctness
#             + relevance
#             + completeness
#         ) / 3

#         return round(
#             overall,
#             2
#         )

#     def judge(
#         self,
#         question,
#         answer,
#         context=None
#     ):

#         prompt = self.build_prompt(
#             question=question,
#             answer=answer,
#             context=context
#         )

#         response = ollama.chat(
#             model=self.model,
#             messages=[
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             ]
#         )

#         content = response["message"]["content"]

#         result = self.parse_response(
#             content
#         )

#         if "error" not in result:

#             result["calculated_overall"] = (
#                 self.calculate_overall(result)
#             )

#         return result

#     def evaluate_prompt(
#         self,
#         prompt
#     ):

#         response = ollama.chat(
#             model=self.model,
#             messages=[
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             ]
#         )

#         content = response["message"]["content"]

#         return self.parse_response(
#             content
#         )

#     def parse_response(
#         self,
#         response
#     ):

#         try:

#             result = json.loads(
#                 response
#             )

#             return result

#         except json.JSONDecodeError:

#             return {
#                 "error": "Judge did not return valid JSON",
#                 "raw_response": response
#             }


import json
import ollama


class LLMJudge:

    def __init__(
        self,
        model_name="llama3.2"
    ):
        self.model_name = model_name

    def evaluate(
        self,
        question,
        context,
        answer,
        expected_answer
    ):

        prompt = f"""
You are an expert evaluator for a RAG application.

Evaluate the generated answer using the
question, context, and expected answer.

QUESTION:
{question}

CONTEXT:
{context}

EXPECTED ANSWER:
{expected_answer}

GENERATED ANSWER:
{answer}

Evaluate these three criteria:

1. Faithfulness:
Is the generated answer supported by the context?

2. Relevance:
Does the generated answer directly answer
the question?

3. Correctness:
Is the generated answer factually correct
compared with the expected answer?

Use a score from 1 to 5.

Scoring:

5 = Excellent
4 = Good
3 = Acceptable
2 = Poor
1 = Very poor

Return ONLY valid JSON using exactly this format:

{{
    "faithfulness": {{
        "score": 1,
        "reason": "..."
    }},
    "relevance": {{
        "score": 1,
        "reason": "..."
    }},
    "correctness": {{
        "score": 1,
        "reason": "..."
    }},
    "overall_score": 1
}}
"""

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        content = (
            response["message"]["content"]
        )

        return self._parse_response(content)

    def _parse_response(self, content):

        try:

            return json.loads(content)

        except json.JSONDecodeError:

            raise ValueError(
                "Judge did not return valid JSON.\n"
                f"Raw response:\n{content}"
            )