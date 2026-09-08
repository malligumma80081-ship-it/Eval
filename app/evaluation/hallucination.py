# HALLUCINATION_PROMPT = """
# You are an evaluator checking whether an LLM answer
# is supported by the provided context.

# Question:
# {question}

# Context:
# {context}

# Answer:
# {answer}

# Determine whether the answer contains unsupported claims.

# Score from 1 to 5:

# 5 = fully supported by context
# 4 = mostly supported
# 3 = partially supported
# 2 = mostly unsupported
# 1 = completely unsupported

# Return only the number.
# """

HALLUCINATION_PROMPT = """
You are evaluating whether an LLM answer is grounded
in the provided context.

Question:
{question}

Context:
{context}

Answer:
{answer}

Break the answer into individual factual claims.

For each claim, determine whether it is supported
by the context.

Return JSON in this format:

{{
  "claims": [
    {{
      "claim": "claim text",
      "supported": true
    }}
  ],
  "grounded": true,
  "score": 5
}}

Score:

5 = all claims supported
4 = almost all claims supported
3 = some claims unsupported
2 = many claims unsupported
1 = mostly unsupported

Do not use outside knowledge.
Only use the provided context.
"""

import ollama


def ask_judge(prompt):

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]

def evaluate_hallucination(
    question,
    context,
    answer
):

    prompt = HALLUCINATION_PROMPT.format(
        question=question,
        context=context,
        answer=answer
    )

    result = ask_judge(prompt)

    return result
