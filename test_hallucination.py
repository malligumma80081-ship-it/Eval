from app.evaluation.hallucination import (
    evaluate_hallucination
)


question = "Who created Python?"

context = """
Python was created by Guido van Rossum.
"""


answer = """
Python was created by Guido van Rossum.
"""

# answer = """
# Python was created by Bill Gates in 1985.
# """


score = evaluate_hallucination(
    question,
    context,
    answer
)


print("Hallucination Evaluation Score:", score)