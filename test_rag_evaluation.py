from app.evaluation.rag_evaluator import (
    evaluate_faithfulness,
    evaluate_relevance
)


question = "Who created Python?"

context = """
Python was created by Guido van Rossum.
Python was first released in 1991.
"""

answer = """
Python was created by Guido van Rossum.
"""


faithfulness = evaluate_faithfulness(
    question,
    context,
    answer
)

relevance = evaluate_relevance(
    question,
    answer
)


print("Faithfulness:", faithfulness)
print("Relevance:", relevance)