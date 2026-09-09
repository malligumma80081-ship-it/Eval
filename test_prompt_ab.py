# from app.evaluation.prompts import PROMPT_A
from app.evaluation.prompts import PROMPT_B
from app.evaluation.prompt_runner import generate_answer


question = "Who created Python?"

context = """
Python was created by Guido van Rossum.
"""


prompt = PROMPT_B.format(
    question=question,
    context=context
)


answer = generate_answer(prompt)

print("PROMPT A")
print("--------")
print(answer)