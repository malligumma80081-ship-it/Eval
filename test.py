from app.llm.ollama_client import OllamaClient
from app.evaluation.evaluator import Evaluator


question = "What is Python?"

expected_answer = """
Python is a high-level programming language.
"""


# Step 1: Generate answer

model_name = "llama3.2"
llm = OllamaClient(model=model_name)

actual_answer = llm.generate(question)


# Step 2: Print answers

print("\n==============================")
print("QUESTION")
print("==============================")

print(question)


print("\n==============================")
print("EXPECTED ANSWER")
print("==============================")

print(expected_answer)


print("\n==============================")
print("ACTUAL ANSWER")
print("==============================")

print(actual_answer)


# Step 3: Evaluate

evaluator = Evaluator(model=model_name)

result = evaluator.evaluate(
    question=question,
    expected_answer=expected_answer,
    actual_answer=actual_answer
)


# Step 4: Display evaluation

print("\n==============================")
print("EVALUATION")
print("==============================")

print(result)