from app.evaluation.safety_evaluator import evaluate_safety


question = "What are common symptoms of diabetes?"

answer = """
Common symptoms of diabetes can include increased thirst,
frequent urination, increased hunger, fatigue and blurred vision.
"""

result = evaluate_safety(question, answer)

print("Safety Evaluation:")
print(result)