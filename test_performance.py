from app.evaluation.performance_evaluator import generate_response


question = "What is hypertension?"

answer, latency = generate_response(question)

print("Answer:")
print(answer)

print("\nLatency:")
print(f"{latency:.2f} seconds")