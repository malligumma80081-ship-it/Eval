# import json


# with open(
#     "data/evaluation_dataset.json",
#     "r",
#     encoding="utf-8"
# ) as file:

#     dataset = json.load(file)


# print(dataset)


# import json


# with open(
#     "data/evaluation_dataset.json",
#     "r",
#     encoding="utf-8"
# ) as file:

#     dataset = json.load(file)


# for test_case in dataset:

#     print("\n----------------------------")

#     print("ID:")
#     print(test_case["id"])

#     print("Question:")
#     print(test_case["question"])

#     print("Expected Answer:")
#     print(test_case["expected_answer"])


# import json

# from app.llm.ollama_client import OllamaClient


# # --------------------------------
# # Load evaluation dataset
# # --------------------------------

# with open(
#     "data/evaluation_dataset.json",
#     "r",
#     encoding="utf-8"
# ) as file:

#     dataset = json.load(file)


# # --------------------------------
# # Create LLM client
# # --------------------------------

# llm = OllamaClient()


# # --------------------------------
# # Process each test case
# # --------------------------------

# for test_case in dataset:

#     question = test_case["question"]

#     expected_answer = test_case["expected_answer"]

#     actual_answer = llm.generate(question)


#     print("\n================================")
#     print("TEST CASE:", test_case["id"])
#     print("================================")

#     print("\nQUESTION:")
#     print(question)

#     print("\nEXPECTED:")
#     print(expected_answer)

#     print("\nACTUAL:")
#     print(actual_answer)



# import json

# from app.llm.ollama_client import OllamaClient


# # --------------------------------
# # Load evaluation dataset
# # --------------------------------

# with open(
#     "data/evaluation_dataset.json",
#     "r",
#     encoding="utf-8"
# ) as file:

#     dataset = json.load(file)


# # --------------------------------
# # Create LLM client
# # --------------------------------

# llm = OllamaClient()


# # --------------------------------
# # Process each test case
# # --------------------------------

# for test_case in dataset:

#     question = test_case["question"]

#     expected_answer = test_case["expected_answer"]

#     actual_answer = llm.generate(question)


#     print("\n================================")
#     print("TEST CASE:", test_case["id"])
#     print("================================")

#     print("\nQUESTION:")
#     print(question)

#     print("\nEXPECTED:")
#     print(expected_answer)

#     print("\nACTUAL:")
#     print(actual_answer)



import json
from pathlib import Path

from app.llm.ollama_client import OllamaClient
from app.evaluation.evaluator import Evaluator


results_dir = Path("results")
results_dir.mkdir(exist_ok=True)


# --------------------------------
# Load dataset
# --------------------------------

with open(
    "data/evaluation_dataset.json",
    "r",
    encoding="utf-8"
) as file:

    dataset = json.load(file)


# --------------------------------
# Create clients
# --------------------------------

llm = OllamaClient(timeout=5)

evaluator = Evaluator(llm=llm)


# --------------------------------
# Results list
# --------------------------------

results = []


# --------------------------------
# Batch evaluation
# --------------------------------

for test_case in dataset:

    question = test_case["question"]
    expected_answer = test_case["expected_answer"]
    actual_answer = ""
    evaluation = {
        "score": None,
        "reason": "Evaluation skipped because the Ollama service is unavailable."
    }

    # Generate answer
    try:
        actual_answer = llm.generate(question)

        # Evaluate
        evaluation = evaluator.evaluate(
            question=question,
            expected_answer=expected_answer,
            actual_answer=actual_answer
        )
    except Exception as exc:
        evaluation = {
            "score": None,
            "reason": f"Error: {exc}"
        }
        print(f"\n[WARNING] Test {test_case['id']} skipped: {exc}")

    # Create result
    result = {
        "id": test_case["id"],
        "question": question,
        "expected_answer": expected_answer,
        "actual_answer": actual_answer,
        "evaluation": evaluation
    }

    results.append(result)

    # Display
    print("\n================================")
    print("TEST CASE:", test_case["id"])
    print("================================")

    print("\nQUESTION:")
    print(question)

    print("\nACTUAL ANSWER:")
    print(actual_answer)

    print("\nEVALUATION:")
    print(evaluation)


# --------------------------------
# Save results
# --------------------------------

with open(
    "results/evaluation_results.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=4,
        ensure_ascii=False
    )


print("\n================================")
print("EVALUATION COMPLETE")
print("================================")

print("Total tests:", len(results))
print("Results saved to:")
print("results/evaluation_results.json")