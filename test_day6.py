import json

from app.evaluation.rag_evaluator import RAGEvaluator


with open(
    "data/rag_dataset.json",
    "r",
    encoding="utf-8"
) as file:

    dataset = json.load(file)


evaluator = RAGEvaluator(
    model="llama3.2"
)


for item in dataset:

    question = item["question"]

    context = item["context"]

    reference_answer = item[
        "reference_answer"
    ]


    # Simulated generated answer
    generated_answer = reference_answer


    context_result = (evaluator.evaluate_context_relevance(
            question=question,
            context=context
        )
    )


    faithfulness_result = (
        evaluator.evaluate_faithfulness(
            question=question,
            context=context,
            answer=generated_answer
        )
    )


    relevance_result = (
        evaluator.evaluate_answer_relevance(
            question=question,
            answer=generated_answer
        )
    )


    print("\n==============================")
    print("QUESTION")
    print("==============================")

    print(question)


    print("\nCONTEXT")
    print("==============================")

    for document in context:
        print("-", document)


    print("\nANSWER")
    print("==============================")

    print(generated_answer)


    print("\nCONTEXT RELEVANCE")
    print(context_result)


    print("\nFAITHFULNESS")
    print(faithfulness_result)


    print("\nANSWER RELEVANCE")
    print(relevance_result)