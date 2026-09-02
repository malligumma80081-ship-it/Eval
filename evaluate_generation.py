import json

from app.rag.rag_retriever import (
    RAGRetriever
)

from app.rag.llm_generator import (
    LlamaGenerator
)

from app.evaluation.generation_metrics import (
    faithfulness_score,
    answer_relevance_score,
    exact_correctness
)


def main():

    retriever = RAGRetriever(
        "data/documents"
    )

    generator = LlamaGenerator(
        model_name="llama3.2"
    )

    with open(
        "data/generation_dataset.json",
        "r",
        encoding="utf-8"
    ) as file:

        dataset = json.load(file)


    results = []


    for item in dataset:

        question = item["question"]

        expected_answer = (
            item["expected_answer"]
        )


        # Step 1: Retrieve context

        retrieved = retriever.retrieve(
            question,
            top_k=3
        )


        # Step 2: Generate answer

        answer = generator.generate(
            question,
            retrieved
        )


        # Step 3: Combine context

        context = "\n\n".join(
            result["text"]
            for result in retrieved
        )


        # Step 4: Evaluate

        faithfulness = (
            faithfulness_score(
                answer,
                context
            )
        )

        relevance = (
            answer_relevance_score(
                question,
                answer
            )
        )

        correctness = (
            exact_correctness(
                answer,
                expected_answer
            )
        )


        result = {
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": answer,
            "faithfulness": faithfulness,
            "relevance": relevance,
            "correctness": correctness
        }


        results.append(result)


        print("\n" + "=" * 70)

        print(
            f"Question:\n{question}"
        )

        print(
            f"\nExpected Answer:\n"
            f"{expected_answer}"
        )

        print(
            f"\nGenerated Answer:\n"
            f"{answer}"
        )

        print(
            f"\nFaithfulness: "
            f"{faithfulness:.2%}"
        )

        print(
            f"Relevance: "
            f"{relevance:.2%}"
        )

        print(
            f"Correctness: "
            f"{correctness:.2%}"
        )


    # Overall scores

    if results:

        avg_faithfulness = (
            sum(
                r["faithfulness"]
                for r in results
            )
            / len(results)
        )

        avg_relevance = (
            sum(
                r["relevance"]
                for r in results
            )
            / len(results)
        )

        avg_correctness = (
            sum(
                r["correctness"]
                for r in results
            )
            / len(results)
        )


        print("\n")
        print("=" * 70)
        print("OVERALL GENERATION EVALUATION")
        print("=" * 70)

        print(
            f"Faithfulness: "
            f"{avg_faithfulness:.2%}"
        )

        print(
            f"Answer Relevance: "
            f"{avg_relevance:.2%}"
        )

        print(
            f"Answer Correctness: "
            f"{avg_correctness:.2%}"
        )


if __name__ == "__main__":
    main()