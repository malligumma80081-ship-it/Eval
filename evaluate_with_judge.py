import json

from app.rag.rag_retriever import (
    RAGRetriever
)

from app.rag.llm_generator import (
    LlamaGenerator
)

from app.evaluation.llm_judge import (
    LLMJudge
)


def main():

    # -------------------------
    # Create components
    # -------------------------

    retriever = RAGRetriever(
        "data/documents"
    )

    generator = LlamaGenerator(
        model_name="llama3.2"
    )

    judge = LLMJudge(
        model_name="llama3.2"
    )


    # -------------------------
    # Load evaluation dataset
    # -------------------------

    with open(
        "data/generation_dataset.json",
        "r",
        encoding="utf-8"
    ) as file:

        dataset = json.load(file)


    results = []


    # -------------------------
    # Evaluate each question
    # -------------------------

    for item in dataset:

        question = item["question"]

        expected_answer = (
            item["expected_answer"]
        )


        # 1. Retrieve context

        retrieved = retriever.retrieve(
            question,
            top_k=3
        )


        # 2. Build context

        context = "\n\n".join(
            result["text"]
            for result in retrieved
        )


        # 3. Generate answer

        generated_answer = (
            generator.generate(
                question,
                retrieved
            )
        )


        # 4. Judge answer

        evaluation = judge.evaluate(
            question=question,
            context=context,
            answer=generated_answer,
            expected_answer=expected_answer
        )


        # 5. Save result

        result = {
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": generated_answer,
            "evaluation": evaluation
        }

        results.append(result)


        # -------------------------
        # Print result
        # -------------------------

        print("\n")
        print("=" * 70)

        print(
            f"QUESTION:\n{question}"
        )

        print(
            f"\nEXPECTED:\n{expected_answer}"
        )

        print(
            f"\nGENERATED:\n{generated_answer}"
        )

        print("\nJUDGE:")

        print(
            f"Faithfulness: "
            f"{evaluation['faithfulness']['score']}/5"
        )

        print(
            f"Relevance: "
            f"{evaluation['relevance']['score']}/5"
        )

        print(
            f"Correctness: "
            f"{evaluation['correctness']['score']}/5"
        )

        print(
            f"Overall: "
            f"{evaluation['overall_score']}/5"
        )


    # -------------------------
    # Calculate average
    # -------------------------

    if results:

        avg_faithfulness = (
            sum(
                r["evaluation"]
                ["faithfulness"]["score"]
                for r in results
            )
            / len(results)
        )

        avg_relevance = (
            sum(
                r["evaluation"]
                ["relevance"]["score"]
                for r in results
            )
            / len(results)
        )

        avg_correctness = (
            sum(
                r["evaluation"]
                ["correctness"]["score"]
                for r in results
            )
            / len(results)
        )

        avg_overall = (
            sum(
                r["evaluation"]
                ["overall_score"]
                for r in results
            )
            / len(results)
        )


        print("\n")
        print("=" * 70)
        print("FINAL EVALUATION")
        print("=" * 70)

        print(
            f"Average Faithfulness: "
            f"{avg_faithfulness:.2f}/5"
        )

        print(
            f"Average Relevance: "
            f"{avg_relevance:.2f}/5"
        )

        print(
            f"Average Correctness: "
            f"{avg_correctness:.2f}/5"
        )

        print(
            f"Average Overall: "
            f"{avg_overall:.2f}/5"
        )


if __name__ == "__main__":
    main()