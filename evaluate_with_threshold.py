import json

from app.rag.rag_retriever import RAGRetriever
from app.rag.llm_generator import LlamaGenerator
from app.evaluation.llm_judge import LLMJudge
from app.evaluation.thresholds import check_thresholds


def main():

    retriever = RAGRetriever(
        "data/documents"
    )

    generator = LlamaGenerator(
        model_name="llama3.2"
    )

    judge = LLMJudge(
        model_name="llama3.2"
    )


    with open(
        "data/golden_dataset.json",
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


        # -------------------------
        # Retrieve
        # -------------------------

        retrieved = retriever.retrieve(
            question,
            top_k=3
        )


        # -------------------------
        # Build context
        # -------------------------

        context = "\n\n".join(
            result["text"]
            for result in retrieved
        )


        # -------------------------
        # Generate
        # -------------------------

        generated_answer = (
            generator.generate(
                question,
                retrieved
            )
        )


        # -------------------------
        # Judge
        # -------------------------

        evaluation = judge.evaluate(
            question=question,
            context=context,
            answer=generated_answer,
            expected_answer=expected_answer
        )


        # -------------------------
        # Threshold
        # -------------------------

        passed = check_thresholds(
            evaluation
        )


        result = {
            "id": item["id"],
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": generated_answer,
            "evaluation": evaluation,
            "status": (
                "PASS"
                if passed
                else "FAIL"
            )
        }


        results.append(result)


        # -------------------------
        # Print
        # -------------------------

        print("\n" + "=" * 70)

        print(
            f"ID: {item['id']}"
        )

        print(
            f"Question: {question}"
        )

        print(
            f"\nExpected:\n"
            f"{expected_answer}"
        )

        print(
            f"\nGenerated:\n"
            f"{generated_answer}"
        )

        print(
            "\nScores:"
        )

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

        print(
            f"\nSTATUS: "
            f"{result['status']}"
        )


    # -------------------------
    # Summary
    # -------------------------

    passed_count = sum(
        1
        for result in results
        if result["status"] == "PASS"
    )

    failed_count = (
        len(results) - passed_count
    )


    print("\n")
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"Total tests : {len(results)}"
    )

    print(
        f"Passed      : {passed_count}"
    )

    print(
        f"Failed      : {failed_count}"
    )


    pass_rate = (
        passed_count / len(results)
        if results
        else 0
    )


    print(
        f"Pass rate   : {pass_rate:.2%}"
    )


    # -------------------------
    # Save results
    # -------------------------

    with open(
        "evaluation_results.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )


if __name__ == "__main__":
    main()