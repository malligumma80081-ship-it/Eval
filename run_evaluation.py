import json
from pathlib import Path

from app.rag.rag_retriever import RAGRetriever
from app.rag.llm_generator import LlamaGenerator
from app.evaluation.llm_judge import LLMJudge
from app.evaluation.thresholds import check_thresholds, extract_metric_score
from app.evaluation.regression import compare_versions


DATASET_PATH = Path("data/golden_dataset.json")

CURRENT_RESULTS_PATH = Path(
    "results/current_results.json"
)

BASELINE_RESULTS_PATH = Path(
    "results/baseline_results.json"
)

REGRESSION_PATH = Path(
    "results/regression_report.json"
)


VERSION = "1.1"


def load_dataset():

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def calculate_average(results, metric):

    scores = [
        extract_metric_score(item["evaluation"], metric)
        for item in results
    ]

    if not scores:
        return 0

    return sum(scores) / len(scores)


def main():

    print("=" * 60)
    print("AUTOMATED LLM/RAG EVALUATION")
    print("=" * 60)

    # -----------------------------------
    # 1. Load dataset
    # -----------------------------------

    dataset = load_dataset()

    print(f"\nLoaded {len(dataset)} test cases")

    # -----------------------------------
    # 2. Initialize RAG
    # -----------------------------------

    retriever = RAGRetriever(
        "data/documents"
    )

    generator = LlamaGenerator(
        model_name="llama3.2"
    )

    judge = LLMJudge(
        model_name="llama3.2"
    )

    results = []

    # -----------------------------------
    # 3. Evaluate every test case
    # -----------------------------------

    for index, item in enumerate(dataset, start=1):

        question = item["question"]
        expected_answer = item["expected_answer"]

        print("\n" + "-" * 60)
        print(f"Test Case {index}: {item['id']}")
        print(f"Question: {question}")

        # Retrieve context
        context = retriever.retrieve(question)

        # Generate answer
        answer = generator.generate(
            question=question,
            contexts=context
        )

        # LLM Judge
        evaluation = judge.evaluate(
            question=question,
            context=context,
            answer=answer,
            expected_answer=expected_answer
        )

        # Threshold
        passed = check_thresholds(evaluation)

        status = "PASS" if passed else "FAIL"

        print(f"Generated Answer: {answer}")
        print(f"Faithfulness: {extract_metric_score(evaluation, 'faithfulness')}")
        print(f"Relevance: {extract_metric_score(evaluation, 'relevance')}")
        print(f"Correctness: {extract_metric_score(evaluation, 'correctness')}")
        print(f"Overall: {extract_metric_score(evaluation, 'overall')}")
        print(f"Status: {status}")

        results.append({
            "id": item["id"],
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": answer,
            "evaluation": evaluation,
            "status": status
        })

    # -----------------------------------
    # 4. Calculate summary
    # -----------------------------------

    total = len(results)

    passed = sum(
        1
        for item in results
        if item["status"] == "PASS"
    )

    failed = total - passed

    pass_rate = passed / total if total else 0

    summary = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(pass_rate, 2),

        "faithfulness": round(
            calculate_average(
                results,
                "faithfulness"
            ),
            2
        ),

        "relevance": round(
            calculate_average(
                results,
                "relevance"
            ),
            2
        ),

        "correctness": round(
            calculate_average(
                results,
                "correctness"
            ),
            2
        ),

        "overall": round(
            calculate_average(
                results,
                "overall"
            ),
            2
        )
    }

    # -----------------------------------
    # 5. Create result object
    # -----------------------------------

    current_results = {
        "version": VERSION,
        "summary": summary,
        "cases": results
    }

    # -----------------------------------
    # 6. Save results
    # -----------------------------------

    Path("results").mkdir(
        exist_ok=True
    )

    with open(
        CURRENT_RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            current_results,
            file,
            indent=4
        )

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"Total Tests : {total}")
    print(f"Passed      : {passed}")
    print(f"Failed      : {failed}")
    print(f"Pass Rate   : {pass_rate:.0%}")

    print(
        f"Faithfulness : {summary['faithfulness']}"
    )

    print(
        f"Relevance    : {summary['relevance']}"
    )

    print(
        f"Correctness  : {summary['correctness']}"
    )

    print(
        f"Overall      : {summary['overall']}"
    )

    # -----------------------------------
    # 7. Regression test
    # -----------------------------------

    if BASELINE_RESULTS_PATH.exists():

        with open(
            BASELINE_RESULTS_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            baseline = json.load(file)

        regression = compare_versions(
            baseline,
            {
                "version": VERSION,
                **summary
            }
        )

        with open(
            REGRESSION_PATH,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                regression,
                file,
                indent=4
            )

        print("\n" + "=" * 60)
        print("REGRESSION RESULT")
        print("=" * 60)

        print(
            f"Status: {regression['overall_status']}"
        )

    else:

        print(
            "\nNo baseline found. "
            "Skipping regression test."
        )


if __name__ == "__main__":
    main()