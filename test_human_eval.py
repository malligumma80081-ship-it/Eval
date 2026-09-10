from app.evaluation.human_eval import evaluate_results


results = [
    {
        "correctness": 5,
        "relevance": 5,
        "faithfulness": 5,
        "completeness": 4,
        "overall": 5
    },
    {
        "correctness": 4,
        "relevance": 5,
        "faithfulness": 5,
        "completeness": 4,
        "overall": 4
    },
    {
        "correctness": 5,
        "relevance": 4,
        "faithfulness": 4,
        "completeness": 5,
        "overall": 4
    }
]


summary = evaluate_results(results)

print(summary)