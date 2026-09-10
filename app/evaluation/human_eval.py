import json
from pathlib import Path


def calculate_average(scores):

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def evaluate_results(results):

    metrics = [
        "correctness",
        "relevance",
        "faithfulness",
        "completeness",
        "overall"
    ]

    summary = {}

    for metric in metrics:

        scores = [
            item[metric]
            for item in results
        ]

        summary[metric] = calculate_average(scores)

    return summary


def load_human_eval_results(path="results/human_eval_results.json"):
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            return evaluate_results(data)
        return {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_human_eval_results(results, path="results/human_eval_results.json"):
    summary = evaluate_results(results)
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary
