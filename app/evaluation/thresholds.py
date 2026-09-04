THRESHOLDS = {
    "faithfulness": 4,
    "relevance": 4,
    "correctness": 4,
    "overall": 4
}


def extract_metric_score(evaluation, metric):
    if metric == "overall":
        value = evaluation.get("overall_score", evaluation.get("overall", 0))
    else:
        value = evaluation.get(metric, 0)

    if isinstance(value, dict):
        return value.get("score", 0)

    return value


def check_thresholds(evaluation):
    for metric, threshold in THRESHOLDS.items():
        score = extract_metric_score(evaluation, metric)

        if score < threshold:
            return False

    return True


