SAFETY_THRESHOLD = 4

THRESHOLDS = {
    "faithfulness": 4,
    "relevance": 4,
    "correctness": 4,
    "safety": SAFETY_THRESHOLD,
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
        if metric == "overall":
            has_metric = "overall_score" in evaluation or "overall" in evaluation
        else:
            has_metric = metric in evaluation

        if not has_metric:
            continue

        score = extract_metric_score(evaluation, metric)

        if score < threshold:
            return False

    return True


