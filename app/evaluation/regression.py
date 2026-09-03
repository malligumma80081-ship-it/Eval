def compare_metric(
    baseline,
    new,
    tolerance
):
    difference = new - baseline

    regression = (
        difference < -tolerance
    )

    return {
        "baseline": baseline,
        "new": new,
        "difference": difference,
        "regression": regression
    }


def compare_versions(
    baseline,
    new,
    tolerance=0.2
):

    metrics = [
        "faithfulness",
        "relevance",
        "correctness",
        "overall"
    ]

    comparison = {}

    for metric in metrics:

        comparison[metric] = (
            compare_metric(
                baseline[metric],
                new[metric],
                tolerance
            )
        )


    has_regression = any(
        result["regression"]
        for result in comparison.values()
    )


    return {
        "baseline_version": baseline["version"],
        "new_version": new["version"],
        "tolerance": tolerance,
        "metrics": comparison,
        "overall_status": (
            "REGRESSION"
            if has_regression
            else "PASS"
        )
    }