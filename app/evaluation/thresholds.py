THRESHOLDS = {
    "faithfulness": 4.5,
    "relevance": 4,
    "correctness": 4.7,
    "overall": 4.2
}


def check_thresholds(evaluation):

    faithfulness = evaluation[
        "faithfulness"
    ]["score"]

    relevance = evaluation[
        "relevance"
    ]["score"]

    correctness = evaluation[
        "correctness"
    ]["score"]

    overall = evaluation[
        "overall_score"
    ]


    passed = (
        faithfulness >= THRESHOLDS["faithfulness"]
        and relevance >= THRESHOLDS["relevance"]
        and correctness >= THRESHOLDS["correctness"]
        and overall >= THRESHOLDS["overall"]
    )


    return passed