from app.evaluation.thresholds import check_thresholds


def test_check_thresholds_accepts_nested_metric_scores():
    evaluation = {
        "faithfulness": {"score": 5, "reason": "supported"},
        "relevance": {"score": 4, "reason": "answers the question"},
        "correctness": {"score": 4, "reason": "factually sound"},
        "overall_score": 4,
    }

    assert check_thresholds(evaluation) is True


def test_check_thresholds_rejects_low_score():
    evaluation = {
        "faithfulness": {"score": 3, "reason": "partially supported"},
        "relevance": {"score": 4, "reason": "answers the question"},
        "correctness": {"score": 4, "reason": "factually sound"},
        "overall_score": 4,
    }

    assert check_thresholds(evaluation) is False
