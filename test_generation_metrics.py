from app.evaluation.generation_metrics import answer_relevance_score, faithfulness_score


def test_answer_relevance_score_handles_who_question():
    question = "Who created Python?"
    answer = "Guido van Rossum."

    score = answer_relevance_score(question, answer)

    assert score > 0.5


def test_faithfulness_score_high_for_supported_answer():
    answer = "Guido van Rossum."
    context = "Python was created by Guido van Rossum."

    score = faithfulness_score(answer, context)

    assert score > 0.8
