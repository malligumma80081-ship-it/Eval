def normalize_text(text: str) -> str:

    return (
        text.lower()
        .replace(".", "")
        .replace(",", "")
        .strip()
    )


def faithfulness_score(
    answer: str,
    context: str
) -> float:

    answer_words = set(
        normalize_text(answer).split()
    )

    context_words = set(
        normalize_text(context).split()
    )

    if not answer_words:
        return 0.0

    supported_words = (
        answer_words & context_words
    )

    return len(supported_words) / len(
        answer_words
    )


def answer_relevance_score(
    question: str,
    answer: str
) -> float:

    question_words = set(
        normalize_text(question).split()
    )

    answer_words = set(
        normalize_text(answer).split()
    )

    if not question_words:
        return 0.0

    matching_words = (
        question_words & answer_words
    )

    return len(matching_words) / len(
        question_words
    )


def exact_correctness(
    answer: str,
    expected_answer: str
) -> float:

    answer = normalize_text(answer)
    expected = normalize_text(
        expected_answer
    )

    return float(
        answer == expected
    )