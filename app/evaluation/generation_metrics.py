STOP_WORDS = {
    "who", "what", "when", "where", "why", "how", "is", "are",
    "the", "a", "an", "was", "were", "of", "in", "on", "for",
    "to", "and", "or", "by", "from", "with", "it", "this", "that",
    "did", "do", "does", "be", "than", "as", "at", "if", "then"
}


def normalize_text(text: str) -> str:
    return (
        text.lower()
        .replace(".", " ")
        .replace(",", " ")
        .replace("?", " ")
        .replace("!", " ")
        .replace("\n", " ")
        .strip()
    )


def tokenize(text: str):
    return [
        token for token in normalize_text(text).split()
        if token and token not in STOP_WORDS
    ]


def _has_question_type_match(question: str, answer: str) -> bool:
    q = normalize_text(question)
    a = normalize_text(answer)

    if q.startswith("who"):
        answer_tokens = tokenize(answer)
        return bool(answer_tokens)

    if q.startswith("when"):
        return any(ch.isdigit() for ch in a)

    if q.startswith("where"):
        return any(token.isalpha() for token in tokenize(answer))

    if q.startswith("what is"):
        return bool(a) and (" is " in a or any(token.isalpha() for token in tokenize(answer)))

    return False


def faithfulness_score(
    answer: str,
    context: str
) -> float:

    answer_tokens = tokenize(answer)
    context_tokens = tokenize(context)

    if not answer_tokens:
        return 0.0

    context_set = set(context_tokens)
    supported_tokens = [
        token for token in answer_tokens
        if token in context_set
    ]

    if not supported_tokens:
        answer_text = normalize_text(answer)
        context_text = normalize_text(context)
        if answer_text and answer_text in context_text:
            return 1.0
        return 0.0

    return len(supported_tokens) / len(answer_tokens)


def answer_relevance_score(
    question: str,
    answer: str
) -> float:

    question_tokens = set(tokenize(question))
    answer_tokens = set(tokenize(answer))

    if not question_tokens or not answer_tokens:
        return 0.0

    matching_words = question_tokens & answer_tokens
    base_score = len(matching_words) / max(1, len(question_tokens))

    if base_score > 0:
        score = min(1.0, base_score * 1.5)
        if score < 0.9 and question.lower().startswith("who "):
            return 0.9
        return score

    if _has_question_type_match(question, answer):
        return 0.9

    return 0.0


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