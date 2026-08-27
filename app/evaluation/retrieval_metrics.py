def context_precision(
    retrieved_sources,
    relevant_sources
):
    """
    Precision:
    Of the retrieved sources,
    how many are relevant?
    """

    if not retrieved_sources:
        return 0.0

    relevant_sources = set(
        relevant_sources
    )

    relevant_count = sum(
        source in relevant_sources
        for source in retrieved_sources
    )

    return relevant_count / len(
        retrieved_sources
    )


def context_recall(
    retrieved_sources,
    relevant_sources
):
    """
    Recall:
    Of all relevant sources,
    how many were retrieved?
    """

    relevant_sources = set(
        relevant_sources
    )

    if not relevant_sources:
        return 0.0

    retrieved_sources = set(
        retrieved_sources
    )

    retrieved_relevant = (
        retrieved_sources
        & relevant_sources
    )

    return len(
        retrieved_relevant
    ) / len(
        relevant_sources
    )


def hit_rate(
    retrieved_sources,
    relevant_sources
):
    """
    Returns:
        1 if at least one relevant
        source was retrieved.

        0 otherwise.
    """

    relevant_sources = set(
        relevant_sources
    )

    for source in retrieved_sources:

        if source in relevant_sources:
            return 1

    return 0