# def context_precision(
#     retrieved_sources,
#     relevant_sources
# ):
#     """
#     Precision:
#     Of the retrieved sources,
#     how many are relevant?
#     """

#     if not retrieved_sources:
#         return 0.0

#     relevant_sources = set(
#         relevant_sources
#     )

#     relevant_count = sum(
#         source in relevant_sources
#         for source in retrieved_sources
#     )

#     return relevant_count / len(
#         retrieved_sources
#     )


# def context_recall(
#     retrieved_sources,
#     relevant_sources
# ):
#     """
#     Recall:
#     Of all relevant sources,
#     how many were retrieved?
#     """

#     relevant_sources = set(
#         relevant_sources
#     )

#     if not relevant_sources:
#         return 0.0

#     retrieved_sources = set(
#         retrieved_sources
#     )

#     retrieved_relevant = (
#         retrieved_sources
#         & relevant_sources
#     )

#     return len(
#         retrieved_relevant
#     ) / len(
#         relevant_sources
#     )


# def hit_rate(
#     retrieved_sources,
#     relevant_sources
# ):
#     """
#     Returns:
#         1 if at least one relevant
#         source was retrieved.

#         0 otherwise.
#     """

#     relevant_sources = set(
#         relevant_sources
#     )

#     for source in retrieved_sources:

#         if source in relevant_sources:
#             return 1

#     return 0



import math


def context_precision(retrieved_sources, relevant_sources):
    if not retrieved_sources:
        return 0.0

    relevant = set(relevant_sources)
    relevant_count = sum(
        1 for source in retrieved_sources if source in relevant
    )

    return relevant_count / len(retrieved_sources)


def context_recall(retrieved_sources, relevant_sources):
    relevant = set(relevant_sources)

    if not relevant:
        return 0.0

    retrieved = set(retrieved_sources)
    retrieved_relevant = retrieved & relevant

    return len(retrieved_relevant) / len(relevant)


def hit_rate(retrieved_sources, relevant_sources):
    relevant = set(relevant_sources)

    for source in retrieved_sources:
        if source in relevant:
            return 1.0

    return 0.0


def precision_at_k(retrieved_docs, relevant_docs, k):
    if k <= 0:
        return 0.0

    retrieved = retrieved_docs[:k]
    relevant = set(relevant_docs)

    relevant_count = sum(
        1 for doc in retrieved
        if doc in relevant
    )

    return relevant_count / k


def recall_at_k(retrieved_docs, relevant_docs, k):
    if not relevant_docs:
        return 0.0

    retrieved = retrieved_docs[:k]
    relevant = set(relevant_docs)

    relevant_count = sum(
        1 for doc in retrieved
        if doc in relevant
    )

    return relevant_count / len(relevant)


def hit_rate_at_k(retrieved_docs, relevant_docs, k):
    retrieved = retrieved_docs[:k]

    for doc in retrieved:
        if doc in relevant_docs:
            return 1.0

    return 0.0


def reciprocal_rank(retrieved_docs, relevant_docs):
    for rank, doc in enumerate(retrieved_docs, start=1):

        if doc in relevant_docs:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(results):

    if not results:
        return 0.0

    scores = []

    for result in results:

        score = reciprocal_rank(
            result["retrieved"],
            result["relevant"]
        )

        scores.append(score)

    return sum(scores) / len(scores)


def dcg_at_k(relevances, k):

    score = 0.0

    for index, relevance in enumerate(
        relevances[:k]
    ):

        rank = index + 1

        score += (
            (2 ** relevance - 1)
            / math.log2(rank + 1)
        )

    return score


def ndcg_at_k(relevances, k):

    actual_dcg = dcg_at_k(
        relevances,
        k
    )

    ideal_relevances = sorted(
        relevances,
        reverse=True
    )

    ideal_dcg = dcg_at_k(
        ideal_relevances,
        k
    )

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg