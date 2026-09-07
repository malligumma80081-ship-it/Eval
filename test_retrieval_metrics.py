from app.evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
    ndcg_at_k
)


retrieved = [
    "machine_learning.txt",
    "python.txt",
    "ai.txt",
    "python_history.txt",
    "database.txt"
]


relevant = [
    "python.txt",
    "python_history.txt"
]


print(
    "Precision@5:",
    precision_at_k(
        retrieved,
        relevant,
        5
    )
)


print(
    "Recall@5:",
    recall_at_k(
        retrieved,
        relevant,
        5
    )
)


print(
    "Hit Rate@5:",
    hit_rate_at_k(
        retrieved,
        relevant,
        5
    )
)


print(
    "Reciprocal Rank:",
    reciprocal_rank(
        retrieved,
        relevant
    )
)


relevance_scores = [
    0,
    3,
    0,
    2,
    0
]


print(
    "NDCG@5:",
    ndcg_at_k(
        relevance_scores,
        5
    )
)