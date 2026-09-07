import json

from app.rag.rag_retriever import (
    RAGRetriever
)

from app.evaluation.retrieval_metrics import (
    context_precision,
    context_recall,
    hit_rate,
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    reciprocal_rank,
    ndcg_at_k
)


def main():

    retriever = RAGRetriever(
        "data/documents"
    )

    with open(
        "data/retrieval_dataset.json",
        "r",
        encoding="utf-8"
    ) as file:

        dataset = json.load(file)


    all_results = []


    for item in dataset:

        question = item["question"]

        relevant_sources = (
            item["relevant_sources"]
        )

        retrieved = retriever.retrieve(
            question,
            top_k=10
        )

        retrieved_sources = [
            result["source"]
            for result in retrieved
        ]


        precision = context_precision(
            retrieved_sources,
            relevant_sources
        )

        recall = context_recall(
            retrieved_sources,
            relevant_sources
        )

        hit = hit_rate(
            retrieved_sources,
            relevant_sources
        )

        precision_k = precision_at_k(
            retrieved_sources,
            relevant_sources,
            5
        )

        recall_k = recall_at_k(
            retrieved_sources,
            relevant_sources,
            5
        )

        hit_k = hit_rate_at_k(
            retrieved_sources,
            relevant_sources,
            5
        )

        rr = reciprocal_rank(
            retrieved_sources,
            relevant_sources
        )

        relevance_scores = [
            1 if source in relevant_sources else 0
            for source in retrieved_sources[:5]
        ]

        ndcg_k = ndcg_at_k(
            relevance_scores,
            5
        )


        result = {
            "question": question,
            "expected_sources": (
                relevant_sources
            ),
            "retrieved_sources": (
                retrieved_sources
            ),
            "precision": precision,
            "recall": recall,
            "hit_rate": hit,
            "precision_at_5": precision_k,
            "recall_at_5": recall_k,
            "hit_rate_at_5": hit_k,
            "reciprocal_rank": rr,
            "ndcg_at_5": ndcg_k
        }


        all_results.append(result)


        print("\n" + "=" * 60)

        print(
            f"Question: {question}"
        )

        print(
            f"Expected: {relevant_sources}"
        )

        print(
            f"Retrieved: {retrieved_sources}"
        )

        print(
            f"Precision: {precision:.2f}"
        )

        print(
            f"Recall: {recall:.2f}"
        )

        print(
            f"Hit Rate: {hit}"
        )

        print(
            f"Precision@5: {precision_k:.2f}"
        )

        print(
            f"Recall@5: {recall_k:.2f}"
        )

        print(
            f"Hit Rate@5: {hit_k:.2f}"
        )

        print(
            f"Reciprocal Rank: {rr:.2f}"
        )

        print(
            f"NDCG@5: {ndcg_k:.2f}"
        )

        print("\nRetrieved chunks:")

        for result in retrieved:

            print(
                f"- {result['chunk_id']}"
            )

            print(
                f"  Score: {result['score']:.4f}"
            )

            print(
                f"  Text: {result['text']}"
            )


    # Overall metrics

    average_precision = (
        sum(
            item["precision"]
            for item in all_results
        )
        / len(all_results)
    )

    average_recall = (
        sum(
            item["recall"]
            for item in all_results
        )
        / len(all_results)
    )

    average_hit_rate = (
        sum(
            item["hit_rate"]
            for item in all_results
        )
        / len(all_results)
    )


    print("\n")
    print("=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)

    print(
        f"Average Precision: "
        f"{average_precision:.2%}"
    )

    print(
        f"Average Recall: "
        f"{average_recall:.2%}"
    )

    print(
        f"Average Hit Rate: "
        f"{average_hit_rate:.2%}"
    )


if __name__ == "__main__":
    main()