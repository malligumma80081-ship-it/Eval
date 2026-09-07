import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.evaluation.retrieval_metrics import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
    recall_at_k,
)
from app.rag.rag_retriever import RAGRetriever


RESULTS_FILE = Path(
    "results/current_results.json"
)

REGRESSION_FILE = Path(
    "results/regression_report.json"
)

RETRIEVAL_DATASET_FILE = Path(
    "data/retrieval_dataset.json"
)


def extract_score(value):
    if isinstance(value, dict):
        return value.get("score", 0)
    return value


@st.cache_data(show_spinner=False)
def compute_retrieval_metrics(top_k=5):
    if not RETRIEVAL_DATASET_FILE.exists():
        return {
            "precision": 0.0,
            "recall": 0.0,
            "hit_rate": 0.0,
            "reciprocal_rank": 0.0,
            "ndcg": 0.0,
            "details": []
        }

    with open(RETRIEVAL_DATASET_FILE, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    retriever = RAGRetriever("data/documents")
    details = []

    precision_values = []
    recall_values = []
    hit_values = []
    rr_values = []
    ndcg_values = []

    for item in dataset:
        question = item["question"]
        relevant_sources = item["relevant_sources"]

        retrieved = retriever.retrieve(question, top_k=top_k)
        retrieved_sources = [result["source"] for result in retrieved]

        precision = precision_at_k(retrieved_sources, relevant_sources, top_k)
        recall = recall_at_k(retrieved_sources, relevant_sources, top_k)
        hit = hit_rate_at_k(retrieved_sources, relevant_sources, top_k)
        rr = reciprocal_rank(retrieved_sources, relevant_sources)
        relevance_scores = [
            1 if source in relevant_sources else 0
            for source in retrieved_sources[:top_k]
        ]
        ndcg = ndcg_at_k(relevance_scores, top_k)

        precision_values.append(precision)
        recall_values.append(recall)
        hit_values.append(hit)
        rr_values.append(rr)
        ndcg_values.append(ndcg)

        details.append({
            "Question": question,
            "Relevant": relevant_sources,
            "Retrieved": retrieved_sources,
            "Precision@K": round(precision, 4),
            "Recall@K": round(recall, 4),
            "HitRate@K": round(hit, 4),
            "Reciprocal": round(rr, 4),
            "NDCG@K": round(ndcg, 4),
        })

    return {
        "precision": sum(precision_values) / len(precision_values) if precision_values else 0.0,
        "recall": sum(recall_values) / len(recall_values) if recall_values else 0.0,
        "hit_rate": sum(hit_values) / len(hit_values) if hit_values else 0.0,
        "reciprocal_rank": sum(rr_values) / len(rr_values) if rr_values else 0.0,
        "ndcg": sum(ndcg_values) / len(ndcg_values) if ndcg_values else 0.0,
        "details": details,
    }


# ---------------------------------------
# Page configuration
# ---------------------------------------

st.set_page_config(
    page_title="LLM Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------------
# Load evaluation results
# ---------------------------------------

if not RESULTS_FILE.exists():

    st.error(
        "current_results.json not found."
    )

    st.info(
        "Run: python run_evaluation.py"
    )

    st.stop()


with open(
    RESULTS_FILE,
    "r",
    encoding="utf-8"
) as file:

    results = json.load(file)


summary = results["summary"]
cases = results["cases"]

summary_overall = summary.get("overall", 0)


# ---------------------------------------
# Title
# ---------------------------------------

st.title(
    "📊 LLM / RAG Evaluation Dashboard"
)

st.write(
    f"Evaluation Version: **{results['version']}**"
)


# ---------------------------------------
# KPI Metrics
# ---------------------------------------

col1, col2, col3, col4, col5 = st.columns(5)


col1.metric(
    "Pass Rate",
    f"{summary['pass_rate']:.0%}"
)

col2.metric(
    "Faithfulness",
    f"{summary['faithfulness']:.2f} / 5"
)

col3.metric(
    "Relevance",
    f"{summary['relevance']:.2f} / 5"
)

col4.metric(
    "Correctness",
    f"{summary['correctness']:.2f} / 5"
)

col5.metric(
    "Overall",
    f"{summary_overall:.2f} / 5"
)


st.divider()


# ---------------------------------------
# Retrieval Metrics
# ---------------------------------------

st.header("Retrieval Metrics")

retrieval_metrics = compute_retrieval_metrics(top_k=5)
retrieval_cols = st.columns(5)
retrieval_cols[0].metric("Precision@5", f"{retrieval_metrics['precision']:.2%}")
retrieval_cols[1].metric("Recall@5", f"{retrieval_metrics['recall']:.2%}")
retrieval_cols[2].metric("Hit Rate@5", f"{retrieval_metrics['hit_rate']:.2%}")
retrieval_cols[3].metric("Reciprocal Rank", f"{retrieval_metrics['reciprocal_rank']:.2%}")
retrieval_cols[4].metric("NDCG@5", f"{retrieval_metrics['ndcg']:.2%}")

retrieval_df = pd.DataFrame(retrieval_metrics["details"])
if not retrieval_df.empty:
    st.dataframe(retrieval_df, use_container_width=True)


# ---------------------------------------
# Evaluation Summary
# ---------------------------------------

st.header("Evaluation Summary")


chart_data = pd.DataFrame({
    "Metric": [
        "Faithfulness",
        "Relevance",
        "Correctness",
        "Overall"
    ],
    "Score": [
        summary.get("faithfulness", 0),
        summary.get("relevance", 0),
        summary.get("correctness", 0),
        summary_overall
    ]
})


st.bar_chart(
    chart_data.set_index("Metric")
)


# ---------------------------------------
# Test Case Results
# ---------------------------------------

st.header("Test Case Results")


table_data = []


for case in cases:

    evaluation = case["evaluation"]

    table_data.append({
        "ID": case["id"],
        "Question": case["question"],
        "Faithfulness": extract_score(evaluation.get("faithfulness", 0)),
        "Relevance": extract_score(evaluation.get("relevance", 0)),
        "Correctness": extract_score(evaluation.get("correctness", 0)),
        "Overall": extract_score(evaluation.get("overall_score", evaluation.get("overall", 0))),
        "Status": case["status"]
    })


df = pd.DataFrame(table_data)


st.dataframe(
    df,
    use_container_width=True
)


# ---------------------------------------
# Failed Test Cases
# ---------------------------------------

st.header("❌ Failed Test Cases")


failed_cases = df[
    df["Status"] == "FAIL"
]


if failed_cases.empty:

    st.success(
        "🎉 All test cases passed!"
    )

else:

    st.dataframe(
        failed_cases,
        use_container_width=True
    )


# ---------------------------------------
# Regression Result
# ---------------------------------------

st.header("Regression Testing")


if REGRESSION_FILE.exists():

    with open(
        REGRESSION_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        regression = json.load(file)


    status = regression["overall_status"]


    if status == "PASS":

        st.success(
            "✅ No regression detected"
        )

    else:

        st.error(
            "🚨 Regression detected"
        )


    regression_rows = []


    for metric, data in regression[
        "metrics"
    ].items():

        regression_rows.append({
            "Metric": metric,
            "Baseline": data["baseline"],
            "New": data["new"],
            "Difference": round(
                data["difference"],
                2
            ),
            "Regression": (
                "YES"
                if data["regression"]
                else "NO"
            )
        })


    regression_df = pd.DataFrame(
        regression_rows
    )


    st.dataframe(
        regression_df,
        use_container_width=True
    )

else:

    st.info(
        "No regression report available."
    )


# ---------------------------------------
# Detailed Answers
# ---------------------------------------

st.header("Detailed Evaluation")


for case in cases:

    with st.expander(
        f"{case['id']} — {case['status']}"
    ):

        st.write(
            "**Question:**"
        )

        st.write(
            case["question"]
        )

        st.write(
            "**Expected Answer:**"
        )

        st.write(
            case["expected_answer"]
        )

        st.write(
            "**Generated Answer:**"
        )

        st.write(
            case["generated_answer"]
        )

        st.write(
            "**Evaluation:**"
        )

        st.json(
            case["evaluation"]
        )