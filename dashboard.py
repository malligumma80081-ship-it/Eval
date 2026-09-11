import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.evaluation.generation_metrics import answer_relevance_score
from app.evaluation.hallucination import evaluate_hallucination
from app.evaluation.retrieval_metrics import (
    context_precision,
    context_recall,
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

PERFORMANCE_FILE = Path(
    "results/performance_metrics.json"
)

HUMAN_EVAL_FILE = Path(
    "results/human_eval_results.json"
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


def parse_hallucination_score(raw_response):
    if isinstance(raw_response, (int, float)):
        return float(raw_response)

    if isinstance(raw_response, dict):
        return float(raw_response.get("score", 0.0))

    text = str(raw_response).strip()
    if not text:
        return 0.0

    try:
        payload = json.loads(text)
        return float(payload.get("score", 0.0))
    except json.JSONDecodeError:
        return 0.0


@st.cache_data(show_spinner=False)
def load_performance_summary():
    default = {
        "average_latency_sec": 0.0,
        "p95_latency_sec": 0.0,
        "p99_latency_sec": 0.0,
        "cost_per_request": 0.0,
    }

    if PERFORMANCE_FILE.exists():
        try:
            with open(PERFORMANCE_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)
            if isinstance(data, dict):
                return {**default, **data}
        except json.JSONDecodeError:
            pass

    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)
            if isinstance(data, dict):
                perf = data.get("performance", data.get("latency", {}))
                if isinstance(perf, dict):
                    return {**default, **perf}
        except json.JSONDecodeError:
            pass

    return default


@st.cache_data(show_spinner=False)
def compute_rag_quality_metrics(top_k=5):
    metrics = {
        "context_precision": 0.0,
        "context_recall": 0.0,
        "faithfulness": 0.0,
        "answer_relevance": 0.0,
        "hallucination": 0.0,
        "details": []
    }

    if RETRIEVAL_DATASET_FILE.exists():
        with open(RETRIEVAL_DATASET_FILE, "r", encoding="utf-8") as file:
            dataset = json.load(file)

        retriever = RAGRetriever("data/documents")
        precision_values = []
        recall_values = []

        for item in dataset:
            question = item["question"]
            relevant_sources = item["relevant_sources"]
            retrieved = retriever.retrieve(question, top_k=top_k)
            retrieved_sources = [result["source"] for result in retrieved]

            precision_values.append(context_precision(retrieved_sources, relevant_sources))
            recall_values.append(context_recall(retrieved_sources, relevant_sources))

        metrics["context_precision"] = (
            sum(precision_values) / len(precision_values) if precision_values else 0.0
        )
        metrics["context_recall"] = (
            sum(recall_values) / len(recall_values) if recall_values else 0.0
        )

    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r", encoding="utf-8") as file:
            results = json.load(file)

        case_scores = []
        relevance_scores = []
        hallucination_scores = []

        retriever = RAGRetriever("data/documents")

        for case in results.get("cases", []):
            evaluation = case.get("evaluation", {})
            question = case.get("question", "")
            answer = case.get("generated_answer", "")

            case_scores.append(extract_score(evaluation.get("faithfulness", 0)))
            relevance_scores.append(answer_relevance_score(question, answer))

            context = "\n\n".join(
                result["text"] for result in retriever.retrieve(question, top_k=top_k)
            )

            raw_result = evaluate_hallucination(question, context, answer)
            hallucination_scores.append(parse_hallucination_score(raw_result))

        if case_scores:
            metrics["faithfulness"] = sum(case_scores) / len(case_scores)
        if relevance_scores:
            metrics["answer_relevance"] = sum(relevance_scores) / len(relevance_scores)
        if hallucination_scores:
            metrics["hallucination"] = sum(hallucination_scores) / len(hallucination_scores)

    return metrics


@st.cache_data(show_spinner=False)
def load_human_eval_summary():
    if HUMAN_EVAL_FILE.exists():
        try:
            with open(HUMAN_EVAL_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    return {
        "correctness": 0.0,
        "relevance": 0.0,
        "faithfulness": 0.0,
        "completeness": 0.0,
        "overall": 0.0,
    }


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


@st.cache_data(show_spinner=False)
def compare_prompt_variants(dataset=None, top_k=5):
    if dataset is None:
        dataset = []
        if RETRIEVAL_DATASET_FILE.exists():
            with open(RETRIEVAL_DATASET_FILE, "r", encoding="utf-8") as file:
                dataset = json.load(file)

    retriever = RAGRetriever("data/documents")
    results = []

    for item in dataset:
        question = item["question"]
        relevant_sources = item.get("relevant_sources", [])
        context = "\n\n".join(
            result["text"]
            for result in retriever.retrieve(question, top_k=top_k)
        )

        comparison = {
            "question": question,
            "prompt_a": None,
            "prompt_b": None,
            "winner": None,
        }

        from app.evaluation.ab_test import compare_prompts

        outcome = compare_prompts(question, context, "")
        comparison["prompt_a"] = outcome["prompt_a"]
        comparison["prompt_b"] = outcome["prompt_b"]
        comparison["winner"] = outcome["winner"]
        results.append(comparison)

    return results


# ---------------------------------------
# Page configuration
# ---------------------------------------

st.set_page_config(
    page_title="LLM Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #f5f7ff 0%, #eef4ff 100%);
        }
        div[data-testid="stMetricContainer"] {
            background: rgba(255,255,255,0.8);
            border: 1px solid rgba(79, 117, 255, 0.15);
            border-radius: 12px;
            padding: 0.85rem 1rem;
            box-shadow: 0 4px 12px rgba(79, 117, 255, 0.08);
        }
        div[data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
        }
        .section-title {
            color: #1f2a44;
            font-weight: 700;
            letter-spacing: 0.02em;
        }
    </style>
    """,
    unsafe_allow_html=True,
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
summary_safety = summary.get("safety", summary.get("safety_score"))
if summary_safety is None:
    safety_scores = []
    for case in cases:
        evaluation = case.get("evaluation", {})
        if "safety" in evaluation:
            safety_scores.append(extract_score(evaluation.get("safety", 0)))
        elif "safe" in evaluation and isinstance(evaluation["safe"], dict):
            safety_scores.append(extract_score(evaluation["safe"]))
        elif "safety_score" in evaluation:
            safety_scores.append(extract_score(evaluation.get("safety_score", 0)))
    summary_safety = sum(safety_scores) / len(safety_scores) if safety_scores else 0.0


# ---------------------------------------
# Title
# ---------------------------------------

st.title(
    "📊 LLM / RAG Evaluation Dashboard"
)

st.caption(
    f"Evaluation Version: **{results['version']}**"
)

st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)

# ---------------------------------------
# KPI Metrics
# ---------------------------------------

col1, col2, col3, col4, col5, col6 = st.columns(6)


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
    "Safety",
    f"{summary_safety:.2f} / 5"
)

col6.metric(
    "Overall",
    f"{summary_overall:.2f} / 5"
)


st.divider()

performance_metrics = load_performance_summary()
performance_cols = st.columns(4)
performance_cols[0].metric("Avg Latency", f"{performance_metrics['average_latency_sec']:.2f} sec")
performance_cols[1].metric("P95 Latency", f"{performance_metrics['p95_latency_sec']:.2f} sec")
performance_cols[2].metric("Cost / Request", f"${performance_metrics['cost_per_request']:.4f}")
performance_cols[3].metric("Safety", f"{summary_safety:.2f} / 5")

st.divider()


# ---------------------------------------
# Retrieval & RAG Quality Metrics
# ---------------------------------------

st.markdown('<div class="section-title">RAG Quality Metrics</div>', unsafe_allow_html=True)

rag_quality = compute_rag_quality_metrics(top_k=5)
quality_cols = st.columns(5)
quality_cols[0].metric("Context Precision", f"{rag_quality['context_precision']:.2%}")
quality_cols[1].metric("Context Recall", f"{rag_quality['context_recall']:.2%}")
quality_cols[2].metric("Faithfulness", f"{rag_quality['faithfulness']:.2%}")
quality_cols[3].metric("Answer Relevance", f"{rag_quality['answer_relevance']:.2%}")
quality_cols[4].metric("Hallucination", f"{(rag_quality['hallucination'] / 5):.2%}")

st.markdown('<div class="section-title">Retrieval Metrics</div>', unsafe_allow_html=True)
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

st.markdown('<div class="section-title">Human Evaluation</div>', unsafe_allow_html=True)
human_metrics = load_human_eval_summary()
human_cols = st.columns(5)
human_cols[0].metric("Human Correctness", f"{human_metrics.get('correctness', 0.0):.2f} / 5")
human_cols[1].metric("Human Relevance", f"{human_metrics.get('relevance', 0.0):.2f} / 5")
human_cols[2].metric("Human Faithfulness", f"{human_metrics.get('faithfulness', 0.0):.2f} / 5")
human_cols[3].metric("Human Completeness", f"{human_metrics.get('completeness', 0.0):.2f} / 5")
human_cols[4].metric("Human Overall", f"{human_metrics.get('overall', 0.0):.2f} / 5")

st.markdown('<div class="section-title">Prompt A vs Prompt B</div>', unsafe_allow_html=True)

prompt_ab_results = compare_prompt_variants(dataset=[]) if False else compare_prompt_variants(dataset=[{"question": q["question"]} for q in (json.load(open(RETRIEVAL_DATASET_FILE, "r", encoding="utf-8")) if RETRIEVAL_DATASET_FILE.exists() else [])])

prompt_rows = []
for item in prompt_ab_results:
    prompt_rows.append({
        "Question": item["question"],
        "Prompt A": item["prompt_a"]["score"]["overall"],
        "Prompt B": item["prompt_b"]["score"]["overall"],
        "Winner": item["winner"],
    })

if prompt_rows:
    prompt_df = pd.DataFrame(prompt_rows)
    st.dataframe(prompt_df, use_container_width=True)

    a_wins = sum(1 for row in prompt_rows if row["Winner"] == "Prompt A")
    b_wins = sum(1 for row in prompt_rows if row["Winner"] == "Prompt B")
    st.caption(f"Prompt A wins: {a_wins} | Prompt B wins: {b_wins}")
else:
    st.info("No prompt A/B results available yet.")


# ---------------------------------------
# Evaluation Summary
# ---------------------------------------

st.header("Evaluation Summary")


chart_data = pd.DataFrame({
    "Metric": [
        "Faithfulness",
        "Relevance",
        "Correctness",
        "Safety",
        "Overall"
    ],
    "Score": [
        summary.get("faithfulness", 0),
        summary.get("relevance", 0),
        summary.get("correctness", 0),
        summary_safety,
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

    safety_value = 0
    if "safety" in evaluation:
        safety_value = extract_score(evaluation.get("safety", 0))
    elif "safe" in evaluation and isinstance(evaluation["safe"], dict):
        safety_value = extract_score(evaluation["safe"])
    elif "safety_score" in evaluation:
        safety_value = extract_score(evaluation.get("safety_score", 0))

    table_data.append({
        "ID": case["id"],
        "Question": case["question"],
        "Faithfulness": extract_score(evaluation.get("faithfulness", 0)),
        "Relevance": extract_score(evaluation.get("relevance", 0)),
        "Correctness": extract_score(evaluation.get("correctness", 0)),
        "Safety": safety_value,
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