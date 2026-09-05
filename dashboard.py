import json
from pathlib import Path

import pandas as pd
import streamlit as st


RESULTS_FILE = Path(
    "results/current_results.json"
)

REGRESSION_FILE = Path(
    "results/regression_report.json"
)


def extract_score(value):
    if isinstance(value, dict):
        return value.get("score", 0)
    return value


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