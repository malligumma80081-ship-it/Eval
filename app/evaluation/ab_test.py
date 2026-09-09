import json
from pathlib import Path

from app.evaluation.generation_metrics import answer_relevance_score, faithfulness_score
from app.evaluation.prompt_runner import generate_answer
from app.evaluation.prompts import PROMPT_A, PROMPT_B


def run_prompt(prompt_template, question, context):
    prompt = prompt_template.format(
        question=question,
        context=context,
    )
    return generate_answer(prompt)


def score_prompt_response(question, context, answer, expected_answer=""):
    faithfulness = faithfulness_score(answer, context)
    relevance = answer_relevance_score(question, answer)

    correctness = 0.0
    if expected_answer:
        correctness = 1.0 if answer.strip().lower() == expected_answer.strip().lower() else 0.0

    overall = (faithfulness + relevance + correctness) / (3 if expected_answer else 2)

    return {
        "faithfulness": round(faithfulness, 4),
        "relevance": round(relevance, 4),
        "correctness": round(correctness, 4),
        "overall": round(overall, 4),
    }


def compare_prompts(question, context, expected_answer=""):
    answer_a = run_prompt(PROMPT_A, question, context)
    answer_b = run_prompt(PROMPT_B, question, context)

    score_a = score_prompt_response(question, context, answer_a, expected_answer)
    score_b = score_prompt_response(question, context, answer_b, expected_answer)

    winner = "Prompt A" if score_a["overall"] >= score_b["overall"] else "Prompt B"

    return {
        "prompt_a": {
            "answer": answer_a,
            "score": score_a,
        },
        "prompt_b": {
            "answer": answer_b,
            "score": score_b,
        },
        "winner": winner,
        "winner_score": max(score_a["overall"], score_b["overall"]),
    }


def run_ab_test(question, context, expected_answer=""):
    return compare_prompts(question, context, expected_answer)


def _load_default_dataset():
    dataset_path = Path("data/golden_dataset.json")
    if not dataset_path.exists():
        return []
    with dataset_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    dataset = _load_default_dataset()
    if not dataset:
        print("No dataset found in data/golden_dataset.json")
        return

    print("Prompt A vs Prompt B evaluation")
    print("=" * 48)

    for item in dataset:
        question = item["question"]
        context = "\n".join(
            [
                "Python is a high-level programming language.",
                "Python was created by Guido van Rossum.",
                "Python was first released in 1991.",
                "Python is widely used in data science, machine learning, web development, automation, and artificial intelligence.",
                "Machine learning is a branch of artificial intelligence.",
                "Machine learning allows computers to learn patterns from data.",
                "Artificial intelligence is the field of building systems that can perform tasks that normally require human intelligence.",
            ]
        )
        expected = item.get("expected_answer", "")
        result = compare_prompts(question, context, expected)

        print(f"Question: {question}")
        print(f"Prompt A: {result['prompt_a']['answer']}")
        print(f"Prompt B: {result['prompt_b']['answer']}")
        print(f"Winner: {result['winner']} ({result['winner_score']:.4f})")
        print("-" * 48)


if __name__ == "__main__":
    main()