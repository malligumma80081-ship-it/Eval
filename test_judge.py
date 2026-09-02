from app.evaluation.llm_judge import (
    LLMJudge
)


def main():

    judge = LLMJudge(
        model_name="llama3.2"
    )

    question = (
        "Who created Python?"
    )

    context = (
        "Python was created by "
        "Guido van Rossum."
    )

    expected_answer = (
        "Python was created by "
        "Guido van Rossum."
    )

    generated_answer = (
        "Guido van Rossum created Python."
    )

    result = judge.evaluate(
        question=question,
        context=context,
        answer=generated_answer,
        expected_answer=expected_answer
    )

    print("\nJUDGE RESULT")
    print("=" * 50)

    print(result)


if __name__ == "__main__":
    main()