# from app.evaluation.semantic import SemanticEvaluator


# evaluator = SemanticEvaluator()


# expected = """
# Paris is the capital of France.
# """


# actual = """
# France's capital city is Paris.
# """


# result = evaluator.evaluate(
#     expected=expected,
#     actual=actual
# )


# print("\n==============================")
# print("SEMANTIC EVALUATION")
# print("==============================")


# print("\nExpected:")
# print(expected)


# print("\nActual:")
# print(actual)


# print("\nSimilarity:")
# print(result["score"])


# print("\nThreshold:")
# print(result["threshold"])


# print("\nPassed:")
# print(result["passed"])


from app.evaluation.semantic import SemanticEvaluator


evaluator = SemanticEvaluator()


test_cases = [

    {
        "expected": "Paris is the capital of France.",
        "actual": "France's capital city is Paris."
    },

    {
        "expected": "Python is a programming language.",
        "actual": "Python is used for software development."
    },

    {
        "expected": "Python is a programming language.",
        "actual": "The weather is very hot today."
    }

]


for index, test_case in enumerate(
    test_cases,
    start=1
):

    result = evaluator.evaluate(
        expected=test_case["expected"],
        actual=test_case["actual"]
    )


    print("\n==============================")

    print("TEST:", index)

    print("==============================")


    print(
        "Expected:",
        test_case["expected"]
    )

    print(
        "Actual:",
        test_case["actual"]
    )

    print(
        "Similarity:",
        round(
            result["score"],
            4
        )
    )

    print(
        "Passed:",
        result["passed"]
    )