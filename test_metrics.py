from app.evaluation.metrics import (
    exact_match,
    accuracy,
    precision,
    recall,
    f1_score
)


# --------------------------------
# Exact Match
# --------------------------------

print("Exact Match:")
print(
    exact_match(
        "Paris",
        "Paris"
    )
)


# --------------------------------
# Classification example
# --------------------------------

predictions = [
    1,
    1,
    0,
    1,
    0
]

actuals = [
    1,
    0,
    0,
    1,
    0
]


print("\nAccuracy:")
print(
    accuracy(
        predictions,
        actuals
    )
)


print("\nPrecision:")
print(
    precision(
        predictions,
        actuals
    )
)


print("\nRecall:")
print(
    recall(
        predictions,
        actuals
    )
)


print("\nF1 Score:")
print(
    f1_score(
        predictions,
        actuals
    )
)