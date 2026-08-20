def exact_match(predicted, expected):

    predicted = predicted.strip().lower()
    expected = expected.strip().lower()

    return predicted == expected


def accuracy(predictions, actuals):

    if len(predictions) != len(actuals):
        raise ValueError(
            "Predictions and actuals must have the same length."
        )

    if len(actuals) == 0:
        return 0.0

    correct = 0

    for predicted, actual in zip(
        predictions,
        actuals
    ):

        if predicted == actual:
            correct += 1

    return correct / len(actuals)


def precision(predictions, actuals):

    true_positive = 0
    false_positive = 0

    for predicted, actual in zip(
        predictions,
        actuals
    ):

        if predicted == 1 and actual == 1:
            true_positive += 1

        elif predicted == 1 and actual == 0:
            false_positive += 1

    denominator = true_positive + false_positive

    if denominator == 0:
        return 0.0

    return true_positive / denominator


def recall(predictions, actuals):

    true_positive = 0
    false_negative = 0

    for predicted, actual in zip(
        predictions,
        actuals
    ):

        if predicted == 1 and actual == 1:
            true_positive += 1

        elif predicted == 0 and actual == 1:
            false_negative += 1

    denominator = true_positive + false_negative

    if denominator == 0:
        return 0.0

    return true_positive / denominator


def f1_score(predictions, actuals):

    p = precision(
        predictions,
        actuals
    )

    r = recall(
        predictions,
        actuals
    )

    if p + r == 0:
        return 0.0

    return 2 * (p * r) / (p + r)