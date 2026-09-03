import json

from app.evaluation.regression import (
    compare_versions
)


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def main():

    baseline = load_json(
        "results/baseline_results.json"
    )

    new = load_json(
        "results/new_results.json"
    )


    result = compare_versions(
        baseline,
        new,
        tolerance=0.2
    )


    print("\n")
    print("=" * 70)
    print("REGRESSION TEST")
    print("=" * 70)

    print(
        f"Baseline Version: "
        f"{result['baseline_version']}"
    )

    print(
        f"New Version: "
        f"{result['new_version']}"
    )

    print(
        f"Tolerance: "
        f"{result['tolerance']}"
    )


    print("\nMetric Comparison")
    print("-" * 70)


    for metric, values in (
        result["metrics"].items()
    ):

        baseline_score = (
            values["baseline"]
        )

        new_score = (
            values["new"]
        )

        difference = (
            values["difference"]
        )


        status = (
            "REGRESSION ❌"
            if values["regression"]
            else "OK ✅"
        )


        print(
            f"{metric:<15} "
            f"{baseline_score:.2f} → "
            f"{new_score:.2f} "
            f"({difference:+.2f}) "
            f"{status}"
        )


    print("\n")
    print(
        f"FINAL STATUS: "
        f"{result['overall_status']}"
    )


if __name__ == "__main__":
    main()