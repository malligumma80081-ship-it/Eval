import statistics
import time

import numpy as np
import ollama


def generate_response(question, timeout_seconds=10):
    start = time.perf_counter()

    try:
        client = ollama.Client(timeout=timeout_seconds)
        response = client.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "user",
                    "content": question,
                }
            ],
            stream=False,
        )
        answer = response["message"]["content"]
    except Exception as exc:
        answer = f"Model unavailable: {exc}"

    end = time.perf_counter()
    return answer, end - start


def calculate_latency_metrics(latencies):
    return {
        "average": float(np.mean(latencies)),
        "minimum": float(np.min(latencies)),
        "maximum": float(np.max(latencies)),
        "p50": float(np.percentile(latencies, 50)),
        "p95": float(np.percentile(latencies, 95)),
        "p99": float(np.percentile(latencies, 99)),
    }


def benchmark_response(question, requests=10):
    latencies = []

    for i in range(requests):
        _, latency = generate_response(question)
        latencies.append(latency)
        print(f"Request {i + 1}: {latency:.2f} seconds")

    print("\nPerformance Summary")
    metrics = calculate_latency_metrics(latencies)
    print(f"Average: {statistics.mean(latencies):.2f} sec")
    print(f"Minimum: {metrics['minimum']:.2f} sec")
    print(f"Maximum: {metrics['maximum']:.2f} sec")
    print(f"P50: {metrics['p50']:.2f} sec")
    print(f"P95: {metrics['p95']:.2f} sec")
    print(f"P99: {metrics['p99']:.2f} sec")

    return metrics


if __name__ == "__main__":
    benchmark_response("What is hypertension?")