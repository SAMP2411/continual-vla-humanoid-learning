"""Regenerate compact-benchmark summaries from completed raw run records."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REQUIRED_SEEDS = {7, 21, 42}


def completed_benchmark_runs(root: Path) -> list[Path]:
    runs = []
    for directory in root.glob("reference-*"):
        config_path = directory / "config.yaml"
        if not (directory / "COMPLETE").is_file() or not config_path.is_file():
            continue
        config = __import__("yaml").safe_load(config_path.read_text(encoding="utf-8"))
        if config.get("seed") in REQUIRED_SEEDS and config.get("test_samples_per_action", 0) >= 30:
            runs.append(directory)
    return sorted(runs)


def ci95(successes: int, count: int) -> float:
    if count == 0:
        return 0.0
    probability = successes / count
    return 1.96 * math.sqrt(probability * (1 - probability) / count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    runs = completed_benchmark_runs(args.results)
    by_seed = {}
    records = []
    for directory in runs:
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        seed = metrics["config"]["seed"]
        by_seed[seed] = metrics
        records.extend(
            json.loads(line)
            for line in (directory / "evaluation_records.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    if set(by_seed) != REQUIRED_SEEDS:
        raise ValueError(
            f"expected exactly benchmark seeds {sorted(REQUIRED_SEEDS)}, found {sorted(by_seed)}"
        )

    grouped = defaultdict(list)
    for record in records:
        grouped[(record["method"], record["stage"], record["task"])].append(record)
    cells = []
    for (method, stage, task), values in sorted(grouped.items()):
        successes = sum(value["success"] for value in values)
        cells.append(
            {
                "method": method,
                "stage": stage,
                "task": task,
                "episodes": len(values),
                "success": successes / len(values),
                "ci95": ci95(successes, len(values)),
                "median_latency_ms": sorted(value["latency_ms"] for value in values)[
                    len(values) // 2
                ],
            }
        )
    methods = sorted(by_seed[7]["results"])
    method_summary = []
    for method in methods:
        finals = [
            by_seed[seed]["results"][method]["summary"]["final_average_accuracy"]
            for seed in REQUIRED_SEEDS
        ]
        forgetting = [
            by_seed[seed]["results"][method]["summary"]["average_forgetting"]
            for seed in REQUIRED_SEEDS
        ]
        parameters = by_seed[7]["results"][method]["parameters"]
        method_summary.append(
            {
                "method": method,
                "seeds": sorted(REQUIRED_SEEDS),
                "final_average_accuracy_mean": sum(finals) / len(finals),
                "average_forgetting_mean": sum(forgetting) / len(forgetting),
                "total_parameters": parameters["total"],
                "trainable_parameters": parameters["trainable"],
                "trainable_percent": 100 * parameters["trainable"] / parameters["total"],
                "replay_occupancy": by_seed[7]["results"][method]["replay_occupancy"],
            }
        )
    output = {"runs": [str(run) for run in runs], "cells": cells, "methods": method_summary}
    (args.results / "summary.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    with (args.results / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=method_summary[0].keys())
        writer.writeheader()
        writer.writerows(method_summary)
    figure, axis = plt.subplots(figsize=(6, 3.4))
    axis.bar(
        [row["method"] for row in method_summary],
        [row["average_forgetting_mean"] for row in method_summary],
    )
    axis.set_ylabel("Average forgetting")
    axis.set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(args.results / "forgetting_comparison.png", dpi=160)
    plt.close(figure)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
