"""Metrics for stage-by-task continual-learning evaluation."""

from __future__ import annotations


def task_forgetting(accuracy_matrix: list[list[float]]) -> list[float]:
    """Return best historical minus final accuracy for each observed task."""
    if not accuracy_matrix:
        return []
    task_count = len(accuracy_matrix[-1])
    forgetting = []
    for task in range(task_count):
        history = [row[task] for row in accuracy_matrix if task < len(row)]
        forgetting.append(max(history) - history[-1])
    return forgetting


def summarize(accuracy_matrix: list[list[float]]) -> dict[str, float | list[float]]:
    if not accuracy_matrix or not accuracy_matrix[-1]:
        return {"final_average_accuracy": 0.0, "average_forgetting": 0.0, "forgetting": []}
    forgetting = task_forgetting(accuracy_matrix)
    return {
        "final_average_accuracy": sum(accuracy_matrix[-1]) / len(accuracy_matrix[-1]),
        "average_forgetting": sum(forgetting) / len(forgetting),
        "forgetting": forgetting,
    }
