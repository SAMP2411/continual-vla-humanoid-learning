"""Run sequential LoRA adaptation with and without Experience Replay."""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .data import ACTIONS, STAGES, SyntheticManipulationDataset
from .metrics import summarize
from .model import TinyVLAPolicy
from .records import EvaluationRecord, append_jsonl
from .replay import ReplayBuffer
from .runs import create_run_directory, mark_complete

REQUIRED_CONFIG = {
    "seed", "image_size", "train_samples_per_action", "test_samples_per_action",
    "batch_size", "epochs_per_stage", "learning_rate", "replay_capacity",
    "replay_ratio", "lora_rank", "lora_alpha", "output_dir",
}


def validate_config(config: dict) -> None:
    """Reject incomplete or nonsensical reference-experiment configurations."""
    missing = REQUIRED_CONFIG - config.keys()
    if missing:
        raise ValueError(f"missing configuration keys: {sorted(missing)}")
    positive_ints = (
        "image_size",
        "train_samples_per_action",
        "test_samples_per_action",
        "batch_size",
        "epochs_per_stage",
        "lora_rank",
    )
    for key in positive_ints:
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["replay_capacity"] < 0 or not 0 <= config["replay_ratio"] <= 1:
        raise ValueError("replay capacity must be non-negative and replay ratio must be in [0, 1]")


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_run_directory(output_dir: str | Path) -> Path:
    """Create a unique, non-overwriting result directory for one invocation."""
    # Compatibility helper for callers that only need an empty unique directory.
    return create_run_directory(output_dir, {}, kind="reference")


def evaluate(
    model: nn.Module,
    datasets: list[SyntheticManipulationDataset],
    method: str,
    train_seed: int,
    stage: int,
) -> tuple[list[float], list[EvaluationRecord]]:
    model.eval()
    scores = []
    records = []
    with torch.no_grad():
        for task_index, dataset in enumerate(datasets, start=1):
            correct = total = 0
            for images, tokens, labels in DataLoader(dataset, batch_size=64):
                started = time.perf_counter()
                predictions = model(images, tokens).argmax(dim=1)
                latency_ms = (time.perf_counter() - started) * 1000 / labels.numel()
                correct += int((predictions == labels).sum())
                total += labels.numel()
                records.extend(
                    EvaluationRecord(
                        method=method,
                        train_seed=train_seed,
                        stage=stage,
                        task=f"stage_{task_index}",
                        success=bool(prediction == label),
                        latency_ms=latency_ms,
                    )
                    for prediction, label in zip(predictions.tolist(), labels.tolist())
                )
            scores.append(correct / max(total, 1))
    return scores, records


def train_stage(
    model: TinyVLAPolicy,
    dataset: SyntheticManipulationDataset,
    replay: ReplayBuffer,
    config: dict,
    use_replay: bool,
) -> None:
    model.train()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=config["learning_rate"]
    )
    loss_function = nn.CrossEntropyLoss()
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    for _ in range(config["epochs_per_stage"]):
        for images, tokens, labels in loader:
            if use_replay and len(replay):
                replay_count = max(1, int(labels.numel() * config["replay_ratio"]))
                old = replay.sample(replay_count)
                images = torch.cat((images, torch.stack([item[0] for item in old])))
                tokens = torch.cat((tokens, torch.stack([item[1] for item in old])))
                labels = torch.cat((labels, torch.stack([item[2] for item in old])))
            optimizer.zero_grad()
            loss = loss_function(model(images, tokens), labels)
            loss.backward()
            optimizer.step()
    for item in dataset:
        replay.add(item)


def run_method(config: dict, use_replay: bool, method: str) -> dict:
    set_deterministic_seed(config["seed"])
    model = TinyVLAPolicy(
        num_actions=len(ACTIONS), rank=config["lora_rank"], alpha=config["lora_alpha"]
    )
    # Simulate a shared pretrained representation by keeping the initialized
    # encoders fixed and adapting only the low-rank policy parameters.
    model.freeze_backbone()
    replay = ReplayBuffer(config["replay_capacity"], config["seed"])
    test_sets = [
        SyntheticManipulationDataset(
            actions,
            config["test_samples_per_action"],
            config["image_size"],
            config["seed"] + 1000 + stage,
        )
        for stage, actions in enumerate(STAGES)
    ]
    matrix = []
    evaluation_records = []
    for stage, actions in enumerate(STAGES):
        train_set = SyntheticManipulationDataset(
            actions,
            config["train_samples_per_action"],
            config["image_size"],
            config["seed"] + stage,
        )
        train_stage(model, train_set, replay, config, use_replay)
        scores, records = evaluate(
            model, test_sets[: stage + 1], method, config["seed"], stage + 1
        )
        matrix.append(scores)
        evaluation_records.extend(records)
    return {
        "accuracy_matrix": matrix,
        "summary": summarize(matrix),
        "parameters": model.parameter_counts(),
        "replay_occupancy": len(replay),
        "evaluation_records": evaluation_records,
    }


def plot_results(results: dict, path: Path) -> None:
    labels = list(results)
    accuracy = [results[name]["summary"]["final_average_accuracy"] for name in labels]
    forgetting = [results[name]["summary"]["average_forgetting"] for name in labels]
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    axes[0].bar(labels, accuracy, color=("#557a95", "#2e8b57"))
    axes[0].set_title("Final average accuracy")
    axes[0].set_ylim(0, 1)
    axes[1].bar(labels, forgetting, color=("#557a95", "#2e8b57"))
    axes[1].set_title("Average forgetting")
    axes[1].set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quick.yaml")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    validate_config(config)
    output = create_run_directory(config["output_dir"], config)
    results = {
        "sequential_peft": run_method(config, use_replay=False, method="sequential_peft"),
        "sequential_peft_replay": run_method(
            config, use_replay=True, method="sequential_peft_replay"
        ),
    }
    serializable_results = {
        method: {key: value for key, value in result.items() if key != "evaluation_records"}
        for method, result in results.items()
    }
    with open(output / "metrics.json", "w", encoding="utf-8") as stream:
        payload = {
            "config": config,
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "device": "cpu",
            },
            "results": serializable_results,
        }
        json.dump(payload, stream, indent=2)
    for value in results.values():
        for record in value["evaluation_records"]:
            append_jsonl(output / "evaluation_records.jsonl", record)
    plot_results(serializable_results, output / "comparison.png")
    mark_complete(output)
    print(json.dumps({"run_dir": str(output), "results": serializable_results}, indent=2))


if __name__ == "__main__":
    main()
