"""Run sequential LoRA adaptation with and without Experience Replay."""

from __future__ import annotations

import argparse
import json
import platform
import random
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
from .records import ReferenceEvaluationRecord, append_jsonl
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


def evaluate(model: nn.Module, datasets: list[SyntheticManipulationDataset]) -> list[float]:
    model.eval()
    scores = []
    with torch.no_grad():
        for dataset in datasets:
            correct = total = 0
            for images, tokens, labels in DataLoader(dataset, batch_size=64):
                predictions = model(images, tokens).argmax(dim=1)
                correct += int((predictions == labels).sum())
                total += labels.numel()
            scores.append(correct / max(total, 1))
    return scores


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


def run_method(config: dict, use_replay: bool) -> dict:
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
    for stage, actions in enumerate(STAGES):
        train_set = SyntheticManipulationDataset(
            actions,
            config["train_samples_per_action"],
            config["image_size"],
            config["seed"] + stage,
        )
        train_stage(model, train_set, replay, config, use_replay)
        matrix.append(evaluate(model, test_sets[: stage + 1]))
    return {
        "accuracy_matrix": matrix,
        "summary": summarize(matrix),
        "parameters": model.parameter_counts(),
        "replay_occupancy": len(replay),
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
        "sequential_peft": run_method(config, use_replay=False),
        "peft_with_replay": run_method(config, use_replay=True),
    }
    with open(output / "metrics.json", "w", encoding="utf-8") as stream:
        payload = {
            "config": config,
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "device": "cpu",
            },
            "results": results,
        }
        json.dump(payload, stream, indent=2)
    for method, value in results.items():
        for stage, scores in enumerate(value["accuracy_matrix"], start=1):
            for task, score in enumerate(scores, start=1):
                append_jsonl(output / "reference_evaluations.jsonl", ReferenceEvaluationRecord(
                    method=method, stage=stage, task=f"stage_{task}", accuracy=score,
                    replay_occupancy=value["replay_occupancy"],
                ))
    plot_results(results, output / "comparison.png")
    mark_complete(output)
    print(json.dumps({"run_dir": str(output), "results": results}, indent=2))


if __name__ == "__main__":
    main()
