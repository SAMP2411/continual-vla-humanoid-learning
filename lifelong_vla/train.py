"""Run sequential LoRA adaptation with and without Experience Replay."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .data import ACTIONS, STAGES, SyntheticManipulationDataset
from .metrics import summarize
from .model import TinyVLAPolicy
from .replay import ReplayBuffer


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
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
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
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    results = {
        "sequential_peft": run_method(config, use_replay=False),
        "peft_with_replay": run_method(config, use_replay=True),
    }
    with open(output / "metrics.json", "w", encoding="utf-8") as stream:
        json.dump({"config": config, "results": results}, stream, indent=2)
    plot_results(results, output / "comparison.png")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
