"""Bounded replay memory using reservoir sampling."""

from __future__ import annotations

import random

import torch

BatchItem = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0) -> None:
        if capacity < 0:
            raise ValueError("capacity cannot be negative")
        self.capacity = capacity
        self.items: list[BatchItem] = []
        self.seen = 0
        self.rng = random.Random(seed)

    def add(self, item: BatchItem) -> None:
        if self.capacity == 0:
            return
        detached = tuple(value.detach().cpu().clone() for value in item)
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(detached)  # type: ignore[arg-type]
            return
        candidate = self.rng.randrange(self.seen)
        if candidate < self.capacity:
            self.items[candidate] = detached  # type: ignore[assignment]

    def sample(self, count: int) -> list[BatchItem]:
        return self.rng.sample(self.items, min(count, len(self.items)))

    def __len__(self) -> int:
        return len(self.items)
