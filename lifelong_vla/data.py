"""Synthetic tabletop observations paired with language commands and actions."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

ACTIONS = ("left", "right", "up", "down", "grasp")
STAGES = (("left", "right"), ("up", "down"), ("grasp",))
COLORS = {
    "red": (0.9, 0.1, 0.1),
    "green": (0.1, 0.8, 0.2),
    "blue": (0.1, 0.3, 0.9),
}
TEMPLATES = (
    "move the {color} object {action}",
    "please move {color} {action}",
    "{action} the {color} block",
)


def tokenize(command: str, max_tokens: int = 8, buckets: int = 256) -> torch.Tensor:
    """Hash words into stable token buckets without a learned vocabulary file."""
    ids = []
    for word in command.lower().split()[:max_tokens]:
        digest = hashlib.blake2b(word.encode("utf-8"), digest_size=2).digest()
        ids.append(1 + int.from_bytes(digest, "little") % (buckets - 1))
    ids.extend([0] * (max_tokens - len(ids)))
    return torch.tensor(ids, dtype=torch.long)


@dataclass(frozen=True)
class Sample:
    image: torch.Tensor
    tokens: torch.Tensor
    label: int
    command: str


class SyntheticManipulationDataset(Dataset[Sample]):
    """Generate simple colored-block scenes deterministically from a seed."""

    def __init__(
        self,
        actions: tuple[str, ...],
        samples_per_action: int,
        image_size: int = 32,
        seed: int = 0,
    ) -> None:
        self.samples: list[Sample] = []
        rng = random.Random(seed)
        for action in actions:
            for _ in range(samples_per_action):
                color_name = rng.choice(tuple(COLORS))
                command = rng.choice(TEMPLATES).format(color=color_name, action=action)
                image = torch.zeros(3, image_size, image_size, dtype=torch.float32)
                size = rng.randint(5, 9)
                x = rng.randint(2, image_size - size - 2)
                y = rng.randint(2, image_size - size - 2)
                image[:, y : y + size, x : x + size] = torch.tensor(
                    COLORS[color_name], dtype=torch.float32
                ).view(3, 1, 1)
                image += 0.025 * torch.randn(
                    image.shape, generator=torch.Generator().manual_seed(rng.randrange(2**31))
                )
                image.clamp_(0.0, 1.0)
                self.samples.append(
                    Sample(image, tokenize(command), ACTIONS.index(action), command)
                )
        rng.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.samples[index]
        return item.image, item.tokens, torch.tensor(item.label, dtype=torch.long)
