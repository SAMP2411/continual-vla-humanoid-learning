"""Small multimodal policy with a parameter-efficient LoRA action head."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as functional


class LoRALinear(nn.Module):
    """Frozen linear projection plus trainable low-rank update."""

    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 8.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        self.bias = (
            nn.Parameter(base.bias.detach().clone(), requires_grad=False)
            if base.bias is not None
            else None
        )
        self.lora_a = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank))
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = functional.linear(inputs, self.weight, self.bias)
        update = functional.linear(functional.linear(inputs, self.lora_a), self.lora_b)
        return base + self.scale * update


class TinyVLAPolicy(nn.Module):
    """Encode an image and command, fuse both, and predict a discrete action."""

    def __init__(self, num_actions: int = 5, rank: int = 4, alpha: float = 8.0) -> None:
        super().__init__()
        self.vision = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.embedding = nn.Embedding(256, 32, padding_idx=0)
        self.fusion = nn.Sequential(nn.Linear(64, 48), nn.ReLU())
        self.action_head = LoRALinear(nn.Linear(48, num_actions), rank, alpha)

    def freeze_backbone(self) -> None:
        for module in (self.vision, self.embedding, self.fusion):
            for parameter in module.parameters():
                parameter.requires_grad = False

    def forward(self, image: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        visual = self.vision(image)
        embedded = self.embedding(tokens)
        mask = tokens.ne(0).unsqueeze(-1)
        text = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        fused = self.fusion(torch.cat((visual, text), dim=1))
        return self.action_head(fused)

    def parameter_counts(self) -> dict[str, int]:
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
