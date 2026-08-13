import unittest

import torch
from torch import nn

from lifelong_vla.data import SyntheticManipulationDataset
from lifelong_vla.model import LoRALinear, TinyVLAPolicy
from lifelong_vla.replay import ReplayBuffer


class ModelTest(unittest.TestCase):
    def test_lora_shape_and_frozen_base(self):
        layer = LoRALinear(nn.Linear(12, 5), rank=3, alpha=6)
        self.assertEqual(layer(torch.randn(4, 12)).shape, (4, 5))
        self.assertFalse(layer.weight.requires_grad)
        self.assertTrue(layer.lora_a.requires_grad)

    def test_end_to_end_shapes(self):
        dataset = SyntheticManipulationDataset(("left",), 2, seed=7)
        image, tokens, _ = dataset[0]
        model = TinyVLAPolicy()
        logits = model(image.unsqueeze(0), tokens.unsqueeze(0))
        self.assertEqual(logits.shape, (1, 5))

    def test_replay_capacity(self):
        dataset = SyntheticManipulationDataset(("left",), 8, seed=7)
        replay = ReplayBuffer(capacity=3, seed=7)
        for item in dataset:
            replay.add(item)
        self.assertEqual(len(replay), 3)


if __name__ == "__main__":
    unittest.main()
