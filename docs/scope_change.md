# Scope change: CPU-compatible compact benchmark

On 2026-09-04, the user authorized replacing the former GPU-only pretrained
humanoid-VLA objective with a locally reproducible compact continual-learning
benchmark. The local machine has a 4 GB GTX 1050 Ti and a CPU-only PyTorch runtime;
the former Isaac Lab/SmolVLA study cannot be completed or validated here.

The complete deliverable is now a deterministic synthetic RGB + text action-policy
benchmark with low-rank adaptation, bounded replay, three sequential task stages,
three fixed seeds, held-out evaluation, raw records, and regenerated summaries. It is
explicitly not a pretrained VLA, robot simulation, speech system, or humanoid project.
