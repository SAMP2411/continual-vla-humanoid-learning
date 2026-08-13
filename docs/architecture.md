# Architecture

The current implementation is a CPU-fast **reference backend**, not a pretrained VLA
and not a robot simulator. A deterministic synthetic scene generator produces a 32x32
RGB tabletop image and a templated text instruction. A small CNN and token embedding
are fused, and a LoRA action head predicts one of five discrete actions.

Only the low-rank action-head matrices are trained after `freeze_backbone()`; the
CNN, text embedding, fusion MLP, and base action projection remain frozen. Sequential
training compares current-stage-only updates with bounded reservoir-sampled replay.

The planned final backend is deliberately separate: live simulator camera and language
inputs will be converted for a documented, genuinely pretrained VLA; LoRA will target
named modules in that model; actions will drive a humanoid upper-body embodiment.
No final-backend integration is present yet.

Run management is shared independently of a backend: `lifelong_vla.runs` creates a
unique directory, snapshots and hashes the resolved configuration, and writes an atomic
`COMPLETE` marker only after metrics, raw reference records, and plot output exist.
`lifelong_vla.records` defines append-only JSONL records. These records are explicitly
labelled as reference-backend evidence and cannot be used as simulator benchmark data.
