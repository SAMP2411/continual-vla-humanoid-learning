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
