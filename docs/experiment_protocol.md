# Headline experiment protocol (frozen before any headline run)

**Protocol version:** 0.1-draft. **Status:** not executable until the selected backend,
dataset, and target GPU worker are validated. Do not report benchmark results under
this protocol until its status is changed with a dated decision record.

## Question and task order

Does LoRA adaptation of `lerobot/smolvla_base` with bounded experience replay reduce
catastrophic forgetting versus sequential LoRA without replay on a Unitree G1
upper-body MuJoCo manipulation benchmark?

The frozen order is `pick_place`, `drawer`, then `stack`. Each task uses a live rendered
camera, a natural-language instruction, randomized object pose, and one randomized
visual factor. A scripted oracle must establish task achievability before collection.

## Controlled methods

- `zero_shot`: same pinned base checkpoint; no adaptation.
- `sequential_peft`: same LoRA configuration and current-stage demonstrations only.
- `sequential_peft_replay`: identical to sequential PEFT plus reservoir replay of
  training episodes only.

All adapted methods begin from the same checkpoint and initialization for each seed.
Replay capacity, episode unit, ratio, task provenance, and sampling seed are saved
with each stage. Evaluation episodes are never placed in replay.

## Seeds, evaluation, and records

Run seeds `17`, `29`, and `43` for both adapted methods. Evaluate each task before
training and after every stage using the same 30 deterministic evaluation episodes per
task and seed across methods. Save episode-level success, termination reason, latency,
camera/video reference, configuration hash, checkpoint identity, and resource samples.
Failed runs remain in the manifest.

## Predeclared summaries

From raw records, calculate success and 95% confidence intervals, final average
success, per-task/average forgetting, forward transfer relative to zero-shot, total and
trainable parameters, peak RAM/VRAM, replay storage, training time, and median/p95
inference latency. Regenerate `results/summary.csv`, `results/summary.json`, and plots
only from the raw manifest.

No claim that replay improves retention is permitted unless its per-seed forgetting is
lower under this frozen protocol.
