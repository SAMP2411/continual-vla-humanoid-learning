# Decisions

## 2026-08-14 — retain a compact reference backend

The repository starts with a deterministic CPU reference to test sequential LoRA,
replay, run isolation, and continual-learning metrics. It is not represented as
pretrained or humanoid. A final stack has not been selected because the required
hardware report and compatibility review are the next Phase-A/stack-selection work.

## 2026-09-04 — authorize CPU-bounded completion scope

The user replaced the unachievable local pretrained humanoid-VLA objective with a
complete compact benchmark. This decision avoids presenting a GPU-only plan as a
finished robot project. The benchmark retains the research mechanics that can be
measured locally: deterministic vision/text inputs, low-rank adaptation, sequential
learning, bounded replay, and forgetting.
