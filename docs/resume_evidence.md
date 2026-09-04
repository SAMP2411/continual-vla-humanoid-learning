# Resume evidence

Only compact-benchmark claims are supported.

| Claim | Evidence |
|---|---|
| Implemented a deterministic three-stage vision-and-language continual-learning benchmark | `lifelong_vla/data.py`, `lifelong_vla/train.py`, `configs/benchmark.yaml` |
| Compared sequential low-rank adaptation with bounded replay across seeds 7, 21, and 42 | `scripts/run_benchmark.py`, `results/summary.json` |
| Replay lowered mean forgetting from 0.1981 to 0.1296 while final average accuracy fell from 0.2074 to 0.1722 | `results/summary.csv`, `results/summary.json` |
| Used 30 held-out examples per action and generated raw evaluation records | `configs/benchmark.yaml`, ignored per-run `evaluation_records.jsonl` files listed in `results/summary.json` |
| Added config hashes, completion markers, lint, and unit tests | `lifelong_vla/runs.py`, `.github/workflows/ci.yml`, `tests/` |

Do not describe this work as a pretrained VLA, humanoid robotics, robot simulation, or
speech-recognition project.
