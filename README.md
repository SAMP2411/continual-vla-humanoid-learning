# Compact Continual Vision-Language Policy Benchmark

**Status: locally complete benchmark; remote CI and release pending.**

This repository measures catastrophic forgetting in a small, CPU-compatible policy
that consumes a generated RGB tabletop observation and a natural-language command,
then predicts one of five discrete actions. It compares sequential low-rank adaptation
with and without bounded Experience Replay.

> Scope: this is a compact synthetic benchmark. It is **not** a pretrained VLA, robot
> simulator, humanoid controller, speech system, or physical-robot project. See
> [scope change](docs/scope_change.md).

## Reproduce

```powershell
python -m pip install -e ".[dev]"
python scripts/system_report.py --output docs/system_report.json
python -m ruff check lifelong_vla scripts tests
python -m unittest discover -s tests -v
python scripts/run_benchmark.py --config configs/benchmark.yaml
python scripts/make_report.py --results results
```

The benchmark uses fixed train seeds `7`, `21`, and `42`, three sequential task stages,
and 30 held-out examples per action. Each run receives a unique directory with an
immutable config hash, raw JSONL evaluation records, metrics, plot, and `COMPLETE`
marker. The report script rejects missing seed coverage and regenerates the compact
summary files from raw records.

## Verified local result

Across the three fixed seeds, sequential PEFT has mean final accuracy 0.2074 and mean
forgetting 0.1981. PEFT with replay has mean final accuracy 0.1722 and mean forgetting
0.1296. Thus replay reduced mean forgetting in this run set but did not improve final
average accuracy. Full values, cell-level 95% normal-approximation intervals, and
latency are in [results/summary.json](results/summary.json).

## Components

- `lifelong_vla/data.py`: deterministic synthetic images and language commands.
- `lifelong_vla/model.py`: frozen compact multimodal backbone and trainable LoRA head.
- `lifelong_vla/replay.py`: bounded reservoir-sampling replay.
- `lifelong_vla/train.py`: sequential training, held-out evaluation, and raw records.
- `scripts/run_benchmark.py`: frozen three-seed benchmark runner.
- `scripts/make_report.py`: raw-record summary/plot generator.

## Limitations

This benchmark’s data, action space, and model are intentionally small. Its results do
not transfer automatically to pretrained VLAs, simulators, or robotic systems.
