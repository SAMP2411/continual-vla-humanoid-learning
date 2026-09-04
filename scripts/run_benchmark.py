"""Run the frozen CPU compact benchmark for all required seeds."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seeds = config.pop("seeds", [7, 21, 42])
    if seeds != [7, 21, 42]:
        raise ValueError("compact benchmark requires frozen seeds [7, 21, 42]")
    for seed in seeds:
        run_config = dict(config, seed=seed)
        temporary = config_path.parent / f".{config_path.stem}-seed-{seed}.yaml"
        temporary.write_text(yaml.safe_dump(run_config, sort_keys=True), encoding="utf-8")
        try:
            subprocess.run(
                [sys.executable, "-m", "lifelong_vla.train", "--config", str(temporary)],
                check=True,
            )
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
