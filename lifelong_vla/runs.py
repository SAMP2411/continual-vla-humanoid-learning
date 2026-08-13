"""Immutable run-directory and completion-marker utilities."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def config_fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def create_run_directory(
    output_dir: str | Path, config: dict[str, Any], kind: str = "reference"
) -> Path:
    """Create a unique run and save an immutable resolved configuration snapshot."""
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = Path(output_dir)
    for suffix in range(1000):
        name = f"{kind}-{stamp}" if suffix == 0 else f"{kind}-{stamp}-{suffix}"
        run_dir = root / name
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError("could not allocate a unique run directory")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    (run_dir / "config.sha256").write_text(config_fingerprint(config) + "\n", encoding="utf-8")
    return run_dir


def assert_config_matches(run_dir: str | Path, config: dict[str, Any]) -> None:
    expected = (Path(run_dir) / "config.sha256").read_text(encoding="utf-8").strip()
    if expected != config_fingerprint(config):
        raise ValueError("run configuration mismatch; refusing to mix artifacts")


def mark_complete(run_dir: str | Path) -> None:
    """Atomically signal that all artifacts for a run were successfully written."""
    directory = Path(run_dir)
    temporary = directory / ".complete.tmp"
    temporary.write_text("complete\n", encoding="utf-8")
    temporary.replace(directory / "COMPLETE")
