"""Versioned, append-only record schema for experiment artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReferenceEvaluationRecord:
    """A compact-backend record; never substitute this for simulator evidence."""

    method: str
    stage: int
    task: str
    accuracy: float
    replay_occupancy: int
    schema_version: int = 1


def append_jsonl(path: str | Path, record: ReferenceEvaluationRecord) -> None:
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(asdict(record), sort_keys=True) + "\n")
