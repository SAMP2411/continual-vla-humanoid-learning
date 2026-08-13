"""Fail-closed asset downloader and checksum verifier for GPU environment use."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path
from typing import Any

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(manifest: dict[str, Any]) -> list[dict[str, str]]:
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("manifest must contain a non-empty files list")
    validated = []
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("each asset must be a mapping")
        required = ("name", "url", "path", "sha256", "revision", "license")
        missing = [key for key in required if not isinstance(item.get(key), str) or not item[key]]
        if missing:
            raise ValueError(f"asset manifest entry is incomplete: {missing}")
        if len(item["sha256"]) != 64:
            raise ValueError(f"asset {item['name']} has an invalid SHA-256")
        validated.append({key: item[key] for key in required})
    return validated


def verify(manifest: dict[str, Any], root: Path) -> None:
    for item in validate_manifest(manifest):
        path = root / item["path"]
        if not path.is_file():
            raise FileNotFoundError(f"missing asset: {path}")
        if sha256(path) != item["sha256"]:
            raise ValueError(f"checksum mismatch: {item['name']}")


def download(manifest: dict[str, Any], root: Path) -> None:
    for item in validate_manifest(manifest):
        destination = root / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        with urllib.request.urlopen(item["url"]) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        if sha256(temporary) != item["sha256"]:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"download checksum mismatch: {item['name']}")
        temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("configs/assets.yaml"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    if args.download:
        download(manifest, args.root)
    verify(manifest, args.root)
    print("asset verification passed")


if __name__ == "__main__":
    main()
