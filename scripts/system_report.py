"""Print and optionally save a portable system report before stack selection."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import shutil
from pathlib import Path

import torch


def installed_ram_bytes() -> int | None:
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("memory_load", ctypes.c_ulong),
                       ("total_physical", ctypes.c_ulonglong), ("available_physical", ctypes.c_ulonglong),
                       ("total_page_file", ctypes.c_ulonglong), ("available_page_file", ctypes.c_ulonglong),
                       ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                       ("available_extended_virtual", ctypes.c_ulonglong)]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.total_physical
    if hasattr(os, "sysconf"):
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return None


def report() -> dict:
    cuda = torch.cuda.is_available()
    gpu = []
    if cuda:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpu.append({"name": properties.name, "vram_bytes": properties.total_memory})
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda,
        "gpu": gpu,
        "ram_bytes": installed_ram_bytes(),
        "disk_free_bytes": shutil.disk_usage(Path.cwd()).free,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    contents = json.dumps(report(), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(contents + "\n", encoding="utf-8")
    print(contents)


if __name__ == "__main__":
    main()
