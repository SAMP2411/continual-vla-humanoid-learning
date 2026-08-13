# Local development setup

## Verified on 2026-08-14

The Windows development host was checked with `nvidia-smi`, `python -m unittest
discover -s tests -v`, and `python -m lifelong_vla.train --config configs/debug.yaml`.
The NVIDIA driver is 581.57 and the GeForce GTX 1050 Ti has 4 GB VRAM. The installed
PyTorch wheel is CPU-only, so it reports `cuda_available: false`; this is expected for
the reference backend and does not establish a GPU runtime.

WSL enumeration currently fails with `Wsl/EnumerateDistros/Service/E_ACCESSDENIED`.
No WSL distribution or Ubuntu version is claimed as installed. This must be resolved
before GPU-environment bootstrap.

## Local commands

```powershell
python -m pip install -r requirements.txt
python scripts/system_report.py --output docs/system_report.json
python -m unittest discover -s tests -v
python -m lifelong_vla.train --config configs/debug.yaml
```

The last command produces ignored reference artifacts under `results/`. It is not the
final simulator benchmark.
