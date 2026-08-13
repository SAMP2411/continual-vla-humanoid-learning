# Stack selection

**Status: selected for integration; not yet installed or verified.** This document is a
design record, not evidence that the final backend works.

## Hardware constraint

[`system_report.json`](system_report.json) records Windows 10, Python 3.10.11,
PyTorch 2.13.0+cpu, 16 GB RAM, and no CUDA device. The current workstation can run
the compact reference tests but is not an appropriate machine for the headline
fine-tuning benchmark. The headline runs require a Linux CUDA worker with sufficient
VRAM and persistent storage; no GPU requirement is claimed as satisfied yet.

## Selected integration stack

| Concern | Selection | Rationale and status |
|---|---|---|
| Pretrained VLA | `lerobot/smolvla_base` at Hugging Face revision `d560bdd24ed1230588eac43aeec23fe5beb73089` | 450M-parameter SmolVLA base checkpoint, documented by its publisher as pretrained. Its 907 MB weights and revision are public. Download/load verification is pending. |
| Policy framework | Hugging Face LeRobot with the `smolvla` extra | Provides `SmolVLAPolicy.from_pretrained` and policy preprocessing. Exact pinned package version is pending compatibility installation. |
| Simulator/embodiment | MuJoCo + MuJoCo Menagerie Unitree G1 with hands | Menagerie ships a BSD-3-Clause Unitree G1 humanoid model. The project will limit control to documented torso/arm/hand actuators and render named cameras. Asset download and actuator mapping are pending. |
| Adaptation | PEFT LoRA on named SmolVLA action-expert linear projections | Exact module paths must be inspected from the pinned LeRobot revision before adapters are attached; no current compact-backend LoRA is evidence for this gate. |
| Speech path | Offline audio-file ASR (faster-whisper) | Audio fixtures and transcripts will be stored locally; selection remains unverified. |
| Tracking | Local JSONL episode records, CSV/JSON summaries, TensorBoard | Required to retain raw results, configuration hashes, timing, and resource measurements. Not implemented. |

## Interface contract to validate

SmolVLA consumes camera image tensors, robot state, and a natural-language task, and
produces an action chunk. The integration adapter will provide a fixed observation
mapping (`observation.images.front`, `observation.state`, task string) and map its
continuous action vector only to declared G1 upper-body/gripper controls. Shapes,
normalization statistics, camera conventions, and the final action mapping must be
recorded from the loaded checkpoint and tested before training.

## Licences and capacity

The selected checkpoint and its model card must be archived with its licence at the
pinned revision before use. The Menagerie repository identifies the Unitree G1 model
as BSD-3-Clause. The model weights alone are 907 MB; demonstrations, checkpoints,
video, and three-seed outputs require a dedicated artifact volume. No storage or VRAM
estimate is yet validated on the target worker.

## Alternatives rejected for this machine

Isaac Lab / GR1 Arena is not selected as the primary local path because the inspected
machine has no CUDA device. OpenVLA-class multi-billion-parameter checkpoints are not
selected because their adaptation footprint is unsuitable for the verified 16 GB
CPU-only environment. These are compatibility decisions, not performance comparisons.

## Sources

- [SmolVLA model card](https://huggingface.co/lerobot/smolvla_base/tree/d560bdd24ed1230588eac43aeec23fe5beb73089)
- [LeRobot SmolVLA implementation](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/modeling_smolvla.py)
- [MuJoCo Menagerie models and licences](https://github.com/google-deepmind/mujoco_menagerie)
