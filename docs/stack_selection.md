# Stack selection

**Status: provisional until a GPU inference smoke test passes.** This is the fixed
project integration target, not a statement that any component is installed locally.

| Concern | Selection | Status |
|---|---|---|
| Pretrained VLA | `lerobot/smolvla-arena-gr1-microwave` | Public pretrained GR1 Arena policy; immutable revision and hashes must be captured by `scripts/download_assets.py`. |
| Policy/data | LeRobot + PyTorch | Pending GPU/Linux compatibility installation. |
| PEFT | Hugging Face PEFT/LoRA through the supported SmolVLA path | Pending inspection of actual instantiated module names. |
| Simulator/embodiment | Isaac Lab Arena + Fourier GR1 | Pending headless rendered GPU smoke test. |
| Logging | Episode JSONL, CSV/JSON summary, TensorBoard | Pending implementation. |
| Speech | Local audio-file speech recognition routed through the policy language interface | Pending implementation and fixtures. |

## Current hardware decision

The local Windows machine has a GTX 1050 Ti with 4 GB VRAM. It is limited to source
development and CPU tests. The final simulator, policy inference, and headline
benchmark require an Ubuntu 22.04 NVIDIA GPU worker (preferably 24 GB VRAM, 32 GB RAM,
and 200 GB persistent disk). This is a hardware constraint, not a performance claim.

## Integration contract to verify

The downloaded policy must be loaded from its immutable revision using official
LeRobot processing. A live Arena camera frame, GR1 state, and the natural-language
instruction must form the policy observation; its decoded action chunk must be mapped
only to documented GR1 actions. The concrete image/state/text/action feature schemas,
normalization, licences, model hash, asset revisions, and LoRA target module names are
recorded only after inspection on the target environment.

## Rejected alternatives

The previous MuJoCo/Unitree draft is superseded by this contract-defined GR1 Arena
stack. The 4 GB local GPU is rejected for Isaac Lab and SmolVLA benchmark execution.

## Sources

- [GR1 Arena pretrained-policy instructions](https://github.com/huggingface/lerobot/blob/main/docs/source/envhub_isaaclab_arena.mdx)
- [SmolVLA implementation](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/modeling_smolvla.py)
