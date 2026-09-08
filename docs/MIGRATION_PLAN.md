# Runtime Integration Status

## Preserved baseline

`src/sparse` is the original project implementation from `master` and remains
the only source of query/prefix sparse behavior. The new modules do not alter
its public `patch_model(model, model_name=...)` contract.

## Integrated surfaces

- `src.dense.patch_model` uses that same contract. LLaDA delegates to the
  checkpoint's own LLaDA 2.1 M2T + T2T decoder; SDAR uses the project's
  standard block-diffusion driver with every sparse feature disabled.
- `src.losa.patch_model` uses the same contract and installs the paper-LoSA
  attention patch. The LoSA PyTorch oracle, Triton kernel, generation driver,
  and optional packed MoE backend are vendored under `src/losa`.
- `src.focus.patch_model` uses the same contract and runs the portable FOCUS
  row-retention path directly on the loaded HF model. It supports both
  `llada2_moe` (LLaDA 2.1) and `sdar`; it does not import or launch the sibling
  FOCUS checkout.

## Decoder rules

- LLaDA2.1 checkpoint defaults are `threshold=0.95` and
  `editing_threshold=0.9`; the model card's speed profile is
  `threshold=0.5`, `editing_threshold=0`. In the official decoder, zero is a
  permissive T2T threshold (`p > 0`), not a switch disabling editing.
- SDAR's public reference profile is b4, dynamic, threshold 0.9. This project
  evaluates the separate `SDAR-8B-Chat-b32` checkpoint with its b32 driver;
  those settings remain explicit and are not LLaDA defaults.

## Verification status

- CPU source-level tests and random official-model equivalence tests are
  included under `tests/`.
- CUDA tests still need to run on a node with a live driver, binding no more
  than two idle GPUs. `nvidia-smi` on the current login host reports no live
  driver; use the site's Slurm allocation to run the GPU smoke commands.
