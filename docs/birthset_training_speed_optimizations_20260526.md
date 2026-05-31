# Birthset Training Speed Optimizations

Date: 2026-05-26
Branch: `yasha-dev-newar`

This memo summarizes the speed work on the OrthoMaM 29-leaf fixed-pair full-path training setup using the birthset topology decoder. The main outcome is that the GPU training step is now fast when batches are already prepared, but real end-to-end training is still dominated by full-path dataloader/collate construction at large external batch sizes.

## Correction From 2026-05-29

The original table below mislabeled `joint_trunk_forward` time as full profiled training-step time. The `0.312s` number came from:

```text
logs/benchmark_full_orthomam29leaf_staticgold_precompute_b16_nocap_20260521_gpu2.log
step=25 joint_trunk_forward=0.3123s autoregressive_step=7.6316s total=8.8534s
```

So that number was useful for isolating the trunk forward, but it was not end-to-end step throughput.

Re-running the old batch-64 benchmark path on 2026-05-29 gave warmed full `training_step` totals of `0.803s` and `1.037s`, while wall-clock was still dominated by dataloader/collate. For the current 4-dataset static-candidate setup, row-level preload alone took `78.6s` and ran slower than the non-preloaded run (`~0.64 it/s` versus `~1.0 it/s` at batch 4). Fully pre-collated batch-64 training exposed another measured bottleneck: Lightning was recursively transferring the huge nested full-path batch before `training_step`. Overriding `transfer_batch_to_device` for full-path batches improved repeated pre-collated batch-64 wall throughput from `0.08 it/s` to `0.49 it/s`.

## Current Throughput

Measured on an RTX A6000 with `num_workers=32`, frozen/precomputed Phyla embeddings, full-path velocity plus birthset AR training, and CUDA-synchronized profiling.

| external batch | internal combined tree states | profiled training step | peak GPU memory | status |
|---:|---:|---:|---:|---|
| 64 | 384 | `0.312s` | `2.2 GB` | clean |
| 128 | 754 | `0.484s` | `5.9 GB` | clean |
| 256 | 1524 | `1.117s` | `7.5 GB` | clean |
| 512 | 3104 | `3.039s` | `27.4 GB` | clean |
| 1024 | 4564 | `7.039s` | `32.0 GB` | one profiled step, timed out before clean completion |

Best GPU-side throughput is around external batch `128`, with batch `256` still reasonable if fewer optimizer steps are preferred. Batch `512+` fits in GPU memory, but throughput worsens because backward scales hard with the expanded internal full-path batch.

## End-To-End Bottleneck

The profiler starts after Lightning has received a batch, so it does not include dataloader/collate latency. Direct `next(loader)` timing with `num_workers=32` showed:

| external batch | expanded velocity / AR samples | first batch fetch |
|---:|---:|---:|
| 64 | 127 / 127 | `18.9s` |
| 128 | 281 / 281 | `55.0s` |
| 256 | 585 / 585 | `91.3s` |
| 512 | no batch within `180s` |
| 1024 | no batch within `180s` |

So there are two different ceilings:

- Prepared-batch GPU training is now fast enough for roughly `0.3-1.1s/step` at practical batch sizes.
- Full-path dataloader/collate construction is still the real wall-clock limiter, especially above batch `128`.

## What Changed

### 1. AR Group Index Precompute

Files:

- `data/dataset.py`
- `utils/bhv_utils.py`
- `model/model.py`
- `run/TrainingModule.py`

Before: `_decode_outputs(... autoregressive=True)` rebuilt explicit structural polytomy group token indices from split masks during the model forward.

After:

- `data/dataset.py::_build_collated_full_path_autoregressive_batch` computes `_cached_autoregressive_group_indices` and `_cached_autoregressive_group_splits` in collate.
- `utils/bhv_utils.py::get_explicit_structural_group_indices_from_edge_splits` maps raw tokenizer edge split masks directly to token indices.
- `model/model.py::_decode_outputs` accepts `autoregressive_group_indices` and `autoregressive_group_splits`.
- `run/TrainingModule.py` passes cached group indices/splits through both joint velocity/AR forward and non-joint AR paths.

Validation:

- Cached AR group indices/splits matched the old fallback on a real collated batch: `mismatches=0`.

### 2. Batched AR Split-Identity Embeddings

File:

- `model/model.py`

Before: AR split-identity embeddings were created per polytomy group inside the Python group loop.

After: all split masks across the AR batch are flattened once, projected once, and sliced per group.

Observed effect:

- `split_identity` in AR decode dropped from tens of milliseconds in heavy batches to sub-millisecond to low-millisecond in normal batches.

### 3. Vectorized Split-Mask Binary Construction

File:

- `model/model.py`

The 29-leaf/int64-safe split-mask binary construction path now vectorizes mask-to-bit expansion for <=63-bit masks and falls back for larger masks. This helps both velocity and AR split identity paths avoid Python bit loops on the common OrthoMaM subset case.

### 4. Cached Direct-Set Velocity Targets

Files:

- `data/dataset.py`
- `run/TrainingModule.py`

Before: the direct-set velocity loss branch reparsed Newick trees and recomputed BHV encodings inside `TrainingModule.step`, even when the shared trunk forward was already precomputed.

After:

- `data/dataset.py::_probe_direct_set_precompute_for_velocity_sample` materializes edge indices, direct-set targets, matched masks, and target-set sizes in collate.
- `run/TrainingModule.py::_cached_probe_direct_set_logs` uses one packed tensor gather and vectorized BCE/softplus calculations for the current `velocity_probe_direct_set_mse_weight=0` setup.
- The original parser/BHV path remains as fallback if cached metadata is absent or if a different velocity-loss mode needs it.

Validation:

- Cached direct-set materialization matched the old tree/BHV filtering: `errors=0`.
- Vectorized direct-set loss matched the old per-sample loop:
  - `loss diff=0.0`
  - `mean_jaccard diff=0.0`
  - `target_negative_loss diff=0.0`

Measured effect:

- Before direct-set vectorization: `velocity_step` around `0.16-0.22s`.
- After direct-set vectorization: `velocity_step` around `0.02-0.03s`.

### 5. Profiling Hooks

Files:

- `model/model.py`
- `run/TrainingModule.py`

Useful profiler outputs:

- `TRAINING_STEP_PROFILE`
- `JOINT_FORWARD_PROFILE`
- `BIRTHSET_STEP_PROFILE`
- `BIRTHSET_PROPOSAL_PROFILE`
- `AR_DECODE_PROFILE` via `PHYLA_AR_DECODE_PROFILE=1`
- `VELOCITY_DECODE_PROFILE` via `PHYLA_VELOCITY_DECODE_PROFILE=1`

For meaningful speed numbers, use CUDA-synchronized profiling and ignore first-batch startup unless specifically measuring dataloader/collate latency.

## Practical Recommendation

For current full-path OrthoMaM 29-leaf training:

- Use external batch `128` as the default throughput setting.
- Use batch `256` only if the dataloader can keep up or if fewer optimizer steps matter more than wall-clock throughput.
- Avoid batch `512+` until full-path sample materialization/collate is redesigned or cached more aggressively.

The next high-value optimization is not another model-forward micro-optimization. It is to remove or cache the expensive full-path replay/sample construction currently happening in the dataloader/collate path.

## Reproduction Notes

Representative benchmark config source:

```bash
/tmp/phylaflow_joint_profile_20260522/b64_condexp_skiplegacy_noorder_workers32.yaml
```

Representative command:

```bash
CUDA_VISIBLE_DEVICES=3 \
PHYLA_VELOCITY_DECODE_PROFILE=1 \
PHYLA_AR_DECODE_PROFILE=1 \
/ewsc/yektefai/envs/envs/pgt/bin/python -u -m run.run <benchmark_config.yaml>
```

Syntax check used after code changes:

```bash
/ewsc/yektefai/envs/envs/pgt/bin/python -m py_compile \
  data/dataset.py \
  model/model.py \
  run/TrainingModule.py \
  utils/bhv_utils.py
```
