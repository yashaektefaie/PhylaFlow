# Phyla Embedding Integration Notes

Last updated: 2026-05-05

This note summarizes the current Phyla embedding integration paths, the experiment findings so far, and the config knobs needed to run new experiments. It is intended as a handoff for agents launching DS1-DS8 and Orthomam/realstream runs.

## Current Recommendation

Use Phyla embeddings as leaf tokens plus cheap global and clade context for the first-hit and autoregressive heads:

```yaml
model:
  phyla_dim: 256
  phyla_use_leaf_tokens: true
  phyla_use_split_tokens: false
  phyla_use_global_context: true
  phyla_global_context_scale: 1.0
  phyla_use_clade_context: true
  phyla_clade_context_scale: 1.0
```

Pair this with the stronger metric-start architecture:

```yaml
model:
  hidden_dim: 128
  embed_dim: 128
  n_layers: 4
  tokenizer_n_layers: 4
  autoregressive_start_topology_code_dim: 64
  first_hit_head_case_dim: 64
  first_hit_frozen_start_case_adapter_mode: mlp2
  first_hit_frozen_start_case_adapter_hidden_dim: 256
  autoregressive_frozen_start_case_adapter_mode: mlp2
  autoregressive_frozen_start_case_adapter_hidden_dim: 512
```

For Orthomam scale-up, the current first scale to test is `s192x5`:

```yaml
model:
  hidden_dim: 192
  embed_dim: 192
  n_layers: 5
  first_hit_head_phase_hidden_dim: 192
trainer:
  training_step_full_path_replay_initial_retry_attempt: 2
```

Do not enable split Phyla tokens by default. They were slower and did not appear to help compared with leaf tokens plus head context.

## What Changed In Code

### Model Fusion Paths

Main file: `model/model.py`

New model config keys:

- `phyla_use_global_context`
- `phyla_global_context_scale`
- `phyla_use_clade_context`
- `phyla_clade_context_scale`

The model now has three independent Phyla fusion paths:

- Leaf token injection: projects per-leaf Phyla embeddings and adds them only to leaf token positions. This is controlled by `phyla_use_leaf_tokens`.
- Global context: mean-pools the active leaf Phyla embeddings, projects the pooled vector, and injects it into first-hit and autoregressive head inputs. This is controlled by `phyla_use_global_context`.
- Clade/head context: computes inside/outside mean Phyla embeddings for each represented split, projects the concatenated `[inside_mean, outside_mean]`, and injects that context into edge/first-hit and autoregressive group head inputs. This is controlled by `phyla_use_clade_context`.

The global context matters because first-hit and autoregressive decisions need an explicit "which sequence family/tree am I working on?" signal, not just per-leaf local signals. The clade context is a cheap way to tell the heads which Phyla content is inside a candidate split and which Phyla content is outside it.

Implementation details:

- `TreeDenoiserTokenGT._compute_global_phyla_context(...)` pools active leaf embeddings and projects to `embed_dim`.
- `TreeDenoiserTokenGT._compute_clade_phyla_token_context(...)` builds per-edge split context from inside/outside Phyla means.
- `_compute_first_hit_logits_from_edges(...)` accepts `phyla_global_context` and adds it to first-hit head inputs through `first_hit_phyla_global_proj`.
- The autoregressive structured subset path adds global Phyla context to group embeddings and adds clade context for the candidate group tokens.
- `return_model(config)` now wires the new Phyla config keys.

### Precomputed Phyla Embeddings

Main file: `run/TrainingModule.py`

The trainer can load precomputed Phyla embeddings from a single file, a directory, or a colon-separated list of paths:

```yaml
trainer:
  phyla_checkpoint_path: null
  phyla_precomputed_embeddings_path: "/path/to/embedding_dir:/path/to/fallback_file.pt"
```

Directory loading understands these file suffixes:

- `_phyla_beta_embeddings.pt`
- `_phyla_beta_sitechunk_w256_s256_embeddings.pt`

The loader now keeps dataset-specific lookup maps. This is important for realstream/Orthomam because multiple banks can share simple numeric leaf names, so leaf name alone is ambiguous.

Expected payload fields in a precomputed embedding file:

- `sequence_names` or `names`
- `embeddings` or `phyla_embeddings`
- optional `dataset_id`

Lookup behavior:

- Prefer dataset-specific lookup when `dataset_id` is available.
- Fall back to global name lookup.
- If numeric leaf names are present, use dataset-specific tensor row lookup as a fallback.

### Realstream / Topology Stream Support

Main file: `data/dataset.py`

Relevant config keys:

```yaml
data:
  topology_stream_index_jsonl_path: /ewsc/yektefai/phylaflow_datasets/real_topology_stream_index_20260502.jsonl
  topology_stream_index_max_cases: 0
  topology_stream_index_max_num_leaves: 120
  overfit_virtual_epoch_size: 12451
  sample_metrics_config_path: /path/to/ds2_eval_config.yaml
```

The dataset now carries `dataset_id`, `ids`, `mappings`, and `num_leaves` through the training batch so the trainer can resolve the correct precomputed Phyla embeddings for each tree.

### Replay Subsampling For Orthomam Memory

Main files: `run/TrainingModule.py`, `run/run.py`

Relevant config key:

```yaml
trainer:
  training_step_full_path_replay_initial_retry_attempt: 1
```

Set this to `1` for `s128x4` Orthomam runs and `2` for `s192x5`. It pre-subsamples full-path replay samples before the first forward pass, rather than waiting for a CUDA OOM retry. This is the reason the `s192x5` Orthomam run currently fits on an A6000.

## Experiment Findings So Far

### Split Tokens

`phyla_use_split_tokens: true` was slow and did not look useful enough to keep. It injects Phyla-derived information into split/edge tokens directly, but it adds work and did not solve the main topology weakness.

Default recommendation:

```yaml
phyla_use_split_tokens: false
```

### Leaf Tokens

`phyla_use_leaf_tokens: true` gave a clear improvement over runs without useful Phyla conditioning. It is cheap and should remain enabled.

However, leaf-only conditioning appeared too weak for topology. It improved DS2 behavior but bottomed out on topological metrics and tree topology.

Default recommendation:

```yaml
phyla_use_leaf_tokens: true
```

### Global Context

Global Phyla context helped because it gives the first-hit and autoregressive heads an explicit sequence-family/tree context, analogous to how frozen start-tree metric embeddings helped the first-hit and AR heads.

Default recommendation:

```yaml
phyla_use_global_context: true
phyla_global_context_scale: 1.0
```

### Clade Context

Clade context should be used through the current cheap head-context path, not through a heavy token expansion. It computes inside/outside Phyla means per candidate split and injects the projected vector into first-hit/AR head inputs.

Default recommendation:

```yaml
phyla_use_clade_context: true
phyla_clade_context_scale: 1.0
```

### Metric-Start Architecture

The better architecture found from DS runs is:

- `scale128x4`
- `metricprobe64`
- `fh64`
- `mlp2cap`
- first-hit adapter hidden `256`
- AR adapter hidden `512`

The older `fh16` / first-hit adapter `128` / AR adapter `256` setup should be treated as a weaker baseline.

### Orthomam Scale

Current running Orthomam/realstream runs:

- `s128x4`: fits but uses most of GPU memory on large batches.
- `s192x5`: currently fits on an A6000 with `training_step_full_path_replay_initial_retry_attempt: 2`.

Do not jump to `s256x6` yet. It is likely to OOM under the same replay settings unless replay is reduced more aggressively or the leaf cap is reduced.

## Representative Configs

DS1-DS8 current recommended config:

```text
configs/local_ds1ds8_smallbank_exactanchors_phy256_leafglobal_cladehead_metricprobe64_fh64_aradd_mlp2cap_s128_lr2e3_ds2eval_mrbayes20k_20260505.yaml
```

Orthomam/realstream current recommended config:

```text
configs/local_realstream_348299_phy256_leafglobal_cladehead_metricprobe64_fh64_aradd_mlp2cap_s128_lr2e3_ds2eval_replaysubset_mrbayes20k_20260505.yaml
```

Orthomam/realstream scale-up config:

```text
configs/local_realstream_348299_phy256_leafglobal_cladehead_metricprobe64_fh64_aradd_mlp2cap_s192x5_lr2e3_ds2eval_replaysubset_mrbayes20k_20260505.yaml
```

The matching launch copies on EWSC live under:

```text
/ewsc/yektefai/30272299/launch_configs_ewsc/configs/
```

## Launching Safely On This Machine

Use the EWSC-safe launcher for local GPU launches:

```bash
scripts/launch_phylaflow_ewsc_safe.sh /ewsc/yektefai/30272299/launch_configs_ewsc/configs/<config>.yaml <gpu_id>
```

This exports temp/cache directories onto `/ewsc/yektefai/phylaflow`, including:

- `TMPDIR`
- `TMP`
- `TEMP`
- `WANDB_DIR`
- `WANDB_CACHE_DIR`
- `WANDB_DATA_DIR`
- `WANDB_ARTIFACT_DIR`
- `TRITON_CACHE_DIR`

This matters because the root filesystem can be mounted read-only and `/tmp` can be unusable. Lightning/fsspec checkpoint saving stages an atomic temp file through Python `tempfile`, so a read-only `/tmp` can crash checkpointing even when `checkpoint_dir` itself is on `/ewsc`.

The launcher also sets:

```bash
PYTHONDONTWRITEBYTECODE=1
```

Do not use `PYTHONPYCACHEPREFIX` on `/ewsc`; it caused slow startup by writing thousands of `.pyc` files over NFS.

The Slurm helper `slurm/run_ds_24h.sbatch` also exports the same scratch/temp variables.

## Validation

Unit test coverage for the model fusion paths is in:

```text
tests/test_model_phyla_fusion.py
```

Useful validation commands:

```bash
/ewsc/yektefai/envs/envs/pgt/bin/python -m py_compile model/model.py run/run.py run/TrainingModule.py data/dataset.py
/ewsc/yektefai/envs/envs/pgt/bin/python -m unittest tests.test_model_phyla_fusion -v
bash -n scripts/launch_phylaflow_ewsc_safe.sh slurm/run_ds_24h.sbatch
```

## Practical Defaults For New Experiments

For DS1-DS8:

```yaml
model:
  hidden_dim: 128
  embed_dim: 128
  n_layers: 4
  phyla_dim: 256
  phyla_use_leaf_tokens: true
  phyla_use_split_tokens: false
  phyla_use_global_context: true
  phyla_use_clade_context: true
  first_hit_head_case_dim: 64
  first_hit_frozen_start_case_adapter_hidden_dim: 256
  autoregressive_frozen_start_case_adapter_hidden_dim: 512
trainer:
  phyla_precomputed_embeddings_path: /home/unix/yektefai/PhylaFlow/analysis/full_sanity_fixedpair_20260401/ds_phyla_embeddings_20260428
```

For Orthomam/realstream:

```yaml
model:
  hidden_dim: 128
  embed_dim: 128
  n_layers: 4
  phyla_dim: 256
  phyla_use_leaf_tokens: true
  phyla_use_split_tokens: false
  phyla_use_global_context: true
  phyla_use_clade_context: true
  first_hit_head_case_dim: 64
  first_hit_frozen_start_case_adapter_hidden_dim: 256
  autoregressive_frozen_start_case_adapter_hidden_dim: 512
trainer:
  phyla_precomputed_embeddings_path: "/ewsc/yektefai/phylaflow_datasets/phyla_embeddings_sitechunk_cpu_20260428:/home/unix/yektefai/PhylaFlow/analysis/full_sanity_fixedpair_20260401/ds_phyla_embeddings_20260428/DS2_phyla_beta_embeddings.pt"
  training_step_full_path_replay_initial_retry_attempt: 1
data:
  topology_stream_index_jsonl_path: /ewsc/yektefai/phylaflow_datasets/real_topology_stream_index_20260502.jsonl
  topology_stream_index_max_num_leaves: 120
```

For Orthomam `s192x5`, change:

```yaml
model:
  hidden_dim: 192
  embed_dim: 192
  n_layers: 5
  first_hit_head_phase_hidden_dim: 192
trainer:
  training_step_full_path_replay_initial_retry_attempt: 2
```

## Open Questions

- Whether `s192x5` improves Orthomam metrics enough to justify the extra memory.
- Whether `s256x6` can fit with a higher replay pre-cap, smaller leaf cap, or both.
- Whether clade context scale should remain `1.0` or be swept against `0.5` and `2.0`.
- Whether global context alone explains most of the gain, or clade context gives independent improvement after enough training.
