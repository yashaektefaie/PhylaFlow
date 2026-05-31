# Frontier Birthset Speed Experiments - 2026-05-30

Branch: `yasha-dev-newar`

This memo records the speed experiments around the packed-frontier birthset
training path after the regression hunt on OrthoMaM 29-leaf fixed-bank runs.
The main point is to separate three costs that were previously getting mixed
together:

- one-time row preload / pre-collated batch construction;
- prepared-batch `training_step` time;
- sampling/eval stalls injected during long training runs.

## Relevant Code Changes

Files touched:

- `data/dataset.py`
- `model/model.py`
- `run/TrainingModule.py`
- `run/run.py`
- `scripts/benchmark_fixed_pair_sampling.py`

Major changes:

- Added `PrecollatedBatchDataset` and `data.full_path_preload_collated_batches`
  so fixed full-path training batches can be materialized once, then repeatedly
  indexed during training.
- Added `TrainingModule.transfer_batch_to_device` fast path for full-path
  batches. This avoids Lightning recursively transferring a large nested batch
  structure before `training_step`.
- Added packed birthset scoring via
  `BirthSetTopologyHead.score_many_packed`, so many candidate birth splits are
  scored in one tensorized call instead of per-group Python loops.
- Added flat grouped BCE for birthset candidates. Rank loss still has a
  per-record path and remains a smaller but visible cost.
- Added packed-frontier precompute in `data/dataset.py`, preserving
  non-binary frontier labels and force-including gold subsets.
- Added `topology_decoder` support for `birthset_frontier` and
  `birthset_binary`, plus `birthset_binary_pair_mode` config plumbing.
- Added birthset sampler blocking for just-collapsed splits so a boundary does
  not immediately reinsert the exact split it just collapsed.
- Added synchronized profiling prints:
  `TRAINING_STEP_PROFILE`, `JOINT_FORWARD_PROFILE`, `ENCODE_ONCE_PROFILE`, and
  `BIRTHSET_STEP_PROFILE`.

## Correction To Earlier Timing

The old `0.312s` number in the earlier memo was not a full training step. It
was `joint_trunk_forward` only:

```text
step=25 joint_trunk_forward=0.3123s autoregressive_step=7.6316s total=8.8534s
```

Re-running and profiling full prepared-batch steps showed the true step timing
was closer to `0.6s` after the latest packed-frontier changes, with additional
startup time for row preload and batch pre-collation.

## Long Run With Sampling

Continuation config:

```text
/ewsc/yektefai/phylaflow/diagnostics/frontier_correctness_20260529/run_configs/orthomam29leaf_single10345_frontierpacked_continue10k_afterblock_20260530.yaml
```

Log:

```text
/ewsc/yektefai/phylaflow/diagnostics/frontier_correctness_20260529/logs/orthomam29leaf_single10345_frontierpacked_continue10k_afterblock_20260530_gpu3.log
```

Measured:

- row preload: `13.5s`;
- pre-collate 4 batches: `311.9s`;
- 7000 training steps: `1:12:49`, reported `1.60 it/s`;
- sampling every 1000 steps created about `51-58s` stalls at each sample step.

After removing those sampling stalls from the interpretation, the steady
prepared-batch training rate was roughly `1.7-1.8 it/s`.

## Controlled 80-Step No-Sampling Benchmark

Baseline config:

```text
/ewsc/yektefai/phylaflow/diagnostics/frontier_speed_ablation_20260530/run_configs/single10345_precollated_nosample_profile80_20260530.yaml
```

Baseline log:

```text
/ewsc/yektefai/phylaflow/diagnostics/frontier_speed_ablation_20260530/logs/single10345_precollated_nosample_profile80_gpu3.log
```

Settings:

- one OrthoMaM 29-leaf dataset: `10345_NT_AL`;
- fixed full-path bank with frozen/precomputed Phyla embeddings;
- external batch size `64`;
- combined internal tree states per step around `384`;
- `limit_train_batches: 80`;
- sampling disabled;
- CUDA-synchronized profiling enabled.

Baseline measured timings:

| component | mean | median | note |
| --- | ---: | ---: | --- |
| full step | `0.6470s` | `0.6031s` | includes one outlier |
| full step excluding step 60 outlier | `0.5870s` | `0.5845s` | `1.70 it/s` |
| `joint_trunk_forward` | `0.2273s` | `0.1735s` | step 60 outlier inflated mean |
| `birthset_loss_direct` | `0.1651s` | `0.1644s` | candidate scoring plus BCE/rank |
| backward | `0.2357s` | `0.2461s` | largest stable component |
| scalar logging | `0.0110s` | `0.0108s` | minor |

Representative final-step nested profile:

```text
ENCODE_ONCE_PROFILE step=80 batch=384 tokens=115
prepare_encoder_inputs=0.0629s encoder_layers=0.0153s

JOINT_FORWARD_PROFILE step=80 velocity_batch=192 autoregressive_batch=192
combined_batch=384 tokens=114 move_tokenized=0.0111s
times_phyla_concat=0.0140s tokenized_concat=0.0003s
encode_once=0.0784s velocity_decode=0.0453s autoregressive_decode=0.0242s
```

This shows the transformer layers are not the bottleneck. The encoder layers
were only about `15ms`; `prepare_encoder_inputs` was about `63ms`.

## Rank-Loss-Off Ablation

Config:

```text
/ewsc/yektefai/phylaflow/diagnostics/frontier_speed_ablation_20260530/run_configs/single10345_precollated_nosample_profile80_rank0_20260530.yaml
```

Log:

```text
/ewsc/yektefai/phylaflow/diagnostics/frontier_speed_ablation_20260530/logs/single10345_precollated_nosample_profile80_rank0_gpu3.log
```

Change: `trainer.birthset_lambda_rank: 0.0`.

Measured:

| component | mean | median |
| --- | ---: | ---: |
| full step | `0.5723s` | `0.5718s` |
| full step excluding step 60 | `0.5658s` | `0.5571s` |
| throughput | `1.77 it/s` |  |
| `joint_trunk_forward` | `0.2003s` | `0.1937s` |
| `birthset_loss_direct` | `0.1632s` | `0.1449s` |
| backward | `0.1871s` | `0.1865s` |

Birthset subprofile over the last 10 logged steps:

| component | time |
| --- | ---: |
| candidate build | `0.0045s` |
| topology head forward | `0.1125s` |
| proposal loss | near `0s` |
| BCE/rank bookkeeping | `0.0105s` |

Conclusion: rank loss is visible, but disabling it only improved prepared-batch
throughput from `1.70 it/s` to `1.77 it/s`. This is not the main wall.

## No Phyla Global/Clade Context Ablation

Config:

```text
/ewsc/yektefai/phylaflow/diagnostics/frontier_speed_ablation_20260530/run_configs/single10345_precollated_nosample_profile80_nophylactx_20260530.yaml
```

Log:

```text
/ewsc/yektefai/phylaflow/diagnostics/frontier_speed_ablation_20260530/logs/single10345_precollated_nosample_profile80_nophylactx_gpu6.log
```

Change:

```yaml
model:
  phyla_use_global_context: false
  phyla_use_clade_context: false
  phyla_use_leaf_tokens: true
```

Measured:

| component | mean | median |
| --- | ---: | ---: |
| full step | `0.4539s` | `0.4674s` |
| full step excluding step 60 | `0.4517s` | `0.4514s` |
| throughput | `2.21 it/s` |  |
| `joint_trunk_forward` | `0.0963s` | `0.0994s` |
| `birthset_loss_direct` | `0.1599s` | `0.1599s` |
| backward | `0.1802s` | `0.1901s` |

Representative final-step nested profile:

```text
ENCODE_ONCE_PROFILE step=80 batch=384 tokens=115
prepare_encoder_inputs=0.0089s encoder_layers=0.0155s

JOINT_FORWARD_PROFILE step=80 velocity_batch=192 autoregressive_batch=192
combined_batch=384 tokens=114 move_tokenized=0.0105s
times_phyla_concat=0.0131s tokenized_concat=0.0003s
encode_once=0.0245s velocity_decode=0.0293s autoregressive_decode=0.0224s
```

Conclusion: Phyla global/clade context construction inside
`_prepare_encoder_inputs` is the biggest measured forward-pass inefficiency.
Turning off that context improved prepared-batch throughput from `1.70 it/s` to
`2.21 it/s`.

## Startup / Pre-Collation Cost

The controlled no-sampling runs intentionally separate pre-collation from
prepared-batch training. Pre-collation itself remains noisy and expensive:

| run | row preload | pre-collate 4 batches |
| --- | ---: | ---: |
| baseline | `13.4s` | `309.0s` |
| rank loss off | `16.5s` | `698.0s` |
| no Phyla global/clade context | `16.3s` | `836.8s` |

Do not compare these startup numbers to model-step timings directly. The
startup path is CPU-heavy materialization of full-path batches; the profiled
model loop starts after those batches already exist.

## Current Bottleneck Read

For prepared batches at batch size 64:

- transformer encoder layers are cheap: about `15ms`;
- Phyla global/clade context prep costs about `50ms` of avoidable forward time;
- birthset candidate construction is now negligible: about `4ms`;
- topology-head scoring is still meaningful: about `85-90ms`;
- BCE/rank bookkeeping is about `60ms` with rank enabled;
- backward is about `190-250ms`.

The most direct next speed target is Phyla context prep. Under frozen Phyla
embeddings, the raw context aggregation is static and can move to dataloader or
pre-collation:

```text
dataloader/pre-collate:
  phyla_global_raw = mean(valid leaf Phyla embeddings)
  phyla_clade_raw  = concat(mean_inside_split, mean_outside_split)

model forward:
  phyla_global_proj(phyla_global_raw)
  phyla_clade_proj(phyla_clade_raw)
```

The learned projection layers should stay in the model unless intentionally
frozen. Precomputing projected contexts would freeze those adapters and would be
a separate ablation, not the main training path.

## Reproduction Commands

Use module execution because `python run/run.py` can hit package shadowing:

```bash
CUDA_VISIBLE_DEVICES=3 \
/ewsc/yektefai/envs/envs/pgt/bin/python -u -m run.run \
  /ewsc/yektefai/phylaflow/diagnostics/frontier_speed_ablation_20260530/run_configs/single10345_precollated_nosample_profile80_20260530.yaml
```

Parse profile rows with the `TRAINING_STEP_PROFILE`,
`JOINT_FORWARD_PROFILE`, `ENCODE_ONCE_PROFILE`, and `BIRTHSET_STEP_PROFILE`
lines in the corresponding log.

Syntax check:

```bash
/ewsc/yektefai/envs/envs/pgt/bin/python -m py_compile \
  data/dataset.py \
  model/model.py \
  run/TrainingModule.py \
  run/run.py \
  scripts/benchmark_fixed_pair_sampling.py
```
