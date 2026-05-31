# Packed Frontier Regression Hunt - 2026-05-30

Branch: `yasha-dev-newar`

## Context

We were testing the packed frontier birthset decoder on a single 29-leaf
OrthoMaM dataset (`10345_NT_AL`). The packed labels were fixed so that
non-binary frontier-level gold merges are retained and force-included during
precompute. After that fix, candidate recall was `1.0` and
`gold_mapping_mismatches` was `0`, but rollout from the 3K checkpoint still did
not overfit to RF zero.

## Label Fix

Relevant file: `data/dataset.py`

The packed frontier precompute now:

- keeps gold frontier labels with `len(merge_indices) >= 2`, not just binary
  labels;
- force-adds any gold frontier subset not already present in that level's
  candidate set;
- updates the teacher-forced active frontier by consuming all active fronts
  contained in a positive subset.

Validation on the 4-dataset bank first 64 rows:

- groups: `195`
- targets: `1717`
- target hits: `1717`
- candidates: `54352`
- positives: `1717`
- `candidate_recall_pre_gold`: `1.0`
- `gold_mapping_mismatches`: `0`

## Phase-Cap Diagnosis

Checkpoint:

`/ewsc/yektefai/phylaflow/checkpoints/frontier_correctness_20260529/orthomam29leaf_single10345_frontierpacked_overfit3k_20260529/2026-05-29_20-55-40/epoch=0-step=003000.ckpt`

Direct eval configs:

`/ewsc/yektefai/phylaflow/diagnostics/frontier_correctness_20260529/eval_configs/orthomam29leaf_single10345_frontierpacked_step3000_directeval_cap{16,32,64}_100pairs_20260530.yaml`

Direct eval output:

| cap | RF mean | RF best | topological KL | tree KL | events mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 | 0.5931 | 0.3077 | 1.0410 | 14.3137 | 16 |
| 32 | 0.5923 | 0.3077 | 1.0408 | 14.3137 | 32 |
| 64 | 0.5927 | 0.3077 | 1.0424 | 14.3137 | 64 |

Conclusion: the cap was not the bottleneck. Increasing the phase cap only
increased the number of boundary events; it did not move the rollout closer.

## No-Op Rebirth Loop

Tracing a single sampled pair showed a collapse/rebirth loop. From phase 5
onward, the velocity head collapsed one or two splits and the birthset decoder
selected the exact same splits for reinsertion. RF returned to the same value
after each birthset step.

Example from source bank index `198`:

- phase 5 collapsed split: `8388610`
- phase 5 selected birth split: `8388610`
- phases 6-11 repeated the same exact rebirth behavior

This is invalid for the intended BHV boundary transition: a split that just
collapsed should not be immediately reborn as the birth split for that boundary.

## Sampler Patch

Relevant file: `run/TrainingModule.py`

Added a `blocked_splits` path through:

- `_plan_birthset_boundary_splits`
- `_plan_birthset_frontier_boundary_splits`
- `_birthset_select_compatible_top_k`
- `_birthset_select_compatible_beam`

The discrete and non-discrete samplers now pass the just-collapsed split masks
into the birthset planner. The planner blocks exact canonical rebirth of those
splits for that boundary, without treating them as compatibility constraints.

After patching, the traced pair no longer reinserted just-collapsed splits. The
old 3K checkpoint did not improve from this patch alone:

- RF mean: `0.6154`
- RF best: `0.4231`
- topological KL: `1.0826`
- tree KL: `14.3137`

This suggests retraining or longer continuation is still needed; the guard only
removes the no-op failure mode.

## Active Continuation Run

Config:

`/ewsc/yektefai/phylaflow/diagnostics/frontier_correctness_20260529/run_configs/orthomam29leaf_single10345_frontierpacked_continue10k_afterblock_20260530.yaml`

Log:

`/ewsc/yektefai/phylaflow/diagnostics/frontier_correctness_20260529/logs/orthomam29leaf_single10345_frontierpacked_continue10k_afterblock_20260530_gpu3.log`

Trace:

`/ewsc/yektefai/phylaflow/diagnostics/frontier_correctness_20260529/orthomam29leaf_single10345_frontierpacked_continue10k_afterblock_20260530_trace.jsonl`

Checkpoint dir:

`/ewsc/yektefai/phylaflow/checkpoints/frontier_correctness_20260529/orthomam29leaf_single10345_frontierpacked_continue10k_afterblock_20260530`

Launched from the 3K checkpoint for 7000 additional batches, sampling every
1000 steps.

Current observed sample metrics:

| global step | RF mean | RF best | topological KL | tree KL | events mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1000 | 0.6065 | 0.3846 | 1.0285 | 14.3578 | 16 |
| 2000 | 0.5485 | 0.3077 | 0.9224 | 14.2471 | 16 |

The 2000-step sample shows movement again, but still not RF-zero overfit.

