# DS2 Frontier Label Mode Bug

Date: 2026-05-31
Branch: `yasha-dev-newar`

## Summary

The DS2 packed-frontier run from 2026-05-30 was invalid because the config
enabled two mutually exclusive full-path label modes:

```yaml
data:
  overfit_full_path_control_birthset_boundary_labels: true
  overfit_full_path_control_frontier_level_labels: true

trainer:
  topology_decoder: birthset_frontier
  birthset_lambda_proposal: 0.0
  birthset_use_pair_prefix_candidates: false
```

In `TreeDataset._build_full_path_control_samples`, boundary birthset labels are
checked before frontier labels:

```python
if self.overfit_full_path_control_birthset_boundary_labels:
    ...
elif self.overfit_full_path_control_frontier_level_labels:
    ...
```

That means the run trained old one-shot boundary birthset targets while sampling
with the new packed-frontier decoder. This was a train/sampler mismatch, not a
clean test of the frontier head.

## Symptom

Invalid run:

```text
/ewsc/yektefai/phylaflow/diagnostics/ds2_frontier_fast_20260530/
```

W&B:

```text
https://wandb.ai/yasha/DS/runs/wkkrpu86
```

The model loss went down, but rollout metrics stayed bad:

```text
best rf_norm_mean: 0.4799
best golden_kl_divergence_topological: 0.7349
best golden_kl_divergence_tree_topology: 19.3391
golden_support_rate: 0.0 throughout
sampled_topology_unique_count: almost always 998-1000
```

The training logs exposed the mismatch:

```text
birthset_stats/candidate_recall_pre_gold ~= 0.686
birthset_stats/avg_required_splits ~= 5.087
birthset_stats/avg_observed_gold_splits ~= 4.825
```

For a correct precomputed frontier run, gold candidates should already be present
and required/observed frontier counts should match.

## Root Cause

The config inherited `overfit_full_path_control_birthset_boundary_labels: true`
from the older DS2 birthset setup. The new run added
`overfit_full_path_control_frontier_level_labels: true`, but the dataset code
silently used the boundary-label branch because it was checked first.

The result was:

```text
training labels: old boundary one-shot birthset labels
sampling decoder: birthset_frontier greedy packed-frontier decoder
proposal loss: disabled
pair-prefix candidates: disabled
```

This explains the zero support and near-max sampled topology diversity. The
model was not being trained on the same action representation used at rollout.

## Fix

`data/dataset.py` now fails fast if both flags are enabled:

```python
if (
    self.overfit_full_path_control_birthset_boundary_labels
    and self.overfit_full_path_control_frontier_level_labels
):
    raise ValueError(...)
```

These modes are intentionally mutually exclusive:

- `overfit_full_path_control_birthset_boundary_labels`: trains one-shot boundary
  birthset targets.
- `overfit_full_path_control_frontier_level_labels`: trains packed-frontier
  merge-decoder targets.

## Corrected Config Pattern

Use frontier labels only with the frontier decoder:

```yaml
data:
  overfit_full_path_control_birthset_boundary_labels: false
  overfit_full_path_control_frontier_level_labels: true
  full_path_precompute_birthset_targets: true
  full_path_precompute_birthset_candidate_info: true

trainer:
  topology_decoder: birthset_frontier
  birthset_lambda_proposal: 0.0
  birthset_use_pair_prefix_candidates: false
```

Corrected DS2 run:

```text
/ewsc/yektefai/phylaflow/diagnostics/ds2_frontier_fast_fixedlabels_20260531/
https://wandb.ai/yasha/DS/runs/lgvcjmr1
```

The corrected run started with the expected canaries:

```text
birthset_stats/candidate_recall_pre_gold = 1.000
birthset_stats/avg_required_splits == birthset_stats/avg_observed_gold_splits
```

First eval at step 1000:

```text
rf_norm_mean: 0.4462
golden_kl_divergence_topological: 0.3305
golden_kl_divergence_tree_topology: 19.3391
golden_support_rate: 0.0
sampled_topology_unique_count: 849
```

This is much better than the invalid frontier run at step 1000:

```text
rf_norm_mean: 0.6963
golden_kl_divergence_topological: 1.4168
sampled_topology_unique_count: 1000
```

It still has not recovered the older DS2 birthset behavior as of step 1000:

```text
old DS2 birthset best rf_norm_mean: ~0.181-0.216
old DS2 birthset best golden_kl_divergence_topological: ~0.076-0.108
old DS2 birthset best golden_kl_divergence_tree_topology: ~2.5
old DS2 birthset support rate: nonzero
```

## Guidance For Future Runs

Before trusting a packed-frontier run, check these fields in the first training
logs:

```text
birthset_stats/candidate_recall_pre_gold should be 1.000
birthset_stats/avg_required_splits should equal avg_observed_gold_splits
birthset_stats/gold_mapping_mismatches should be 0
```

If `candidate_recall_pre_gold` is around `0.68` on this DS2 setup, the run is
probably using boundary labels or otherwise training the wrong target path.

Do not use the 2026-05-30 `wkkrpu86` run as evidence that the frontier head
regressed. It was a config/label-mode mismatch. Use the corrected 2026-05-31
run and later fixed-label runs for frontier-head comparisons.
