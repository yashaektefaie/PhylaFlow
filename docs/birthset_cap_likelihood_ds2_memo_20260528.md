# Birthset Generalization, Cap, and Likelihood Findings

Date: 2026-05-28
Branch: `yasha-dev-newar`

This memo records the current state of the birthset decoder experiments after the OrthoMaM 29-leaf single-dataset c264 bank, the phase-cap sweep, and the raw likelihood probe. The purpose is to make the current code/results reproducible for future agents.

## Code State

The branch keeps the birthset topology decoder active while preserving the legacy AR path. Recent code changes are focused on correctness and observability rather than changing the scientific objective.

Tracked files changed:

- `data/dataset.py`
- `run/TrainingModule.py`

Important changes:

- Full-path AR samples now carry `num_leaves` from the actual event tree, avoiding ROOT_DUMMY/off-by-one mismatches when collate builds explicit structural group indices.
- Full-path collate has `_full_path_structural_num_leaves(...)` so structural helpers use the tree leaf count implied by component masks.
- Sample metrics now preserve `dataset_id`, `selected_sequence_names`, and per-dataset name mappings for fixed-pair and joint-bank cases. This matters for OrthoMaM subset banks, where all rows have the same selected 29 taxa but are not the full 120-taxon dataset.
- Harness sampling first tries selected sequence names for precomputed Phyla embeddings, then falls back to tree/name-mapping order.
- Birthset under-resolution is now logged as under-resolution, not as `stopped_for_no_valid_merge`. New trace fields include `birthset_underresolved_boundary`, `birthset_underresolved_boundary_count`, `trace_birthset_underresolved_polytomies`, and `sampled_tree_has_polytomy`.
- Birthset proposal loss is still computed when candidate precompute is unavailable, rather than silently dropping proposal supervision.

## OrthoMaM Single-Dataset c264 Bank

Dataset: `10345_NT_AL`, 29 selected leaves.

The old c210 bank was biased: 210 rows, 140 unique target topologies, missing 30 of 170 retained posterior topologies, missing posterior mass 15.9%, and TV distance 0.234 from the retained empirical posterior.

The corrected all-posterior bank is:

`/ewsc/yektefai/phylaflow/orthomam_subset_banks/orthomam29leaf_single_10345_c264_allposterior_seed20260518/orthomam29leaf_single_10345_c264_allposterior_train.jsonl`

Validation:

- 264 rows, one for each retained MrBayes posterior sample after burn-in.
- 170 unique retained 29-leaf target topologies.
- Missing posterior topology mass: 0.0.
- Empirical TV distance to retained posterior: 0.0.
- SHA256: `512901b18481e0926943ccccdec8b44694ab7ebd85ec91ecbc68c5921d007af4`.

## OrthoMaM c264 Training Result

Run config:

`/ewsc/yektefai/phylaflow/diagnostics/regression_hunt_20260528/run_configs/orthomam29leaf_single10345_c264_allposterior_fromscratch_30k_20260528.yaml`

Final checkpoint:

`/ewsc/yektefai/phylaflow/checkpoints/full_sanity_fixedpair_20260401/orthomam29leaf_single10345_c264_allposterior_fromscratch_30k_20260528/2026-05-28_05-32-25/epoch=0-step=030000.ckpt`

At 30K with `sampling_discrete_phase_max_phases: 16`:

- RF mean: 0.317745
- RF median: 0.269231
- split/topological KL: 0.215519
- tree-topology KL: 15.011207
- sampled unique topologies: 139
- posterior topology support recall: 0.0
- sampled tree polytomy rate: 0.0
- trace birthset events mean: 16.0

The split-level metrics moved, but exact posterior topology support did not. The sampled trees are mostly off-support exact topologies.

## Phase-Cap Sweep

Checkpoint:

`/ewsc/yektefai/phylaflow/checkpoints/full_sanity_fixedpair_20260401/orthomam29leaf_single10345_c264_allposterior_fromscratch_30k_20260528/2026-05-28_05-32-25/epoch=0-step=030000.ckpt`

Output directory:

`/ewsc/yektefai/phylaflow/diagnostics/cap_sweep_20260528`

Summary table:

`/ewsc/yektefai/phylaflow/diagnostics/cap_sweep_20260528/cap_sweep_summary.tsv`

Key results:

| phase cap | RF mean | split KL | tree-topology KL | posterior support recall | shared topologies | polytomy rate |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.665561 | 0.508042 | 14.6039 | 0.020833 | 2 | 0.590909 |
| 2 | 0.345040 | 0.139349 | 13.7902 | 0.041667 | 4 | 0.325758 |
| 3 | 0.327831 | 0.188704 | 13.2941 | 0.083333 | 8 | 0.011364 |
| 4 | 0.322407 | 0.189023 | 12.7881 | 0.083333 | 8 | 0.000000 |
| 5 | 0.317262 | 0.188322 | 13.0381 | 0.093750 | 9 | 0.003788 |
| 6 | 0.321096 | 0.195988 | 14.4976 | 0.020833 | 2 | 0.000000 |
| 7 | 0.317016 | 0.199433 | 14.4976 | 0.020833 | 2 | 0.000000 |
| 8 | 0.315268 | 0.203173 | 13.8935 | 0.041667 | 4 | 0.000000 |
| 16 | 0.314540 | 0.218862 | 15.0478 | 0.010417 | 1 | 0.000000 |

Oracle geodesic path lengths for the c264 bank:

- path events mean: 1.49
- path events median: 1
- path events max: 8

Interpretation: cap 16 overshoots. RF continues to improve slightly with more phases, but exact topology KL degrades after caps 4-5. The model learns useful local split movement, but without a terminal/stop rule it keeps composing locally plausible moves into off-posterior exact topologies.

## Raw JC Likelihood Probe

Raw JC likelihoods were computed on the final sampled trees using the 29 selected OrthoMaM sequences from `10345_NT_AL`. Branch lengths were not optimized, so these numbers are diagnostic only.

Summary path:

`/ewsc/yektefai/phylaflow/diagnostics/cap_sweep_20260528/cap_sweep_raw_jc_likelihood_summary.tsv`

| phase cap | mean raw log-likelihood | RF mean | split KL | tree KL |
|---:|---:|---:|---:|---:|
| 1 | -17188.9 | 0.665561 | 0.508042 | 14.6039 |
| 2 | -16789.3 | 0.345040 | 0.139349 | 13.7902 |
| 3 | -17047.6 | 0.327831 | 0.188704 | 13.2941 |
| 4 | -17169.1 | 0.322407 | 0.189023 | 12.7881 |
| 5 | -17294.9 | 0.317262 | 0.188322 | 13.0381 |
| 8 | -17297.6 | 0.315268 | 0.203173 | 13.8935 |
| 16 | -17505.5 | 0.314540 | 0.218862 | 15.0478 |

Raw likelihood also suggests overshooting. It peaks around cap 2, then worsens as more topology phases are applied. Random starts are much worse, with mean raw likelihood about -25583; target posterior trees are much better, with mean raw likelihood about -7507.

A small fixed-topology MrBayes branch-relax probe was attempted with 500 generations on one 29-leaf sampled tree. It was still running after about 3.5 minutes and was killed. This is too slow for an online stopping rule. MrBayes-relaxed likelihood should be used offline to validate a stopping heuristic, not inside sampling.

## DS2 Prior Result For Contrast

The older DS2 birthset distribution run used:

`configs/local_ds2_210bank_birthset_top32_s128_lr2e3_unseen1000_20260517.yaml`

Metrics:

`/ewsc/yektefai/phylaflow/metrics/full_sanity_fixedpair_20260401/ds2_210bank_birthset_top32_s128_lr2e3_unseen1000_20260517_metrics.jsonl`

This run did show exact topology movement on DS2, unlike the OrthoMaM c264 single-dataset run:

- Step 23K: `kl_divergence_tree_topology` 7.27, RF mean 0.260
- Step 26K: `kl_divergence_tree_topology` 5.98, RF mean 0.169
- Step 39K: `kl_divergence_tree_topology` 4.79, RF mean 0.161
- Step 40K: `kl_divergence_tree_topology` 4.35, RF mean 0.212

The DS2 short/golden tree-KL also dropped to around 2.1-3.5 in the best late checkpoints. DS2 therefore remains the key sanity case showing that exact topology KL can move under this decoder.

## Current Read

- Birthset decoding is fast enough to sample many trees, but terminal stopping is still unsolved.
- On OrthoMaM 29-leaf c264, fixed cap 16 is too high and creates off-support exact topologies.
- A phase cap around 4-5 is better for exact topology on the c264 single-dataset bank, while cap 2 is best for raw split KL and raw JC likelihood.
- MrBayes-relaxed likelihood is too expensive for online sampling but useful as an offline validation of stopping heuristics.
- The next experiment is a fresh DS2 current-code run to confirm whether exact tree-KL still moves with the current branch and updated trace semantics.
