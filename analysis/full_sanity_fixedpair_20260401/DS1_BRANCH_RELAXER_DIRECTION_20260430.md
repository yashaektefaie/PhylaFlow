# DS1 Branch Relaxer Finding and Training Direction

Date: 2026-04-30 UTC

## Short Read

The main DS1 lesson from the last set of experiments is that raw generated
Tree-KL is not telling the whole story. Several model-generated starts looked
mediocre by immediate topology KL, but after a learned branch-length relaxer
adjusted only branch lengths, their log-likelihoods moved much closer to the
posterior basin. When MrBayes starts from those relaxed model-generated trees,
it reaches much better topology distributions much faster.

Concretely, the branch relaxer did not need to change topology to help. The
corrected helper preserves the input topology and only edits branch lengths. On
the 234 DS1 `1280bank_nocond` model-generated starts:

| Tree list | Count | Same topology | Mean LL before | Mean LL after | Mean delta | Improved | Worse |
|---|---:|---:|---:|---:|---:|---:|---:|
| `nocond_splitguided_first234` | 234 | 234 | -9948.402 | -7564.266 | +2384.136 | 234 | 0 |
| `nocond_terminal_first234` | 234 | 234 | -9922.365 | -7499.001 | +2423.364 | 234 | 0 |

Higher log-likelihood here means better likelihood under the DS1 JC scoring
model. If phrased as loss, this is a large negative-log-likelihood decrease.

## Why This Changed the Direction

Before branch relaxation, the best 20K count-matched no-conditioning result was
already encouraging:

| Start source | Count | Tree-KL @ 20K | Tail-half Tree-KL | Support | Posterior recall |
|---|---:|---:|---:|---:|---:|
| `1280bank_nocond` split-guided starts | 234 | 0.743369 | 0.647354 | 0.716637 | 1.000000 |
| `1280bank_nocond` terminal endpoints | 234 | 0.858870 | 0.651847 | 0.625836 | 1.000000 |
| matched random unseen starts | 234 | 1.027645 | 0.775817 | 0.480156 | 0.987342 |

The unrelaxed model starts were therefore already useful initial conditions for
MrBayes, especially early in the chain. But their branch lengths were bad:
generation-0 mean MrBayes log-likelihood was around `-9948` for split-guided
starts, versus the good basin near `-6886` after burn-in.

After topology-preserving branch relaxation, the same starts became much
stronger MrBayes initial states:

| Generation | Relaxed terminal Tree-KL | Relaxed split-guided Tree-KL | Unrelaxed split-guided Tree-KL | Pre-existing random Tree-KL |
|---:|---:|---:|---:|---:|
| 0 | 5.215021 | 5.882294 | 5.882294 | 16.540891 |
| 20K | 0.309899 | 0.344487 | 0.794843 | 0.960733 |
| 40K | 0.240477 | 0.270979 | 0.651244 | 0.726032 |
| 60K | 0.204354 | 0.238591 | 0.585163 | 0.601207 |
| 80K | 0.186642 | 0.220257 | 0.537088 | 0.519239 |
| 100K | 0.174632 | 0.207002 | 0.499590 | 0.452979 |

Tail-half Tree-KL was also best for the relaxed terminal starts:

| Start set | Tail-half Tree-KL | First generation below 2 | First generation below 1 | Support recall at 100K |
|---|---:|---:|---:|---:|
| Relaxed terminal | 0.136126 | 1200 | 2400 | 1.000000 |
| Relaxed split-guided | 0.164908 | 1400 | 2800 | 1.000000 |
| Unrelaxed split-guided | 0.399932 | 2600 | 10800 | 1.000000 |
| Pre-existing random | 0.291147 | 4400 | 18000 | 1.000000 |

This means the model is often producing useful topology basins earlier than our
old training diagnostic suggests. If branch lengths are poor, immediate Tree-KL
and split KL can make the model look worse than it is for the workflow we
actually care about: quickly seeding posterior sampling.

## What We Added to Training Eval

The DS1 training eval now has three important metrics in addition to the older
raw topology/split KL metrics:

1. `sample_metrics/relaxed_log_likelihood_mean`

   Generate unseen start trees, apply the standalone branch relaxer to the
   model-generated outputs, then score the relaxed tree list with
   `GenericJCLikelihood`. For speed and signal quality we currently log only the
   mean.

2. `sample_metrics/mrbayes20k_tree_kl`

   Run a short MrBayes chain from the generated starts and compare its topology
   distribution to the golden DS1 topology distribution. Current default in the
   DS1 configs is 64 starts/chains for speed and eval every 500 training steps.

3. `sample_metrics/mrbayes20k_tail_tree_kl`

   Same MrBayes short-run evaluation, but using the tail half of the sampled
   chain. This is useful because a good initialization can still show burn-in in
   cumulative KL.

These metrics are intentionally expensive but much closer to the downstream
objective. The current conclusion is: do not demote raw Tree-KL/split-KL, but
do not optimize solely for them either. The main score to watch now is whether
generated trees, after branch relaxation, already have good likelihood and seed
20K MrBayes into a good topology distribution.

## Important Validity Notes

- Only model-generated/model-derived outputs were branch-relaxed in the key
  comparison. The random baseline was not relaxed.
- The first relaxed 100K run was invalid because an early helper rebuilt trees
  from split masks and accidentally changed topologies. Do not use:
  `nocond_splitguided_first234_relaxed_g100000_curve.json`.
- The corrected helper writes trees by editing the input Newick in-place and
  preserving topology. In the corrected run, all 234 relaxed trees retained the
  same topology as their input.
- The current strongest result is the corrected topology-preserving relaxed
  terminal run, not the invalid earlier relaxed split-guided curve.

## Core Files

Training/eval code:

- `run/TrainingModule.py`
  - Adds branch-relax support used by training/eval.
  - Adds unseen-start sample eval.
  - Adds relaxed likelihood metric.
  - Adds embedded MrBayes20K sample metric.
  - Gates expensive topology-repeat trace summaries behind
    `sample_metrics_trace_topology_repeats_enabled: false`.
  - Gates chatty per-step logs behind
    `training_step_verbose_logging_enabled: false`.

- `run/run.py`
  - Wires the new trainer config keys into all `TrainingModule(...)` call sites.
  - Defaults sampler inner logging to disabled.

- `model/model.py`
  - Adds optional `mlp2` adapters for frozen start-case embeddings:
    `first_hit_frozen_start_case_adapter_mode: mlp2` and
    `autoregressive_frozen_start_case_adapter_mode: mlp2`.
  - Default remains `linear`, so existing configs keep old behavior unless they
    opt in.

Offline helpers:

- `analysis/full_sanity_fixedpair_20260401/apply_branch_relaxer_and_score_tree_list_20260430.py`
  - Applies the standalone branch relaxer to a tree list.
  - Preserves topology by editing Newick branch lengths in place.
  - Writes `<label>_relaxed_trees.txt`, `<label>_per_tree.jsonl`, and
    `<label>_summary.json`.

- `analysis/full_sanity_fixedpair_20260401/benchmark_mrbayes_fixed_start_generic.py`
  - Runs MrBayes from a fixed start tree or a newline-delimited start-tree list.
  - Computes Tree-KL curves against the golden posterior topology distribution.
  - Uses a local `job.nex` basename inside the work directory to avoid the
    MrBayes long-filename failure.

- `analysis/full_sanity_fixedpair_20260401/train_standalone_branch_relaxer_20260429.py`
  - Trains the standalone branch relaxer checkpoint used by the current eval.

Branch relaxer checkpoint:

- `analysis/full_sanity_fixedpair_20260401/standalone_branch_relaxer_ds1_ds8_phyla_leafonly_balanced_nocase_20260429/best.pt`

## Configs to Run

Main current DS1 anchored baseline with unseen-start eval, relaxed likelihood,
and MrBayes20K:

```bash
python /home/yektefai/.codex/skills/phylaflow-launch/scripts/launch_train.py \
  /home/yektefai/PhylaFlow/configs/local_ds1_1280bank_nocond_s128_lr2e3_unseeneval256_20260430.yaml
```

Clean no-anchor/regression-only comparison:

```bash
python /home/yektefai/.codex/skills/phylaflow-launch/scripts/launch_train.py \
  /home/yektefai/PhylaFlow/configs/local_ds1_1280bank_nocond_noanchors_regonly_s128_lr2e3_unseeneval64_mrbayes20k_20260430.yaml
```

Do not treat this no-anchor/full-aux config as a recommended recipe; it was the
bad exploding-loss comparison:

```bash
configs/local_ds1_1280bank_nocond_noanchors_s128_lr2e3_unseeneval64_mrbayes20k_20260430.yaml
```

Standalone/integrated branch-relax training config from the earlier likelihood
experiment:

```bash
configs/local_ds1_branchrelax_likelihoodonly_20260429.yaml
```

Key trainer flags in the DS1 eval configs:

```yaml
training_sampling_start: 500
training_sampling_frequency: 500
training_sampling_mode: harness_sanity
sample_metrics_num_pairs: 64
sample_metrics_unseen_start_eval: true
sample_metrics_unseen_pair_selection_mode: random_bank
sample_metrics_relaxed_likelihood_enabled: true
sample_metrics_branch_relaxer_checkpoint_path: /home/yektefai/PhylaFlow/analysis/full_sanity_fixedpair_20260401/standalone_branch_relaxer_ds1_ds8_phyla_leafonly_balanced_nocase_20260429/best.pt
sample_metrics_mrbayes20k_enabled: true
sample_metrics_mrbayes20k_num_starts: 64
sample_metrics_mrbayes20k_ngen: 20000
sample_metrics_mrbayes20k_samplefreq: 200
sample_metrics_mrbayes20k_max_workers: 12
wandb_project: DS
```

The 1280-bank training configs currently reference this local bank artifact:

```text
analysis/full_sanity_fixedpair_20260401/ds1_multipair1280_topofreqcover_base256x5_20260429_velocity_anchors.json
```

That combined anchor JSON is about 71 MB and is treated as a dataset artifact,
not source code. It must be present locally or restored from the DS1 artifact
store before launching those configs.

## Stored Result Files

Branch-relaxed trees and likelihood summaries:

- `analysis/full_sanity_fixedpair_20260401/ds1_branchrelax_start_ll_20260430/summary.md`
- `analysis/full_sanity_fixedpair_20260401/ds1_branchrelax_start_ll_20260430/nocond_splitguided_first234_relaxed_trees.txt`
- `analysis/full_sanity_fixedpair_20260401/ds1_branchrelax_start_ll_20260430/nocond_terminal_first234_relaxed_trees.txt`
- `analysis/full_sanity_fixedpair_20260401/ds1_branchrelax_start_ll_20260430/nocond_splitguided_first234_summary.json`
- `analysis/full_sanity_fixedpair_20260401/ds1_branchrelax_start_ll_20260430/nocond_terminal_first234_summary.json`

Corrected 100K MrBayes curves from relaxed starts:

- `analysis/full_sanity_fixedpair_20260401/ds1_branchrelax_start_ll_20260430/nocond_splitguided_first234_relaxed_topopreserve_g100000_curve.json`
- `analysis/full_sanity_fixedpair_20260401/ds1_branchrelax_start_ll_20260430/nocond_terminal_first234_relaxed_topopreserve_g100000_curve.json`
- `analysis/full_sanity_fixedpair_20260401/ds1_branchrelax_start_ll_20260430/relaxed_splitguided_100k_summary.md`
- `analysis/full_sanity_fixedpair_20260401/ds1_branchrelax_start_ll_20260430/relaxed_terminal_100k_summary.md`

Unrelaxed model-start and random baseline summaries:

- `analysis/full_sanity_fixedpair_20260401/ds1_1280bank_nocond_step20000_mrbayes_all256_20260430/summary.md`
- `analysis/full_sanity_fixedpair_20260401/ds1_1280bank_nocond_step20000_mrbayes_all234_from_all256_20260430/comparison_234.md`
- `analysis/full_sanity_fixedpair_20260401/ds1_1280bank_nocond_step20000_mrbayes_all234_from_all256_20260430/splitguided_vs_random_100k_234.md`

## How to Reproduce the Offline Branch-Relaxed Evaluation

Apply the branch relaxer to a saved tree list:

```bash
python analysis/full_sanity_fixedpair_20260401/apply_branch_relaxer_and_score_tree_list_20260430.py \
  --dataset-id DS1 \
  --label nocond_terminal_first234 \
  --input-tree-list analysis/full_sanity_fixedpair_20260401/ds1_1280bank_nocond_step20000_mrbayes_all234_from_all256_20260430/nocond_terminal_first234_start_trees.txt \
  --out-dir analysis/full_sanity_fixedpair_20260401/ds1_branchrelax_start_ll_20260430
```

Run 100K MrBayes from the relaxed tree list:

```bash
python analysis/full_sanity_fixedpair_20260401/benchmark_mrbayes_fixed_start_generic.py \
  --dataset-id DS1 \
  --dataset-pickle /home/yektefai/30272299/DS1.pickle \
  --golden-root /home/yektefai/30272299/golden_run_data_DS1-8/DS1 \
  --start-tree-list analysis/full_sanity_fixedpair_20260401/ds1_branchrelax_start_ll_20260430/nocond_terminal_first234_relaxed_trees.txt \
  --label nocond_terminal_first234_relaxed_topopreserve \
  --num-runs 234 \
  --ngen 100000 \
  --samplefreq 200 \
  --printfreq 5000 \
  --max-workers 12 \
  --curve-interval 20000 \
  --work-dir /tmp/nocond_terminal_first234_relaxed_topopreserve_mrbayes \
  --output analysis/full_sanity_fixedpair_20260401/ds1_branchrelax_start_ll_20260430/nocond_terminal_first234_relaxed_topopreserve_g100000_curve.json
```

## Current Recommendation

Continue DS1 training around the 1280-bank no-conditioning recipe. Watch
`relaxed_log_likelihood_mean`, `mrbayes20k_tree_kl`, and
`mrbayes20k_tail_tree_kl` every 500 steps on 64 unseen starts. Raw Tree-KL is
still important, but the downstream win condition is now clearer: generate
starts that, after branch-length relaxation, are already in a high-likelihood
posterior basin and allow short MrBayes runs to recover the topology
distribution quickly.
