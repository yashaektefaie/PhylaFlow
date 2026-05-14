# PhylaFlow Generalization Experiments

Last updated: 2026-05-14 07:38 EDT

This note summarizes the generalization debugging work on `yasha-dev`, the code/config changes made to support it, and the experiments that are currently running.

## Motivation

The initial observation was that PhylaFlow could learn within related DS1-DS8 settings, but transfer to held-out distributions such as DS2 and OrthoMaM-derived subsets was weak or absent. That suggested the model was not learning a reusable posterior-transport rule from the Phyla conditioning alone.

The main question became:

> Can PhylaFlow learn to map a random start tree plus sequence-derived Phyla embeddings into a posterior-like 10-leaf tree distribution for held-out OrthoMaM datasets?

The 10-leaf setting was chosen as a sanity check before returning to 29-leaf and DS2 transfer.

## Experiments Tried

### OrthoMaM and DS2 Transfer

We first tried training on OrthoMaM and evaluating on DS2-like settings. These runs showed little to no generalization. In several cases training appeared to make DS2 evaluation worse, which pointed away from a simple undertraining explanation.

### 29-Leaf Matching

We investigated whether poor transfer was due to leaf count mismatch. The idea was to train/evaluate with OrthoMaM datasets pruned to match the DS2 leaf count. We also considered taking 29-leaf subsets from larger OrthoMaM posterior trees and subsetting Phyla embeddings rather than recomputing them.

The implementation direction moved toward doing this on the fly in the dataset rather than pre-materializing all subset examples. This eventually led to the more controlled 10-leaf online posterior-subset stream.

### 10-Leaf OrthoMaM Holdout

The main sanity check became:

- sample 10-leaf subsets from OrthoMaM posterior datasets,
- train on 80% of the OrthoMaM dataset IDs,
- hold out 20% of dataset IDs,
- test whether PhylaFlow can generalize to unseen 10-leaf OrthoMaM datasets.

The early precomputed-embedding run:

```text
orthomam10leaf_train80_holdout20_phy256_leafglobal_cladehead_base_s128_lr2e3_20260511
```

used precomputed Phyla-beta embeddings and a prebuilt topology-stream index. It is not directly comparable to the current live-Phyla runs because it used a finite prebuilt stream rather than online posterior-subset sampling.

## Main Code Changes

### Live Phyla End-to-End Training

`TrainingModule` now supports using a live Phyla-beta model during PhylaFlow training:

- `live_phyla_checkpoint_path`
- `live_phyla_unfreeze`
- `live_phyla_lr`
- `live_phyla_input_mode`
- `live_phyla_max_input_tokens`

When `live_phyla_lr` is set, the optimizer uses two parameter groups:

- PhylaFlow/tree model params at `trainer.lr`
- live Phyla params at `trainer.live_phyla_lr`

When `live_phyla_lr: null`, all trainable params share the main LR.

### Online Posterior Subset Dataset

`data/dataset.py` now supports online posterior-subset training/eval:

- `posterior_subset_num_leaves`
- `posterior_subset_max_input_tokens`
- `same_dataset_batch`
- `same_dataset_batches_per_epoch`
- `full_path_preparse_structural_trees`
- `full_path_precompute_tokenizer_raw_graphs`

This lets the loader sample a dataset ID, select a 10-leaf subset, sample random start and posterior target trees, and return the path labels without relying on the old finite topology-stream index.

### Tokenizer and Batch Construction Speed Work

Several changes moved deterministic tree preprocessing out of the training step and into the dataset side where possible:

- Newick/structure preprocessing
- raw graph fields for tokenizer input
- masks and structural arrays needed for full-path training

We also tried a joint velocity/autoregressive tokenization path. Benchmarks showed it did not beat the simpler shared-trunk/separate-tokenization path, so the active configs keep:

```yaml
training_step_joint_tokenize_velocity_ar: false
```

### Logging and Disk Safety

The current long runs write W&B and stdout logs under `/ewsc`. `run/run.py` now supports:

- `default_root_dir`
- `disable_lightning_logger`

This avoids Lightning CSV logs growing under `/home/unix/yektefai`.

## Important Configs

Current long baseline:

```text
configs/local_orthomam10leaf_train80_randomposterior_samebatch_livephyla_rawfull_e2e_100m_resume40k_20260513.yaml
```

Original scratch config for that run:

```text
configs/local_orthomam10leaf_train80_randomposterior_samebatch_livephyla_rawfull_e2e_100m_scratch_wandb_20260513.yaml
```

Aggressive one-LR Phyla run:

```text
configs/local_orthomam10leaf_train80_randomposterior_samebatch_livephyla_rawfull_e2e_100m_aggressive_onelr_lr1e3_20260514.yaml
```

Holdout eval config for the online live-Phyla setup:

```text
configs/local_orthomam10leaf_holdout20_randomposterior_livephyla_rawfull_e2e_datasetpatch_20260513.yaml
```

Early precomputed-embedding comparison run:

```text
configs/local_orthomam10leaf_train80_holdout20_phy256_leafglobal_cladehead_base_s128_lr2e3_20260511.yaml
configs/local_orthomam10leaf_holdout20_eval_phy256_leafglobal_cladehead_base_s128_lr2e3_20260511.yaml
```

## Current Running Experiments

### Baseline Live-Phyla Resume

```text
PID: 1492587
GPU: 5
W&B: https://wandb.ai/yasha/orthomam/runs/pcn0joxw
Config: configs/local_orthomam10leaf_train80_randomposterior_samebatch_livephyla_rawfull_e2e_100m_resume40k_20260513.yaml
Log: /ewsc/yektefai/phylaflow/logs/orthomam10leaf_train80_livephyla_resume40k_gpu5_20260513.log
Checkpoint source: epoch=0-step=040000.ckpt
```

This run resumes the live-Phyla online posterior-subset training from step 40k. It uses:

```yaml
lr: 0.001
live_phyla_lr: 1.0e-6
batch_size: 4
num_workers: 8
posterior_subset_num_leaves: 10
same_dataset_batch: true
```

Current interpretation:

- `kl_divergence_topological` improves slowly and is often below 2.
- `rf_norm_mean` is mostly flat around 0.85-0.90.
- exact posterior topology support recall is essentially zero.
- the model appears to learn topology-diversity statistics before learning true posterior modes.

Recent bucketed metrics from the holdout trace:

```text
0-50k steps:
  KLtop mean: 2.3505
  RFmean mean: 0.8813
  unique sampled topologies mean: 10.3

50-100k steps:
  KLtop mean: 1.9492
  RFmean mean: 0.8751
  unique sampled topologies mean: 21.7

100-145k steps:
  KLtop mean: 1.9122
  RFmean mean: 0.8824
  unique sampled topologies mean: 23.6
```

### Aggressive One-LR Live-Phyla Run

```text
PID: 1638741
GPU: 6
W&B: https://wandb.ai/yasha/orthomam/runs/7qt33k0r
Config: configs/local_orthomam10leaf_train80_randomposterior_samebatch_livephyla_rawfull_e2e_100m_aggressive_onelr_lr1e3_20260514.yaml
Log: /ewsc/yektefai/phylaflow/logs/orthomam10leaf_train80_livephyla_aggressive_onelr_lr1e3_gpu6_20260514.log
```

This run tests the hypothesis that `live_phyla_lr: 1e-6` is effectively too frozen. The aggressive config sets:

```yaml
lr: 0.001
live_phyla_lr: null
```

That puts PhylaFlow and live Phyla-beta in the same optimizer LR group at `1e-3`.

The first eval at step 50 was:

```text
kl_divergence_topological: 2.9415
rf_norm_mean: 0.9432
n_unique_sampled_topologies: 1
```

This early result is only a startup baseline. The question is whether this run learns much faster than the conservative baseline or destabilizes Phyla.

## Current Read

The strongest signal so far is not RF improvement. It is that the live-Phyla setup slowly learns to produce a less collapsed topology distribution, visible in `kl_divergence_topological` and unique sampled topology counts. However, support recall and exact topology overlap remain near zero, so this is not yet a convincing generalization result.

The two most plausible explanations are:

1. Phyla embeddings contain some useful information, but the Phyla-to-tree interface needs much more adaptation than `live_phyla_lr: 1e-6` allows.
2. The model is learning global/topological entropy statistics rather than dataset-specific clade structure.

The aggressive one-LR run is meant to separate those: if it improves RF/support quickly, the issue was under-adapting Phyla. If it collapses or remains RF-flat, the blocker is deeper than LR.

## Operational Notes

`/home/unix/yektefai` filled during the long run. Rebuildable caches were removed, and `.vscode-server` was deleted after confirmation, freeing several GB. The active runs now log under `/ewsc` to avoid filling home again.

Generated artifacts, local weights, metrics dumps, and W&B files are intentionally not part of this branch unless explicitly needed later.
