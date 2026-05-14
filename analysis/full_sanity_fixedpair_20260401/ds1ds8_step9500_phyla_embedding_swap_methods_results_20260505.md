# DS1-DS8 Phyla Embedding Swap Analysis - Methods and Results

Date: 2026-05-05

## Question

This analysis tested whether the DS1-DS8 PhylaFlow checkpoint uses sequence-derived Phyla embeddings as an active conditioning signal for topology generation. The specific question was:

If we keep the same trained flow, the same target dataset, and the same start trees fixed, but swap only the Phyla embedding bank, do generated trees remain in the target dataset posterior basin or move elsewhere?

## Short Answer

The generated trees move elsewhere when the Phyla embeddings are swapped. For every target dataset, the native condition, where the target dataset and Phyla embedding bank match, has low split-KL. Valid off-diagonal swaps increase split-KL by roughly 3 to 4 nats in the strongest comparisons.

The cleanest interpretation is that the checkpoint is not only learning a generic topology-improvement flow from starts. It is using the Phyla conditioning vectors to choose the basin.

## Definitions

- Target dataset: the dataset whose fixed unseen start trees are used and whose golden posterior is used for scoring.
- Condition Phyla embedding ID: the dataset ID of the precomputed Phyla embedding bank passed to the sampler.
- Native condition: target dataset ID equals condition Phyla embedding ID, for example target DS3 with DS3 Phyla embeddings.
- Swap condition: target dataset ID differs from condition Phyla embedding ID, for example target DS3 with DS7 Phyla embeddings.
- Split-KL: `golden_kl_divergence_topological`, the KL divergence between split or bipartition distributions from generated trees and the target golden posterior.
- Tree-topology KL: `golden_kl_divergence_tree_topology`, the KL divergence between full topology distributions from generated trees and the target golden posterior.

## Inputs

### Checkpoint

The analysis used the DS1-DS8 checkpoint at step 9500:

`/ewsc/yektefai/phylaflow/checkpoints/full_sanity_fixedpair_20260401/ds1ds8_smallbank_exactanchors_phy256_leafglobal_cladehead_metricprobe64_fh64_aradd_mlp2cap_s128_lr2e3_ds2eval_mrbayes20k_20260505/2026-05-05_06-54-39/sample-metrics-epoch=1-step=009500.ckpt`

This checkpoint came from the DS1-DS8 run:

- W&B run: `yasha/DS/37duw99j`
- Run name: `ds1ds8_smallbank_exactanchors_phy256_leafglobal_cladehead_metricprobe64_fh64_aradd_mlp2cap_s128_lr2e3_ds2eval_mrbayes20k_20260505`
- It was selected because the original DS2 sample-metrics trace reached its best DS2 split-KL around step 9500.

### Base Model Configuration

Base config:

`/ewsc/yektefai/30272299/launch_configs_ewsc/configs/local_ds1ds8_smallbank_exactanchors_phy256_leafglobal_cladehead_metricprobe64_fh64_aradd_mlp2cap_s128_lr2e3_ds2eval_mrbayes20k_20260505.yaml`

Key architecture/configuration properties:

- Backbone hidden size: `s128`
- Transformer layers: `4`
- Phyla embedding dimension: `256`
- Phyla conditioning mode: leaf/global with clade head
- First-hit head case dimension: `fh64`
- First-hit frozen start-case adapter hidden dimension: `256`
- Autoregressive frozen start-case adapter hidden dimension: `512`
- Learning rate in original run: `2e-3`
- Dataset scope: DS1 through DS8

### Target-Specific Sample-Metrics Configurations

For each target dataset, the runner borrowed the target dataset's sample-metrics configuration so that each target was scored against its own posterior references and fixed-start machinery.

| Target | Sample config |
|---|---|
| DS1 | `local_ds1_1280bank_metricprobe64_fh16_aradd_mlp2_s128_lr2e3_unseeneval234_mrbayes20k_20260501.yaml` |
| DS2 | `local_ds2_210bank_metricprobe64_fh16_aradd_mlp2_s128_lr2e3_unseeneval42_mrbayes20k_20260501.yaml` |
| DS3 | `local_ds3_1215bank_metricprobe64_fh16_aradd_mlp2_s128_lr2e3_unseeneval243_mrbayes20k_20260501.yaml` |
| DS4 | `local_ds4_573bank_metricprobe64_fh16_aradd_mlp2_s128_lr2e3_unseeneval573_mrbayes20k_smallbank_20260502.yaml` |
| DS5 | `local_ds5_525bank_metricprobe64_fh16_aradd_mlp2_s128_lr2e3_unseeneval525_mrbayes20k_smallbank_20260502.yaml` |
| DS6 | `local_ds6_219bank_metricprobe64_fh16_aradd_mlp2_s128_lr2e3_unseeneval219_mrbayes20k_smallbank_20260502.yaml` |
| DS7 | `local_ds7_1344bank_metricprobe64_fh16_aradd_mlp2_s128_lr2e3_unseeneval1344_mrbayes20k_smallbank_20260502.yaml` |
| DS8 | `local_ds8_1122bank_metricprobe64_fh16_aradd_mlp2_s128_lr2e3_unseeneval1122_mrbayes20k_smallbank_20260502.yaml` |

### Phyla Embedding Banks

The analysis used the eight precomputed beta Phyla embedding files in:

`/home/unix/yektefai/PhylaFlow/analysis/full_sanity_fixedpair_20260401/ds_phyla_embeddings_20260428`

Leaf capacities:

| Embedding bank | Leaves |
|---|---:|
| DS1 | 27 |
| DS2 | 29 |
| DS3 | 36 |
| DS4 | 41 |
| DS5 | 50 |
| DS6 | 50 |
| DS7 | 59 |
| DS8 | 64 |

### Start-Topology Conditioning Encoders

The model uses frozen start-topology case tables. Therefore the analysis encoded the fixed unseen starts for each target with the same start-tree metric encoder family used in the sample-metrics harness.

The packaged artifact includes the eight target-specific metric encoders:

- `ds1_quick_metric_encoder_100step.pt`
- `ds2_metric_encoder_100step_20260501.pt`
- `ds3_metric_encoder_100step_20260501.pt`
- `ds4_metric_encoder_100step_20260501.pt`
- `ds5_metric_encoder_100step_20260501.pt`
- `ds6_quick_metric_encoder_100step.pt`
- `ds7_metric_encoder_100step_20260501.pt`
- `ds8_metric_encoder_100step_20260501.pt`

## Methods

### Runner

The analysis was implemented in:

`analysis/full_sanity_fixedpair_20260401/ds1ds8_phyla_embedding_swap_analysis.py`

The command used for the full sweep was:

```bash
cd /home/unix/yektefai/PhylaFlow
export CUDA_VISIBLE_DEVICES=3 PYTHONFAULTHANDLER=1
/ewsc/yektefai/envs/envs/pgt/bin/python -u \
  analysis/full_sanity_fixedpair_20260401/ds1ds8_phyla_embedding_swap_analysis.py \
  --num-pairs 32 \
  --device cuda
```

The run was launched in tmux session:

`ds1ds8_phyla_swap_20260505`

### Step-by-Step Procedure

For each target dataset DS1 through DS8:

1. Load the DS1-DS8 base config.
2. Overlay target-specific sample-metrics settings from the target sample config.
3. Load the same DS1-DS8 checkpoint at step 9500.
4. Build one fixed set of 32 unseen start/target tree pairs for the target dataset.
5. Save the fixed start/target pairs to `fixed_unseen_start_pairs.jsonl`.
6. Encode the 32 start trees once with the target start-tree metric encoder.
7. Replace the model frozen start-case tables with those 32 encoded start vectors.
8. For each condition embedding bank DS1 through DS8:
   - If the embedding bank has too few leaves for the target trees, record `skipped_insufficient_embedding_leaves`.
   - Otherwise, force `pair["dataset_id"]` to the condition embedding ID before sampling.
   - Sample one generated tree from each of the 32 fixed starts.
   - Dump generated trees under the target/condition `generated_trees/` folder.
   - Score generated trees against the target dataset's golden and short posterior references.
   - Save per-condition `summary.json`.
9. After all targets finish, write `summary_matrix.csv` and `summary_matrix.json`.

### Critical Control

Within a target row, the following were fixed:

- the trained checkpoint,
- the model weights,
- the 32 start trees,
- the 32 target trees,
- the frozen start-case conditioning table,
- the posterior reference used for scoring,
- the sampling harness settings.

The intended independent variable was only the Phyla embedding bank.

### Handling Different Taxon Sets

The DS datasets have different leaf counts and different biological sequence names. Because exact Newick starts cannot be shared across DS1, DS2, DS3, and so on, the analysis used fixed starts per target dataset.

For native conditions, biological names match the target embedding bank, so the native Phyla embedding lookup is biologically meaningful.

For off-diagonal conditions, biological sequence names generally do not match. The runner therefore falls back to numeric leaf-position lookup when forcing a different dataset bank. This makes the off-diagonal swap a conditioning stress test rather than a biological taxon-identity preserving experiment.

If a condition embedding bank had fewer leaves than the target tree, the cell was skipped. This is why the final matrix is upper-triangular with respect to leaf count.

### Output Artifacts

Each completed condition has:

- `summary.json`
- `generated_trees/step00000000_stepper00000001_train_trees.jsonl`
- `generated_trees/step00000000_stepper00000001_train_sampled_trees.txt`

Each target has:

- `effective_config.yaml`
- `fixed_unseen_start_pairs.jsonl`
- posterior reference cache files used by the scorer

The aggregate outputs are:

- `summary_matrix.csv`
- `summary_matrix.json`

## Results

The full sweep produced:

- 64 target/condition records
- 37 completed records
- 27 skipped records due to insufficient embedding-bank leaf capacity
- 74 generated-tree output files from completed cells

### Split-KL Matrix

Values are `golden_kl_divergence_topological`. The diagonal is the native condition. `skip` means the condition embedding bank had too few leaves for the target.

| Target \\ Embedding | DS1 | DS2 | DS3 | DS4 | DS5 | DS6 | DS7 | DS8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DS1 | 0.473 | 4.015 | 4.135 | 3.647 | 3.928 | 3.475 | 3.965 | 3.748 |
| DS2 | skip | 0.126 | 3.452 | 3.806 | 3.173 | 3.101 | 3.731 | 3.864 |
| DS3 | skip | skip | 0.087 | 3.839 | 3.595 | 3.748 | 3.507 | 3.526 |
| DS4 | skip | skip | skip | 0.392 | 4.021 | 3.539 | 3.724 | 3.834 |
| DS5 | skip | skip | skip | skip | 0.185 | 4.139 | 4.036 | 4.134 |
| DS6 | skip | skip | skip | skip | 4.212 | 0.176 | 3.623 | 3.995 |
| DS7 | skip | skip | skip | skip | skip | skip | 0.215 | 3.889 |
| DS8 | skip | skip | skip | skip | skip | skip | skip | 0.278 |

### Native Versus Best Valid Swap

| Target | Native split-KL | Native tree-topology KL | Best swap | Best swap split-KL | Best swap tree-topology KL | Split-KL gap |
|---|---:|---:|---|---:|---:|---:|
| DS1 | 0.472620 | 14.551 | DS6 | 3.475207 | 14.551 | 3.002587 |
| DS2 | 0.126370 | 6.358 | DS6 | 3.101321 | 15.897 | 2.974951 |
| DS3 | 0.087494 | 8.631 | DS7 | 3.506540 | 15.581 | 3.419046 |
| DS4 | 0.391699 | 14.115 | DS6 | 3.539359 | 14.157 | 3.147659 |
| DS5 | 0.185142 | 12.194 | DS7 | 4.036193 | 12.608 | 3.851052 |
| DS6 | 0.175638 | 14.929 | DS7 | 3.622534 | 14.929 | 3.446895 |
| DS7 | 0.215032 | 12.272 | DS8 | 3.889010 | 12.679 | 3.673978 |
| DS8 | 0.277910 | 12.850 | none valid | NA | NA | NA |

### Interpretation

The native condition is the best split-KL condition for every target. In the strongest rows, the split-KL gap from native to the best valid swap is about 3 to 4 nats.

This directly supports the basin-switch interpretation:

1. The same checkpoint can generate low split-KL trees for multiple target datasets when conditioned on the matching Phyla embeddings.
2. Holding target starts fixed and swapping only the Phyla embeddings causes the generated split distribution to move away from the target golden posterior.
3. The movement is large and systematic, not a small perturbation.
4. Therefore, the Phyla embeddings are acting as a meaningful control signal for topology generation.

The DS2 and DS3 rows are particularly clean because they were the original motivation:

- DS2 native split-KL: `0.126370`
- DS2 best valid swapped split-KL: `3.101321`
- DS3 native split-KL: `0.087494`
- DS3 best valid swapped split-KL: `3.506540`

### Important Caveats

The off-diagonal swaps are not taxon-identity preserving because the DS datasets do not share the same biological leaf sets. Off-diagonal results should therefore be described as a forced embedding-bank swap or conditioning stress test, not as a biological substitution experiment where the same taxa receive different sequence embeddings.

The start trees are fixed within each target row, but they are not identical across target rows. This is necessary because DS1, DS2, DS3, and the other DS datasets have different numbers of leaves and different labels.

Tree-topology KL is not always improved in the same way split-KL is. This checkpoint was selected for low DS2 split-KL, and the analysis is primarily about split-distribution basin movement.

## Reproduction

The packaged artifact contains all small-to-medium dependencies needed to rerun the analysis, including:

- checkpoint used for analysis,
- base DS1-DS8 config,
- target sample configs,
- Phyla embedding banks,
- start-tree metric encoders,
- runner script,
- full result tree,
- logs,
- checksums.

The posterior reference data and dataset pickle files are part of the broader `/ewsc/yektefai/30272299` dataset bundle and are referenced by path in the effective configs.

From the repo checkout:

```bash
cd /home/unix/yektefai/PhylaFlow
export CUDA_VISIBLE_DEVICES=<GPU> PYTHONFAULTHANDLER=1
/ewsc/yektefai/envs/envs/pgt/bin/python -u \
  analysis/full_sanity_fixedpair_20260401/ds1ds8_phyla_embedding_swap_analysis.py \
  --base-config /ewsc/yektefai/30272299/analysis/full_sanity_fixedpair_20260401/ds1ds8_step9500_phyla_embedding_swap_20260505/configs/base_ds1ds8_step9500_config.yaml \
  --checkpoint /ewsc/yektefai/30272299/analysis/full_sanity_fixedpair_20260401/ds1ds8_step9500_phyla_embedding_swap_20260505/checkpoint/ds1ds8_step009500_best_ds2_split_kl.ckpt \
  --num-pairs 32 \
  --device cuda
```

The script defaults to the original source paths. For a fully relocated rerun, update the default sample config map or pass equivalent sample configs that point to the packaged dependencies and the existing 302 posterior reference data.

## Local and GCS Locations

Local package:

`/ewsc/yektefai/30272299/analysis/full_sanity_fixedpair_20260401/ds1ds8_step9500_phyla_embedding_swap_20260505`

GCS package:

`gs://phyla/30272299/analysis/full_sanity_fixedpair_20260401/ds1ds8_step9500_phyla_embedding_swap_20260505/`
