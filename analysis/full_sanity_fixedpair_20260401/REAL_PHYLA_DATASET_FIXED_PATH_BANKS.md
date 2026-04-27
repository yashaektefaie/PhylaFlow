# Real PhylaFlow Dataset Fixed-Path Banks

This runbook is for generating fixed start/target case banks from the real
`phylaflow_datasets` corpus:

```text
/home/yektefai/phylaflow_datasets/nexus/<dataset_id>.nex
/home/yektefai/phylaflow_datasets/runs/<dataset_id>/*.t
/home/yektefai/phylaflow_datasets/runs/<dataset_id>/*.mcmc
```

Run from the PhylaFlow repo root:

```bash
cd /home/yektefai/PhylaFlow
```

## Script

```bash
python analysis/full_sanity_fixedpair_20260401/build_real_fixed_path_bank.py --help
```

The script writes the same artifact shape used by the DS fixed-path recipe:

- one start-tree JSON per case
- one target-tree JSON per case
- one combined full-path velocity-anchor JSON
- one manifest with config fields for training

It checks the final MrBayes `AvgStdDev(s)` from `.mcmc` logs by default. The
recommended threshold for this corpus is:

```bash
--asdsf-threshold 0.05
```

Do not use `--max-posterior-trees` for production preprocessing. The default is
no cap, matching the DS preprocessing behavior. That flag exists only for quick
debug runs.

## Recommended Real-Corpus Mode

For the real corpus, generate one case per unique posterior topology and do
posterior-frequency oversampling later from the manifest metadata:

```bash
DATASET_ID=10816_NT_AL
OUT=/path/to/real_fixed_path_banks

python analysis/full_sanity_fixedpair_20260401/build_real_fixed_path_bank.py \
  --dataset-id "$DATASET_ID" \
  --output-root "$OUT" \
  --target-schedule unique-topologies \
  --asdsf-threshold 0.05 \
  --full-path-anchor-count 4 \
  --min-boundary-paths 3
```

If `--num-cases` is omitted in `unique-topologies` mode, the script uses every
unique posterior topology exactly once.

Each case in the manifest records:

- `topology_count`: number of post-burn-in posterior samples with that topology
- `topology_probability`: `topology_count / posterior_sample_count`
- `posterior_index`: representative posterior tree used as the target

Those fields are enough to oversample cases posthoc and recover the posterior
topology distribution.

## DS-Style Weighted Mode

To match the DS bank construction more closely, use weighted full-support
allocation:

```bash
DATASET_ID=10816_NT_AL
OUT=/path/to/real_fixed_path_banks

python analysis/full_sanity_fixedpair_20260401/build_real_fixed_path_bank.py \
  --dataset-id "$DATASET_ID" \
  --output-root "$OUT" \
  --target-schedule weighted-topologies \
  --support-multiplier 3 \
  --asdsf-threshold 0.05 \
  --full-path-anchor-count 4 \
  --min-boundary-paths 3
```

If `--num-cases` is omitted in `weighted-topologies` mode, the script uses:

```text
num_cases = unique_topology_count * support_multiplier
```

When `num_cases >= unique_topology_count`, every unique topology is represented
at least once, then remaining cases are allocated in proportion to posterior
frequency.

## Outputs

For a bank named `real_<dataset>_<mode>_fixedpath_auto`, outputs are written
under:

```text
<output-root>/real_<dataset>_<mode>_fixedpath_auto/
```

Expected files:

```text
*_case00_start.json
*_case00_target.json
...
*_velocity_anchors.json
*_manifest.json
```

The manifest includes:

```json
"training_config_fields": {
  "data.overfit_fixed_pair_start_tree_json_paths": ["..."],
  "data.overfit_fixed_pair_target_tree_json_paths": ["..."],
  "data.overfit_full_path_control_extra_velocity_samples_json_path": "...",
  "data.overfit_virtual_epoch_size": 28,
  "trainer.sample_metrics_num_pairs": 28,
  "model.first_hit_head_num_cases": 28
}
```

For training, copy these fields into the fixed-path recipe config. The three
case-count fields must all match the number of start/target pairs.

## Slurm Array Shape

Create a file with one accepted dataset id per line, for example
`datasets_asdsf005.txt`. Then use one Slurm array task per dataset:

```bash
#!/bin/bash
#SBATCH --job-name=real-phylaflow-bank
#SBATCH --array=0-937%100
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/real_bank_%A_%a.out
#SBATCH --error=logs/real_bank_%A_%a.err

set -euo pipefail

cd /home/yektefai/PhylaFlow

DATASET_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" datasets_asdsf005.txt)
OUT=/path/to/real_fixed_path_banks

python analysis/full_sanity_fixedpair_20260401/build_real_fixed_path_bank.py \
  --dataset-id "$DATASET_ID" \
  --output-root "$OUT" \
  --target-schedule unique-topologies \
  --asdsf-threshold 0.05 \
  --full-path-anchor-count 4 \
  --min-boundary-paths 3
```

Adjust `--array=0-N%100` to match the dataset-list length and cluster limit.

If the dataset directory is not at `/home/yektefai/phylaflow_datasets`, pass:

```bash
--nexus-root /path/to/phylaflow_datasets/nexus \
--mrbayes-root /path/to/phylaflow_datasets/runs
```

## Smoke Result

The wrapper was smoke-tested on `10816_NT_AL`:

```text
ASDSF max:                    0.015503
post-burn-in posterior trees: 340
unique posterior topologies:  28
unique-topology cases:        28
```

The smoke command was:

```bash
python analysis/full_sanity_fixedpair_20260401/build_real_fixed_path_bank.py \
  --dataset-id 10816_NT_AL \
  --output-root analysis/full_sanity_fixedpair_20260401/real_fixed_path_bank_smoke \
  --target-schedule unique-topologies \
  --min-boundary-paths 3 \
  --full-path-anchor-count 4 \
  --bank-name smoke_10816_NT_AL_unique_fixedpath
```

## Common Failures

- ASDSF failure: the dataset did not pass `--asdsf-threshold`; skip it unless
  intentionally relaxing the filter.
- Pair-selection failure: increase `--max-start-tries-per-target`; for very
  small datasets, lowering `--min-boundary-paths` is possible but changes the
  recipe assumption.
- Slow topology counting: this is expected for large diffuse posterior samples.
  Run on the cluster; avoid `--max-posterior-trees` for production.
