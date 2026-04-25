# DS1/DS5 Repro Package, 2026-04-25

This package records the current reproducible path for the good DS1 result and
the current DS5 run. It is meant to be used from the repository root:

```bash
cd /home/yektefai/PhylaFlow
```

The training entry point is positional. Do not launch these configs with
`run.py` directly and do not pass `--config`.

```bash
python -m run.run /home/yektefai/PhylaFlow/configs/<config>.yaml
```

## Committed Files

Core code needed by the recipe:

- `data/dataset.py`
- `model/model.py`
- `run/TrainingModule.py`
- `run/run.py`
- `analysis/full_sanity_fixedpair_20260401/make_multi_singlepath_parity_bank.py`
- `analysis/full_sanity_fixedpair_20260401/make_singlepath_parity_case.py`

Training configs:

- `configs/ds1_short_multipair234_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptboth_6000_20260421.yaml`
- `configs/ds5_short_multipair525_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptfhonly_6000_currentrecipe_20260424.yaml`

Analysis scripts:

- `analysis/full_sanity_fixedpair_20260401/benchmark_ds_golden_posterior_ckpt.py`
- `analysis/full_sanity_fixedpair_20260401/benchmark_mrbayes_fixed_start_ds1.py`
- `analysis/full_sanity_fixedpair_20260401/benchmark_mrbayes_initializations_ds1.py`
- `analysis/full_sanity_fixedpair_20260401/run_ds1_mrbayes_generation_curves.py`
- `analysis/full_sanity_fixedpair_20260401/make_sampled_start_bank_from_ckpt.py`
- `analysis/full_sanity_fixedpair_20260401/remap_posterior_to_harness_lexindex.py`
- `analysis/full_sanity_fixedpair_20260401/build_weighted_fullsupport_ds_banks.py`
- `analysis/full_sanity_fixedpair_20260401/rebuild_bank_full_path_anchors.py`
- `analysis/full_sanity_fixedpair_20260401/build_ds1_usher_matopt_start.py`
- `analysis/full_sanity_fixedpair_20260401/build_ds1_usher_matopt_multistart.py`

Artifact manifest:

- `analysis/full_sanity_fixedpair_20260401/ds1_ds5_repro_artifacts_20260425.json`

The manifest summarizes the external artifact groups that should live on GCP:
training banks, checkpoints, exact MrBayes start/result artifacts, optional
ML/parsimony starts, and sampled-analysis JSONs. The configs currently use
absolute local paths under `/home/yektefai/PhylaFlow`; if the repo is checked out
elsewhere, rewrite those paths or restore artifacts to the same path.

## Regenerating Training Banks

The DS1-DS8 weighted full-support bank generator is:

```bash
python analysis/full_sanity_fixedpair_20260401/build_weighted_fullsupport_ds_banks.py
```

It calls `make_multi_singlepath_parity_bank.py` to write the per-case start,
target, manifest, and velocity-anchor JSONs. The DS1 `fullpathanchors4`
aggregate anchor file used by the current case-adapted recipe can be rebuilt
from an existing bank manifest with:

```bash
python analysis/full_sanity_fixedpair_20260401/rebuild_bank_full_path_anchors.py \
  --source-manifest analysis/full_sanity_fixedpair_20260401/ds1_short_multipair234_topofreqcover_discretephase_terminal_probeparity_wandbclean_6000_20260417_manifest.json \
  --source-config configs/ds1_short_multipair234_topofreqcover_discretephase_terminal_probeparity_wandbclean_6000_20260417.yaml \
  --output-name ds1_short_multipair234_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_20260418 \
  --full-path-anchor-count 4
```

## External Artifacts To Put On GCP

These are required by the configs but are intentionally treated as external
generated artifacts:

- DS1 caseadaptboth training bank: 469 files, 12.79 MB total.
  - One velocity-anchor JSON:
    `analysis/full_sanity_fixedpair_20260401/ds1_short_multipair234_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_20260418_velocity_anchors.json`
  - 234 start JSON files:
    `analysis/full_sanity_fixedpair_20260401/ds1_short_multipair234_topofreqcover_discretephase_terminal_probeparity_wandbclean_6000_20260417_caseXX_start.json`
  - 234 target JSON files:
    `analysis/full_sanity_fixedpair_20260401/ds1_short_multipair234_topofreqcover_discretephase_terminal_probeparity_wandbclean_6000_20260417_caseXX_target.json`

- DS5 current-recipe training bank: 1051 files, 51.93 MB total.
  - One velocity-anchor JSON:
    `analysis/full_sanity_fixedpair_20260401/ds5_short_multipair525_topofreqcover_discretephase_terminal_probeparity_wandbclean_6000_20260417_velocity_anchors.json`
  - 525 start JSON files:
    `analysis/full_sanity_fixedpair_20260401/ds5_short_multipair525_topofreqcover_discretephase_terminal_probeparity_wandbclean_6000_20260417_caseXXX_start.json`
  - 525 target JSON files:
    `analysis/full_sanity_fixedpair_20260401/ds5_short_multipair525_topofreqcover_discretephase_terminal_probeparity_wandbclean_6000_20260417_caseXXX_target.json`

MrBayes analysis also expects:

- `/home/yektefai/30272299/DS1.pickle`
- `/home/yektefai/30272299/golden_run_data_DS1-8/DS1/rep_*/DS1.trprobs`
- MrBayes binary at `/opt/conda/envs/phylaflow-mrbayes/bin/mb`, or pass
  `--mrbayes-bin` to the benchmark scripts.

The following generated artifacts are also intentionally kept on GCP rather
than committed:

- DS1/DS5 checkpoints listed below.
- DS1 full234 MrBayes exact start trees and result JSON/summary files:
  `analysis/full_sanity_fixedpair_20260401/ds1_mrbayes_generation_curves_100k_caseadaptboth_step22000_full234_20260425/`
- DS1 first78 historical MrBayes starts/results:
  `analysis/full_sanity_fixedpair_20260401/ds1_mrbayes_generation_curves_100k_caseadaptboth_step22000_20260425/`
- DS1 IQ-TREE ML and UShER/matOptimize baseline start artifacts:
  `analysis/full_sanity_fixedpair_20260401/ds1_iqtree_mfp_runs10_ml_20260424.treefile`,
  `analysis/full_sanity_fixedpair_20260401/ds1_usher_matopt_mp_20260424/`,
  and `analysis/full_sanity_fixedpair_20260401/ds1_usher_matopt_mp_multistart78_20260424/`
- DS1 step-22000 sampled-analysis JSONs:
  `analysis/full_sanity_fixedpair_20260401/ds1_caseadaptboth_step22000_*.json`

## DS1 Training Repro

Launch:

```bash
python -m run.run /home/yektefai/PhylaFlow/configs/ds1_short_multipair234_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptboth_6000_20260421.yaml
```

Important config settings:

- `model.first_hit_head_mode: case_adapted_mlp`
- `model.first_hit_head_num_cases: 234`
- `trainer.sample_metrics_num_pairs: 234`
- `trainer.training_sampling_frequency: 1000`
- `data.overfit_virtual_epoch_size: 234`
- `data.overfit_full_path_control_mode: true`

Metric trace:

- `analysis/full_sanity_fixedpair_20260401/ds1_short_multipair234_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptboth_6000_20260421_metrics.jsonl`

Key rows from the trace:

| step | mean normRF | tree KL | sampled unique topologies |
| ---: | ---: | ---: | ---: |
| 20000 | 0.169076 | 4.523037 | 63 |
| 21000 | 0.169504 | 3.292762 | 78 |
| 22000 | 0.169707 | 3.265710 | 72 |
| 23000 | 0.200582 | 3.899574 | 78 |
| 24000 | 0.133121 | 5.682637 | 48 |

For the MrBayes comparison below, use the step 22000 checkpoint because it had
the best tree-topology KL among the late DS1 checkpoints:

```text
checkpoints/full_sanity_fixedpair_20260401/ds1_short_multipair234_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptboth_6000_20260421/2026-04-21_02-28-40/epoch=94-step=022000.ckpt
```

The step 24000 checkpoint is also kept because it had the lowest mean normRF in
the trace:

```text
checkpoints/full_sanity_fixedpair_20260401/ds1_short_multipair234_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptboth_6000_20260421/2026-04-21_02-28-40/epoch=102-step=024000.ckpt
```

## Current DS5 Run

Launch:

```bash
python -m run.run /home/yektefai/PhylaFlow/configs/ds5_short_multipair525_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptfhonly_6000_currentrecipe_20260424.yaml
```

This is the current DS5 recipe that was running on 2026-04-25:

- W&B run id: `m9hr1xeo`
- checkpoint dir:
  `checkpoints/full_sanity_fixedpair_20260401/ds5_short_multipair525_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptfhonly_6000_currentrecipe_20260424/2026-04-24_23-44-20`
- metric trace:
  `analysis/full_sanity_fixedpair_20260401/ds5_short_multipair525_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptfhonly_6000_currentrecipe_20260424_metrics.jsonl`

Snapshot through step 5000:

| step | mean normRF | best normRF | tree KL | sampled unique topologies |
| ---: | ---: | ---: | ---: | ---: |
| 1000 | 0.898720 | 0.021277 | 15.405948 | 127 |
| 2000 | 0.607816 | 0.021277 | 15.090823 | 186 |
| 3000 | 0.620157 | 0.000000 | 14.657651 | 135 |
| 4000 | 0.698558 | 0.000000 | 13.017185 | 95 |
| 5000 | 0.672963 | 0.000000 | 14.700751 | 74 |

Step 5000 checkpoint:

```text
checkpoints/full_sanity_fixedpair_20260401/ds5_short_multipair525_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptfhonly_6000_currentrecipe_20260424/2026-04-24_23-44-20/epoch=09-step=005000.ckpt
```

## DS1 MrBayes Analysis

The paper-critical check is whether PhylaFlow starts reduce MrBayes burn-in
relative to random starts on the DS1 posterior topology distribution.

Restore the exact start trees used for the 234-chain comparison from GCP to:

```text
analysis/full_sanity_fixedpair_20260401/ds1_mrbayes_generation_curves_100k_caseadaptboth_step22000_full234_20260425/starts/
```

Run PhylaFlow full234:

```bash
python analysis/full_sanity_fixedpair_20260401/benchmark_mrbayes_fixed_start_ds1.py \
  --start-tree-list analysis/full_sanity_fixedpair_20260401/ds1_mrbayes_generation_curves_100k_caseadaptboth_step22000_full234_20260425/starts/phylaflow_sampled_caseadaptboth_step22000_full234_start_trees.txt \
  --label "PhylaFlow caseadaptboth step22000 full234" \
  --num-runs 234 \
  --ngen 100000 \
  --samplefreq 200 \
  --printfreq 5000 \
  --curve-interval 20000 \
  --max-workers 24 \
  --ds1-pickle /home/yektefai/30272299/DS1.pickle \
  --golden-root /home/yektefai/30272299/golden_run_data_DS1-8/DS1 \
  --mrbayes-bin /opt/conda/envs/phylaflow-mrbayes/bin/mb \
  --work-dir /tmp/mrbayes_ds1_caseadaptboth_step22000_full234/phylaflow \
  --output analysis/full_sanity_fixedpair_20260401/ds1_mrbayes_generation_curves_100k_caseadaptboth_step22000_full234_20260425/phylaflow_caseadaptboth_step22000_full234_g100000_curve.json
```

Run random full234:

```bash
python analysis/full_sanity_fixedpair_20260401/benchmark_mrbayes_fixed_start_ds1.py \
  --start-tree-list analysis/full_sanity_fixedpair_20260401/ds1_mrbayes_generation_curves_100k_caseadaptboth_step22000_full234_20260425/starts/random_original_caseadaptboth_step22000_full234_start_trees.txt \
  --label "Random original full234" \
  --num-runs 234 \
  --ngen 100000 \
  --samplefreq 200 \
  --printfreq 5000 \
  --curve-interval 20000 \
  --max-workers 24 \
  --ds1-pickle /home/yektefai/30272299/DS1.pickle \
  --golden-root /home/yektefai/30272299/golden_run_data_DS1-8/DS1 \
  --mrbayes-bin /opt/conda/envs/phylaflow-mrbayes/bin/mb \
  --work-dir /tmp/mrbayes_ds1_caseadaptboth_step22000_full234/random \
  --output analysis/full_sanity_fixedpair_20260401/ds1_mrbayes_generation_curves_100k_caseadaptboth_step22000_full234_20260425/random_original_full234_g100000_curve.json
```

Initial DS1 MrBayes result:

| method | KL@0 | KL@20000 | KL@40000 | KL@60000 | KL@80000 | KL@100000 | tail KL | support | first KL < 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PhylaFlow full234 | 3.265710 | 0.745528 | 0.555898 | 0.480639 | 0.436806 | 0.402686 | 0.305834 | 0.830356 | 11600 |
| Random full234 | 16.540891 | 0.960733 | 0.726032 | 0.601207 | 0.519239 | 0.452979 | 0.291147 | 0.741645 | 18000 |

This is the cleanest result to report right now: PhylaFlow starts are already
much closer at generation 0 and reach KL < 1 roughly 6400 generations earlier
than random in the equal-chain 234 vs 234 comparison.

The result files are also restored from GCP:

- `analysis/full_sanity_fixedpair_20260401/ds1_mrbayes_generation_curves_100k_caseadaptboth_step22000_full234_20260425/full234_random_vs_phylaflow_equal_chain_summary.md`
- `analysis/full_sanity_fixedpair_20260401/ds1_mrbayes_generation_curves_100k_caseadaptboth_step22000_full234_20260425/full234_vs_78_baselines_summary.md`
- `analysis/full_sanity_fixedpair_20260401/ds1_mrbayes_generation_curves_100k_caseadaptboth_step22000_full234_20260425/phylaflow_caseadaptboth_step22000_full234_g100000_curve.json`
- `analysis/full_sanity_fixedpair_20260401/ds1_mrbayes_generation_curves_100k_caseadaptboth_step22000_full234_20260425/random_original_full234_g100000_curve.json`

## Optional Baselines

The helper scripts for maximum parsimony style starts are included:

```bash
python analysis/full_sanity_fixedpair_20260401/build_ds1_usher_matopt_start.py
python analysis/full_sanity_fixedpair_20260401/build_ds1_usher_matopt_multistart.py
```

The UShER/matOptimize start artifacts used by the older 78-chain comparison are
kept on GCP:

- `analysis/full_sanity_fixedpair_20260401/ds1_usher_matopt_mp_20260424/matopt_optimized.nwk`
- `analysis/full_sanity_fixedpair_20260401/ds1_usher_matopt_mp_20260424/summary.json`
- `analysis/full_sanity_fixedpair_20260401/ds1_usher_matopt_mp_multistart78_20260424/start_trees.txt`
- `analysis/full_sanity_fixedpair_20260401/ds1_usher_matopt_mp_multistart78_20260424/summary.json`

The IQ-TREE ML start artifact is:

- `analysis/full_sanity_fixedpair_20260401/ds1_iqtree_mfp_runs10_ml_20260424.treefile`

## Sanity Checks

After restoring external artifacts, verify the high-level artifact groups from
the manifest:

```bash
python - <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("analysis/full_sanity_fixedpair_20260401/ds1_ds5_repro_artifacts_20260425.json").read_text()
)
for group in manifest["gcp_artifact_groups"]:
    files = []
    for pattern in group["patterns"]:
        files.extend(Path(".").glob(pattern))
    files = [path for path in files if path.is_file()]
    print(
        f"{group['name']}: found={len(files)} expected={group['local_file_count']}"
    )
PY
```

Compile-check the touched Python files:

```bash
python -m py_compile \
  data/dataset.py \
  model/model.py \
  run/TrainingModule.py \
  run/run.py \
  analysis/full_sanity_fixedpair_20260401/benchmark_ds_golden_posterior_ckpt.py \
  analysis/full_sanity_fixedpair_20260401/benchmark_mrbayes_fixed_start_ds1.py \
  analysis/full_sanity_fixedpair_20260401/benchmark_mrbayes_initializations_ds1.py
```
