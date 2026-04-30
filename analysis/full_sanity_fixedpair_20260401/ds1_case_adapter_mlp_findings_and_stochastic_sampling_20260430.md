# DS1 Case Adapter MLP Findings And Stochastic Sampling Notes

Date: 2026-04-30

This memo summarizes what we found about DS1 case conditioning, frozen start-tree embeddings, metric/PhyloVAE start embeddings, and why stochastic decoding is now the most likely next angle to test.

## Short Version

The strongest DS1 in-bank result still comes from case-conditioned training, but the evidence now suggests this is not a clean start-tree representation. The case signal appears to route the model toward posterior/end-topology behavior tied to a fixed start-target/path bank.

The frozen probe table is less obviously bad than a trainable `nn.Embedding`, because it is computed from the start tree, but its pretraining objective was case classification over the same 234 DS1 starts. That makes it a strong case-identity proxy.

The metric-probe table is cleaner: it was pretrained from sampled random trees using tree metric objectives, not DS1 target labels. It also genuinely started to work in-bank: the 234-case metric-probe MLP2 run reached tree KL `1.7138` live at step `8000`. But the only saved checkpoint available for the unseen-start diagnostic was the earlier step-5000 checkpoint, where in-bank tree KL was about `2.29`; under unseen random starts with freshly generated metric embeddings it degraded to about `4.98` tree KL. So the metric approach is promising, but the existing generalization test is not a clean test of the best live state.

The current sampler is mostly deterministic at inference. Larger start-target banks help reduce small-bank memorization, but without stochastic decoding, one fixed conditioning input still produces one greedy trajectory/mode.

## Important Runs And Artifacts

### Best case-conditioned DS1 checkpoint

- Local checkpoint: `/home/yektefai/PhylaFlow/checkpoints/gcs_30272299/DS1/best_04_27/best_topological_kl_step031000.ckpt`
- Config: `/home/yektefai/PhylaFlow/configs/ds1_short_multipair234_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptboth_arref1_arcase_6000_currentrecipe_20260426.yaml`
- In-bank generated-tree metric summary: `/home/yektefai/PhylaFlow/analysis/full_sanity_fixedpair_20260401/ds1_arcase_best_topokl_step031000_checkpoint_samples_20260427/ds1_arcase_best_topokl_step031000_metric_summary.json`
- In-bank tree KL: `0.762408`
- This is the strongest DS1 number we have, but it uses case conditioning tied to the fixed DS1 bank.

### Frozen-probe / frozen case-table run

- W&B run: `ds1_frozenprobe64_fh16_aradd_s128_lr2e3_local`
- W&B id: `ct617nwy`
- Config: `/home/yektefai/PhylaFlow/configs/local_ds1_frozenprobe64_fh16_aradd_scale128x4_lr2e3_20260428.yaml`
- Frozen table: `/home/yektefai/PhylaFlow/artifacts/start_case_probe/ds1_start_case_probe_emb64_20260428.pt`
- Best logged tree KL: `1.025789` at step `29000`
- Best saved local checkpoint found: step `25000`, tree KL `1.252486`
- Saved checkpoint: `/home/yektefai/PhylaFlow/checkpoints/full_sanity_fixedpair_20260401/ds1_frozenprobe64_fh16_aradd_s128_lr2e3_local/2026-04-28_23-56-59/epoch=106-step=025000.ckpt`

This run replaced trainable case IDs with a frozen `234 x 64` start-tree table. The table came from a probe trained to classify the DS1 start tree from split-mask features. The probe reached 100% start-case classification accuracy. The frozen table was injected through two MLP adapters:

- First-hit: `64 -> 128 -> 16`, concatenated into first-hit edge representations.
- AR: `64 -> 256 -> 128`, added into the AR representation.

The table itself was frozen during PhylaFlow training. Only the adapters trained.

### Metric-probe run

- W&B/run name: `ds1_metricprobe64_fh16_aradd_mlp2_s128_lr2e3_local`
- Config: `/home/yektefai/PhylaFlow/configs/local_ds1_metricprobe64_fh16_aradd_mlp2_s128_lr2e3_20260429.yaml`
- Frozen table: `/home/yektefai/PhylaFlow/artifacts/start_tree_metric_encoder/ds1_quick_metric_start_table_100step.pt`
- Source encoder checkpoint: `/home/yektefai/PhylaFlow/artifacts/start_tree_metric_encoder/ds1_quick_metric_encoder_100step.pt`
- Best tree KL seen in local metrics: `1.713830` at step `8000`
- Best saved checkpoint found locally: step `5000`
- Step-5000 checkpoint: `/home/yektefai/PhylaFlow/checkpoints/full_sanity_fixedpair_20260401/ds1_metricprobe64_fh16_aradd_mlp2_s128_lr2e3_local/2026-04-29_20-46-33/epoch=21-step=005000.ckpt`

This table is cleaner than the frozen case-classification table. It was pretrained with random prior trees and tree-geometry targets:

- sizes: `[8, 12, 16, 24, 32, 50]`
- objective: normalized RF/cosine target, distance regression, RF-bin CE, VICReg-style regularization
- no DS1 target tree supervision
- no DS1 case-label CE

Concretely, the metric probe is a split-set encoder trained by `scripts/pretrain_start_tree_metric_encoder.py`.

Input representation:

- Parse each tree with ETE.
- Convert every internal bipartition into a canonical split bitmask.
- Represent each split as a fixed-width bit vector with `max_bits=64`.
- Add two size features: normalized taxon count and normalized log taxon count.

Encoder architecture:

- Per-split MLP: `LayerNorm(64) -> Linear(64, 128) -> GELU -> Linear(128, 128) -> GELU`.
- Pool split embeddings with sum, mean, and max.
- Encode size features with a small MLP.
- Concatenate `[pooled_sum, pooled_mean, pooled_max, size_embedding]`.
- Final tree MLP maps this to a `64D` tree embedding, followed by `LayerNorm`.
- Exported tables are L2-normalized, so each row has norm about `1.0`.

Training objective:

- Sample tree pairs from the known random-start prior, not from DS1 endpoint labels.
- Pair modes in the 100-step run:
  - identical tree with probability `0.10`
  - label-swap perturbation with probability `0.40`
  - independent prior sample otherwise
- Compute normalized RF distance:

```text
d_norm = RF(T_i, T_j) / max_RF(n_taxa)
```

- Train pair heads to predict:
  - cosine similarity target `exp(-d_norm / tau)` with `tau=0.25`
  - normalized RF distance by MSE
  - RF distance bin by cross-entropy, thresholds `[0.05, 0.15, 0.30, 0.50, 0.75]`
- Add VICReg-style variance/covariance regularization to discourage collapse.

The quick metric encoder was intentionally small/cheap:

- steps: `100`
- batch size: `32` pairs
- hidden dim: `128`
- embedding dim: `64`
- final distance Pearson: `0.926692`
- final cosine-target Pearson: `0.718582`
- final RF-bin accuracy: `0.8125`

What the exported tables looked like:

- 234-case DS1 table: shape `(234, 64)`, norm mean `1.0`, effective rank `21.945`, top-PC variance `0.1374`, off-diagonal cosine mean `0.1874`, min `-0.6012`, max `0.8821`.
- 1280-case DS1 table: shape `(1280, 64)`, norm mean `1.0`, effective rank `23.358`, top-PC variance `0.1375`, off-diagonal cosine mean `0.1958`, min `-0.6637`, max `0.9207`.

This is why the metric probe looked better than the case-classification probe on paper: the embeddings were not one-hot case IDs, were not trained against DS1 targets, had nontrivial effective rank, and encoded smooth RF geometry. The failure mode is downstream: after freezing this geometry table and training PhylaFlow on a fixed 234-row bank, the model can still learn a smooth embedding-to-target routing function.

However, once exported as a fixed `234 x 64` table and used with the 234 DS1 training cases, downstream PhylaFlow can still learn a metric-space routing map.

Important caveat: the later live state looked substantially better than the checkpoint we were able to test. The local metrics were:

- step `5000`: in-bank golden tree KL `2.292349`
- step `7000`: in-bank golden tree KL `2.135743`
- step `8000`: in-bank golden tree KL `1.713830`

But only the step-5000 checkpoint was saved locally. The unseen-start diagnostic therefore loaded the step-5000 checkpoint, generated new random DS1 start trees, encoded those starts with the metric encoder, temporarily replaced the frozen table rows with those generated embeddings, and evaluated the sampled endpoints. That test measured:

- 64 unseen starts
- generated embedding effective rank `18.801`
- generated embedding off-diagonal cosine mean `0.1745`
- unseen golden tree KL `4.985922`
- unseen short tree KL `4.979824`
- unseen RF mean `0.582585`
- unseen sampled topology unique count `52`

So the metric approach did "sort of work": it learned an in-bank route that got down into the low-2s and then `1.71` tree KL live. The negative result is narrower: an earlier `2.29` checkpoint degraded to roughly `5` tree KL on unseen random starts. We should not claim the best live metric-probe state generalizes or fails until we have a saved checkpoint from the `1.71` regime and run the same unseen-start diagnostic on that exact checkpoint.

### True non-case baseline

If we mean no case/start embedding conditioning at all, the best DS1 local result found was:

- Metrics: `/home/yektefai/PhylaFlow/analysis/full_sanity_fixedpair_20260401/ds1_short_multipair234_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_6000_20260417_metrics.jsonl`
- Checkpoint: `/home/yektefai/PhylaFlow/checkpoints/full_sanity_fixedpair_20260401/ds1_short_multipair234_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_6000_20260417/2026-04-17_22-53-17/epoch=114-step=026750.ckpt`
- Best tree KL: `3.804201` at step `26750`

So the rough gap is:

- case-conditioned best: about `0.76`
- frozen case-probe best logged: about `1.03`
- metric-probe best live metric: about `1.71`
- metric-probe saved checkpoint tested on unseen starts: in-bank about `2.29`, unseen about `4.98`
- no case/start conditioning: about `3.80`

## Generalization Diagnostics

We tested unseen random DS1 start trees with the best case-conditioned checkpoint.

Script:

- `/home/yektefai/PhylaFlow/analysis/full_sanity_fixedpair_20260401/eval_ds1_unseen_start_generalization_20260429.py`

### Unseen random starts + random seen case IDs

Output:

- `/home/yektefai/PhylaFlow/analysis/full_sanity_fixedpair_20260401/ds1_unseen_start_generalization_20260429/summary.json`

Result:

- tree KL: `1.968812`
- topological KL: `0.044711`
- mean RF: `0.589312`
- support recall: `0.430380`
- shared golden topologies: `34`

This was better than expected but still not true extrapolation, because the case IDs were sampled from the seen 234-case vocabulary.

### Unseen random starts + fixed case ID 0 for all starts

Output:

- `/home/yektefai/PhylaFlow/analysis/full_sanity_fixedpair_20260401/ds1_unseen_start_fixedcase000_20260430/summary.json`

Result:

- tree KL: `8.176177`
- topological KL: `0.099535`
- mean RF: `0.293091`
- support recall: `0.075949`
- shared golden topologies: `6`

This is the key diagnostic. Fixing the case ID destroys posterior support even though mean RF is not terrible. The case signal appears to route posterior/topology family, not merely represent a start tree.

## Interpretation

The case adapter MLP likely works because it gives the model a stable, high-separation identifier for the fixed DS1 case/path. In this setup, case identity is tied to a specific start-target/path pair, so it can indirectly encode where the trajectory should end.

The frozen probe table reduces trainable memorization, but because its pretraining objective was classifying the same 234 starts, it remains a strong case-identity representation.

The metric-probe table is the most defensible of the embedding attempts because its pretraining did not use endpoint/topology labels. But downstream training can still use it as a smooth lookup key:

`start-tree metric embedding -> local DS1 target/path behavior`

This is not target leakage from the embedding pretraining. It is ordinary overfitting/routing by the downstream model.

## Larger DS1 Banks

Larger DS1 banks do exist locally:

- `/home/yektefai/PhylaFlow/analysis/full_sanity_fixedpair_20260401/ds1_multipair1280_topofreqcover_base256x5_20260429_manifest.json`
- `/home/yektefai/PhylaFlow/analysis/full_sanity_fixedpair_20260401/ds1_multipair1280_topofreqcover_base256x5_20260429_velocity_anchors.json`
- `/home/yektefai/PhylaFlow/analysis/full_sanity_fixedpair_20260401/ds1_multipair256_topofreqcover_base256_20260429_manifest.json`
- `/home/yektefai/PhylaFlow/analysis/full_sanity_fixedpair_20260401/ds1_multipair256_topofreqcover_base256_20260429_velocity_anchors.json`

Counts found:

- `ds1_multipair1280_topofreqcover_base256x5_20260429`: 1280 starts, 1280 targets, 1280 per-case anchor files, plus combined anchor JSON.
- `ds1_multipair256_topofreqcover_base256_20260429`: 256 starts, 256 targets, 256 per-case anchor files, plus combined anchor JSON.
- Older `ds1_short_multipair10000_..._1epoch_20260417` prefix appears partial locally: 427 starts, 427 targets, 427 anchor files.

The larger bank should help reduce small-bank memorization, especially without trainable case IDs. But by itself it does not make inference stochastic.

## Current Stochasticity Assessment

The current DS1 rollout path is mostly deterministic at inference:

- First-hit edge logits are thresholded; if none pass threshold, the best logit is selected.
- Velocity boundary step size is deterministically computed from predicted first-hit edges.
- AR merge planning uses argmax for starter pair and subset size.
- The rollout applies the top planned merge only.

Randomness currently enters mostly through training/evaluation data selection:

- sampling which bank pair/start-target pair to train/evaluate
- sampling which event along a fixed path is supervised
- replay/anchor subsampling in some paths

For a fixed checkpoint, fixed start tree, and fixed conditioning embedding, the current sampler is much closer to a greedy MAP decoder than a posterior sampler.

## Working Hypothesis

Large bank plus deterministic decoding can learn a better deterministic transport field. It may improve RF and maybe some KL, but it cannot represent multiple posterior modes for the same conditioning input unless the conditioning itself changes.

Large bank plus stochastic decoding is the cleaner next experiment. The architecture already emits logits. The minimal test is to sample from those logits instead of always thresholding/argmaxing them.

## Detailed Stochastic Sampling Investigation

This section records the concrete sampler path inspection from 2026-04-30.

### Deterministic decode points

Current inference is deterministic in the places that matter for topology choice.

- First-hit selection is in `run/TrainingModule.py:859`. `_predict_first_hit_mask_from_logits` chooses every candidate edge with logit above threshold. If none pass, it chooses the single argmax candidate. The wrapper at `run/TrainingModule.py:884` currently calls the raw selector and returns the raw mask.
- The discrete-phase rollout calls this selector at `run/TrainingModule.py:5541-5557`. The main `sample()` path calls it at `run/TrainingModule.py:14048` and again at `run/TrainingModule.py:14143`.
- The velocity boundary step is deterministic. `dt_target` is computed from the selected collapsing edges at `run/TrainingModule.py:5573-5585`.
- Structured AR merge subset size uses argmax at `run/TrainingModule.py:1441`.
- Structured AR starter pair uses argmax at `run/TrainingModule.py:1539`.
- Structured AR extra members are chosen by sorting member logits and taking the top members at `run/TrainingModule.py:1471-1481`.
- The AR planner calls `_decode_structured_merge_subset` at `run/TrainingModule.py:1921-1925`.
- Both rollout paths keep only the first planned AR merge: `run/TrainingModule.py:5654-5655` and `run/TrainingModule.py:14650-14651`.

The module has sampling knobs, but they are not posterior sampling knobs:

- `velocity_first_hit_sampling_max_edges`
- `velocity_first_hit_sampling_fallback_threshold`
- `velocity_first_hit_sampling_fallback_top_k`
- `sampling_use_top_merge_planner`

These control truncation/fallback/top-merge behavior. They do not provide temperature, categorical sampling, Bernoulli sampling, or stochastic AR pair/size/member selection.

### Existing stochastic scripts

Two existing analysis scripts try stochastic variants:

- `analysis/full_sanity_fixedpair_20260401/benchmark_ds_pairconditioned_stochastic_ar.py`
- `analysis/full_sanity_fixedpair_20260401/benchmark_ds_pairconditioned_stochastic_firsthit.py`

The AR script monkeypatches `_plan_autoregressive_boundary_merges`. It can softmax-sample between planner candidates and it has random-pair perturbation modes. But for the current `structured_subset` decoder, the normal path still calls `_decode_structured_merge_subset`, which already took the argmax pair and argmax size. Therefore temperature-only AR sampling can still have no real diversity.

The first-hit script monkeypatches `_predict_first_hit_mask_with_fallback`. It first runs the deterministic threshold/argmax selector, then subsamples the active predicted first-hit set. That is a crude perturbation, not a calibrated sampler over all candidate edges.

### Old stochastic outputs

Existing outputs are old and should not be treated as evidence for the current best DS1 checkpoint:

- `analysis/full_sanity_fixedpair_20260401/ds1_pairconditioned_stochastic_ar_*.json`
- `analysis/full_sanity_fixedpair_20260401/ds1_pairconditioned_stochastic_firsthit_*.json`

They used:

- Config: `analysis/full_sanity_fixedpair_20260401/randomstart50_posteriorbank_hybridbank_predsimoverrun_replayall_DS1_20260413_nowandb.yaml`
- Checkpoint: `checkpoints/full_sanity_fixedpair_20260401/randomstart50_posteriorbank_hybridbank_predsimoverrun_replayall_DS1_20260413_nowandb/2026-04-13_19-57-28/epoch=1199-step=002400.ckpt`

Summary:

- AR temperature/random-every-N runs usually had `unique_sampled_topologies: 1`, even when target topologies varied. That is mode collapse, not useful posterior sampling.
- First-hit subsampling produced more unique sampled topologies, but it destroyed posterior support. Examples:
  - `ds1_pairconditioned_stochastic_firsthit_k1_t10_16_20260414.json`: tree KL about `13.82`, support recall about `0.013`, mean final RF about `0.922`.
  - `ds1_pairconditioned_stochastic_firsthit_k2_t10_12_20260414.json`: tree KL about `13.56`, support recall about `0.013`, mean final RF about `0.927`.

Interpretation: naive perturbation can create diversity, but mostly creates wrong trees. The useful test is not "add noise anywhere"; it is "sample from the same distributions the model was trained to predict, before greedy collapse."

### Current-checkpoint smoke attempt

I created a diagnostic local-path config copy:

- `analysis/full_sanity_fixedpair_20260401/ds1_arcase_best_20260426_localpaths_for_stochastic_probe.yaml`

It is a copy of:

- `configs/ds1_short_multipair234_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptboth_arref1_arcase_6000_currentrecipe_20260426.yaml`

with the gcloud prefix replaced:

- from `/n/holylfs06/LABS/mzitnik_lab/Users/yektefaie/DS_data/30272299`
- to `/home/yektefai/30272299`

Then I tried a tiny current-checkpoint AR stochastic smoke:

```bash
python analysis/full_sanity_fixedpair_20260401/benchmark_ds_pairconditioned_stochastic_ar.py \
  --config analysis/full_sanity_fixedpair_20260401/ds1_arcase_best_20260426_localpaths_for_stochastic_probe.yaml \
  --checkpoint checkpoints/gcs_30272299/DS1/best_04_27/best_topological_kl_step031000.ckpt \
  --num-trials 8 \
  --temperature 1.0 \
  --seed 20260430 \
  --device cuda \
  --output analysis/full_sanity_fixedpair_20260401/ds1_arcase031000_stochastic_ar_t10_8_20260430.json
```

The smoke failed before sampling:

```text
KeyError: "Leaf '27' is missing from the harness ordering map."
```

This is a harness/parity issue, not a stochastic model result. The current best config's local start tree includes leaf `27`, while the benchmark's posterior reference ordering map built from the default DS1 posterior root did not. Before trusting any KL/RF result here, the benchmark needs the same leaf-label space as the checkpoint/config artifacts.

### What the stochastic angle means

The first useful change is sampling behavior, not architecture. The model already emits logits for first-hit and AR decisions. We need a config-gated stochastic decoder that preserves deterministic defaults.

Recommended config additions:

- `sampling_stochastic_enabled: false`
- `sampling_rng_seed`
- `sampling_first_hit_mode`: `threshold_argmax`, `bernoulli`, `categorical_k`, `topk_without_replacement`
- `sampling_first_hit_temperature`
- `sampling_first_hit_k`
- `sampling_ar_mode`: `argmax`, `categorical_pair`, `categorical_pair_and_size`, `categorical_pair_size_members`
- `sampling_ar_temperature`
- `sampling_ar_top_k`

For first-hit:

- `threshold_argmax` should remain the default.
- `bernoulli` would sample candidate edges from sigmoid logits, with safeguards for at least one negative-velocity edge.
- `categorical_k` or `topk_without_replacement` would sample a fixed or predicted number of first-hit edges from candidate logits.

For structured AR:

- sample starter pair from `starter_pair_logits` before argmax;
- sample subset size from `subset_size_logits` before argmax;
- optionally sample extra members from `member_logits` instead of always taking sorted top members;
- keep invalid split/new-split filtering exactly as the deterministic path does.

### Larger bank implication

The 1280-case DS1 bank is still useful, but it answers a different question.

Large bank training can reduce memorization of 78 or 234 fixed routes, especially if case IDs are removed or replaced with start-derived embeddings. But if the sampler remains greedy, each fixed start/conditioning input still maps to one dominant trajectory. That can improve deterministic transport, but it will not by itself become a posterior sampler.

The clean experiment is:

1. Add config-gated stochastic decoding with deterministic defaults.
2. Fix the benchmark leaf-label/parity issue for the current `0.762` checkpoint.
3. Compare deterministic vs stochastic decoding on the same checkpoint and same sampler path.
4. If logits are too sharp or stochastic samples are poor, train on the 1280 bank with stochastic-compatible objectives or regularization.

So the immediate change is sampling, not architecture. If stochastic decoding exposes that the learned logits are too sharp or not calibrated, then the next training change is loss/calibration, not another embedding table.
