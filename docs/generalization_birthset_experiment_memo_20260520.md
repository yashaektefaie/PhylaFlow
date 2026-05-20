# Generalization and Birthset Decoder Experiment Memo - 2026-05-20

## Summary

This note summarizes the recent PhylaFlow generalization work on
`yasha-dev-newar`. The main arc was:

1. OrthoMaM-to-DS2 transfer with the older topology decoder did not generalize.
2. Matching the leaf count and moving to 10-leaf/29-leaf sanity settings did not
   by itself solve the problem.
3. The sequential AR topology resolver was replaced with a one-shot birth-split
   decoder, which fixed the sampling bottleneck and can overfit DS2.
4. The 29-leaf OrthoMaM generalization runs show some within-OrthoMaM signal,
   but still do not transfer to DS2.
5. Diagnostics suggest the model leans heavily on an OrthoMaM positional/topology
   prior. The newest run adds direct component-level Phyla conditioning to the
   birthset heads so candidate split scores depend on the sequences occupying a
   split, not only the split identity.

The current active run is:

```text
W&B: https://wandb.ai/yasha/orthomam/runs/l3fbbhau
Config: configs/orthomam29leaf_fixedpair_c210_birthset_top32_livephyla_rawfull_unfrozen_cap80k_2gpu_phylacond_s128_lr1e3_holdout1000_20260520.yaml
Log: logs/orthomam29leaf_livephyla_unfrozen_cap80k_2gpu_phylacond_20260520_gpu0_gpu1.log
PID: 3559627
GPUs: 0/1
```

## Core Question

The scientific question is whether PhylaFlow can learn a reusable posterior
transport rule:

```text
random start tree + sequence context -> posterior-like tree distribution
```

The DS1-DS8 experiments showed that conditioning can work in related settings.
The failure mode here is that OrthoMaM pretraining improves OrthoMaM-like
heldout metrics but does not produce useful DS2 transfer. In some runs, applying
the OrthoMaM-trained model to DS2 made the output actively worse than random
starts.

## Dataset Setups

### DS2

DS2 has 29 leaves in the current experiments. Its posterior is concentrated:

- top posterior topology mass: about `0.5235`
- effective Simpson topology count: about `3.0`
- strong splits with posterior support >= 0.99: `24/26`

DS2 sequence/domain diagnostics:

- DS2 overlaps the selected OrthoMaM taxa only by `Homo_sapiens`; `28/29` DS2
  taxa are absent from the OrthoMaM selected taxa.
- DS2 median pairwise identity: `0.786`
- OrthoMaM truncated 29-leaf median pairwise identity: `0.883`
- DS2 is at about the 7th percentile among OrthoMaM medians.
- DS2 gap fraction: `30.5%`; OrthoMaM median gap fraction: `6.1%`.

In frozen Phyla embedding space, DS2 was not extreme out-of-domain:

- DS2 to OrthoMaM centroid distance median: `0.0254`
- OrthoMaM to OrthoMaM centroid distance median: `0.0161`
- OrthoMaM to OrthoMaM p90: `0.0531`

So DS2 is sequence-domain shifted, but not obviously impossible in the frozen
embedding geometry.

### OrthoMaM 29-Leaf Fixed-Pair Bank

The current 29-leaf OrthoMaM bank is:

```text
/ewsc/yektefai/phylaflow/orthomam_subset_banks/orthomam29leaf_fixedpair_c210_seed20260518_livephyla_cap80k/
```

It contains:

- train: `785` dataset IDs, `164850` rows
- test/heldout: `200` dataset IDs, `42000` rows
- fixed pairs per dataset: `210`

The active training configs use:

```yaml
batch_size: 4
num_workers: 8
same_dataset_batch: true
same_dataset_batches_per_epoch: 100000000
train_all_configured_posterior_dataset_ids: true
train_from_test_ids: false
posterior_subset_max_input_tokens: 80000
overfit_fixed_pair_joint_bank_jsonl_path: ...train.jsonl
```

The eval samples 1000 heldout pairs/starts.

## Earlier Generalization Runs

### 10-Leaf OrthoMaM Sanity

We moved to 10-leaf OrthoMaM subsets to ask whether the model could generalize
in a much smaller topology space. The setup was:

- sample 10-leaf subsets from OrthoMaM posterior datasets
- train on a subset of OrthoMaM dataset IDs
- hold out dataset IDs
- use Phyla embeddings
- measure RF and topology KL on heldout 10-leaf OrthoMaM posteriors

Result:

- no convincing heldout RF generalization
- topology KL moved somewhat, but exact topology/support recall stayed weak
- a misleading `rf_norm_best = 0` observation was traced to metric semantics,
  not evidence of real posterior recovery

Interpretation:

The model could learn distributional/topological entropy structure before it
could learn dataset-specific clade structure. This pushed us away from "leaf
count alone explains the failure".

### Live Phyla Unfreezing

We tested live Phyla-beta end-to-end training rather than frozen/precomputed
embeddings. The live path:

- strips gaps from raw sequences
- enforces a token cap
- runs Phyla-beta inside the training step
- optionally unfreezes Phyla

Important implementation detail:

- `live_phyla_lr: null` means live Phyla parameters are in the main optimizer
  group at the main LR.
- The 80K-token, two-GPU run puts PhylaFlow on visible `cuda:0` and Phyla on
  visible `cuda:1`.

Result before the newest component-conditioning patch:

- live/unfrozen Phyla followed a trajectory similar to frozen/precomputed Phyla
- Phyla weights did change, so this was not a "Phyla was accidentally frozen"
  bug
- unfreezing alone did not solve DS2 transfer

## Birthset Decoder Work

The old sampler resolved boundary polytomies by repeatedly calling the graph
transformer, predicting one AR merge, inserting one split, rebuilding the tree,
and repeating. This was the main sampling bottleneck.

The new decoder changes the boundary topology step to:

```text
collapsed boundary tree
-> score candidate birth splits once
-> select compatible top-(G - 3) splits
-> insert them all
```

Key config used in the current birthset runs:

```yaml
topology_decoder: birthset
birthset_fallback: none
birthset_use_train_birth_split_bank: false
birthset_use_small_polytomy_enumeration: true
birthset_use_pair_prefix_candidates: true
birthset_pair_prefix_top_pairs: 32
birthset_proposal_pair_target_mode: strict_minimal
birthset_lambda_birth: 1.0
birthset_lambda_rank: 0.1
birthset_lambda_proposal: 1.0
```

The train birth-split bank was disabled because it leaked train-distribution
candidate splits and made the decoder a ranker over a bank rather than a true
candidate generator.

### Birthset Correctness Fixes

Major bugs fixed during this pass:

- canonical unrooted split handling for labels, de-duplication, and selection
- misleading polytomy/incomplete-tree checks under the rooted/dummy
  representation
- batch slicing of list-valued tokenizer fields
- top-k proposal target definition, using strict/minimal constructive pairs
  rather than any pair contained in a gold split

These fixes were needed before the single-path overfit sanity could reach exact
topology recovery.

### Single DS2 Path Overfit

The fixed DS2 single-path birthset sanity can overfit:

```text
Checkpoint:
/ewsc/yektefai/phylaflow/checkpoints/birthset_sanity/ds2_case00_strictpair_top32_canonfix_retrain_20260516/2026-05-16_20-42-50/epoch=0-step=001500.ckpt

Eval config:
/ewsc/yektefai/phylaflow/configs/local_ds2_case00_strictpair_top32_canonfix_retrain_20260516_eval_maxphase3.yaml
```

With `sampling_discrete_phase_max_phases: 3`:

- normalized RF: `0.0`
- sampled-target tree KL: `0.0`
- sampled-target topological KL: `0.0`
- birthset events: `3`
- inserted/required birth splits: `26/26`
- legacy AR fallback calls: `0`

This validated the birthset topology resolver locally. With a larger phase cap,
the model can overrun the target, so phase/terminal stopping remains a separate
modeling problem.

### Birthset Sampling Speed

Corrected distinct-start top32 timing:

| Batch size | Total time | Per tree | Candidate splits |
| ---: | ---: | ---: | ---: |
| 8 | `0.773s` | `0.0966s` | `4531` |
| 16 | `1.68s` | `0.105s` | `12981` |
| 32 | `3.31s` | `0.104s` | `31666` |
| 64 | `6.44s` | `0.101s` | `54894` |
| 100 | `10.39s` | `0.104s` | `84139` |
| 200 | `20.27s` | `0.101s` | `182473` |
| 210 | `21.36s` | `0.102s` | `189815` |

Actual 1000-sample timing:

- total wall time: `104.55s`
- per tree: `0.1046s`
- throughput: `9.56 trees/sec`
- total candidate splits: `900805`
- total birthset events: `3000`

This is roughly `1.7 minutes` per 1000 trees, compared with the old sequential
AR regime at roughly tens of minutes per 1000 trees.

## DS2 210-Bank Distribution Result

Config:

```text
configs/local_ds2_210bank_birthset_top32_s128_lr2e3_unseen1000_20260517.yaml
```

Metrics:

```text
/ewsc/yektefai/phylaflow/metrics/full_sanity_fixedpair_20260401/ds2_210bank_birthset_top32_s128_lr2e3_unseen1000_20260517_metrics.jsonl
```

The run evaluates 1000 unseen random starts. It substantially learns the DS2
distribution:

- start RF mean: about `0.993`
- best RF mean: `0.149` at step `16000`
- best topology KL: `0.0668` at step `18000`
- best golden topology KL: `0.0647` at step `18000`
- best golden tree-topology KL: `1.102` at step `33000`
- best posterior topology support recall: `0.643` at step `27000`
- best golden posterior topology support recall: `1.0`

The last recorded point at step `42000` had drifted:

- topology KL: `0.0969`
- RF mean: `0.216`
- support recall: `0.357`

Conclusion:

The birthset decoder is capable of learning DS2 with fixed DS2 training pairs
and unseen random starts. That means the remaining problem is not that birthset
cannot represent the posterior move. The failure is transfer/generalization.

## 29-Leaf OrthoMaM Generalization Results

### Frozen/precomputed Phyla baseline

Config:

```text
configs/orthomam29leaf_fixedpair_c210_birthset_top32_phy256_s128_lr1e3_holdout1000_20260518.yaml
```

Metrics:

```text
/ewsc/yektefai/phylaflow/metrics/full_sanity_fixedpair_20260401/orthomam29leaf_fixedpair_c210_birthset_top32_phy256_s128_lr1e3_holdout1000_20260518_metrics.jsonl
```

Best heldout metrics:

- best topology KL: `1.030` at step `51000`
- best RF mean: `0.945` at step `131000`
- posterior topology support recall: `0.0`

This is better than random starts in RF and topology KL, but still not learning
exact heldout posterior modes.

### Live Phyla, unfrozen, 80K cap

Config:

```text
configs/orthomam29leaf_fixedpair_c210_birthset_top32_livephyla_rawfull_unfrozen_cap80k_2gpu_s128_lr1e3_holdout1000_20260519.yaml
```

Metrics:

```text
/ewsc/yektefai/phylaflow/metrics/full_sanity_fixedpair_20260401/orthomam29leaf_fixedpair_c210_birthset_top32_livephyla_rawfull_unfrozen_cap80k_2gpu_s128_lr1e3_holdout1000_20260519_metrics.jsonl
```

Best heldout metrics before stopping:

- best topology KL: `1.032` at step `34000`
- best RF mean: `0.937` at step `40000`
- posterior topology support recall: `0.0`

This was not a clear improvement over frozen/precomputed Phyla. It suggested
that unfreezing Phyla is not enough if the birthset heads only see sequence
context indirectly.

## Mean OrthoMaM Prior Diagnostics

The strongest diagnostic evidence so far is that the OrthoMaM-trained models
often emit repeated, globally common OrthoMaM positional splits rather than
dataset-specific posterior-supported splits.

### DS2 generated trees from older OrthoMaM pretraining

Tree dump:

```text
/ewsc/yektefai/phylaflow/metrics/full_sanity_fixedpair_20260401/orthomam29leaf_507cases_phy256_leafglobal_cladehead_metricprobe64_fh64_aradd_mlp2cap_s128_lr2e3_ds2eval_mrbayes20k_20260507_generated_trees/step00099100_stepper00099101_train_trees.jsonl
```

Findings:

- sampled topologies were unique, but repeated wrong local splits appeared
- repeated generated splits had DS2 posterior support `0`
- examples:
  - generated DS2 split position `[18,19]`: generated support about `0.5`,
    DS2 support `0`, global OrthoMaM support about `0.196`
  - `[14,15]`: generated support about `0.5`, DS2 support `0`, global
    OrthoMaM support about `0.183`
  - `[16,17]`: generated support about `0.476`, DS2 support `0`, global
    OrthoMaM support about `0.220`

This is direct evidence that OrthoMaM training can impose a numeric-position
clade prior on DS2.

### Heldout OrthoMaM diagnostic

Diagnostic output:

```text
/ewsc/yektefai/phylaflow/metrics/full_sanity_fixedpair_20260401/orthomam29leaf_meanprior_diag_step141k_20260520
```

Only two heldout datasets were sampled before stopping the diagnostic, but they
were informative:

- `10266_NT_AL`: model KL `3.42` vs random start KL `3.94`; RF model `0.965`
  vs start `0.994`
- `114625_NT_AL`: model KL `3.87` vs random start KL `3.79`; RF model `0.963`
  vs start `0.987`

Generated repeated splits were more globally OrthoMaM-prior-like than random
starts:

- generated split global-prior support mean: `0.0333`
- random start global-prior support mean: `0.0079`
- heldout posterior global-prior support mean: `0.0508`

Case-level repeated splits:

- `10266_NT_AL`: 4 repeated generated splits >= 0.25 support; `0/4` were
  heldout-posterior-supported; `4/4` were global-only
- `114625_NT_AL`: 8 repeated generated splits >= 0.25 support; `2/8` were
  heldout-supported; `6/8` were global-only

Interpretation:

On heldout OrthoMaM, the global prior is sometimes useful because heldout
OrthoMaM shares mammalian positional/clade regularities. On DS2, the same
position prior is harmful because the taxa occupying those slots are different.

## Architecture Diagnosis

Before the newest patch, the birthset topology head scored candidate local
subsets from:

```text
component embeddings
graph context
candidate size features
```

The component embeddings do contain some Phyla information:

- leaf Phyla token additions enter the trunk
- global Phyla context can be added to AR group embeddings
- clade Phyla context can be added to AR group embeddings

But the birthset scorer itself did not directly receive "which sequences are in
this candidate split." It could therefore lean on split identity and positional
structure. This is not wrong for representing BHV state, but it is a shortcut
for cross-dataset transfer.

The relevant split identity path embeds the binary split mask and adds it to
edge/group embeddings using:

```python
self.split_identity_scale = 0.75
```

We decided not to remove split identity yet, because it is useful BHV coordinate
information. Instead, the next experiment makes the topology head score:

```text
split structure + sequence content occupying that structure
```

## New Component-Level Phyla Conditioning Patch

New flag:

```yaml
birthset_use_component_phyla_conditioning: true
```

Implemented behavior:

1. For each polytomy component, pool raw Phyla embeddings over leaves in that
   component:

   ```text
   p_i = mean(Phyla embeddings for leaves in component i)
   ```

2. For each candidate split `I | I^c`, compute:

   ```text
   p_in  = mean(p_i for i in I)
   p_out = mean(p_i for i not in I)
   ```

3. Add Phyla candidate features to both the final birthset scorer and the
   pair-prefix proposal scorer:

   ```text
   p_in + p_out
   abs(p_in - p_out)
   p_in * p_out
   structural_sum * phyla_sum
   structural_abs_diff * phyla_abs_diff
   ```

4. Keep the same structural features, split identity, graph context, and size
   features.

The important point is that top-k pair proposal is also sequence-conditioned.
Otherwise the final scorer might see sequence context only after the true split
was filtered out.

Implementation locations:

- `model/model.py`: `BirthSetTopologyHead` now accepts
  `component_phyla_embeddings`
- `model/model.py`: the model forward attaches pooled component Phyla embeddings
  to each AR/birthset group
- `run/TrainingModule.py`: training and sampling pass those tensors into both
  the final birthset head and proposal head
- `run/run.py`: config plumbing for
  `birthset_use_component_phyla_conditioning`

The new heads initialize at `242K` parameters each, up from `108K`, confirming
that the direct Phyla-conditioned feature path is active.

## Active Run

Current run:

```text
PID: 3559627
W&B: https://wandb.ai/yasha/orthomam/runs/l3fbbhau
Config: configs/orthomam29leaf_fixedpair_c210_birthset_top32_livephyla_rawfull_unfrozen_cap80k_2gpu_phylacond_s128_lr1e3_holdout1000_20260520.yaml
Eval config: configs/orthomam29leaf_fixedpair_c210_birthset_top32_livephyla_rawfull_unfrozen_cap80k_2gpu_phylacond_s128_lr1e3_holdout1000_20260520_eval.yaml
Log: logs/orthomam29leaf_livephyla_unfrozen_cap80k_2gpu_phylacond_20260520_gpu0_gpu1.log
Checkpoint dir: /ewsc/yektefai/phylaflow/checkpoints/full_sanity_fixedpair_20260401/orthomam29leaf_fixedpair_c210_birthset_top32_livephyla_rawfull_unfrozen_cap80k_2gpu_phylacond_s128_lr1e3_holdout1000_20260520
```

This run uses:

```yaml
live_phyla_checkpoint_path: /home/unix/yektefai/PhylaFlow/weights/11564369
live_phyla_unfreeze: true
live_phyla_lr: null
live_phyla_device: cuda:1
live_phyla_max_input_tokens: 80000
birthset_use_component_phyla_conditioning: true
```

Early training is running normally at about `0.9 it/s` after startup. No eval
has completed yet at the time of this memo.

## Current Interpretation

The evidence supports this read:

- Birthset decoding fixes the sampling bottleneck and can learn DS2 when trained
  on DS2 fixed pairs.
- OrthoMaM 29-leaf heldout training learns some family-level/topological prior,
  visible in RF/topology-KL movement.
- The same learned prior does not transfer cleanly to DS2.
- The model appears to overuse numeric-position/clade priors and underuse the
  identity of the sequences occupying a split.
- Unfreezing Phyla alone does not fix the interface problem.
- Direct candidate-level/component-level Phyla conditioning is the next best
  test because it makes every birthset decision depend explicitly on sequence
  context.

If the active Phyla-conditioned run improves heldout OrthoMaM while reducing the
DS2 harm, that would support the hypothesis that the missing piece was the
birthset head's weak conditioning interface. If it behaves like the previous
live-Phyla run, then the next intervention should be stronger anti-position
shortcut pressure, such as split identity dropout or leaf permutation
augmentation.
