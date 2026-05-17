# Birthset Decoder Findings - 2026-05-17

## Context

This branch replaces the slow sequential topology resolver with a birth-split set
decoder. The old sampler resolved a boundary polytomy by repeatedly feeding the
partially resolved tree back through the graph transformer and inserting one AR
split at a time. The birthset path scores candidate split insertions for a
boundary tree and inserts a compatible set in one step.

The current implementation keeps the legacy AR path available, but the active
sanity experiments use:

- `topology_decoder: birthset`
- `birthset_fallback: none`
- pair-prefix candidates enabled
- top pair cap `birthset_pair_prefix_top_pairs: 32`
- candidate cap `birthset_max_candidates_per_polytomy: 2048`
- strict/minimal proposal pair targets

## Correctness Bugs Fixed

1. Canonical unrooted split handling:
   - Rooted masks for the same biological split could both be selected.
   - This wasted birth slots and prevented exact decoding in top-k mode.
   - The birthset path now canonicalizes unrooted splits before labeling,
     de-duplicating, selecting, and measuring precision/recall.

2. Polytomy detection after birthset insertion:
   - The sampler was using rooted polytomy checks in places where an unrooted
     binary tree has a degree-3 root.
   - This produced misleading `stopped_for_no_valid_merge` flags after a fully
     resolved birthset decode.
   - Birthset sampling now uses the unrooted-ok polytomy predicate consistently.

3. Batched boundary-tokenizer slicing:
   - In the batched birthset rollout, tensor-valued tokenizer fields were sliced
     per boundary tree, but list/tuple-valued fields were not.
   - The split-mask field is list-valued, so later batch items could rebuild
     from batch item 0's active split masks.
   - This made batched distinct-start sampling follow a different topology path
     from serial sampling and inflated candidate work.
   - The batched path now slices list/tuple tokenizer fields too.

4. Unseen-start metrics cap:
   - The old unseen-start eval capped `sample_metrics_num_pairs` at the training
     bank size.
   - For DS2 this limited eval to 210 starts.
   - If `sample_metrics_num_pairs` exceeds the bank size, unseen-start eval now
     samples source target cases with replacement while still generating unseen
     random starts.

## Single-Path Sanity Result

Checkpoint:

`/ewsc/yektefai/phylaflow/checkpoints/birthset_sanity/ds2_case00_strictpair_top32_canonfix_retrain_20260516/2026-05-16_20-42-50/epoch=0-step=001500.ckpt`

Config:

`/ewsc/yektefai/phylaflow/configs/local_ds2_case00_strictpair_top32_canonfix_retrain_20260516_eval_maxphase3.yaml`

With `sampling_discrete_phase_max_phases: 3`, the fixed DS2 case reaches:

- normalized RF: `0.0`
- sampled-target tree KL: `0.0`
- sampled-target topological KL: `0.0`
- birthset events: `3`
- inserted/required birth splits: `26/26`
- no legacy AR fallback calls

With a larger phase cap, the same model can overrun the target topology. This
means phase stopping remains a real modeling issue; the birthset decoder fixes
the local topology-resolution bottleneck but does not itself solve terminal
phase prediction.

## Speed Benchmarks

Corrected distinct-start timing for the top32 birthset sampler on GPU 4:

| Batch size | Total time | Per tree | Candidate splits |
| ---: | ---: | ---: | ---: |
| 8 | `0.773s` | `0.0966s` | `4531` |
| 16 | `1.68s` | `0.105s` | `12981` |
| 32 | `3.31s` | `0.104s` | `31666` |
| 64 | `6.44s` | `0.101s` | `54894` |
| 100 | `10.39s` | `0.104s` | `84139` |
| 200 | `20.27s` | `0.101s` | `182473` |
| 210 | `21.36s` | `0.102s` | `189815` |

The current ceiling is not GPU memory. GPU memory stayed low even at batch 210.
The bottleneck is still CPU/topology/candidate construction. Larger batches
mainly preserve throughput; they do not materially improve per-tree speed beyond
batch 8-16.

Actual 1000-sample timing, run as five loaded `batch_size=200` calls:

- total wall time: `104.55s`
- per tree: `0.1046s`
- throughput: `9.56 trees/sec`
- total candidate splits: `900805`
- total birthset events: `3000`

This is about `1.7 minutes` per 1000 trees for the current top32/max3 sanity
configuration.

Corrected comparison against the earlier broad/pre-top-k birthset decoder on
batch-8 distinct starts:

| Decoder | Total time | Per tree | Candidate splits | Birthset events |
| --- | ---: | ---: | ---: | ---: |
| Broad/pre-top-k | `2.43s` | `0.304s` | `23400` | `64` |
| Current top32 | `0.773s` | `0.0966s` | `4531` | `24` |

That is roughly a 3.1x wall-time improvement and a 5.2x reduction in candidate
count. Compared with the old sequential AR regime, this is at least an 8x speed
improvement under the optimistic legacy benchmark and closer to 17x against the
rough 30-minute-per-1000 regime.

## Next DS2 Distribution Run

New config:

`configs/local_ds2_210bank_birthset_top32_s128_lr2e3_unseen1000_20260517.yaml`

New compact bank artifact:

`artifacts/birthset_sanity/ds2_multipair210_topofreqcover_x5_20260429_joint_bank.jsonl`

This config uses the same 210 fixed DS2 start/target cases from the previous
DS2 210-bank overfit run, represented as one joint JSONL bank. The target bank
is the previous `topofreqcover_x5` posterior-weighted/coverage set.

Key changes from the old DS2 210-bank AR config:

- birthset topology decoder with strict top32 proposal training
- no train birth-split bank leakage
- no legacy AR fallback during sampling
- birthset boundary labels from full-path control data
- unseen-start eval increased from 42 to 1000
- 1000 eval starts are generated as random starts not seen in training
- tree KL and topological KL are computed from the 1000 generated samples
- MrBayes downstream eval is disabled for this first birthset distribution run

The 210 training paths have true boundary phase counts:

- 175 paths with 3 phases
- 33 paths with 4 phases
- 2 paths with 5 phases

The DS2 distribution config therefore uses `sampling_discrete_phase_max_phases:
5` rather than the single-path speed setting of 3.
