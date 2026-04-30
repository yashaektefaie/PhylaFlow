# DS1 Model-Generated Start Tree Branch Relaxation

Standalone branch relaxer checkpoint:

`analysis/full_sanity_fixedpair_20260401/standalone_branch_relaxer_ds1_ds8_phyla_leafonly_balanced_nocase_20260429/best.pt`

Only model-generated/model-derived outputs were scored. Random start lists were not relaxed or rerun here.

## Corrected topology-preserving results

The clean reruns are `nocond_splitguided_first234` and `nocond_terminal_first234`. The branch relaxer now edits branch lengths in-place on the input Newick tree, preserving the original topology.

Split-guided input:

`analysis/full_sanity_fixedpair_20260401/ds1_1280bank_nocond_step20000_mrbayes_all234_from_all256_20260430/nocond_splitguided_first234_start_trees.txt`

Split-guided relaxed output:

`analysis/full_sanity_fixedpair_20260401/ds1_branchrelax_start_ll_20260430/nocond_splitguided_first234_relaxed_trees.txt`

Terminal input:

`analysis/full_sanity_fixedpair_20260401/ds1_1280bank_nocond_step20000_mrbayes_all234_from_all256_20260430/nocond_terminal_first234_start_trees.txt`

Terminal relaxed output:

`analysis/full_sanity_fixedpair_20260401/ds1_branchrelax_start_ll_20260430/nocond_terminal_first234_relaxed_trees.txt`

| Tree list | Count | Same topology | Mean LL before | Mean LL after | Mean delta | Median delta | Improved | Worse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `nocond_splitguided_first234` | 234 | 234 | -9948.402 | -7564.266 | +2384.136 | +2444.915 | 234 | 0 |
| `nocond_terminal_first234` | 234 | 234 | -9922.365 | -7499.001 | +2423.364 | +2473.620 | 234 | 0 |

## Important invalidated output

`nocond_splitguided_first234_relaxed_g100000_curve.json` is not a valid branch-length-only result. It was produced before the topology-preserving fix, and the helper rebuilt trees from split masks in a way that changed topologies. Do not use it for conclusions about branch relaxation.

Use this corrected 100K trajectory instead:

`analysis/full_sanity_fixedpair_20260401/ds1_branchrelax_start_ll_20260430/nocond_splitguided_first234_relaxed_topopreserve_g100000_curve.json`

The corrected terminal 100K trajectory is:

`analysis/full_sanity_fixedpair_20260401/ds1_branchrelax_start_ll_20260430/nocond_terminal_first234_relaxed_topopreserve_g100000_curve.json`

## Other relaxed lists

The metric-probe relaxed tree files in this directory were generated during the earlier exploratory pass. Rerun them with `apply_branch_relaxer_and_score_tree_list_20260430.py` before treating them as branch-length-only comparisons.
