# DS1 Relaxed Terminal 100K MrBayes Trajectory

This is the topology-preserving branch-relaxed run for the `1280bank_nocond` terminal starts. All 234 chains completed with zero failures.

Corrected terminal relaxed trajectory:

`analysis/full_sanity_fixedpair_20260401/ds1_branchrelax_start_ll_20260430/nocond_terminal_first234_relaxed_topopreserve_g100000_curve.json`

Corrected split-guided relaxed trajectory:

`analysis/full_sanity_fixedpair_20260401/ds1_branchrelax_start_ll_20260430/nocond_splitguided_first234_relaxed_topopreserve_g100000_curve.json`

Pre-existing random trajectory:

`analysis/full_sanity_fixedpair_20260401/ds1_mrbayes_generation_curves_100k_caseadaptboth_step22000_full234_20260425/random_original_full234_g100000_curve.json`

## Tree-KL

| Generation | Relaxed terminal | Relaxed split-guided | Unrelaxed split-guided | Pre-existing random |
|---:|---:|---:|---:|---:|
| 0 | 5.215021 | 5.882294 | 5.882294 | 16.540891 |
| 20K | 0.309899 | 0.344487 | 0.794843 | 0.960733 |
| 40K | 0.240477 | 0.270979 | 0.651244 | 0.726032 |
| 60K | 0.204354 | 0.238591 | 0.585163 | 0.601207 |
| 80K | 0.186642 | 0.220257 | 0.537088 | 0.519239 |
| 100K | 0.174632 | 0.207002 | 0.499590 | 0.452979 |

Tail-half Tree-KL:

| Start set | Tail-half Tree-KL | First generation below 2 | First generation below 1 | Support recall at 100K |
|---|---:|---:|---:|---:|
| Relaxed terminal | 0.136126 | 1200 | 2400 | 1.000000 |
| Relaxed split-guided | 0.164908 | 1400 | 2800 | 1.000000 |
| Unrelaxed split-guided | 0.399932 | 2600 | 10800 | 1.000000 |
| Pre-existing random | 0.291147 | 4400 | 18000 | 1.000000 |

The direct 20K unrelaxed terminal comparison was already available:

| Start set | Generation 0 Tree-KL | 20K Tree-KL | Tail-half Tree-KL |
|---|---:|---:|---:|
| Unrelaxed terminal | 5.215021 | 0.858870 | 0.651847 |
| Relaxed terminal | 5.215021 | 0.309899 | 0.136126 |

## Mean MrBayes LnL

| Generation | Relaxed terminal | Relaxed split-guided |
|---:|---:|---:|
| 0 | -7463.657 | -7529.630 |
| 20K | -6887.097 | -6886.984 |
| 40K | -6885.534 | -6887.163 |
| 60K | -6885.694 | -6886.442 |
| 80K | -6885.092 | -6885.808 |
| 100K | -6885.524 | -6885.878 |

## Read

The terminal starts are now the strongest result in this branch-relaxed DS1 set. They begin from a better topology distribution than split-guided starts, get the largest branch-length likelihood boost, and end with lower Tree-KL and lower tail-half Tree-KL than the relaxed split-guided starts.
