# DS1 Relaxed Split-Guided 100K MrBayes Trajectory

This compares the corrected topology-preserving relaxed `1280bank_nocond` split-guided starts against the previous unrelaxed split-guided run and the pre-existing random 234-run trace. No new random run was launched for this comparison.

Corrected relaxed trajectory:

`analysis/full_sanity_fixedpair_20260401/ds1_branchrelax_start_ll_20260430/nocond_splitguided_first234_relaxed_topopreserve_g100000_curve.json`

Unrelaxed split-guided trajectory:

`analysis/full_sanity_fixedpair_20260401/ds1_1280bank_nocond_step20000_mrbayes_all234_from_all256_20260430/nocond_splitguided_first234_g100000_curve.json`

Pre-existing random trajectory:

`analysis/full_sanity_fixedpair_20260401/ds1_mrbayes_generation_curves_100k_caseadaptboth_step22000_full234_20260425/random_original_full234_g100000_curve.json`

All three curves have 234 completed chains and zero failures.

## Tree-KL

| Generation | Relaxed split-guided | Unrelaxed split-guided | Pre-existing random |
|---:|---:|---:|---:|
| 0 | 5.882294 | 5.882294 | 16.540891 |
| 20K | 0.344487 | 0.794843 | 0.960733 |
| 40K | 0.270979 | 0.651244 | 0.726032 |
| 60K | 0.238591 | 0.585163 | 0.601207 |
| 80K | 0.220257 | 0.537088 | 0.519239 |
| 100K | 0.207002 | 0.499590 | 0.452979 |

Tail-half Tree-KL:

| Start set | Tail-half Tree-KL | First generation below 2 | First generation below 1 | Support recall at 100K |
|---|---:|---:|---:|---:|
| Relaxed split-guided | 0.164908 | 1400 | 2800 | 1.000000 |
| Unrelaxed split-guided | 0.399932 | 2600 | 10800 | 1.000000 |
| Pre-existing random | 0.291147 | 4400 | 18000 | 1.000000 |

## Mean MrBayes LnL

| Generation | Relaxed split-guided | Unrelaxed split-guided | Pre-existing random |
|---:|---:|---:|---:|
| 0 | -7529.630 | -9948.401 | -40494.097 |
| 20K | -6886.984 | -6889.843 | -6893.991 |
| 40K | -6887.163 | -6888.801 | -6890.958 |
| 60K | -6886.442 | -6888.354 | -6889.492 |
| 80K | -6885.808 | -6888.354 | -6888.597 |
| 100K | -6885.878 | -6887.889 | -6886.862 |

## Read

Branch relaxation changes the trajectory substantially. The starting topology distribution is identical to the unrelaxed split-guided starts, but the improved branch lengths move MrBayes into the good topology region much faster and end with much lower Tree-KL at 100K.

The earlier spectacular relaxed curve was invalid because topology changed during relaxation. This corrected run preserves all 234 topologies and is the result to use.
