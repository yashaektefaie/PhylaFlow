# DS1 no-conditioning checkpoint, MrBayes 20K from 256 starts

Checkpoint source:
`ds1_1280bank_nocond_s128_lr2e3_unseeneval256_local`, step 20000.

Question: what changes if the MrBayes experiment uses all 256 generated/eval
starts instead of the earlier 64-start slice?

All runs below used DS1, `ngen=20000`, `samplefreq=200`, 101 samples per
chain, and the golden DS1 posterior tree-topology distribution.

| method | chains | initial tree-KL | final 20K tree-KL | tail-half tree-KL | support | posterior recall | sampled unique | first gen <2 | first gen <1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| terminal endpoints, first 64 | 64 | 6.218959 | 0.865891 | 0.743818 | 0.600093 | 0.949367 | 1216 | 6000 | 16000 |
| terminal endpoints, all 256 | 256 | 5.199915 | 0.859653 | 0.652903 | 0.625503 | 1.000000 | 3822 | 4000 | 14000 |
| split-guided endpoints, all 256 | 256 | 5.893293 | 0.732605 | 0.632437 | 0.714225 | 1.000000 | 2546 | 2000 | 8000 |
| random unseen starts, all 256 | 256 | 16.630747 | 1.031644 | 0.786954 | 0.479966 | 0.987342 | 4380 | 6000 | n/a |

Main read:

- Moving terminal endpoints from 64 to 256 mostly improves coverage and speed,
  not final cumulative KL: 0.865891 -> 0.859653.
- The all-256 terminal run reaches every golden posterior topology by 20K,
  while the first-64 run only reaches 94.9 percent posterior-topology recall.
- The split-guided all-256 endpoints are best in this test: final tree-KL
  0.732605, support 0.714225, all posterior topologies recovered, and tree-KL
  drops below 1 by 8000 generations.
- The all-256 random-start baseline is worse in cumulative 20K tree-KL
  (1.031644), despite high raw diversity. Its tail-half tree-KL is better
  (0.786954), which suggests ordinary random starts need more burn-in.

Important convention note:

The terminal all-256 MrBayes run uses a sanitized/split-compatible version of
the generated terminal trees because one raw generated tree had a unary/double
bracket artifact that MrBayes rejected. The sanitized terminal list reproduces
the checkpoint endpoint tree-KL at generation 0: 5.199915.

Artifacts:

- `nocond_terminal_all256_safe_g20000_curve.json`
- `nocond_splitguided_all256_g20000_curve.json`
- `nocond_random_unseenstarts_all256_g20000_curve.json`
- `starts/nocond_terminal_all256_mrbayes_safe_start_trees.txt`
- `starts/nocond_splitguided_all256_start_trees.txt`
