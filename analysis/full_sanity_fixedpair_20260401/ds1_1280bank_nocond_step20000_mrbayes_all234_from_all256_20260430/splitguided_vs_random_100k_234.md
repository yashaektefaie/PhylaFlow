# DS1 100K MrBayes: Split-Guided 234 vs Existing Random 234

The split-guided run uses the first 234 `1280bank_nocond` split-guided starts from the all256 start set, run fresh to 100K generations. The random comparator is the pre-existing `Random original full234` 100K run.

| Start source | Count | Tree-KL @ 20K | Tree-KL @ 60K | Tree-KL @ 100K | Tail-half Tree-KL | Support @ 100K | Unique sampled topologies | First < 2 | First < 1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1280bank_nocond split-guided | 234 | 0.794843 | 0.585163 | 0.499590 | 0.399932 | 0.849933 | 2757 | 2600 | 10800 |
| Existing random original full234 | 234 | 0.960733 | 0.601207 | 0.452979 | 0.291147 | 0.741645 | 4526 | 4400 | 18000 |

Selected curve:

| Generation | Split-guided Tree-KL | Random Tree-KL |
|---:|---:|---:|
| 0 | 5.882294 | 16.540891 |
| 200 | 6.567085 | 17.234037 |
| 1000 | 3.793854 | 12.303424 |
| 2000 | 2.470217 | 3.530097 |
| 5000 | 1.396094 | 1.835617 |
| 10000 | 1.020227 | 1.282348 |
| 20000 | 0.794843 | 0.960733 |
| 40000 | 0.651244 | 0.726032 |
| 60000 | 0.585163 | 0.601207 |
| 80000 | 0.537088 | 0.519239 |
| 100000 | 0.499590 | 0.452979 |

Read: split-guided gives a much better initial condition and reaches low Tree-KL faster. By 100K, the existing random run has better final Tree-KL and much better tail-half Tree-KL. The split-guided run has higher posterior support rate at 100K but fewer unique sampled topologies, which suggests it is more concentrated by the end.

Artifacts:

- `nocond_splitguided_first234_g100000_curve.json`
- `nocond_splitguided_first234_start_trees.txt`
- Pre-existing random comparator: `../ds1_mrbayes_generation_curves_100k_caseadaptboth_step22000_full234_20260425/random_original_full234_g100000_curve.json`
