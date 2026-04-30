# DS1 20K MrBayes Comparison, Count-Matched at 234 Starts

All rows use 234 starts/chains. The no-cond rows are first-234 subsets recomputed from the completed all256 MrBayes work dirs.

| Start source | Count | Tree-KL @ 20K | Tail-half Tree-KL | Support | Posterior recall | Unique sampled topologies | First < 2 | First < 1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1280bank_nocond split-guided starts | 234 | 0.743369 | 0.647354 | 0.716637 | 1.000000 | 2367 | 2000 | 10000 |
| 1280bank_nocond terminal endpoints | 234 | 0.858870 | 0.651847 | 0.625836 | 1.000000 | 3555 | 4000 | 14000 |
| metric-probe terminal endpoints | 234 | 0.905342 | 0.712771 | 0.657231 | 0.987342 | 3350 | 5000 | 16000 |
| 1280bank_nocond matched random unseen starts | 234 | 1.027645 | 0.775817 | 0.480156 | 0.987342 | 4074 | 6000 | - |
| metric-probe split-guided starts | 234 | 1.017231 | 0.886394 | 0.730092 | 0.974684 | 2483 | 4000 | - |
| metric-probe matched random starts | 234 | 1.075355 | 0.769703 | 0.484133 | 0.974684 | 4009 | 8000 | - |

Main read: the best count-matched 20K result is still the no-conditioning 1280-bank split-guided starts at Tree-KL 0.743369. The previous all256 value was 0.732605, so dropping to the first 234 chains does not change the conclusion.

Artifacts:

- `nocond_splitguided_first234_reuse_g20000_curve.json`
- `nocond_terminal_first234_reuse_g20000_curve.json`
- `nocond_random_unseenstarts_first234_reuse_g20000_curve.json`
