# PhyLa CPU Embedding Generation for `phylaflow_datasets`

This note documents how to run PhyLa-beta embeddings for `/home/yektefai/phylaflow_datasets` on CPU-only workers, how the runtime estimates were produced, and how to launch a 100-worker embedding batch.

## What Is Being Generated

Source alignments:

```bash
/home/yektefai/phylaflow_datasets/nexus/*.nex
```

There are `1,163` NEXUS alignments. Each NEXUS header has `NTAX` and `NCHAR`, which were used for runtime estimation.

Embedding method:

- Model: `phyla-beta`
- Checkpoint: `/home/yektefai/PhylaFlow/weights/11564369`
- PhyLa repo: `/tmp/Phyla`
- Mamba source: `/tmp/mamba_src/mamba_ssm-2.3.1`
- Windowing: all taxa, contiguous alignment-column chunks
- Window size: `256`
- Stride: `256`
- Aggregation: length-weighted mean over window embeddings

Each forward pass sees every taxon for one site window, for example columns `0:256`, then `256:512`. This is the intended comparator-style usage; it is not taxa chunking.

Each output `.pt` file contains:

- `embeddings`: pooled tensor, shape `[1, num_taxa, 256]`
- `chunk_embeddings`: per-window tensor, shape `[num_windows, num_taxa, 256]`
- `windows`: per-window timing and coordinate metadata
- `sequence_names`
- source/checkpoint/model metadata

## CPU Kernel Requirement

Direct CPU execution fails in this environment because the PhyLa/Mamba code still tries to call Triton/CUDA fused layernorm kernels on CPU tensors:

```text
ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
```

The CPU path used here disables the fused norm path and forces reference/PyTorch kernels:

```python
cfg.model.fused_add_norm = False
phyla_model.RMSNorm = torch.nn.RMSNorm
mamba_simple.RMSNorm = torch.nn.RMSNorm
mamba_simple.causal_conv1d_fn = None
selective_scan_interface.selective_scan_cuda = None
```

Use the provided scripts instead of calling PhyLa directly; they apply this patch when `--cpu-reference-kernels` or `--force-cpu-reference-kernels` is set.

The CPU-only worker still needs a Python environment that can import the local PhyLa and Mamba source trees. The scripts add these paths explicitly:

```text
/tmp/Phyla
/tmp/Phyla/phyla
/tmp/mamba_src/mamba_ssm-2.3.1
/home/yektefai/PhylaFlow
```

If a new CPU machine does not have those paths, copy or install the same PhyLa/Mamba sources and update the constants at the top of the scripts.

Also force one thread per process. This matters for 100 workers; otherwise PyTorch/BLAS thread pools will oversubscribe the machine.

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TORCH_NUM_THREADS=1
```

## Benchmark Scripts

Benchmark one or more windows:

```bash
cd /home/yektefai/PhylaFlow

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_NUM_THREADS=1 \
python analysis/full_sanity_fixedpair_20260401/benchmark_phylaflow_nexus_sitechunk_phyla.py \
  10125_NT_AL \
  --window-size 256 \
  --stride 256 \
  --max-windows 1 \
  --device cpu \
  --disable-fused-add-norm \
  --force-cpu-reference-kernels \
  --output-json analysis/full_sanity_fixedpair_20260401/phylaflow_datasets_10125_cpu_w256_reference_benchmark_20260428.json
```

Generate embeddings:

```bash
cd /home/yektefai/PhylaFlow

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_NUM_THREADS=1 \
python analysis/full_sanity_fixedpair_20260401/generate_phylaflow_nexus_sitechunk_phyla_embeddings.py \
  --datasets 10816_NT_AL \
  --window-size 256 \
  --stride 256 \
  --max-windows 1 \
  --device cpu \
  --cpu-reference-kernels \
  --output-dir analysis/full_sanity_fixedpair_20260401/tmp_phylaflow_cpu_generator_smoke_20260428 \
  --overwrite
```

The smoke run above completed and wrote:

```text
10816_NT_AL_phyla_beta_sitechunk_w256_s256_embeddings.pt
embeddings shape:       [1, 10, 256]
chunk_embeddings shape: [1, 10, 256]
```

## Runtime Measurements

Calibration dataset:

```text
10125_NT_AL
161 taxa
2409 sites
10 windows at window_size=256
```

Measured single-window runtimes:

| Path | Window | Time |
| --- | ---: | ---: |
| GPU optimized, A100 | 161 taxa x 256 sites | `232.1046s` |
| CPU reference, 1 thread | 161 taxa x 32 sites | `32.2071s` |
| CPU reference, 1 thread | 161 taxa x 64 sites | `60.5633s` |
| CPU reference, 1 thread | 161 taxa x 256 sites | `265.4601s` |

CPU/GPU numerical agreement was checked on a 32-site window:

```text
shape:         [1, 161, 256]
flat cosine:   0.9999997616
max abs diff:  1.7136335e-07
mean abs diff: 2.4547047e-08
```

The CPU estimate uses:

```text
cpu seconds per taxa-site = 265.46013390179724 / (161 * 256)
                          = 0.006440705888533512
```

Dataset summary for all `1,163` NEXUS files:

```text
total windows:    8,733
total taxa-sites: 315,592,446
taxa median:      156
taxa p95:         161
taxa max:         162
sites median:     1,329
sites p95:        4,848
sites max:        15,114
```

Estimated total CPU time:

```text
serial CPU, 1 thread: 23.53 days
```

Greedy balanced makespan estimates:

| Workers | Estimated wall time |
| ---: | ---: |
| 1 | `564.62 h` |
| 12 | `47.06 h` |
| 25 | `22.60 h` |
| 50 | `11.31 h` |
| 75 | `7.55 h` |
| 100 | `5.67 h` |
| 150 | `3.81 h` |
| 200 | `3.81 h` |

The largest single dataset is estimated at `3.81 h`, so adding workers beyond roughly 150 has little benefit unless a dataset is split across workers.

Estimate files:

```bash
analysis/full_sanity_fixedpair_20260401/phylaflow_datasets_cpu_runtime_estimate_20260428.json
analysis/full_sanity_fixedpair_20260401/phylaflow_datasets_cpu_runtime_estimate_20260428.csv
analysis/full_sanity_fixedpair_20260401/phylaflow_datasets_sitechunk_runtime_estimate_20260428.json
analysis/full_sanity_fixedpair_20260401/phylaflow_datasets_sitechunk_runtime_estimate_20260428.csv
```

## Memory Planning

The CPU 32-site reference run reported peak RSS around `1047 MB`. The full 256-site CPU run was not remeasured with RSS after adding RSS logging, so request more than the observed minimum.

Practical guidance:

- Minimum planning floor: `~1 GB/process`
- Safer scheduler request: `2-4 GB/process`
- For 100 workers, plan for at least `200 GB` RAM if using `2 GB/process`; `400 GB` is safer if the cluster can afford it.

Do not launch 100 local processes on the current `/home/yektefai/PhylaFlow` machine. It exposes only `12` logical CPUs, so local 100-way CPU parallelism would oversubscribe badly.

## 100-Worker Launch

The generator supports balanced sharding:

```bash
--num-shards 100
--shard-index <0..99>
--estimate-csv analysis/full_sanity_fixedpair_20260401/phylaflow_datasets_cpu_runtime_estimate_20260428.csv
```

When the estimate CSV is present, datasets are assigned greedily by estimated CPU seconds, which is what produced the `~5.67 h` 100-worker makespan estimate.

Set an output directory on shared storage:

```bash
export OUTPUT_DIR=/home/yektefai/PhylaFlow/analysis/full_sanity_fixedpair_20260401/phylaflow_datasets_phyla_embeddings_sitechunk_cpu_20260428
mkdir -p "$OUTPUT_DIR/logs"
```

### GNU Parallel or xargs

Use this only on a machine or allocation with 100 real CPU cores and enough RAM:

```bash
cd /home/yektefai/PhylaFlow

seq 0 99 | xargs -I{} -P100 bash -lc '
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export TORCH_NUM_THREADS=1
  python analysis/full_sanity_fixedpair_20260401/generate_phylaflow_nexus_sitechunk_phyla_embeddings.py \
    --nexus-root /home/yektefai/phylaflow_datasets/nexus \
    --output-dir "$OUTPUT_DIR" \
    --checkpoint /home/yektefai/PhylaFlow/weights/11564369 \
    --window-size 256 \
    --stride 256 \
    --device cpu \
    --cpu-reference-kernels \
    --num-shards 100 \
    --shard-index {} \
    --estimate-csv analysis/full_sanity_fixedpair_20260401/phylaflow_datasets_cpu_runtime_estimate_20260428.csv \
    --continue-on-error \
    > "$OUTPUT_DIR/logs/shard_{}.log" 2>&1
'
```

The script skips existing output files unless `--overwrite` is passed, so it is resume-safe.

### SLURM Array

Example array job:

```bash
cat > cpu_phyla_embeddings_100.sbatch <<'EOF'
#!/usr/bin/env bash
#SBATCH --job-name=phyla_cpu_embeddings
#SBATCH --array=0-99%100
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=/home/yektefai/PhylaFlow/analysis/full_sanity_fixedpair_20260401/phylaflow_datasets_phyla_embeddings_sitechunk_cpu_20260428/logs/slurm_%A_%a.out
#SBATCH --error=/home/yektefai/PhylaFlow/analysis/full_sanity_fixedpair_20260401/phylaflow_datasets_phyla_embeddings_sitechunk_cpu_20260428/logs/slurm_%A_%a.err

set -euo pipefail

cd /home/yektefai/PhylaFlow

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TORCH_NUM_THREADS=1

OUTPUT_DIR=/home/yektefai/PhylaFlow/analysis/full_sanity_fixedpair_20260401/phylaflow_datasets_phyla_embeddings_sitechunk_cpu_20260428
mkdir -p "$OUTPUT_DIR/logs"

python analysis/full_sanity_fixedpair_20260401/generate_phylaflow_nexus_sitechunk_phyla_embeddings.py \
  --nexus-root /home/yektefai/phylaflow_datasets/nexus \
  --output-dir "$OUTPUT_DIR" \
  --checkpoint /home/yektefai/PhylaFlow/weights/11564369 \
  --window-size 256 \
  --stride 256 \
  --device cpu \
  --cpu-reference-kernels \
  --num-shards 100 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --estimate-csv analysis/full_sanity_fixedpair_20260401/phylaflow_datasets_cpu_runtime_estimate_20260428.csv \
  --continue-on-error
EOF

mkdir -p /home/yektefai/PhylaFlow/analysis/full_sanity_fixedpair_20260401/phylaflow_datasets_phyla_embeddings_sitechunk_cpu_20260428/logs
sbatch cpu_phyla_embeddings_100.sbatch
```

## Validation After Launch

Check file count and failures:

```bash
cd /home/yektefai/PhylaFlow

python - <<'PY'
from pathlib import Path
import json
import torch

out = Path("analysis/full_sanity_fixedpair_20260401/phylaflow_datasets_phyla_embeddings_sitechunk_cpu_20260428")
pt_files = sorted(out.glob("*_phyla_beta_sitechunk_w256_s256_embeddings.pt"))
manifests = sorted(out.glob("manifest_shard*_of_100.json"))
failures = []
for path in manifests:
    payload = json.loads(path.read_text())
    failures.extend(payload.get("failures", []))

print("embedding files", len(pt_files))
print("manifests", len(manifests))
print("failures", len(failures))
if failures:
    print(failures[:5])

sample = torch.load(pt_files[0], map_location="cpu")
print(pt_files[0].name)
print("embeddings", tuple(sample["embeddings"].shape), torch.isfinite(sample["embeddings"]).all().item())
print("chunks", tuple(sample["chunk_embeddings"].shape), torch.isfinite(sample["chunk_embeddings"]).all().item())
PY
```

Expected final count:

```text
embedding files 1163
manifests 100
failures 0
```

If failures occurred, rerun the same 100-worker command. Existing successful `.pt` files will be skipped unless `--overwrite` is supplied.
