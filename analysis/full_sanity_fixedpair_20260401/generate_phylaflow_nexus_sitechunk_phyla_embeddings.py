"""Generate PhyLa site-chunk embeddings for phylaflow_datasets NEXUS files.

Each forward pass sees all taxa for one contiguous alignment-window. Window
embeddings are length-weighted into one pooled [1, N, D] tensor per dataset.
The script can run on GPU, or on CPU by forcing PyTorch/reference Mamba kernels.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmark_phylaflow_nexus_sitechunk_phyla import (
    DEFAULT_CHECKPOINT,
    DEFAULT_NEXUS_ROOT,
    configure_imports,
    iter_windows,
    load_nexus_alignment,
)


REPO_ROOT = Path("/home/yektefai/PhylaFlow")
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "analysis"
    / "full_sanity_fixedpair_20260401"
    / "phylaflow_datasets_phyla_embeddings_sitechunk_cpu_20260428"
)
DEFAULT_ESTIMATE_CSV = (
    REPO_ROOT
    / "analysis"
    / "full_sanity_fixedpair_20260401"
    / "phylaflow_datasets_cpu_runtime_estimate_20260428.csv"
)


def dataset_stem(value: str) -> str:
    path = Path(value.strip())
    return path.stem if path.suffix else value.strip()


def load_dataset_list(args: argparse.Namespace) -> list[str]:
    selected: list[str]
    if args.dataset_list is not None:
        selected = [
            dataset_stem(line)
            for line in args.dataset_list.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    elif args.datasets:
        selected = [dataset_stem(item) for item in args.datasets]
    else:
        selected = sorted(path.stem for path in args.nexus_root.glob("*.nex"))

    if args.num_shards <= 1:
        return selected
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")

    selected_set = set(selected)
    if args.estimate_csv is not None and args.estimate_csv.exists():
        rows = []
        with args.estimate_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                stem = dataset_stem(row["dataset"])
                if stem in selected_set:
                    rows.append((float(row.get("cpu_seconds", 0.0)), stem))
        if len(rows) != len(selected_set):
            missing = sorted(selected_set - {stem for _, stem in rows})
            raise ValueError(
                f"Estimate CSV missing {len(missing)} selected datasets; first missing={missing[:5]}"
            )
        loads = [(0.0, shard, []) for shard in range(args.num_shards)]
        heapq.heapify(loads)
        for seconds, stem in sorted(rows, reverse=True):
            load, shard, stems = heapq.heappop(loads)
            stems.append(stem)
            heapq.heappush(loads, (load + seconds, shard, stems))
        assignments = {shard: stems for _, shard, stems in loads}
        return sorted(assignments[args.shard_index])

    return [
        stem
        for idx, stem in enumerate(sorted(selected))
        if idx % args.num_shards == args.shard_index
    ]


def apply_cpu_reference_kernels() -> None:
    import mamba_ssm.modules.mamba_simple as mamba_simple
    import mamba_ssm.ops.selective_scan_interface as selective_scan_interface
    import phyla.model.model as phyla_model

    phyla_model.RMSNorm = torch.nn.RMSNorm
    mamba_simple.RMSNorm = torch.nn.RMSNorm
    mamba_simple.causal_conv1d_fn = None
    selective_scan_interface.selective_scan_cuda = None


def compute_sitechunk_embeddings(
    model,
    names: list[str],
    sequences: list[str],
    device: str,
    window_size: int,
    stride: int,
    max_windows: int | None,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    alignment_length = len(sequences[0])
    windows = iter_windows(alignment_length, window_size, stride)
    if max_windows is not None:
        windows = windows[: int(max_windows)]
    if not windows:
        raise ValueError("No windows selected")

    chunk_embeddings = []
    metadata = []
    weighted_sum = None
    total_weight = 0.0
    compute_start = time.perf_counter()
    with torch.inference_mode():
        for idx, (start, end) in enumerate(windows, start=1):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            window_start = time.perf_counter()
            chunk_sequences = [sequence[start:end] for sequence in sequences]
            encoded_aa, cls_token_mask, sequence_mask, _ = model.encode(chunk_sequences, names)
            embedding = model(
                encoded_aa.to(device),
                sequence_mask.to(device),
                cls_token_mask.to(device),
            ).detach().cpu().float()
            chunk_embeddings.append(embedding.squeeze(0))

            weight = float(end - start)
            weighted_sum = embedding * weight if weighted_sum is None else weighted_sum + embedding * weight
            total_weight += weight
            elapsed = time.perf_counter() - window_start
            max_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_cuda_memory_mb = (
                torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else None
            )
            metadata.append(
                {
                    "window_index": int(idx - 1),
                    "start": int(start),
                    "end": int(end),
                    "length": int(end - start),
                    "seconds": float(elapsed),
                    "taxa": int(len(names)),
                    "taxa_x_sites": int(len(names) * (end - start)),
                    "max_rss_mb": float(max_rss_kb / 1024.0),
                    "peak_cuda_memory_mb": None
                    if peak_cuda_memory_mb is None
                    else float(peak_cuda_memory_mb),
                }
            )
            print(
                f"embedded window {idx}/{len(windows)} columns {start}:{end} "
                f"in {elapsed:.2f}s",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if weighted_sum is None or total_weight <= 0.0:
        raise RuntimeError("No embeddings produced")
    pooled = (weighted_sum / total_weight).float()
    total_seconds = time.perf_counter() - compute_start
    for item in metadata:
        item["dataset_embedding_seconds"] = float(total_seconds)
    return pooled, torch.stack(chunk_embeddings, dim=0).float(), metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nexus-root", type=Path, default=DEFAULT_NEXUS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--dataset-list", type=Path, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--estimate-csv", type=Path, default=DEFAULT_ESTIMATE_CSV)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cpu-reference-kernels", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    configure_imports()
    if args.cpu_reference_kernels:
        apply_cpu_reference_kernels()
    from phyla.model.model import Config, Phyla

    cfg = Config()
    cfg.model.model_name = "phyla-beta"
    if args.cpu_reference_kernels:
        cfg.model.fused_add_norm = False
    model = Phyla(cfg, device=args.device).load(checkpoint_file=str(args.checkpoint)).to(args.device)
    model.eval()

    created_at = datetime.now(timezone.utc).isoformat()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_datasets = load_dataset_list(args)
    manifest = {
        "created_at_utc": created_at,
        "nexus_root": str(args.nexus_root),
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "model_name": "phyla-beta",
        "chunk_axis": "alignment_columns",
        "window_size": int(args.window_size),
        "stride": int(args.stride),
        "aggregation": "length_weighted_mean",
        "cpu_reference_kernels": bool(args.cpu_reference_kernels),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "selected_datasets": selected_datasets,
        "datasets": {},
        "failures": [],
    }

    for dataset_id in selected_datasets:
        dataset_start = time.perf_counter()
        nexus_path = args.nexus_root / f"{dataset_id}.nex"
        output_path = (
            args.output_dir
            / f"{dataset_id}_phyla_beta_sitechunk_w{int(args.window_size)}_s{int(args.stride)}_embeddings.pt"
        )
        if output_path.exists() and not args.overwrite:
            print(f"{dataset_id}: skipping existing {output_path}", flush=True)
            manifest["datasets"][dataset_id] = {
                "path": str(output_path),
                "source_nexus": str(nexus_path),
                "skipped_existing": True,
            }
            continue

        try:
            names, sequences = load_nexus_alignment(nexus_path)
            print(
                f"{dataset_id}: loaded {len(names)} sequences length {len(sequences[0])}",
                flush=True,
            )
            embeddings, chunk_embeddings, windows = compute_sitechunk_embeddings(
                model,
                names,
                sequences,
                args.device,
                int(args.window_size),
                int(args.stride),
                max_windows=None if int(args.max_windows) <= 0 else int(args.max_windows),
            )
            torch.save(
                {
                    "dataset_id": dataset_id,
                    "sequence_names": names,
                    "embeddings": embeddings,
                    "chunk_embeddings": chunk_embeddings,
                    "windows": windows,
                    "source_nexus": str(nexus_path),
                    "checkpoint_path": str(args.checkpoint),
                    "model_name": "phyla-beta",
                    "chunk_axis": "alignment_columns",
                    "aggregation": "length_weighted_mean",
                    "cpu_reference_kernels": bool(args.cpu_reference_kernels),
                    "created_at_utc": created_at,
                },
                output_path,
            )
            dataset_seconds = time.perf_counter() - dataset_start
            window_seconds = [float(window["seconds"]) for window in windows]
            manifest["datasets"][dataset_id] = {
                "path": str(output_path),
                "source_nexus": str(nexus_path),
                "num_sequences": len(names),
                "sequence_length": len(sequences[0]),
                "num_windows": len(windows),
                "embedding_shape": list(embeddings.shape),
                "chunk_embedding_shape": list(chunk_embeddings.shape),
                "total_seconds": float(dataset_seconds),
                "embedding_seconds": float(windows[0]["dataset_embedding_seconds"]) if windows else 0.0,
                "mean_window_seconds": float(sum(window_seconds) / len(window_seconds))
                if window_seconds
                else 0.0,
                "min_window_seconds": float(min(window_seconds)) if window_seconds else 0.0,
                "max_window_seconds": float(max(window_seconds)) if window_seconds else 0.0,
                "taxa_x_sites": int(len(names) * len(sequences[0])),
            }
            print(
                f"{dataset_id}: wrote {output_path} total {dataset_seconds:.2f}s",
                flush=True,
            )
        except Exception as exc:
            failure = {
                "dataset_id": dataset_id,
                "source_nexus": str(nexus_path),
                "error": repr(exc),
            }
            manifest["failures"].append(failure)
            print(json.dumps({"failure": failure}), file=sys.stderr, flush=True)
            if not args.continue_on_error:
                raise

    manifest_path = args.output_dir / f"manifest_shard{int(args.shard_index):03d}_of_{int(args.num_shards):03d}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest_path": str(manifest_path), "failures": len(manifest["failures"])}), flush=True)


if __name__ == "__main__":
    main()
