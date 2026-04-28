"""Benchmark PhyLa site-window embedding calls on phylaflow_datasets NEXUS files."""

from __future__ import annotations

import argparse
import json
import resource
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


REPO_ROOT = Path("/home/yektefai/PhylaFlow")
PHYLA_REPO = Path("/tmp/Phyla")
MAMBA_SRC = Path("/tmp/mamba_src/mamba_ssm-2.3.1")
DEFAULT_NEXUS_ROOT = Path("/home/yektefai/phylaflow_datasets/nexus")
DEFAULT_CHECKPOINT = REPO_ROOT / "weights" / "11564369"


def configure_imports() -> None:
    sys.path.insert(0, str(PHYLA_REPO))
    sys.path.insert(1, str(PHYLA_REPO / "phyla"))
    sys.path.insert(2, str(MAMBA_SRC))
    sys.path.insert(3, str(REPO_ROOT))


def load_nexus_alignment(path: Path) -> tuple[list[str], list[str]]:
    names: list[str] = []
    chunks_by_name: dict[str, list[str]] = {}
    in_matrix = False
    comment_pattern = re.compile(r"\[[^\]]*\]")

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = comment_pattern.sub("", raw_line).strip()
            if not line:
                continue
            upper = line.upper()
            if not in_matrix:
                if upper == "MATRIX" or upper.startswith("MATRIX "):
                    in_matrix = True
                continue
            if line.startswith(";"):
                break
            if ";" in line:
                line = line.split(";", 1)[0].strip()
                if not line:
                    break
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            name, sequence = parts
            sequence = "".join(sequence.split()).upper()
            if not sequence:
                continue
            if name not in chunks_by_name:
                names.append(name)
                chunks_by_name[name] = []
            chunks_by_name[name].append(sequence)

    if not names:
        raise ValueError(f"No MATRIX rows parsed from {path}")
    sequences = ["".join(chunks_by_name[name]) for name in names]
    lengths = {len(sequence) for sequence in sequences}
    if len(lengths) != 1:
        raise ValueError(f"Parsed sequences have inconsistent lengths in {path}: {sorted(lengths)[:5]}")
    return names, sequences


def iter_windows(length: int, window_size: int, stride: int) -> list[tuple[int, int]]:
    windows = []
    start = 0
    while start < length:
        end = min(start + window_size, length)
        windows.append((start, end))
        if end == length:
            break
        start += stride
    return windows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Dataset stem such as 10725_NT_AL, or a .nex path")
    parser.add_argument("--nexus-root", type=Path, default=DEFAULT_NEXUS_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--max-windows", type=int, default=1)
    parser.add_argument("--start-window", type=int, default=0)
    parser.add_argument("--disable-fused-add-norm", action="store_true")
    parser.add_argument(
        "--force-cpu-reference-kernels",
        action="store_true",
        help="Use PyTorch/reference CPU ops instead of Triton/CUDA extension paths where possible.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    configure_imports()
    import mamba_ssm.modules.mamba_simple as mamba_simple
    import mamba_ssm.ops.selective_scan_interface as selective_scan_interface
    import phyla.model.model as phyla_model
    from phyla.model.model import Config, Phyla

    dataset_arg = Path(args.dataset)
    nexus_path = dataset_arg if dataset_arg.suffix else args.nexus_root / f"{args.dataset}.nex"
    names, sequences = load_nexus_alignment(nexus_path)
    alignment_length = len(sequences[0])
    windows = iter_windows(alignment_length, int(args.window_size), int(args.stride))
    selected_windows = windows[int(args.start_window) :]
    if int(args.max_windows) > 0:
        selected_windows = selected_windows[: int(args.max_windows)]
    if not selected_windows:
        raise ValueError("No windows selected")

    cfg = Config()
    cfg.model.model_name = "phyla-beta"
    if args.disable_fused_add_norm:
        cfg.model.fused_add_norm = False
    if args.force_cpu_reference_kernels:
        phyla_model.RMSNorm = torch.nn.RMSNorm
        mamba_simple.RMSNorm = torch.nn.RMSNorm
        mamba_simple.causal_conv1d_fn = None
        selective_scan_interface.selective_scan_cuda = None
    model = Phyla(cfg, device=args.device).load(checkpoint_file=str(args.checkpoint)).to(args.device)
    model.eval()

    timings = []
    with torch.inference_mode():
        for index, (start, end) in enumerate(selected_windows, start=1):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            window_start = time.perf_counter()
            chunk_sequences = [sequence[start:end] for sequence in sequences]
            encoded_aa, cls_token_mask, sequence_mask, _ = model.encode(chunk_sequences, names)
            output = model(
                encoded_aa.to(args.device),
                sequence_mask.to(args.device),
                cls_token_mask.to(args.device),
            )
            output.detach().cpu()
            elapsed = time.perf_counter() - window_start
            peak_memory_mb = (
                torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else None
            )
            max_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            item = {
                "window_index": int(args.start_window + index - 1),
                "start": int(start),
                "end": int(end),
                "length": int(end - start),
                "seconds": float(elapsed),
                "taxa": int(len(names)),
                "taxa_x_sites": int(len(names) * (end - start)),
                "peak_cuda_memory_mb": None if peak_memory_mb is None else float(peak_memory_mb),
                "max_rss_mb": float(max_rss_kb / 1024.0),
            }
            timings.append(item)
            print(json.dumps(item), flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": nexus_path.stem,
        "nexus_path": str(nexus_path),
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "model_name": "phyla-beta",
        "taxa": len(names),
        "sites": alignment_length,
        "window_size": int(args.window_size),
        "stride": int(args.stride),
        "total_windows": len(windows),
        "benchmarked_windows": timings,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
