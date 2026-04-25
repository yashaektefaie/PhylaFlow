#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence


ROOT = Path("/home/yektefai/PhylaFlow")
ANALYSIS_DIR = ROOT / "analysis/full_sanity_fixedpair_20260401"
BENCHMARK = ANALYSIS_DIR / "benchmark_mrbayes_fixed_start_ds1.py"
REPLAY_JSON = (
    ANALYSIS_DIR
    / "ds1_multipair78_caseadaptfhonly_step15000_replay_vs_old10_step900_keep_trees_20260424.json"
)
IQTREE_ML_TREE = ANALYSIS_DIR / "ds1_iqtree_mfp_runs10_ml_20260424.treefile"
USHER_SINGLE_TREE = ANALYSIS_DIR / "ds1_usher_matopt_mp_20260424/matopt_optimized.nwk"
USHER_MULTISTART_LIST = ANALYSIS_DIR / "ds1_usher_matopt_mp_multistart78_20260424/start_trees.txt"
MRBAYES_MP_20K_WORK = Path("/tmp/mrbayes_ds1_78_parsimony_g20000_20260424/parsimony")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.metric_utils import canonicalize_topology_newick  # noqa: E402


@dataclass(frozen=True)
class MethodSpec:
    slug: str
    label: str
    start_arg: str
    start_path: Path
    num_runs: int | None = None


def _ensure_semicolon(tree: str) -> str:
    tree = tree.strip()
    return tree if tree.endswith(";") else tree + ";"


def _load_replay_rows(num_runs: int) -> List[dict]:
    data = json.loads(REPLAY_JSON.read_text())
    rows = data["caseadapt78_step15000"]["rows"]
    if len(rows) < num_runs:
        raise ValueError(f"Replay only has {len(rows)} rows, requested {num_runs}")
    return list(rows[:num_runs])


def _write_replay_start_list(
    *,
    rows: Sequence[Mapping[str, object]],
    key: str,
    start_dir: Path,
    prefix: str,
) -> Path:
    start_dir.mkdir(parents=True, exist_ok=True)
    list_path = start_dir / f"{prefix}_start_trees.txt"
    lines: List[str] = []
    for idx, row in enumerate(rows):
        tree = _ensure_semicolon(str(row[key]))
        tree_path = start_dir / f"{prefix}_{idx:04d}.nwk"
        tree_path.write_text(tree + "\n")
        lines.append(str(tree_path))
    list_path.write_text("\n".join(lines) + "\n")
    return list_path


def _write_phylaflow_dedup_random_fill_list(
    *,
    rows: Sequence[Mapping[str, object]],
    start_dir: Path,
    num_runs: int,
) -> Path:
    start_dir.mkdir(parents=True, exist_ok=True)
    list_path = start_dir / "phylaflow_dedup_randomfill_start_trees.txt"
    manifest_path = start_dir / "phylaflow_dedup_randomfill_manifest.json"
    lines: List[str] = []
    seen_topologies: set[str] = set()
    manifest_rows: List[dict] = []

    for idx, row in enumerate(rows):
        tree = _ensure_semicolon(str(row["sampled_tree"]))
        key = canonicalize_topology_newick(tree)
        if key in seen_topologies:
            continue
        seen_topologies.add(key)
        tree_path = start_dir / f"phylaflow_dedup_{len(lines):04d}_row{idx:04d}.nwk"
        tree_path.write_text(tree + "\n")
        lines.append(str(tree_path))
        manifest_rows.append(
            {
                "source": "phylaflow_unique",
                "row_index": idx,
                "tree_path": str(tree_path),
            }
        )

    phylaflow_unique_count = len(lines)
    for idx, row in enumerate(rows):
        if len(lines) >= num_runs:
            break
        tree = _ensure_semicolon(str(row["start_tree"]))
        key = canonicalize_topology_newick(tree)
        if key in seen_topologies:
            continue
        seen_topologies.add(key)
        tree_path = start_dir / f"random_fill_{len(lines) - phylaflow_unique_count:04d}_row{idx:04d}.nwk"
        tree_path.write_text(tree + "\n")
        lines.append(str(tree_path))
        manifest_rows.append(
            {
                "source": "random_fill",
                "row_index": idx,
                "tree_path": str(tree_path),
            }
        )

    if len(lines) != num_runs:
        raise ValueError(
            f"Hybrid start list has {len(lines)} unique starts, expected {num_runs}"
        )
    list_path.write_text("\n".join(lines) + "\n")
    manifest_path.write_text(
        json.dumps(
            {
                "method": "phylaflow_dedup_randomfill",
                "requested_runs": int(num_runs),
                "phylaflow_unique_count": int(phylaflow_unique_count),
                "random_fill_count": int(num_runs - phylaflow_unique_count),
                "unique_topology_count": int(len(seen_topologies)),
                "rows": manifest_rows,
            },
            indent=2,
        )
        + "\n"
    )
    return list_path


def _write_phylaflow_plus_random_list(
    *,
    rows: Sequence[Mapping[str, object]],
    start_dir: Path,
    phylaflow_count: int,
    random_extra_count: int,
) -> Path:
    start_dir.mkdir(parents=True, exist_ok=True)
    total_count = int(phylaflow_count) + int(random_extra_count)
    prefix = f"phylaflow_full_plus_random{random_extra_count}"
    list_path = start_dir / f"{prefix}_start_trees.txt"
    manifest_path = start_dir / f"{prefix}_manifest.json"
    lines: List[str] = []
    manifest_rows: List[dict] = []

    if len(rows) < max(phylaflow_count, random_extra_count):
        raise ValueError(
            f"Need at least {max(phylaflow_count, random_extra_count)} rows, found {len(rows)}"
        )

    for idx, row in enumerate(rows[:phylaflow_count]):
        tree = _ensure_semicolon(str(row["sampled_tree"]))
        tree_path = start_dir / f"{prefix}_phylaflow_{idx:04d}.nwk"
        tree_path.write_text(tree + "\n")
        lines.append(str(tree_path))
        manifest_rows.append(
            {
                "source": "phylaflow_full",
                "row_index": idx,
                "tree_path": str(tree_path),
            }
        )

    for idx, row in enumerate(rows[:random_extra_count]):
        tree = _ensure_semicolon(str(row["start_tree"]))
        tree_path = start_dir / f"{prefix}_random_{idx:04d}.nwk"
        tree_path.write_text(tree + "\n")
        lines.append(str(tree_path))
        manifest_rows.append(
            {
                "source": "random_extra",
                "row_index": idx,
                "tree_path": str(tree_path),
            }
        )

    if len(lines) != total_count:
        raise ValueError(f"Hybrid start list has {len(lines)} starts, expected {total_count}")
    list_path.write_text("\n".join(lines) + "\n")
    manifest_path.write_text(
        json.dumps(
            {
                "method": prefix,
                "phylaflow_count": int(phylaflow_count),
                "random_extra_count": int(random_extra_count),
                "total_count": int(total_count),
                "rows": manifest_rows,
            },
            indent=2,
        )
        + "\n"
    )
    return list_path


def _mrbayes_tree_line_to_newick(line: str) -> str:
    if "=" not in line:
        raise ValueError(f"Not a MrBayes tree line: {line[:100]}")
    newick = line.split("=", 1)[1].strip()
    for prefix in ("[&U]", "[&R]"):
        if newick.startswith(prefix):
            newick = newick[len(prefix) :].strip()
    return _ensure_semicolon(newick)


def _extract_mrbayes_mp_start_list(*, start_dir: Path, num_runs: int) -> Path:
    run_files = sorted(MRBAYES_MP_20K_WORK.glob("run_*/run.t"))
    if len(run_files) < num_runs:
        raise ValueError(
            f"Found {len(run_files)} MrBayes MP run.t files under {MRBAYES_MP_20K_WORK}, "
            f"requested {num_runs}"
        )
    start_dir.mkdir(parents=True, exist_ok=True)
    list_path = start_dir / "mrbayes_mp_start_trees.txt"
    lines: List[str] = []
    for idx, run_file in enumerate(run_files[:num_runs]):
        gen0 = None
        with run_file.open() as handle:
            for line in handle:
                if line.lstrip().startswith("tree gen.0"):
                    gen0 = _mrbayes_tree_line_to_newick(line)
                    break
        if gen0 is None:
            raise ValueError(f"No tree gen.0 line found in {run_file}")
        tree_path = start_dir / f"mrbayes_mp_{idx:04d}.nwk"
        tree_path.write_text(gen0 + "\n")
        lines.append(str(tree_path))
    list_path.write_text("\n".join(lines) + "\n")
    return list_path


def _build_method_specs(*, num_runs: int, output_dir: Path) -> List[MethodSpec]:
    rows = _load_replay_rows(num_runs)
    start_dir = output_dir / "starts"
    random_list = _write_replay_start_list(
        rows=rows,
        key="start_tree",
        start_dir=start_dir,
        prefix="random",
    )
    phylaflow_list = _write_replay_start_list(
        rows=rows,
        key="sampled_tree",
        start_dir=start_dir,
        prefix="phylaflow_step15000",
    )
    hybrid_list = _write_phylaflow_dedup_random_fill_list(
        rows=rows,
        start_dir=start_dir,
        num_runs=num_runs,
    )
    phylaflow_plus_random20_list = _write_phylaflow_plus_random_list(
        rows=rows,
        start_dir=start_dir,
        phylaflow_count=num_runs,
        random_extra_count=20,
    )
    phylaflow_plus_random39_list = _write_phylaflow_plus_random_list(
        rows=rows,
        start_dir=start_dir,
        phylaflow_count=num_runs,
        random_extra_count=39,
    )
    mp_list = _extract_mrbayes_mp_start_list(start_dir=start_dir, num_runs=num_runs)

    for required in [IQTREE_ML_TREE, USHER_SINGLE_TREE, USHER_MULTISTART_LIST]:
        if not required.exists():
            raise FileNotFoundError(required)

    return [
        MethodSpec("random", "Random", "--start-tree-list", random_list),
        MethodSpec("mrbayes_mp", "MrBayes MP", "--start-tree-list", mp_list),
        MethodSpec("iqtree_ml", "IQ-TREE ML", "--start-tree", IQTREE_ML_TREE),
        MethodSpec("usher_matopt_single", "UShER+matOpt single", "--start-tree", USHER_SINGLE_TREE),
        MethodSpec(
            "usher_matopt_multistart",
            "UShER+matOpt multistart",
            "--start-tree-list",
            USHER_MULTISTART_LIST,
        ),
        MethodSpec("phylaflow", "PhylaFlow", "--start-tree-list", phylaflow_list),
        MethodSpec(
            "phylaflow_dedup_randomfill",
            "PhylaFlow dedup + random fill",
            "--start-tree-list",
            hybrid_list,
        ),
        MethodSpec(
            "phylaflow_plus_random20",
            "PhylaFlow full + random20",
            "--start-tree-list",
            phylaflow_plus_random20_list,
            num_runs=num_runs + 20,
        ),
        MethodSpec(
            "phylaflow_plus_random39",
            "PhylaFlow full + random39",
            "--start-tree-list",
            phylaflow_plus_random39_list,
            num_runs=num_runs + 39,
        ),
    ]


def _run_one_method(
    *,
    method: MethodSpec,
    args: argparse.Namespace,
    output_dir: Path,
    work_root: Path,
) -> Path:
    output_json = output_dir / f"{method.slug}_g{args.ngen}_curve.json"
    log_path = output_dir / f"{method.slug}_g{args.ngen}_curve.log"
    if output_json.exists() and not args.force:
        print(f"{method.label}: using existing {output_json}", flush=True)
        return output_json
    num_runs = int(method.num_runs or args.num_runs)

    cmd = [
        sys.executable,
        str(BENCHMARK),
        "--label",
        method.label,
        "--num-runs",
        str(num_runs),
        "--ngen",
        str(args.ngen),
        "--samplefreq",
        str(args.samplefreq),
        "--printfreq",
        str(args.printfreq),
        "--max-workers",
        str(args.max_workers),
        "--curve-interval",
        str(args.curve_interval),
        "--work-dir",
        str(work_root / method.slug),
        "--output",
        str(output_json),
        method.start_arg,
        str(method.start_path),
    ]

    print(f"{method.label}: launching {num_runs} chains to {args.ngen} generations", flush=True)
    last_report = -1
    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()
            try:
                progress = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(progress, dict):
                continue
            if not {"completed", "failed", "total"}.issubset(progress):
                continue
            completed = int(progress["completed"])
            failed = int(progress["failed"])
            total = int(progress["total"])
            if completed == total or completed // 10 != last_report:
                last_report = completed // 10
                print(
                    f"{method.label}: {completed}/{total} chains complete, {failed} failed",
                    flush=True,
                )
        returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(f"{method.label} failed with code {returncode}; see {log_path}")
    return output_json


def _curve_rows_for_method(result: Mapping[str, object], interval: int, ngen: int) -> List[dict]:
    rows = []
    for row in result["selected_cumulative_by_generation"]:
        generation = int(row["generation"])
        if generation % interval == 0 and generation <= ngen:
            rows.append(dict(row))
    return rows


def _fmt_float(value: object) -> str:
    number = float(value)
    if math.isnan(number):
        return "nan"
    return f"{number:.6f}"


def _write_tsv_and_plots(
    *,
    methods: Sequence[MethodSpec],
    result_paths: Mapping[str, Path],
    output_dir: Path,
    interval: int,
    ngen: int,
) -> None:
    loaded = {method.slug: json.loads(result_paths[method.slug].read_text()) for method in methods}
    curve_tsv = output_dir / "treeKL_curve.tsv"
    fields = [
        "method",
        "generation",
        "samples_per_run",
        "kl_divergence_tree_topology",
        "n_shared_topologies",
        "n_unique_posterior_topologies",
        "posterior_topology_support_recall",
        "support_rate_samples",
        "sampled_topology_mode_mass",
        "n_unique_sampled_topologies",
    ]
    lines = ["\t".join(fields)]
    for method in methods:
        for row in _curve_rows_for_method(loaded[method.slug], interval, ngen):
            line = [
                method.label,
                str(int(row["generation"])),
                str(int(row["samples_per_run"])),
                _fmt_float(row["kl_divergence_tree_topology"]),
                str(int(float(row["n_shared_topologies"]))),
                str(int(float(row["n_unique_posterior_topologies"]))),
                _fmt_float(row["posterior_topology_support_recall"]),
                _fmt_float(row["support_rate_samples"]),
                _fmt_float(row["sampled_topology_mode_mass"]),
                str(int(float(row["n_unique_sampled_topologies"]))),
            ]
            lines.append("\t".join(line))
    curve_tsv.write_text("\n".join(lines) + "\n")

    summary_md = output_dir / "curve_summary.md"
    generations = list(range(0, ngen + 1, interval))
    header = ["method", *[f"KL@{generation}" for generation in generations], "best_KL", "first<2", "first<1"]
    table = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    print("\nCurve summary")
    print("\t".join(header))
    for method in methods:
        result = loaded[method.slug]
        rows_by_gen = {
            int(row["generation"]): row
            for row in _curve_rows_for_method(result, interval, ngen)
        }
        kl_values = []
        for generation in generations:
            row = rows_by_gen.get(generation)
            kl_values.append(_fmt_float(row["kl_divergence_tree_topology"]) if row else "")
        best = result.get("best_cumulative", {})
        summary = [
            method.label,
            *kl_values,
            _fmt_float(best.get("kl_divergence_tree_topology", float("nan"))),
            str(result.get("first_generation_below_2")),
            str(result.get("first_generation_below_1")),
        ]
        print("\t".join(summary))
        table.append("| " + " | ".join(summary) + " |")
    summary_md.write_text("\n".join(table) + "\n")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"Skipping plots because matplotlib import failed: {exc}", flush=True)
        return

    def plot_metric(metric: str, ylabel: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        for method in methods:
            rows = _curve_rows_for_method(loaded[method.slug], interval, ngen)
            x = [int(row["generation"]) for row in rows]
            y = [float(row[metric]) for row in rows]
            ax.plot(x, y, marker="o", linewidth=1.8, markersize=4, label=method.label)
        ax.set_xlabel("MrBayes generations")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, ngen)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    plot_metric("kl_divergence_tree_topology", "Topology KL to DS1 posterior", "treeKL_curve.png")
    plot_metric("posterior_topology_support_recall", "Posterior topology support recall", "support_recall_curve.png")
    plot_metric("support_rate_samples", "Sample mass on posterior support", "support_rate_curve.png")
    print(f"\nWrote {curve_tsv}", flush=True)
    print(f"Wrote {summary_md}", flush=True)
    print(f"Wrote {output_dir / 'treeKL_curve.png'}", flush=True)
    print(f"Wrote {output_dir / 'support_recall_curve.png'}", flush=True)
    print(f"Wrote {output_dir / 'support_rate_curve.png'}", flush=True)


def _filter_methods(methods: Iterable[MethodSpec], requested: Sequence[str]) -> List[MethodSpec]:
    requested_set = set(requested)
    selected = [method for method in methods if method.slug in requested_set or method.label in requested_set]
    missing = requested_set.difference({method.slug for method in selected}).difference(
        {method.label for method in selected}
    )
    if missing:
        raise ValueError(f"Unknown methods: {', '.join(sorted(missing))}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ngen", type=int, default=100000)
    parser.add_argument("--curve-interval", type=int, default=20000)
    parser.add_argument("--samplefreq", type=int, default=200)
    parser.add_argument("--printfreq", type=int, default=5000)
    parser.add_argument("--num-runs", type=int, default=78)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument(
        "--output-dir",
        default=str(ANALYSIS_DIR / "ds1_mrbayes_generation_curves_100k_20260424"),
    )
    parser.add_argument(
        "--work-root",
        default="/tmp/mrbayes_ds1_generation_curves_100k_20260424",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "random",
            "mrbayes_mp",
            "iqtree_ml",
            "usher_matopt_single",
            "usher_matopt_multistart",
            "phylaflow",
            "phylaflow_dedup_randomfill",
            "phylaflow_plus_random20",
            "phylaflow_plus_random39",
        ],
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    work_root = Path(args.work_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    methods = _filter_methods(
        _build_method_specs(num_runs=int(args.num_runs), output_dir=output_dir),
        args.methods,
    )

    result_paths = {}
    for method in methods:
        result_paths[method.slug] = _run_one_method(
            method=method,
            args=args,
            output_dir=output_dir,
            work_root=work_root,
        )
    _write_tsv_and_plots(
        methods=methods,
        result_paths=result_paths,
        output_dir=output_dir,
        interval=int(args.curve_interval),
        ngen=int(args.ngen),
    )


if __name__ == "__main__":
    main()
