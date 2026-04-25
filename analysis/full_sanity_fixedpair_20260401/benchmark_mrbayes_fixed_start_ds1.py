#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path("/home/yektefai/PhylaFlow")
ANALYSIS_DIR = ROOT / "analysis/full_sanity_fixedpair_20260401"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from benchmark_mrbayes_initializations_ds1 import (  # noqa: E402
    _build_ds1_alignment_nexus,
    _collect_tree_files,
    _extract_newicks,
    _load_ds1_posterior_trees,
    _run_one,
)
from ete3 import Tree as EteTree  # noqa: E402
from utils.metric_utils import (  # noqa: E402
    canonicalize_topology_newick,
)


def _strip_internal_node_names(newick: str) -> str:
    tree = EteTree(newick, format=1)
    for node in tree.traverse():
        if not node.is_leaf():
            node.name = ""
    return tree.write(format=1)


def _topology_counts(trees: Sequence[str], cache: Dict[str, str]) -> Counter:
    counts: Counter = Counter()
    for tree in trees:
        key = cache.get(tree)
        if key is None:
            key = canonicalize_topology_newick(tree)
            cache[tree] = key
        counts[key] += 1
    return counts


def _tree_distribution_metrics_from_counts(
    posterior_counts: Counter,
    sampled_counts: Counter,
    *,
    alpha: float = 1e-6,
) -> Dict[str, float]:
    support = set(posterior_counts).union(sampled_counts)
    if not support:
        return {
            "kl_divergence_tree_topology": 0.0,
            "n_unique_posterior_topologies": 0.0,
            "n_unique_sampled_topologies": 0.0,
            "n_shared_topologies": 0.0,
            "posterior_topology_support_recall": 1.0,
            "support_rate_samples": 0.0,
            "sampled_topology_mode_mass": 0.0,
            "sample_count": 0.0,
        }

    posterior_total = float(sum(posterior_counts.values()))
    sample_count = sum(sampled_counts.values())
    sampled_total = float(sample_count)
    zp = posterior_total + alpha * len(support)
    zq = sampled_total + alpha * len(support)
    kl = 0.0
    for key in support:
        p = (float(posterior_counts.get(key, 0.0)) + alpha) / zp
        q = (float(sampled_counts.get(key, 0.0)) + alpha) / zq
        kl += p * math.log(p / q)

    shared = set(posterior_counts).intersection(sampled_counts)
    posterior_unique = len(posterior_counts)
    return {
        "kl_divergence_tree_topology": float(kl),
        "n_unique_posterior_topologies": float(posterior_unique),
        "n_unique_sampled_topologies": float(len(sampled_counts)),
        "n_shared_topologies": float(len(shared)),
        "posterior_topology_support_recall": (
            float(len(shared)) / float(posterior_unique) if posterior_unique else 1.0
        ),
        "support_rate_samples": (
            float(sum(sampled_counts[key] for key in shared)) / float(sample_count)
            if sample_count
            else 0.0
        ),
        "sampled_topology_mode_mass": (
            float(max(sampled_counts.values())) / float(sample_count)
            if sample_count
            else 0.0
        ),
        "sample_count": float(sample_count),
    }


def _tail_half(items: Sequence[str]) -> List[str]:
    return list(items[len(items) // 2 :]) if items else []


def _run_task(task: dict) -> dict:
    trees = _run_one(
        mrbayes_bin=Path(task["mrbayes_bin"]),
        work_dir=Path(task["run_dir"]),
        alignment_nexus_text=str(task["alignment_nexus_text"]),
        start_tree=str(task["start_tree"]),
        ngen=int(task["ngen"]),
        samplefreq=int(task["samplefreq"]),
        printfreq=int(task["printfreq"]),
        nchains=1,
        nruns=1,
        nperts=0,
    )
    return {"run_index": int(task["run_index"]), "sampled_trees": trees}


def _selected_cumulative(
    *,
    posterior_counts: Counter,
    sampled_trees_by_run: Sequence[Sequence[str]],
    samplefreq: int,
    generations: Sequence[int],
) -> Tuple[List[dict], int | None, int | None]:
    if not sampled_trees_by_run:
        return [], None, None
    max_samples = min(len(run) for run in sampled_trees_by_run)
    generation_set = set(int(generation) for generation in generations)
    rows_by_generation: Dict[int, dict] = {}
    seen_counts: Counter = Counter()
    cache: Dict[str, str] = {}
    below_2 = None
    below_1 = None
    for sample_idx in range(max_samples):
        for run in sampled_trees_by_run:
            tree = run[sample_idx]
            key = cache.get(tree)
            if key is None:
                key = canonicalize_topology_newick(tree)
                cache[tree] = key
            seen_counts[key] += 1
        generation = int(sample_idx * samplefreq)
        row = None
        if generation in generation_set:
            row = _tree_distribution_metrics_from_counts(posterior_counts, seen_counts)
            row["generation"] = generation
            row["samples_per_run"] = int(sample_idx + 1)
            rows_by_generation[generation] = row
        if below_2 is None or below_1 is None:
            if row is None:
                row = _tree_distribution_metrics_from_counts(posterior_counts, seen_counts)
            kl = float(row["kl_divergence_tree_topology"])
            if below_2 is None and kl < 2.0:
                below_2 = generation
            if below_1 is None and kl < 1.0:
                below_1 = generation
    return [rows_by_generation[generation] for generation in generations if generation in rows_by_generation], below_2, below_1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-tree")
    parser.add_argument(
        "--start-tree-list",
        help="Optional newline-delimited list of per-run Newick files. Overrides --start-tree.",
    )
    parser.add_argument("--label", default="fixed_start")
    parser.add_argument("--num-runs", type=int, default=78)
    parser.add_argument("--ngen", type=int, default=20000)
    parser.add_argument("--samplefreq", type=int, default=200)
    parser.add_argument("--printfreq", type=int, default=1000)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--posterior-samples-per-rep", type=int, default=1000)
    parser.add_argument(
        "--curve-interval",
        type=int,
        default=0,
        help="If positive, include cumulative metrics every N generations.",
    )
    parser.add_argument("--ds1-pickle", default="/home/yektefai/30272299/DS1.pickle")
    parser.add_argument("--golden-root", default="/home/yektefai/30272299/golden_run_data_DS1-8/DS1")
    parser.add_argument("--mrbayes-bin", default="/opt/conda/envs/phylaflow-mrbayes/bin/mb")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    start_tree_paths: List[Path]
    if args.start_tree_list:
        list_path = Path(args.start_tree_list).resolve()
        start_tree_paths = [
            Path(line.strip()).resolve()
            for line in list_path.read_text().splitlines()
            if line.strip()
        ]
        if not start_tree_paths:
            raise ValueError(f"No start tree paths found in {list_path}")
    elif args.start_tree:
        start_tree_paths = [Path(args.start_tree).resolve()]
    else:
        raise ValueError("Pass either --start-tree or --start-tree-list")
    start_trees = [
        _strip_internal_node_names(path.read_text().strip())
        for path in start_tree_paths
    ]
    work_dir = Path(args.work_dir).resolve()
    output_path = Path(args.output).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    alignment_path = work_dir / "DS1_real_alignment.nex"
    alignment_nexus_text, ordered_taxa = _build_ds1_alignment_nexus(
        ds1_pickle=Path(args.ds1_pickle).resolve(),
        translation_source=Path(args.golden_root).resolve() / "rep_1" / "DS1.trprobs",
        output_path=alignment_path,
    )
    posterior_trees = _load_ds1_posterior_trees(
        golden_root=Path(args.golden_root).resolve(),
        per_file_sample_count=int(args.posterior_samples_per_rep),
    )
    tree_key_cache: Dict[str, str] = {}
    posterior_counts = _topology_counts(posterior_trees, tree_key_cache)

    tasks = [
        {
            "run_index": run_index,
            "run_dir": str(work_dir / f"run_{run_index:04d}"),
            "mrbayes_bin": str(Path(args.mrbayes_bin).resolve()),
            "alignment_nexus_text": alignment_nexus_text,
            "start_tree": start_trees[run_index % len(start_trees)],
            "ngen": int(args.ngen),
            "samplefreq": int(args.samplefreq),
            "printfreq": int(args.printfreq),
        }
        for run_index in range(int(args.num_runs))
    ]

    completed: List[dict] = []
    failures: List[dict] = []
    max_workers = max(1, min(int(args.max_workers), len(tasks)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(_run_task, task): task for task in tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "run_index": int(task["run_index"]),
                        "run_dir": str(task["run_dir"]),
                        "error": str(exc),
                    }
                )
                print(
                    json.dumps(
                        {
                            "completed": len(completed),
                            "failed": len(failures),
                            "total": len(tasks),
                            "run_index": int(task["run_index"]),
                            "error": str(exc),
                        }
                    ),
                    flush=True,
                )
                continue
            completed.append(result)
            print(
                json.dumps(
                    {
                        "completed": len(completed),
                        "failed": len(failures),
                        "total": len(tasks),
                        "run_index": int(result["run_index"]),
                        "num_sampled_trees": len(result["sampled_trees"]),
                    }
                ),
                flush=True,
            )

    completed.sort(key=lambda item: int(item["run_index"]))
    sampled_by_run = [list(item["sampled_trees"]) for item in completed]
    all_samples = [tree for run in sampled_by_run for tree in run]
    tail_samples = [tree for run in sampled_by_run for tree in _tail_half(run)]
    initial_starts = [start_trees[int(item["run_index"]) % len(start_trees)] for item in completed]
    selected_generations = [
        generation
        for generation in [0, 200, 1000, 2000, 5000, 10000, int(args.ngen)]
        if generation <= int(args.ngen)
    ]
    if int(args.curve_interval) > 0:
        selected_generations.extend(
            range(0, int(args.ngen) + 1, int(args.curve_interval))
        )
    if int(args.ngen) not in selected_generations:
        selected_generations.append(int(args.ngen))
    selected_generations = sorted(set(selected_generations))
    cumulative, below_2, below_1 = _selected_cumulative(
        posterior_counts=posterior_counts,
        sampled_trees_by_run=sampled_by_run,
        samplefreq=int(args.samplefreq),
        generations=selected_generations,
    )
    final_cumulative = cumulative[-1] if cumulative else {}
    initial_counts = _topology_counts(initial_starts, tree_key_cache)
    all_counts = _topology_counts(all_samples, tree_key_cache)
    tail_counts = _topology_counts(tail_samples, tree_key_cache)

    output = {
        "label": str(args.label),
        "work_root": str(work_dir),
        "start_tree_path": str(start_tree_paths[0]),
        "start_tree_count": int(len(start_tree_paths)),
        "start_tree_paths": [str(path) for path in start_tree_paths],
        "num_runs": int(args.num_runs),
        "completed_runs": int(len(completed)),
        "failures": failures,
        "samplefreq": int(args.samplefreq),
        "ngen": int(args.ngen),
        "posterior_tree_count": int(len(posterior_trees)),
        "posterior_unique_topologies": int(len(posterior_counts)),
        "ordered_taxa": ordered_taxa,
        "initial_starts": _tree_distribution_metrics_from_counts(posterior_counts, initial_counts),
        "all_samples": _tree_distribution_metrics_from_counts(posterior_counts, all_counts),
        "tail_half_samples": _tree_distribution_metrics_from_counts(posterior_counts, tail_counts),
        "selected_cumulative_by_generation": cumulative,
        "final_cumulative_by_generation": final_cumulative,
        "first_generation_below_2": below_2,
        "first_generation_below_1": below_1,
        "per_run_sample_counts": [len(run) for run in sampled_by_run],
    }
    if cumulative:
        output["best_cumulative"] = min(
            cumulative,
            key=lambda row: float(row["kl_divergence_tree_topology"]),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
