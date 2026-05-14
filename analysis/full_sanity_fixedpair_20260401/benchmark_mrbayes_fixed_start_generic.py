#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import pickle
import random
import re
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ete3 import Tree as EteTree

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.metric_utils import canonicalize_topology_newick  # noqa: E402


def _extract_newick_from_line(line: str) -> str:
    start = line.find("(")
    if start == -1:
        return ""
    end = line.rfind(";")
    if end == -1:
        end = len(line)
    else:
        end += 1
    return line[start:end].strip()


def _extract_tree_weight(line: str) -> Optional[float]:
    match = re.search(r"p\s*=\s*([0-9.eE+-]+)", line)
    if match:
        return float(match.group(1))
    match = re.search(r"&W\s*([0-9.eE+-]+)", line)
    if match:
        return float(match.group(1))
    return None


def _parse_translate_block(path: Path) -> Dict[int, str]:
    text = path.read_text().splitlines()
    in_translate = False
    mapping: Dict[int, str] = {}
    for raw_line in text:
        line = raw_line.strip()
        if not in_translate:
            if line.lower().startswith("translate"):
                in_translate = True
            continue
        if not line:
            continue
        if line.endswith(";"):
            line = line[:-1]
            done = True
        else:
            done = False
        line = line.rstrip(",")
        match = re.match(r"(\d+)\s+(.+)$", line)
        if match:
            mapping[int(match.group(1))] = match.group(2).strip().strip(",")
        if done:
            break
    if not mapping:
        raise ValueError(f"Failed to parse translate block from {path}")
    return mapping


def _build_alignment_nexus(
    *,
    dataset_pickle: Path,
    translation_source: Path,
    output_path: Path,
) -> Tuple[str, List[str]]:
    seqs: Dict[str, str] = pickle.load(open(dataset_pickle, "rb"))
    translate = _parse_translate_block(translation_source)
    ordered_taxa = [translate[i] for i in sorted(translate)]
    missing = [name for name in ordered_taxa if name not in seqs]
    if missing:
        raise ValueError(f"{dataset_pickle} is missing taxa: {missing}")

    lengths = {len(seqs[name]) for name in ordered_taxa}
    if len(lengths) != 1:
        raise ValueError(f"Sequences are not aligned to one length: {sorted(lengths)}")
    nchar = next(iter(lengths))

    rows = [f"{i}    {seqs[name]}" for i, name in enumerate(ordered_taxa, start=1)]
    nexus_text = (
        "#NEXUS\n\n"
        "BEGIN DATA;\n"
        f"    DIMENSIONS NTAX={len(rows)} NCHAR={nchar};\n"
        "    FORMAT DATATYPE=DNA MISSING=? GAP=-;\n"
        "    MATRIX\n"
        + "\n".join(rows)
        + "\n    ;\nEND;\n"
    )
    output_path.write_text(nexus_text)
    return nexus_text, ordered_taxa


def _expand_weighted_trees(
    trees: Sequence[str],
    weights: Sequence[Optional[float]],
    sample_count: int,
) -> List[str]:
    if sample_count <= 0 or not trees:
        return list(trees)
    clean_weights = [
        max(0.0, float(weight)) if weight is not None else 0.0
        for weight in weights
    ]
    total = sum(clean_weights)
    if total <= 0.0:
        return list(trees)

    scaled = [(weight / total) * sample_count for weight in clean_weights]
    counts = [int(math.floor(value)) for value in scaled]
    remainder = sample_count - sum(counts)
    if remainder > 0:
        order = sorted(
            range(len(scaled)),
            key=lambda idx: (scaled[idx] - counts[idx], clean_weights[idx]),
            reverse=True,
        )
        for idx in order[:remainder]:
            counts[idx] += 1

    expanded: List[str] = []
    for tree, count in zip(trees, counts):
        if count > 0:
            expanded.extend([tree] * count)
    return expanded


def _load_posterior_trees(
    *,
    golden_root: Path,
    dataset_id: str,
    per_file_sample_count: int,
) -> List[str]:
    trprobs_paths = sorted(golden_root.glob(f"rep_*/{dataset_id}.trprobs"))
    if not trprobs_paths:
        raise ValueError(f"No {dataset_id}.trprobs files found under {golden_root}")

    all_trees: List[str] = []
    for path in trprobs_paths:
        file_trees: List[str] = []
        file_weights: List[Optional[float]] = []
        for raw_line in path.read_text().splitlines():
            newick = _extract_newick_from_line(raw_line)
            if not newick:
                continue
            file_trees.append(newick)
            file_weights.append(_extract_tree_weight(raw_line))
        all_trees.extend(
            _expand_weighted_trees(file_trees, file_weights, per_file_sample_count)
        )
    if not all_trees:
        raise ValueError(f"Posterior tree set is empty under {golden_root}")
    return all_trees


def _collect_tree_files(prefix: Path) -> List[Path]:
    candidates = [
        prefix.with_suffix(".t"),
        prefix.parent / f"{prefix.name}.run1.t",
        prefix.parent / f"{prefix.name}.run2.t",
    ]
    found = [path for path in candidates if path.exists()]
    if not found:
        found = sorted(prefix.parent.glob(f"{prefix.name}*.t"))
    return found


def _extract_newicks(path: Path) -> List[str]:
    trees: List[str] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        newick = _extract_newick_from_line(line)
        if newick:
            trees.append(newick)
    return trees


def _build_run_nexus_text(
    *,
    alignment_nexus_text: str,
    start_tree: str,
    filename_prefix: str,
    ngen: int,
    samplefreq: int,
    printfreq: int,
    nchains: int,
    nruns: int,
    nperts: int,
    mrbayes_pre_lines: Sequence[str] = (),
    mrbayes_extra_lines: Sequence[str] = (),
) -> str:
    pre_text = "".join(f"    {line.rstrip()}\n" for line in mrbayes_pre_lines if line.strip())
    extra_text = "".join(f"    {line.rstrip()}\n" for line in mrbayes_extra_lines if line.strip())
    return (
        "#NEXUS\n\n"
        + alignment_nexus_text
        + "\nBEGIN TREES;\n"
        + f"    TREE init = {start_tree}\n"
        + "END;\n\n"
        + "BEGIN MRBAYES;\n"
        + "    set autoclose=yes nowarn=yes quitonerror=yes;\n"
        + pre_text
        + "    startvals tau=init;\n"
        + "    startvals v=init;\n"
        + f"    mcmcp filename={filename_prefix} nruns={nruns} nchains={nchains} "
        + f"ngen={ngen} samplefreq={samplefreq} printfreq={printfreq} "
        + f"diagnfreq={max(samplefreq, printfreq)} checkpoint=no append=no "
        + f"starttree=current nperts={nperts};\n"
        + extra_text
        + "    mcmc;\n"
        + "END;\n"
    )


def _remap_numeric_zero_based_leaves(tree: EteTree, num_taxa: int) -> None:
    leaf_names = [str(leaf.name) for leaf in tree.iter_leaves()]
    if not leaf_names:
        return
    try:
        numeric = [int(name) for name in leaf_names]
    except ValueError:
        return
    numeric_set = set(numeric)
    one_based = set(range(1, int(num_taxa) + 1))
    zero_based = set(range(0, int(num_taxa)))
    if numeric_set == one_based:
        return
    if numeric_set == zero_based:
        for leaf in tree.iter_leaves():
            leaf.name = str(int(leaf.name) + 1)
        return
    if 0 in numeric_set and max(numeric_set) < int(num_taxa):
        for leaf in tree.iter_leaves():
            leaf.name = str(int(leaf.name) + 1)


def _sanitize_start_tree(
    newick: str,
    *,
    num_taxa: int,
    min_branch_length: float = 1e-6,
) -> str:
    tree = EteTree(newick, format=1)
    _remap_numeric_zero_based_leaves(tree, num_taxa=int(num_taxa))
    for node in tree.traverse("postorder"):
        if not node.is_leaf():
            node.name = ""
            node.children.sort(
                key=lambda child: ",".join(
                    sorted(str(name) for name in child.get_leaf_names())
                )
            )
    random.seed(1)
    tree.resolve_polytomy(
        default_dist=float(min_branch_length),
        default_support=0.0,
        recursive=True,
    )
    for node in tree.traverse():
        if not math.isfinite(float(node.dist)) or float(node.dist) < min_branch_length:
            node.dist = float(min_branch_length)
    return tree.write(format=1)


def _run_one(
    *,
    mrbayes_bin: Path,
    work_dir: Path,
    alignment_nexus_text: str,
    start_tree: str,
    ngen: int,
    samplefreq: int,
    printfreq: int,
    nchains: int,
    nruns: int,
    nperts: int,
    num_taxa: int,
    mrbayes_pre_lines: Sequence[str] = (),
    mrbayes_extra_lines: Sequence[str] = (),
    reuse_existing: bool = False,
) -> List[str]:
    work_dir.mkdir(parents=True, exist_ok=True)
    prefix = work_dir / "run"
    if reuse_existing:
        existing_trees: List[str] = []
        for tree_file in _collect_tree_files(prefix):
            existing_trees.extend(_extract_newicks(tree_file))
        if existing_trees:
            return existing_trees
    nexus_path = work_dir / "job.nex"
    safe_start_tree = _sanitize_start_tree(start_tree, num_taxa=int(num_taxa))
    nexus_path.write_text(
        _build_run_nexus_text(
            alignment_nexus_text=alignment_nexus_text,
            start_tree=safe_start_tree,
            filename_prefix="run",
            ngen=ngen,
            samplefreq=samplefreq,
            printfreq=printfreq,
            nchains=nchains,
            nruns=nruns,
            nperts=nperts,
            mrbayes_pre_lines=mrbayes_pre_lines,
            mrbayes_extra_lines=mrbayes_extra_lines,
        )
    )
    log_path = work_dir / "stdout.log"
    with log_path.open("w") as log_file:
        result = subprocess.run(
            [str(mrbayes_bin), nexus_path.name],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(work_dir),
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"MrBayes failed for {work_dir} with code {result.returncode}. See {log_path}"
        )

    trees: List[str] = []
    for tree_file in _collect_tree_files(prefix):
        trees.extend(_extract_newicks(tree_file))
    if not trees:
        raise RuntimeError(f"No tree samples found under {work_dir}")
    return trees


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
        nchains=int(task["nchains"]),
        nruns=int(task["nruns"]),
        nperts=int(task["nperts"]),
        num_taxa=int(task["num_taxa"]),
        mrbayes_pre_lines=list(task.get("mrbayes_pre_lines", [])),
        mrbayes_extra_lines=list(task.get("mrbayes_extra_lines", [])),
        reuse_existing=bool(task.get("reuse_existing", False)),
    )
    return {"run_index": int(task["run_index"]), "sampled_trees": trees}


def _selected_cumulative(
    *,
    posterior_counts: Counter,
    sampled_trees_by_run: Sequence[Sequence[str]],
    samplefreq: int,
    generations: Sequence[int],
    threshold_check_selected_only: bool = False,
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
        should_check_thresholds = (
            below_2 is None or below_1 is None
        ) and (row is not None or not threshold_check_selected_only)
        if should_check_thresholds:
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
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-pickle", required=True)
    parser.add_argument("--golden-root", required=True)
    parser.add_argument("--start-tree")
    parser.add_argument(
        "--start-tree-list",
        help="Optional newline-delimited list of per-run Newick files. Overrides --start-tree.",
    )
    parser.add_argument("--label", default="fixed_start")
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--ngen", type=int, default=100000)
    parser.add_argument("--samplefreq", type=int, default=200)
    parser.add_argument("--printfreq", type=int, default=5000)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--posterior-samples-per-rep", type=int, default=1000)
    parser.add_argument("--curve-interval", type=int, default=20000)
    parser.add_argument("--mrbayes-bin", default="/opt/conda/envs/phylaflow-mrbayes/bin/mb")
    parser.add_argument("--nchains", type=int, default=1)
    parser.add_argument("--nruns", type=int, default=1)
    parser.add_argument("--nperts", type=int, default=0)
    parser.add_argument(
        "--mrbayes-extra-line",
        action="append",
        default=[],
        help="Additional line to insert after mcmcp and before mcmc, e.g. a propset command.",
    )
    parser.add_argument(
        "--mrbayes-extra-lines-file",
        action="append",
        default=[],
        help="File containing additional MrBayes block lines to insert after mcmcp and before mcmc.",
    )
    parser.add_argument(
        "--mrbayes-pre-line",
        action="append",
        default=[],
        help="Additional line to insert after set and before startvals.",
    )
    parser.add_argument(
        "--mrbayes-pre-lines-file",
        action="append",
        default=[],
        help="File containing additional MrBayes block lines to insert after set and before startvals.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing run.<t> files under --work-dir instead of rerunning MrBayes.",
    )
    parser.add_argument(
        "--threshold-check-selected-only",
        action="store_true",
        help=(
            "Only evaluate first_generation_below_1/2 at requested curve generations. "
            "This keeps curve metrics identical while avoiding an expensive per-sample scan."
        ),
    )
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.start_tree_list:
        list_path = Path(args.start_tree_list).resolve()
        start_trees = []
        for raw_line in list_path.read_text().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("(") or line.startswith("[") or line.endswith(";"):
                start_trees.append(line)
                continue
            maybe_path = Path(line).resolve()
            try:
                is_existing_path = maybe_path.exists()
            except OSError:
                is_existing_path = False
            if is_existing_path:
                start_trees.append(maybe_path.read_text().strip())
            else:
                start_trees.append(line)
        if not start_trees:
            raise ValueError(f"No start trees found in {list_path}")
    elif args.start_tree:
        start_trees = [Path(args.start_tree).resolve().read_text().strip()]
    else:
        raise ValueError("Pass either --start-tree or --start-tree-list")
    mrbayes_pre_lines = list(args.mrbayes_pre_line)
    for pre_file in args.mrbayes_pre_lines_file:
        mrbayes_pre_lines.extend(
            line.strip()
            for line in Path(pre_file).read_text().splitlines()
            if line.strip()
        )
    mrbayes_extra_lines = list(args.mrbayes_extra_line)
    for extra_file in args.mrbayes_extra_lines_file:
        mrbayes_extra_lines.extend(
            line.strip()
            for line in Path(extra_file).read_text().splitlines()
            if line.strip()
        )
    work_dir = Path(args.work_dir).resolve()
    output_path = Path(args.output).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    golden_root = Path(args.golden_root).resolve()
    dataset_id = str(args.dataset_id)
    alignment_path = work_dir / f"{dataset_id}_real_alignment.nex"
    alignment_nexus_text, ordered_taxa = _build_alignment_nexus(
        dataset_pickle=Path(args.dataset_pickle).resolve(),
        translation_source=golden_root / "rep_1" / f"{dataset_id}.trprobs",
        output_path=alignment_path,
    )
    posterior_trees = _load_posterior_trees(
        golden_root=golden_root,
        dataset_id=dataset_id,
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
            "nchains": int(args.nchains),
            "nruns": int(args.nruns),
            "nperts": int(args.nperts),
            "num_taxa": int(len(ordered_taxa)),
            "mrbayes_pre_lines": list(mrbayes_pre_lines),
            "mrbayes_extra_lines": list(mrbayes_extra_lines),
            "reuse_existing": bool(args.reuse_existing),
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
        threshold_check_selected_only=bool(args.threshold_check_selected_only),
    )
    final_cumulative = cumulative[-1] if cumulative else {}
    initial_counts = _topology_counts(initial_starts, tree_key_cache)
    all_counts = _topology_counts(all_samples, tree_key_cache)
    tail_counts = _topology_counts(tail_samples, tree_key_cache)

    output = {
        "label": str(args.label),
        "dataset_id": dataset_id,
        "work_root": str(work_dir),
        "start_tree_source": str(args.start_tree_list or args.start_tree),
        "start_tree_count": int(len(start_trees)),
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
