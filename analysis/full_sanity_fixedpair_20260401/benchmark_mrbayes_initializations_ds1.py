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
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ete3 import Tree as EteTree

ROOT = Path("/home/yektefai/PhylaFlow")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.metric_utils import (  # noqa: E402
    calculate_norm_rf,
    canonicalize_topology_newick,
)


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


def _build_ds1_alignment_nexus(
    *,
    ds1_pickle: Path,
    translation_source: Path,
    output_path: Path,
) -> Tuple[str, List[str]]:
    seqs: Dict[str, str] = pickle.load(open(ds1_pickle, "rb"))
    translate = _parse_translate_block(translation_source)
    ordered_taxa = [translate[i] for i in sorted(translate)]
    missing = [name for name in ordered_taxa if name not in seqs]
    if missing:
        raise ValueError(f"DS1 pickle is missing taxa: {missing}")

    lengths = {len(seqs[name]) for name in ordered_taxa}
    if len(lengths) != 1:
        raise ValueError(f"DS1 sequences are not aligned to one length: {sorted(lengths)}")
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
    counts = [int(math.floor(x)) for x in scaled]
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


def _load_ds1_posterior_trees(
    *,
    golden_root: Path,
    per_file_sample_count: int,
) -> List[str]:
    trprobs_paths = sorted(golden_root.glob("rep_*/DS1.trprobs"))
    if not trprobs_paths:
        raise ValueError(f"No DS1.trprobs files found under {golden_root}")

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
) -> str:
    return (
        "#NEXUS\n\n"
        + alignment_nexus_text
        + "\nBEGIN TREES;\n"
        + f"    TREE init = {start_tree}\n"
        + "END;\n\n"
        + "BEGIN MRBAYES;\n"
        + "    set autoclose=yes nowarn=yes quitonerror=yes;\n"
        + "    startvals tau=init;\n"
        + "    startvals v=init;\n"
        + f"    mcmcp filename={filename_prefix} nruns={nruns} nchains={nchains} "
        + f"ngen={ngen} samplefreq={samplefreq} printfreq={printfreq} "
        + f"diagnfreq={max(samplefreq, printfreq)} checkpoint=no append=no "
        + f"starttree=current nperts={nperts};\n"
        + "    mcmc;\n"
        + "END;\n"
    )


def _sanitize_start_tree(newick: str, min_branch_length: float = 1e-6) -> str:
    tree = EteTree(newick, format=1)
    for node in tree.traverse("postorder"):
        if not node.is_leaf():
            node.children.sort(
                key=lambda child: ",".join(sorted(str(name) for name in child.get_leaf_names()))
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
) -> List[str]:
    work_dir.mkdir(parents=True, exist_ok=True)
    prefix = work_dir / "run"
    nexus_path = work_dir / "job.nex"
    safe_start_tree = _sanitize_start_tree(start_tree)
    nexus_text = _build_run_nexus_text(
        alignment_nexus_text=alignment_nexus_text,
        start_tree=safe_start_tree,
        filename_prefix="run",
        ngen=ngen,
        samplefreq=samplefreq,
        printfreq=printfreq,
        nchains=nchains,
        nruns=nruns,
        nperts=nperts,
    )
    nexus_path.write_text(nexus_text)
    log_path = work_dir / "stdout.log"
    with log_path.open("w") as log_file:
        result = subprocess.run(
            [str(mrbayes_bin), str(nexus_path)],
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


def _min_rf_to_reference(tree: str, reference_trees: Sequence[str]) -> float:
    return min(float(calculate_norm_rf(tree, ref)) for ref in reference_trees)


def _support_rate(sampled_trees: Sequence[str], posterior_support: set[str]) -> float:
    if not sampled_trees:
        return 0.0
    hits = sum(
        1 for tree in sampled_trees if canonicalize_topology_newick(tree) in posterior_support
    )
    return float(hits) / float(len(sampled_trees))


def _unique_topology_representatives(trees: Sequence[str]) -> List[str]:
    reps: Dict[str, str] = {}
    for tree in trees:
        key = canonicalize_topology_newick(tree)
        if key not in reps:
            reps[key] = tree
    return list(reps.values())


def _tail_half(items: Sequence[str]) -> List[str]:
    if not items:
        return []
    return list(items[len(items) // 2 :])


def _group_phylaflow_topologies(rows: Sequence[dict]) -> List[dict]:
    grouped: Dict[str, dict] = {}
    for row in rows:
        key = canonicalize_topology_newick(str(row["_sampled_tree"]))
        bucket = grouped.setdefault(
            key,
            {
                "topology_key": key,
                "sample_count": 0,
                "pair_indices": [],
                "representative_row": row,
            },
        )
        bucket["sample_count"] += 1
        bucket["pair_indices"].append(int(row["pair_index"]))
        if int(row["pair_index"]) < int(bucket["representative_row"]["pair_index"]):
            bucket["representative_row"] = row

    return sorted(
        grouped.values(),
        key=lambda item: (-int(item["sample_count"]), int(item["representative_row"]["pair_index"])),
    )


def _select_phylaflow_diverse_rows(
    rows: Sequence[dict],
    *,
    num_pairs: int,
    candidate_pool_size: int,
) -> Tuple[List[dict], List[dict]]:
    candidates = _group_phylaflow_topologies(rows)
    if not candidates:
        return [], []

    pool = candidates[: max(int(num_pairs), int(candidate_pool_size))]
    selected: List[dict] = [pool.pop(0)]

    while len(selected) < int(num_pairs) and pool:
        best_idx = None
        best_score = None
        for idx, candidate in enumerate(pool):
            candidate_tree = str(candidate["representative_row"]["_sampled_tree"])
            min_diversity = min(
                float(calculate_norm_rf(candidate_tree, str(sel["representative_row"]["_sampled_tree"])))
                for sel in selected
            )
            score = (float(min_diversity), int(candidate["sample_count"]), -int(candidate["representative_row"]["pair_index"]))
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
        selected.append(pool.pop(best_idx))

    return [item["representative_row"] for item in selected], selected


def _normalize_replay_row(row: dict) -> dict:
    normalized = dict(row)
    if "pair_index" not in normalized and "case_index" in normalized:
        normalized["pair_index"] = int(normalized["case_index"])
    if "_sampled_tree" not in normalized and "sampled_tree" in normalized:
        normalized["_sampled_tree"] = normalized["sampled_tree"]
    if "start_rf_norm" not in normalized and "start_rf_to_target" in normalized:
        normalized["start_rf_norm"] = float(normalized["start_rf_to_target"])
    if "rf_norm" not in normalized and "sampled_rf_to_target" in normalized:
        normalized["rf_norm"] = float(normalized["sampled_rf_to_target"])
    required = [
        "pair_index",
        "start_tree",
        "target_tree",
        "_sampled_tree",
        "start_rf_norm",
        "rf_norm",
    ]
    missing = [key for key in required if key not in normalized]
    if missing:
        raise KeyError(f"Replay row is missing required fields {missing}: {sorted(row)}")
    return normalized


def _aggregate_metrics(
    *,
    sampled_trees: List[str],
    posterior_support: set[str],
) -> Dict[str, float]:
    if not sampled_trees:
        return {
            "sample_count": 0.0,
            "support_rate": 0.0,
            "n_unique_sampled_topologies": 0.0,
            "sampled_topology_mode_mass": 0.0,
        }
    topo_counts: Dict[str, int] = {}
    for tree in sampled_trees:
        key = canonicalize_topology_newick(tree)
        topo_counts[key] = topo_counts.get(key, 0) + 1
    support_hits = [key for key in topo_counts if key in posterior_support]
    return {
        "sample_count": float(len(sampled_trees)),
        "support_rate": float(_support_rate(sampled_trees, posterior_support)),
        "n_unique_sampled_topologies": float(len(topo_counts)),
        "sampled_topology_mode_mass": float(max(topo_counts.values())) / float(len(sampled_trees)),
        "sampled_support_topology_count": float(len(support_hits)),
        "posterior_support_topology_recall": (
            float(len(support_hits)) / float(len(posterior_support)) if posterior_support else 1.0
        ),
    }


def _cumulative_generation_metrics(
    *,
    sampled_trees_by_run: Sequence[Sequence[str]],
    reference_trees: Sequence[str],
    posterior_support: set[str],
    samplefreq: int,
) -> List[Dict[str, float]]:
    if not sampled_trees_by_run:
        return []
    max_samples = min(len(run) for run in sampled_trees_by_run)
    if max_samples <= 0:
        return []

    trajectory: List[Dict[str, float]] = []
    for sample_idx in range(max_samples):
        current_trees = [run[sample_idx] for run in sampled_trees_by_run]
        seen_trees: List[str] = []
        for run in sampled_trees_by_run:
            seen_trees.extend(run[: sample_idx + 1])

        current_min_rfs = [
            float(_min_rf_to_reference(tree, reference_trees)) for tree in current_trees
        ]
        run_hit_fraction = float(
            sum(
                1
                for run in sampled_trees_by_run
                if any(canonicalize_topology_newick(tree) in posterior_support for tree in run[: sample_idx + 1])
            )
        ) / float(len(sampled_trees_by_run))

        topo_counts: Dict[str, int] = {}
        for tree in seen_trees:
            key = canonicalize_topology_newick(tree)
            topo_counts[key] = topo_counts.get(key, 0) + 1
        support_hits = [key for key in topo_counts if key in posterior_support]

        trajectory.append(
            {
                "generation": int(sample_idx * samplefreq),
                "samples_per_run": int(sample_idx + 1),
                "run_hit_fraction": float(run_hit_fraction),
                "mean_current_min_rf_to_posterior": float(statistics.mean(current_min_rfs)),
                "best_current_min_rf_to_posterior": float(min(current_min_rfs)),
                "cumulative_sample_count": float(len(seen_trees)),
                "cumulative_unique_topologies": float(len(topo_counts)),
                "cumulative_support_topology_count": float(len(support_hits)),
                "cumulative_posterior_support_recall": (
                    float(len(support_hits)) / float(len(posterior_support))
                    if posterior_support
                    else 1.0
                ),
                "cumulative_support_rate": float(_support_rate(seen_trees, posterior_support)),
            }
        )
    return trajectory


def _per_run_summary(
    *,
    start_tree: str,
    sampled_trees: List[str],
    reference_trees: List[str],
    posterior_support: set[str],
    samplefreq: int,
) -> Dict[str, object]:
    start_min_rf = _min_rf_to_reference(start_tree, reference_trees)
    if not sampled_trees:
        return {
            "num_sampled_trees": 0.0,
            "start_min_rf_to_posterior": float(start_min_rf),
            "final_min_rf_to_posterior": float("nan"),
            "best_min_rf_over_samples": float("nan"),
            "support_hit_rate": 0.0,
            "first_support_hit_sample": None,
            "first_support_hit_generation": None,
            "min_rf_trajectory": [],
        }

    min_rf_trajectory = [
        float(_min_rf_to_reference(tree, reference_trees)) for tree in sampled_trees
    ]
    first_support_hit_sample = None
    for idx, tree in enumerate(sampled_trees):
        if canonicalize_topology_newick(tree) in posterior_support:
            first_support_hit_sample = idx
            break

    return {
        "num_sampled_trees": float(len(sampled_trees)),
        "start_min_rf_to_posterior": float(start_min_rf),
        "final_min_rf_to_posterior": float(min_rf_trajectory[-1]),
        "best_min_rf_over_samples": float(min(min_rf_trajectory)),
        "support_hit_rate": float(_support_rate(sampled_trees, posterior_support)),
        "first_support_hit_sample": first_support_hit_sample,
        "first_support_hit_generation": (
            int(first_support_hit_sample * samplefreq)
            if first_support_hit_sample is not None
            else None
        ),
        "min_rf_trajectory": min_rf_trajectory,
    }


def _run_one_method_task(task: dict) -> dict:
    sampled_trees = _run_one(
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
    )
    return {
        "method": str(task["method"]),
        "row": task["row"],
        "sampled_trees": sampled_trees,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replay-json",
        default=str(
            ROOT
            / "analysis/full_sanity_fixedpair_20260401/ds1_caseadaptboth_step22000_tree_level_replay_exact_20260421.json"
        ),
    )
    parser.add_argument(
        "--rows-key",
        default="",
        help="Optional top-level key containing a {'rows': [...]} replay block.",
    )
    parser.add_argument(
        "--ds1-pickle",
        default="/home/yektefai/30272299/DS1.pickle",
    )
    parser.add_argument(
        "--golden-root",
        default="/home/yektefai/30272299/golden_run_data_DS1-8/DS1",
    )
    parser.add_argument(
        "--mrbayes-bin",
        default="/opt/conda/envs/phylaflow-mrbayes/bin/mb",
    )
    parser.add_argument("--num-pairs", type=int, default=4)
    parser.add_argument(
        "--selection-mode",
        choices=["first_k", "phylaflow_topofreq_diverse"],
        default="first_k",
    )
    parser.add_argument("--candidate-pool-size", type=int, default=24)
    parser.add_argument("--ngen", type=int, default=100)
    parser.add_argument("--samplefreq", type=int, default=10)
    parser.add_argument("--printfreq", type=int, default=50)
    parser.add_argument("--nruns", type=int, default=1)
    parser.add_argument("--nchains", type=int, default=1)
    parser.add_argument("--nperts", type=int, default=0)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["phylaflow", "random"],
        default=["phylaflow", "random"],
    )
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--posterior-samples-per-rep", type=int, default=1000)
    parser.add_argument(
        "--work-dir",
        default="/tmp/mrbayes_init_benchmark_ds1",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    replay_json = Path(args.replay_json).resolve()
    ds1_pickle = Path(args.ds1_pickle).resolve()
    golden_root = Path(args.golden_root).resolve()
    mrbayes_bin = Path(args.mrbayes_bin).resolve()
    work_dir = Path(args.work_dir).resolve()
    output_path = Path(args.output).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    replay = json.loads(replay_json.read_text())
    row_source = replay
    if args.rows_key:
        row_source = replay[str(args.rows_key)]
    elif "rows" not in row_source:
        row_keys = [
            key
            for key, value in replay.items()
            if isinstance(value, dict) and isinstance(value.get("rows"), list)
        ]
        if len(row_keys) == 1:
            row_source = replay[row_keys[0]]
        else:
            raise KeyError(
                "Replay JSON has no top-level 'rows'. Pass --rows-key with one of: "
                + ", ".join(row_keys)
            )
    all_rows = [_normalize_replay_row(row) for row in row_source["rows"]]
    selection_summary: Optional[List[dict]] = None
    if args.selection_mode == "phylaflow_topofreq_diverse":
        rows, selected_meta = _select_phylaflow_diverse_rows(
            all_rows,
            num_pairs=int(args.num_pairs),
            candidate_pool_size=int(args.candidate_pool_size),
        )
        selection_summary = [
            {
                "pair_index": int(item["representative_row"]["pair_index"]),
                "sample_count": int(item["sample_count"]),
                "pair_indices": list(item["pair_indices"]),
            }
            for item in selected_meta
        ]
    else:
        rows = all_rows[: int(args.num_pairs)]
    if not rows:
        raise ValueError("No replay rows selected.")

    alignment_path = work_dir / "DS1_real_alignment.nex"
    alignment_nexus_text, ordered_taxa = _build_ds1_alignment_nexus(
        ds1_pickle=ds1_pickle,
        translation_source=golden_root / "rep_1" / "DS1.trprobs",
        output_path=alignment_path,
    )

    posterior_trees = _load_ds1_posterior_trees(
        golden_root=golden_root,
        per_file_sample_count=int(args.posterior_samples_per_rep),
    )
    reference_trees = _unique_topology_representatives(posterior_trees)
    posterior_support = {canonicalize_topology_newick(tree) for tree in posterior_trees}
    num_leaves = len(ordered_taxa)

    methods = {method: [] for method in args.methods}
    per_run: Dict[str, List[dict]] = {method: [] for method in args.methods}

    tasks = []
    for row in rows:
        pair_index = int(row["pair_index"])
        start_specs = {
            "phylaflow": str(row["_sampled_tree"]),
            "random": str(row["start_tree"]),
        }
        for method in args.methods:
            start_tree = start_specs[method]
            tasks.append(
                {
                    "method": method,
                    "row": row,
                    "run_dir": str(work_dir / method / f"pair_{pair_index:04d}"),
                    "mrbayes_bin": str(mrbayes_bin),
                    "alignment_nexus_text": alignment_nexus_text,
                    "start_tree": start_tree,
                    "ngen": int(args.ngen),
                    "samplefreq": int(args.samplefreq),
                    "printfreq": int(args.printfreq),
                    "nchains": int(args.nchains),
                    "nruns": int(args.nruns),
                    "nperts": int(args.nperts),
                }
            )

    max_workers = max(1, min(int(args.max_workers), len(tasks)))
    failures = []
    if max_workers == 1:
        completed = []
        for task in tasks:
            try:
                completed.append(_run_one_method_task(task))
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "method": str(task["method"]),
                        "pair_index": int(task["row"]["pair_index"]),
                        "run_dir": str(task["run_dir"]),
                        "error": str(exc),
                    }
                )
    else:
        completed = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(_run_one_method_task, task): task for task in tasks
            }
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    failures.append(
                        {
                            "method": str(task["method"]),
                            "pair_index": int(task["row"]["pair_index"]),
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
                                "method": str(task["method"]),
                                "pair_index": int(task["row"]["pair_index"]),
                                "error": str(exc),
                            }
                        ),
                        flush=True,
                    )
                    continue
                completed.append(result)
                row = result["row"]
                print(
                    json.dumps(
                        {
                            "completed": len(completed),
                            "total": len(tasks),
                            "method": result["method"],
                            "pair_index": int(row["pair_index"]),
                            "num_sampled_trees": len(result["sampled_trees"]),
                        }
                    ),
                    flush=True,
                )

    completed.sort(key=lambda item: (str(item["method"]), int(item["row"]["pair_index"])))
    for item in completed:
        method = item["method"]
        row = item["row"]
        pair_index = int(row["pair_index"])
        sampled_trees = list(item["sampled_trees"])
        start_tree = str(row["_sampled_tree"] if method == "phylaflow" else row["start_tree"])
        methods[method].extend(sampled_trees)
        per_run[method].append(
            {
                "pair_index": pair_index,
                "_sampled_trees": sampled_trees,
                "source_target_tree": str(row["target_tree"]),
                "source_random_start_tree": str(row["start_tree"]),
                "source_phylaflow_tree": str(row["_sampled_tree"]),
                "source_random_start_rf_to_target": float(row["start_rf_norm"]),
                "source_phylaflow_rf_to_target": float(row["rf_norm"]),
                **_per_run_summary(
                    start_tree=start_tree,
                    sampled_trees=sampled_trees,
                    reference_trees=reference_trees,
                    posterior_support=posterior_support,
                    samplefreq=int(args.samplefreq),
                ),
            }
        )

    results = {
        "benchmark": {
            "replay_json": str(replay_json),
            "checkpoint": replay.get("checkpoint"),
            "ds1_pickle": str(ds1_pickle),
            "golden_root": str(golden_root),
            "alignment_nexus": str(alignment_path),
            "mrbayes_bin": str(mrbayes_bin),
            "num_pairs": int(args.num_pairs),
            "selection_mode": str(args.selection_mode),
            "candidate_pool_size": int(args.candidate_pool_size),
            "ngen": int(args.ngen),
            "samplefreq": int(args.samplefreq),
            "printfreq": int(args.printfreq),
            "nruns": int(args.nruns),
            "nchains": int(args.nchains),
            "nperts": int(args.nperts),
            "methods": list(args.methods),
            "max_workers": int(max_workers),
            "failures": failures,
            "posterior_samples_per_rep": int(args.posterior_samples_per_rep),
            "num_reference_trees": int(len(posterior_trees)),
            "num_reference_unique_topologies": int(len(posterior_support)),
            "num_reference_rf_topologies": int(len(reference_trees)),
            "num_taxa": int(num_leaves),
            "ordered_taxa": ordered_taxa,
            "selected_pair_indices": [int(row["pair_index"]) for row in rows],
        },
        "methods": {},
    }
    if selection_summary is not None:
        results["benchmark"]["selection_summary"] = selection_summary

    for method, sampled_trees in methods.items():
        tail_trees = _tail_half(sampled_trees)
        run_rows = per_run[method]
        if not run_rows:
            results["methods"][method] = {
                "all_samples": _aggregate_metrics(
                    sampled_trees=[],
                    posterior_support=posterior_support,
                ),
                "tail_half_samples": _aggregate_metrics(
                    sampled_trees=[],
                    posterior_support=posterior_support,
                ),
                "cumulative_by_generation": [],
                "per_run": [],
                "per_run_mean_start_min_rf_to_posterior": float("nan"),
                "per_run_mean_final_min_rf_to_posterior": float("nan"),
                "per_run_mean_best_min_rf_over_samples": float("nan"),
                "best_min_rf_any_sample": float("nan"),
                "per_run_mean_support_hit_rate": float("nan"),
                "per_run_support_hit_fraction": 0.0,
                "per_run_mean_first_support_hit_generation": None,
            }
            continue
        support_hit_rows = [
            row["first_support_hit_generation"]
            for row in run_rows
            if row["first_support_hit_generation"] is not None
        ]
        results["methods"][method] = {
            "all_samples": _aggregate_metrics(
                sampled_trees=sampled_trees,
                posterior_support=posterior_support,
            ),
            "tail_half_samples": _aggregate_metrics(
                sampled_trees=tail_trees,
                posterior_support=posterior_support,
            ),
            "cumulative_by_generation": _cumulative_generation_metrics(
                sampled_trees_by_run=[row["_sampled_trees"] for row in run_rows],
                reference_trees=reference_trees,
                posterior_support=posterior_support,
                samplefreq=int(args.samplefreq),
            ),
            "per_run": [
                {k: v for k, v in row.items() if k != "_sampled_trees"} for row in run_rows
            ],
            "per_run_mean_start_min_rf_to_posterior": float(
                statistics.mean(row["start_min_rf_to_posterior"] for row in run_rows)
            ),
            "per_run_mean_final_min_rf_to_posterior": float(
                statistics.mean(row["final_min_rf_to_posterior"] for row in run_rows)
            ),
            "per_run_mean_best_min_rf_over_samples": float(
                statistics.mean(row["best_min_rf_over_samples"] for row in run_rows)
            ),
            "best_min_rf_any_sample": float(
                min((row["best_min_rf_over_samples"] for row in run_rows), default=float("nan"))
            ),
            "per_run_mean_support_hit_rate": float(
                statistics.mean(row["support_hit_rate"] for row in run_rows)
            ),
            "per_run_support_hit_fraction": float(len(support_hit_rows)) / float(len(run_rows)),
            "per_run_mean_first_support_hit_generation": (
                float(statistics.mean(support_hit_rows)) if support_hit_rows else None
            ),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
