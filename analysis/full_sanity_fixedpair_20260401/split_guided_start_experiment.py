#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from ete3 import Tree as EteTree

ROOT = Path("/home/yektefai/PhylaFlow")
ANALYSIS_DIR = ROOT / "analysis/full_sanity_fixedpair_20260401"
BENCHMARK = ANALYSIS_DIR / "benchmark_mrbayes_fixed_start_generic.py"

DS1_REPLAY = (
    ANALYSIS_DIR
    / "ds1_multipair78_caseadaptfhonly_step15000_replay_vs_old10_step900_keep_trees_20260424.json"
)
DS5_NORMRF_EXPORT = (
    ANALYSIS_DIR
    / "ds5_checkpoint_samples_20260426/ds5_normrf_step24000_checkpoint_samples.json"
)
DS5_TREEKL_EXPORT = (
    ANALYSIS_DIR
    / "ds5_checkpoint_samples_20260426/ds5_treekl_step12000_checkpoint_samples.json"
)

DATASETS = {
    "DS1": {
        "pickle": Path("/home/yektefai/30272299/DS1.pickle"),
        "golden_root": Path("/home/yektefai/30272299/golden_run_data_DS1-8/DS1"),
        "num_taxa": 27,
        "default_runs": 78,
    },
    "DS5": {
        "pickle": Path("/home/yektefai/30272299/DS5.pickle"),
        "golden_root": Path("/home/yektefai/30272299/golden_run_data_DS1-8/DS5"),
        "num_taxa": 50,
        "default_runs": 525,
    },
}

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.full_sanity_fixedpair_20260401.benchmark_mrbayes_fixed_start_generic import (  # noqa: E402
    _load_posterior_trees,
    _topology_counts,
    _tree_distribution_metrics_from_counts,
)


Split = Tuple[str, ...]


@dataclass(frozen=True)
class StartSet:
    slug: str
    label: str
    trees: List[str]
    source: str
    stats: Mapping[str, float] | None = None


def _ensure_semicolon(tree: str) -> str:
    tree = tree.strip()
    return tree if tree.endswith(";") else tree + ";"


def _label_sort_key(name: str) -> Tuple[int, str]:
    try:
        return (0, f"{int(name):020d}")
    except Exception:
        return (1, str(name))


def _sorted_labels(labels: Iterable[str]) -> List[str]:
    return sorted((str(label) for label in labels), key=_label_sort_key)


def _normalise_numeric_zero_based(tree: EteTree) -> None:
    leaf_names = [str(leaf.name) for leaf in tree.iter_leaves()]
    try:
        numeric = [int(name) for name in leaf_names]
    except ValueError:
        return
    n = len(numeric)
    if set(numeric) == set(range(n)):
        for leaf in tree.iter_leaves():
            leaf.name = str(int(leaf.name) + 1)


def _parse_tree(tree: str) -> EteTree:
    parsed = EteTree(_ensure_semicolon(tree), format=1, quoted_node_names=True)
    _normalise_numeric_zero_based(parsed)
    return parsed


def _canonical_split(side: Iterable[str], universe: frozenset[str]) -> Split | None:
    left = frozenset(str(item) for item in side)
    right = universe.difference(left)
    if len(left) <= 1 or len(right) <= 1:
        return None
    left_key = (len(left), tuple(_sorted_labels(left)))
    right_key = (len(right), tuple(_sorted_labels(right)))
    chosen = left if left_key <= right_key else right
    return tuple(_sorted_labels(chosen))


def _extract_splits(tree: str) -> Tuple[frozenset[str], List[Split]]:
    parsed = _parse_tree(tree)
    universe = frozenset(str(leaf.name) for leaf in parsed.iter_leaves())
    splits: List[Split] = []
    seen: set[Split] = set()
    for node in parsed.traverse("postorder"):
        if node is parsed or node.is_leaf():
            continue
        split = _canonical_split(node.get_leaf_names(), universe)
        if split is None or split in seen:
            continue
        seen.add(split)
        splits.append(split)
    return universe, splits


def _split_counts(trees: Sequence[str]) -> Tuple[frozenset[str], Counter]:
    universe: frozenset[str] | None = None
    counts: Counter = Counter()
    for tree in trees:
        tree_universe, splits = _extract_splits(tree)
        if universe is None:
            universe = tree_universe
        elif universe != tree_universe:
            raise ValueError("Tree leaf sets differ inside one split-count batch")
        counts.update(splits)
    return universe or frozenset(), counts


def _split_kl_from_counts(
    posterior_counts: Counter,
    sampled_counts: Counter,
    *,
    alpha: float = 1e-6,
) -> Dict[str, float]:
    support = set(posterior_counts).union(sampled_counts)
    if not support:
        return {
            "split_kl": 0.0,
            "n_unique_posterior_splits": 0.0,
            "n_unique_sampled_splits": 0.0,
            "n_shared_splits": 0.0,
            "posterior_split_support_recall": 1.0,
        }
    posterior_total = float(sum(posterior_counts.values()))
    sampled_total = float(sum(sampled_counts.values()))
    zp = posterior_total + alpha * len(support)
    zq = sampled_total + alpha * len(support)
    kl = 0.0
    for split in support:
        p = (float(posterior_counts.get(split, 0.0)) + alpha) / zp
        q = (float(sampled_counts.get(split, 0.0)) + alpha) / zq
        kl += p * math.log(p / q)
    shared = set(posterior_counts).intersection(sampled_counts)
    return {
        "split_kl": float(kl),
        "n_unique_posterior_splits": float(len(posterior_counts)),
        "n_unique_sampled_splits": float(len(sampled_counts)),
        "n_shared_splits": float(len(shared)),
        "posterior_split_support_recall": (
            float(len(shared)) / float(len(posterior_counts))
            if posterior_counts
            else 1.0
        ),
    }


def _compatible_split(a: Split, b: Split, universe: frozenset[str]) -> bool:
    left_a = frozenset(a)
    left_b = frozenset(b)
    right_a = universe.difference(left_a)
    right_b = universe.difference(left_b)
    return (
        not left_a.intersection(left_b)
        or not left_a.intersection(right_b)
        or not right_a.intersection(left_b)
        or not right_a.intersection(right_b)
    )


def _orient_clades(
    splits: Sequence[Split],
    *,
    universe: frozenset[str],
    root_label: str,
) -> List[frozenset[str]]:
    clades: List[frozenset[str]] = []
    for split in splits:
        side = frozenset(split)
        clade = universe.difference(side) if root_label in side else side
        if 1 < len(clade) < len(universe):
            clades.append(clade)
    clades.sort(key=lambda clade: (len(clade), tuple(_sorted_labels(clade))))
    return clades


def _random_join(children: List[str], rng: random.Random) -> str:
    if not children:
        raise ValueError("Cannot join an empty child list")
    rng.shuffle(children)
    while len(children) > 1:
        a = children.pop()
        b = children.pop()
        children.append(f"({a},{b}):0.001")
        rng.shuffle(children)
    return children[0]


def _tree_from_compatible_splits(
    *,
    universe: frozenset[str],
    splits: Sequence[Split],
    rng: random.Random,
) -> str:
    root_label = _sorted_labels(universe)[0]
    clades = _orient_clades(splits, universe=universe, root_label=root_label)
    clade_set = set(clades)

    def direct_child_clades(node_set: frozenset[str]) -> List[frozenset[str]]:
        candidates = [
            clade
            for clade in clades
            if clade < node_set and clade in clade_set
        ]
        direct: List[frozenset[str]] = []
        for candidate in candidates:
            if not any(candidate < other < node_set for other in candidates):
                direct.append(candidate)
        direct.sort(key=lambda clade: (len(clade), tuple(_sorted_labels(clade))))
        return direct

    def emit(node_set: frozenset[str], *, is_root: bool = False) -> str:
        if len(node_set) == 1:
            return f"{next(iter(node_set))}:0.1"
        child_clades = direct_child_clades(node_set)
        covered = frozenset().union(*child_clades) if child_clades else frozenset()
        children = [emit(clade) for clade in child_clades]
        children.extend(f"{label}:0.1" for label in _sorted_labels(node_set.difference(covered)))
        if len(children) == 1:
            body = children[0]
        elif len(children) == 2:
            body = f"({children[0]},{children[1]})"
        else:
            body = _random_join(children, rng)
            if body.endswith(":0.001"):
                body = body[: -len(":0.001")]
        return body if is_root else f"{body}:0.001"

    newick = emit(universe, is_root=True) + ";"
    parsed = EteTree(newick, format=1)
    parsed.resolve_polytomy(default_dist=0.001, default_support=0.0, recursive=True)
    for node in parsed.traverse():
        if not math.isfinite(float(node.dist)) or float(node.dist) <= 0.0:
            node.dist = 0.001
    return parsed.write(format=1)


def _split_frequencies(trees: Sequence[str]) -> Tuple[frozenset[str], Dict[Split, float]]:
    universe, counts = _split_counts(trees)
    total_trees = float(len(trees))
    if total_trees <= 0:
        return universe, {}
    return universe, {split: float(count) / total_trees for split, count in counts.items()}


def _gumbel(rng: random.Random) -> float:
    u = min(max(rng.random(), 1e-12), 1.0 - 1e-12)
    return -math.log(-math.log(u))


def _sample_split_guided_trees(
    *,
    guide_trees: Sequence[str],
    num_trees: int,
    seed: int,
    max_candidates: int,
    temperature: float,
    diversity_strength: float,
    greedy: bool,
) -> Tuple[List[str], Dict[str, float]]:
    universe, frequencies = _split_frequencies(guide_trees)
    if not frequencies:
        raise ValueError("No guide splits available")
    candidates = sorted(
        frequencies.items(),
        key=lambda item: (-item[1], item[0]),
    )[:max_candidates]
    target_splits = max(0, len(universe) - 3)
    rng = random.Random(seed)
    used_counts: Counter = Counter()
    trees: List[str] = []
    accepted_split_counts: List[int] = []

    for tree_index in range(num_trees):
        if greedy:
            ordered = list(candidates)
        else:
            generated_so_far = max(1, tree_index)
            ordered = sorted(
                candidates,
                key=lambda item: (
                    math.log(max(item[1], 1e-12))
                    + temperature * _gumbel(rng)
                    - diversity_strength
                    * (float(used_counts[item[0]]) / float(generated_so_far))
                ),
                reverse=True,
            )

        accepted: List[Split] = []
        for split, _freq in ordered:
            if len(accepted) >= target_splits:
                break
            if all(_compatible_split(split, existing, universe) for existing in accepted):
                accepted.append(split)
        for split in accepted:
            used_counts[split] += 1
        accepted_split_counts.append(len(accepted))
        trees.append(
            _tree_from_compatible_splits(
                universe=universe,
                splits=accepted,
                rng=rng,
            )
        )

    return trees, {
        "guide_split_count": float(len(frequencies)),
        "candidate_split_count": float(len(candidates)),
        "mean_accepted_guide_splits": (
            float(sum(accepted_split_counts)) / float(len(accepted_split_counts))
            if accepted_split_counts
            else 0.0
        ),
        "target_internal_splits": float(target_splits),
        "guide_tree_count": float(len(guide_trees)),
    }


def _topology_metrics(
    *,
    posterior_counts: Counter,
    posterior_split_counts: Counter,
    trees: Sequence[str],
    cache: Dict[str, str],
) -> Dict[str, float]:
    tree_counts = _topology_counts(trees, cache)
    _universe, split_counts = _split_counts(trees)
    metrics = _tree_distribution_metrics_from_counts(posterior_counts, tree_counts)
    metrics.update(_split_kl_from_counts(posterior_split_counts, split_counts))
    return metrics


def _load_ds1_sets(num_runs: int) -> List[StartSet]:
    data = json.loads(DS1_REPLAY.read_text())
    rows = list(data["caseadapt78_step15000"]["rows"])[:num_runs]
    if len(rows) < num_runs:
        raise ValueError(f"DS1 replay has {len(rows)} rows, requested {num_runs}")
    return [
        StartSet(
            slug="random",
            label="Random",
            trees=[_ensure_semicolon(str(row["start_tree"])) for row in rows],
            source=str(DS1_REPLAY),
        ),
        StartSet(
            slug="phylaflow",
            label="PhylaFlow terminal",
            trees=[_ensure_semicolon(str(row["sampled_tree"])) for row in rows],
            source=str(DS1_REPLAY),
        ),
    ]


def _load_export_rows(path: Path, num_runs: int) -> List[dict]:
    data = json.loads(path.read_text())
    rows = list(data["rows"])[:num_runs]
    if len(rows) < num_runs:
        raise ValueError(f"{path} has {len(rows)} rows, requested {num_runs}")
    return rows


def _load_ds5_sets(num_runs: int) -> List[StartSet]:
    normrf_rows = _load_export_rows(DS5_NORMRF_EXPORT, num_runs)
    treekl_rows = _load_export_rows(DS5_TREEKL_EXPORT, num_runs)
    return [
        StartSet(
            slug="random",
            label="Random",
            trees=[_ensure_semicolon(str(row["start_tree"])) for row in normrf_rows],
            source=str(DS5_NORMRF_EXPORT),
        ),
        StartSet(
            slug="phylaflow_normrf",
            label="PhylaFlow normRF ckpt",
            trees=[_ensure_semicolon(str(row["sampled_tree"])) for row in normrf_rows],
            source=str(DS5_NORMRF_EXPORT),
        ),
        StartSet(
            slug="phylaflow_treekl",
            label="PhylaFlow treeKL ckpt",
            trees=[_ensure_semicolon(str(row["sampled_tree"])) for row in treekl_rows],
            source=str(DS5_TREEKL_EXPORT),
        ),
    ]


def _write_start_set(output_dir: Path, start_set: StartSet) -> Path:
    start_dir = output_dir / "starts" / start_set.slug
    start_dir.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for idx, tree in enumerate(start_set.trees):
        tree_path = start_dir / f"{start_set.slug}_{idx:04d}.nwk"
        tree_path.write_text(_ensure_semicolon(tree) + "\n")
        lines.append(str(tree_path))
    list_path = output_dir / "starts" / f"{start_set.slug}_start_trees.txt"
    list_path.write_text("\n".join(lines) + "\n")
    return list_path


def _format_float(value: object) -> str:
    number = float(value)
    if math.isnan(number):
        return "nan"
    return f"{number:.6f}"


def _write_initial_tables(rows: Sequence[Mapping[str, object]], output_dir: Path) -> None:
    fields = [
        "dataset",
        "method",
        "n",
        "kl_divergence_tree_topology",
        "split_kl",
        "n_shared_topologies",
        "posterior_topology_support_recall",
        "support_rate_samples",
        "sampled_topology_mode_mass",
        "n_unique_sampled_topologies",
        "n_shared_splits",
        "posterior_split_support_recall",
        "n_unique_sampled_splits",
        "mean_accepted_guide_splits",
        "guide_split_count",
    ]
    lines = ["\t".join(fields)]
    for row in rows:
        line: List[str] = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                line.append(_format_float(value))
            elif field.startswith("n_") or field == "mean_accepted_guide_splits":
                try:
                    line.append(_format_float(value))
                except Exception:
                    line.append(str(value))
            else:
                line.append(str(value))
        lines.append("\t".join(line))
    (output_dir / "initial_metrics.tsv").write_text("\n".join(lines) + "\n")

    md_fields = [
        "dataset",
        "method",
        "KL-tree",
        "KL-split",
        "shared topo",
        "support rate",
        "mode mass",
        "unique topo",
    ]
    table = [
        "| " + " | ".join(md_fields) + " |",
        "| " + " | ".join(["---"] * len(md_fields)) + " |",
    ]
    for row in rows:
        table.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["method"]),
                    _format_float(row["kl_divergence_tree_topology"]),
                    _format_float(row["split_kl"]),
                    str(int(float(row["n_shared_topologies"]))),
                    _format_float(row["support_rate_samples"]),
                    _format_float(row["sampled_topology_mode_mass"]),
                    str(int(float(row["n_unique_sampled_topologies"]))),
                ]
            )
            + " |"
        )
    (output_dir / "initial_metrics.md").write_text("\n".join(table) + "\n")


def _generate_dataset(
    *,
    dataset_id: str,
    num_runs: int,
    output_dir: Path,
    posterior_samples_per_rep: int,
    seed: int,
    max_candidates: int,
    temperature: float,
    diversity_strength: float,
) -> Tuple[List[dict], List[StartSet]]:
    dataset = DATASETS[dataset_id]
    posterior_trees = _load_posterior_trees(
        golden_root=Path(dataset["golden_root"]),
        dataset_id=dataset_id,
        per_file_sample_count=posterior_samples_per_rep,
    )
    cache: Dict[str, str] = {}
    posterior_counts = _topology_counts(posterior_trees, cache)
    _posterior_universe, posterior_split_counts = _split_counts(posterior_trees)

    baseline_sets = _load_ds1_sets(num_runs) if dataset_id == "DS1" else _load_ds5_sets(num_runs)
    generated_sets: List[StartSet] = []

    for start_set in baseline_sets:
        if start_set.slug == "random":
            continue
        trees, stats = _sample_split_guided_trees(
            guide_trees=start_set.trees,
            num_trees=num_runs,
            seed=seed + len(generated_sets) * 997 + (1 if dataset_id == "DS1" else 5000),
            max_candidates=max_candidates,
            temperature=temperature,
            diversity_strength=diversity_strength,
            greedy=False,
        )
        generated_sets.append(
            StartSet(
                slug=f"{start_set.slug}_splitguided",
                label=f"{start_set.label} split-guided",
                trees=trees,
                source=f"split marginals from {start_set.slug}",
                stats=stats,
            )
        )

        greedy_trees, greedy_stats = _sample_split_guided_trees(
            guide_trees=start_set.trees,
            num_trees=num_runs,
            seed=seed + len(generated_sets) * 997 + (2 if dataset_id == "DS1" else 6000),
            max_candidates=max_candidates,
            temperature=0.0,
            diversity_strength=0.0,
            greedy=True,
        )
        generated_sets.append(
            StartSet(
                slug=f"{start_set.slug}_topsplits",
                label=f"{start_set.label} top-splits",
                trees=greedy_trees,
                source=f"top compatible splits from {start_set.slug}",
                stats=greedy_stats,
            )
        )

    oracle_trees, oracle_stats = _sample_split_guided_trees(
        guide_trees=posterior_trees,
        num_trees=num_runs,
        seed=seed + (10000 if dataset_id == "DS1" else 20000),
        max_candidates=max_candidates,
        temperature=temperature,
        diversity_strength=diversity_strength,
        greedy=False,
    )
    generated_sets.append(
        StartSet(
            slug="posterior_split_oracle",
            label="Posterior split oracle",
            trees=oracle_trees,
            source="golden posterior split marginals",
            stats=oracle_stats,
        )
    )

    rows: List[dict] = []
    for start_set in [*baseline_sets, *generated_sets]:
        metrics = _topology_metrics(
            posterior_counts=posterior_counts,
            posterior_split_counts=posterior_split_counts,
            trees=start_set.trees,
            cache=cache,
        )
        stats = dict(start_set.stats or {})
        row = {
            "dataset": dataset_id,
            "method": start_set.label,
            "slug": start_set.slug,
            "n": int(len(start_set.trees)),
            "source": start_set.source,
            **metrics,
            **stats,
        }
        rows.append(row)

    dataset_dir = output_dir / dataset_id.lower()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for start_set in [*baseline_sets, *generated_sets]:
        list_path = _write_start_set(dataset_dir, start_set)
        rows_by_slug = [row for row in rows if row["slug"] == start_set.slug]
        if rows_by_slug:
            rows_by_slug[0]["start_tree_list"] = str(list_path)

    (dataset_dir / "initial_metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    _write_initial_tables(rows, dataset_dir)
    return rows, [*baseline_sets, *generated_sets]


def _run_benchmark(
    *,
    dataset_id: str,
    method_slug: str,
    method_label: str,
    start_tree_list: Path,
    num_runs: int,
    output_dir: Path,
    work_root: Path,
    ngen: int,
    samplefreq: int,
    printfreq: int,
    curve_interval: int,
    max_workers: int,
    force: bool,
) -> Path:
    dataset = DATASETS[dataset_id]
    output_json = output_dir / f"{method_slug}_g{ngen}_curve.json"
    log_path = output_dir / f"{method_slug}_g{ngen}_curve.log"
    if output_json.exists() and not force:
        print(f"{dataset_id} {method_label}: using existing {output_json}", flush=True)
        return output_json

    cmd = [
        sys.executable,
        str(BENCHMARK),
        "--dataset-id",
        dataset_id,
        "--dataset-pickle",
        str(dataset["pickle"]),
        "--golden-root",
        str(dataset["golden_root"]),
        "--label",
        method_label,
        "--num-runs",
        str(num_runs),
        "--ngen",
        str(ngen),
        "--samplefreq",
        str(samplefreq),
        "--printfreq",
        str(printfreq),
        "--max-workers",
        str(max_workers),
        "--curve-interval",
        str(curve_interval),
        "--work-dir",
        str(work_root / dataset_id.lower() / method_slug),
        "--output",
        str(output_json),
        "--start-tree-list",
        str(start_tree_list),
    ]

    print(f"{dataset_id} {method_label}: launching {num_runs} chains to {ngen}", flush=True)
    last_bucket = -1
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
            bucket = completed // max(1, total // 10)
            if bucket != last_bucket or completed == total:
                last_bucket = bucket
                print(
                    f"{dataset_id} {method_label}: {completed}/{total} chains complete, {failed} failed",
                    flush=True,
                )
        returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(f"{dataset_id} {method_label} failed; see {log_path}")
    return output_json


def _write_curve_tsv(result_paths: Sequence[Tuple[str, str, Path]], output_dir: Path) -> None:
    fields = [
        "dataset",
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
    for dataset_id, label, path in result_paths:
        result = json.loads(path.read_text())
        for row in result["selected_cumulative_by_generation"]:
            lines.append(
                "\t".join(
                    [
                        dataset_id,
                        label,
                        str(int(row["generation"])),
                        str(int(row["samples_per_run"])),
                        _format_float(row["kl_divergence_tree_topology"]),
                        str(int(float(row["n_shared_topologies"]))),
                        str(int(float(row["n_unique_posterior_topologies"]))),
                        _format_float(row["posterior_topology_support_recall"]),
                        _format_float(row["support_rate_samples"]),
                        _format_float(row["sampled_topology_mode_mass"]),
                        str(int(float(row["n_unique_sampled_topologies"]))),
                    ]
                )
            )
    (output_dir / "mrbayes_curves.tsv").write_text("\n".join(lines) + "\n")


def _selected_methods_for_mrbayes(rows: Sequence[Mapping[str, object]]) -> List[Mapping[str, object]]:
    selected: List[Mapping[str, object]] = []
    by_dataset: Dict[str, List[Mapping[str, object]]] = {}
    for row in rows:
        by_dataset.setdefault(str(row["dataset"]), []).append(row)
    for dataset_id, dataset_rows in by_dataset.items():
        for row in dataset_rows:
            slug = str(row["slug"])
            if slug == "random" or slug.endswith("_splitguided"):
                selected.append(row)
        split_rows = [row for row in dataset_rows if str(row["slug"]).endswith("_splitguided")]
        if split_rows:
            best = min(split_rows, key=lambda row: float(row["kl_divergence_tree_topology"]))
            if best not in selected:
                selected.append(best)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["DS1", "DS5"], choices=sorted(DATASETS))
    parser.add_argument(
        "--output-dir",
        default=str(ANALYSIS_DIR / "split_guided_search_ds1_ds5_20260426"),
    )
    parser.add_argument("--posterior-samples-per-rep", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument("--max-candidates", type=int, default=5000)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--diversity-strength", type=float, default=1.5)
    parser.add_argument("--run-mrbayes", action="store_true")
    parser.add_argument("--ngen", type=int, default=20000)
    parser.add_argument("--samplefreq", type=int, default=200)
    parser.add_argument("--printfreq", type=int, default=5000)
    parser.add_argument("--curve-interval", type=int, default=20000)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[dict] = []
    for dataset_id in args.datasets:
        rows, _sets = _generate_dataset(
            dataset_id=dataset_id,
            num_runs=int(DATASETS[dataset_id]["default_runs"]),
            output_dir=output_dir,
            posterior_samples_per_rep=int(args.posterior_samples_per_rep),
            seed=int(args.seed),
            max_candidates=int(args.max_candidates),
            temperature=float(args.temperature),
            diversity_strength=float(args.diversity_strength),
        )
        all_rows.extend(rows)

    (output_dir / "initial_metrics.json").write_text(json.dumps(all_rows, indent=2) + "\n")
    _write_initial_tables(all_rows, output_dir)
    print((output_dir / "initial_metrics.tsv").read_text())

    if not args.run_mrbayes:
        return

    curve_paths: List[Tuple[str, str, Path]] = []
    work_root = Path(f"/tmp/split_guided_search_ds1_ds5_g{args.ngen}_20260426")
    for row in _selected_methods_for_mrbayes(all_rows):
        start_tree_list = Path(str(row["start_tree_list"]))
        dataset_id = str(row["dataset"])
        curve_path = _run_benchmark(
            dataset_id=dataset_id,
            method_slug=str(row["slug"]),
            method_label=str(row["method"]),
            start_tree_list=start_tree_list,
            num_runs=int(row["n"]),
            output_dir=output_dir / dataset_id.lower() / "mrbayes",
            work_root=work_root,
            ngen=int(args.ngen),
            samplefreq=int(args.samplefreq),
            printfreq=int(args.printfreq),
            curve_interval=int(args.curve_interval),
            max_workers=int(args.max_workers),
            force=bool(args.force),
        )
        curve_paths.append((dataset_id, str(row["method"]), curve_path))
    _write_curve_tsv(curve_paths, output_dir)
    print((output_dir / "mrbayes_curves.tsv").read_text())


if __name__ == "__main__":
    main()
