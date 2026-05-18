#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
from ete3 import Tree as EteTree

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset import TreeDataset  # noqa: E402
from utils.metric_utils import canonicalize_topology_newick  # noqa: E402


DEFAULT_NEXUS_ROOT = Path("/ewsc/yektefai/phylaflow_datasets/nexus")
DEFAULT_MRBAYES_ROOT = Path("/ewsc/yektefai/phylaflow_datasets/runs")
DEFAULT_OUTPUT_ROOT = (
    Path("/ewsc/yektefai/phylaflow/orthomam_subset_banks")
)


def _numeric_name_sort_key(value: Any):
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def _tree_with_semicolon(tree: str) -> str:
    tree = str(tree).strip()
    return tree if tree.endswith(";") else f"{tree};"


def _clean_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _available_ids(nexus_root: Path) -> List[str]:
    ids = []
    for path in sorted(nexus_root.iterdir()):
        if path.suffix.lower() in {".nex", ".nexus"}:
            ids.append(path.stem)
    return ids


def _ids_from_config(config_path: Path) -> List[str]:
    config = yaml.safe_load(config_path.read_text())
    data_cfg = config.get("data") or {}
    configured = data_cfg.get("posterior_dataset_ids") or data_cfg.get(
        "short_run_dataset_ids"
    )
    if configured:
        if isinstance(configured, str):
            return [item for item in re.split(r"[,\s]+", configured) if item]
        return [str(item) for item in configured]
    single = data_cfg.get("posterior_dataset_id") or data_cfg.get("short_run_dataset_id")
    if single:
        return [str(single)]
    return _available_ids(Path(data_cfg["nexus_root"]))


def _split_ids(ids: List[str], seed: int) -> Tuple[List[str], List[str]]:
    ids = list(ids)
    rng = random.Random(int(seed))
    rng.shuffle(ids)
    split = int(0.8 * len(ids)) if len(ids) >= 2 else len(ids)
    return ids[:split], ids[split:]


def _resolve_sequence_for_label(
    label: str,
    seqs: Dict[str, str],
    taxa_order: List[str],
    translate_map: Dict[str, str],
) -> Tuple[str, str]:
    label = str(label)
    taxon_name = translate_map.get(label)
    if taxon_name is not None:
        return taxon_name, seqs.get(taxon_name, "")
    if label in seqs:
        return label, seqs.get(label, "")
    try:
        idx = int(label)
    except ValueError:
        return label, seqs.get(label, "")
    if 1 <= idx <= len(taxa_order):
        taxon_name = str(taxa_order[idx - 1])
        return taxon_name, seqs.get(taxon_name, "")
    if 0 <= idx < len(taxa_order):
        taxon_name = str(taxa_order[idx])
        return taxon_name, seqs.get(taxon_name, "")
    return label, seqs.get(label, "")


def _strip_live_phyla_sequence(sequence: str) -> str:
    return str(sequence or "").replace("-", "").replace(".", "")


def _choose_leaf_subset(
    *,
    leaf_labels: List[str],
    seqs: Dict[str, str],
    taxa_order: List[str],
    translate_map: Dict[str, str],
    num_leaves: int,
    max_input_tokens: int,
    seed: str,
) -> Tuple[List[str], List[str], int]:
    if len(leaf_labels) < int(num_leaves):
        raise RuntimeError(
            f"Only {len(leaf_labels)} leaves are available; requested {num_leaves}."
        )

    def input_tokens(labels: List[str]) -> int:
        total = len(labels)
        for label in labels:
            _name, sequence = _resolve_sequence_for_label(
                label, seqs, taxa_order, translate_map
            )
            total += len(_strip_live_phyla_sequence(sequence))
        return int(total)

    rng = random.Random(seed)
    best_labels = sorted(
        rng.sample(leaf_labels, int(num_leaves)),
        key=_numeric_name_sort_key,
    )
    best_tokens = input_tokens(best_labels)
    chosen = list(best_labels)

    if int(max_input_tokens) > 0 and best_tokens > int(max_input_tokens):
        for _attempt in range(64):
            candidate = sorted(
                rng.sample(leaf_labels, int(num_leaves)),
                key=_numeric_name_sort_key,
            )
            candidate_tokens = input_tokens(candidate)
            if candidate_tokens < best_tokens:
                best_labels = list(candidate)
                best_tokens = int(candidate_tokens)
            if candidate_tokens <= int(max_input_tokens):
                chosen = list(candidate)
                break
        else:
            lengths = []
            for label in leaf_labels:
                _name, sequence = _resolve_sequence_for_label(
                    label, seqs, taxa_order, translate_map
                )
                lengths.append((len(_strip_live_phyla_sequence(sequence)), label))
            chosen = sorted(
                [label for _length, label in sorted(lengths)[: int(num_leaves)]],
                key=_numeric_name_sort_key,
            )
            best_tokens = input_tokens(chosen)
    else:
        chosen = list(best_labels)

    if int(max_input_tokens) > 0 and int(best_tokens) > int(max_input_tokens):
        raise RuntimeError(
            f"Shortest {num_leaves}-leaf subset still has {best_tokens} tokens, "
            f"above max_input_tokens={max_input_tokens}."
        )

    selected_names = [
        _resolve_sequence_for_label(label, seqs, taxa_order, translate_map)[0]
        for label in chosen
    ]
    return chosen, selected_names, int(input_tokens(chosen))


def _prune_and_relabel_tree(newick: str, chosen_labels: List[str]) -> str:
    tree = EteTree(str(newick), format=1)
    tree.prune(list(chosen_labels), preserve_branch_length=True)
    relabel = {str(old): str(idx) for idx, old in enumerate(chosen_labels)}
    for leaf in tree.iter_leaves():
        leaf.name = relabel[str(leaf.name)]
    return _tree_with_semicolon(tree.write(format=1))


def _random_pair_newick(left: str, right: str, rng: random.Random) -> str:
    return f"({left}:{rng.uniform(0.1, 1.0):.12g},{right}:{rng.uniform(0.1, 1.0):.12g})"


def _random_rooted_start_tree(num_leaves: int, rng: random.Random) -> str:
    labels = [str(i) for i in range(int(num_leaves))]
    if len(labels) < 3:
        raise RuntimeError("Need at least three leaves for a random start tree.")
    # Mimic the dataset random-start convention: leaf 0 is attached at the root,
    # and the remaining leaves form a random binary subtree.
    clusters = list(labels[1:])
    rng.shuffle(clusters)
    while len(clusters) > 1:
        i = rng.randrange(len(clusters))
        left = clusters.pop(i)
        j = rng.randrange(len(clusters))
        right = clusters.pop(j)
        clusters.append(_random_pair_newick(left, right, rng))
    root_length = rng.uniform(0.1, 1.0)
    return f"({clusters[0]}:{root_length:.12g},0:0);"


def _build_weighted_schedule(
    pruned_trees: List[str],
    *,
    num_cases: int,
    ensure_all_topologies_if_possible: bool,
    seed: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    topology_to_indices: Dict[str, List[int]] = {}
    for idx, tree in enumerate(pruned_trees):
        key = canonicalize_topology_newick(str(tree).strip())
        topology_to_indices.setdefault(str(key), []).append(int(idx))

    total_count = len(pruned_trees)
    items = []
    for topology_key, indices in sorted(
        topology_to_indices.items(), key=lambda item: item[1][0]
    ):
        items.append(
            {
                "topology_key": str(topology_key),
                "posterior_index": int(indices[0]),
                "topology_count": int(len(indices)),
                "topology_probability": float(len(indices)) / float(total_count),
            }
        )

    num_cases = int(num_cases)
    base_alloc = [0 for _item in items]
    full_support = bool(ensure_all_topologies_if_possible and num_cases >= len(items))
    if full_support:
        base_alloc = [1 for _item in items]

    extra_cases = num_cases - sum(base_alloc)
    if extra_cases < 0:
        raise RuntimeError(
            f"num_cases={num_cases} is smaller than required base allocation "
            f"{sum(base_alloc)}."
        )
    scaled = [
        (float(item["topology_count"]) / float(total_count)) * float(extra_cases)
        for item in items
    ]
    allocated = [
        int(base) + int(math.floor(value))
        for base, value in zip(base_alloc, scaled)
    ]
    remaining = num_cases - sum(allocated)
    if remaining > 0:
        order = sorted(
            range(len(items)),
            key=lambda idx: (
                scaled[idx] - math.floor(scaled[idx]),
                items[idx]["topology_count"],
            ),
            reverse=True,
        )
        for idx in order[:remaining]:
            allocated[idx] += 1

    schedule = []
    represented = 0
    for item, count in zip(items, allocated):
        if count <= 0:
            continue
        represented += 1
        for _ in range(int(count)):
            schedule.append(dict(item))

    if len(schedule) != num_cases:
        raise RuntimeError(
            f"Built {len(schedule)} scheduled cases, expected {num_cases}."
        )
    random.Random(seed).shuffle(schedule)
    return schedule, {
        "posterior_sample_count": int(total_count),
        "unique_topology_count": int(len(items)),
        "num_cases": int(num_cases),
        "represented_topology_count": int(represented),
        "full_support": bool(represented == len(items)),
        "ensure_all_topologies_if_possible": bool(ensure_all_topologies_if_possible),
    }


def _build_one_dataset(payload: Dict[str, Any]) -> Dict[str, Any]:
    dataset_id = str(payload["dataset_id"])
    num_leaves = int(payload["num_leaves"])
    num_cases = int(payload["num_cases"])
    nexus_root = Path(payload["nexus_root"])
    mrbayes_root = Path(payload["mrbayes_root"])
    max_input_tokens = int(payload["max_input_tokens"])
    seed = int(payload["seed"])
    split_name = str(payload["split_name"])
    bank_name = str(payload["bank_name"])
    ensure_all_topologies_if_possible = bool(
        payload["ensure_all_topologies_if_possible"]
    )

    try:
        dataset = TreeDataset(
            str(nexus_root),
            str(mrbayes_root),
            filter_ids=[dataset_id],
        )
        if len(dataset._index) != 1:
            raise RuntimeError(f"Expected one index row, found {len(dataset._index)}.")
        meta = dataset._index[0]
        if not meta.get("tree_paths"):
            raise RuntimeError("No MrBayes .t tree paths found.")
        posterior_trees = dataset.return_posterior_trees(0)
        if not posterior_trees:
            raise RuntimeError("No posterior trees loaded after burn-in.")
        seqs, taxa_order = dataset.parse_nexus(meta["nexus_path"])
        translate_map = dataset.parse_translate_block(meta["tree_paths"][0])
        first_tree = EteTree(str(posterior_trees[0]), format=1)
        leaf_labels = sorted(
            [str(leaf.name) for leaf in first_tree.iter_leaves()],
            key=_numeric_name_sort_key,
        )
        chosen_labels, selected_names, input_tokens = _choose_leaf_subset(
            leaf_labels=leaf_labels,
            seqs=seqs,
            taxa_order=taxa_order,
            translate_map=translate_map,
            num_leaves=num_leaves,
            max_input_tokens=max_input_tokens,
            seed=f"{seed}:{dataset_id}:subset{num_leaves}",
        )
        pruned_trees = [
            _prune_and_relabel_tree(tree, chosen_labels)
            for tree in posterior_trees
        ]
        schedule, schedule_summary = _build_weighted_schedule(
            pruned_trees,
            num_cases=num_cases,
            ensure_all_topologies_if_possible=ensure_all_topologies_if_possible,
            seed=f"{seed}:{dataset_id}:schedule",
        )
        rng = random.Random(f"{seed}:{dataset_id}:starts")
        rows = []
        for case_idx, entry in enumerate(schedule):
            group_key = (
                f"{bank_name}_{split_name}_{_clean_tag(dataset_id)}_case"
                f"{case_idx:04d}"
            )
            start_tree = _random_rooted_start_tree(num_leaves, rng)
            target_tree = pruned_trees[int(entry["posterior_index"])]
            rows.append(
                {
                    "dataset_id": str(dataset_id).upper(),
                    "bank_group_key": group_key,
                    "group_key": group_key,
                    "subset_key": f"{_clean_tag(dataset_id)}_subset{num_leaves}_seed{seed}",
                    "subset_size": int(num_leaves),
                    "selected_original_labels": list(chosen_labels),
                    "selected_sequence_names": list(selected_names),
                    "selected_input_tokens": int(input_tokens),
                    "start_tree": start_tree,
                    "target_tree": target_tree,
                    "posterior_index": int(entry["posterior_index"]),
                    "topology_key": str(entry["topology_key"]),
                    "topology_count": int(entry["topology_count"]),
                    "topology_probability": float(entry["topology_probability"]),
                    "case_index": int(case_idx),
                    "target_schedule_mode": "weighted_topologies",
                }
            )
        return {
            "dataset_id": dataset_id,
            "status": "ok",
            "rows": rows,
            "summary": {
                "dataset_id": dataset_id,
                "num_rows": len(rows),
                "num_leaves": int(num_leaves),
                "selected_original_labels": list(chosen_labels),
                "selected_sequence_names": list(selected_names),
                "selected_input_tokens": int(input_tokens),
                **schedule_summary,
            },
        }
    except Exception as exc:
        return {
            "dataset_id": dataset_id,
            "status": "error",
            "error": str(exc),
            "rows": [],
            "summary": {
                "dataset_id": dataset_id,
                "error": str(exc),
            },
        }


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def _write_split_bank(
    *,
    split_name: str,
    dataset_ids: List[str],
    args: argparse.Namespace,
    bank_name: str,
    output_dir: Path,
) -> Dict[str, Any]:
    bank_path = output_dir / f"{bank_name}_{split_name}.jsonl"
    summaries = []
    errors = []
    total_rows = 0
    payloads = [
        {
            "dataset_id": dataset_id,
            "nexus_root": str(args.nexus_root),
            "mrbayes_root": str(args.mrbayes_root),
            "num_leaves": int(args.num_leaves),
            "num_cases": int(args.cases_per_dataset),
            "max_input_tokens": int(args.max_input_tokens),
            "seed": int(args.seed),
            "split_name": split_name,
            "bank_name": bank_name,
            "ensure_all_topologies_if_possible": bool(
                args.ensure_all_topologies_if_possible
            ),
        }
        for dataset_id in dataset_ids
    ]

    with bank_path.open("w", encoding="utf-8") as handle:
        if int(args.workers) <= 1:
            iterator = map(_build_one_dataset, payloads)
            for result in iterator:
                if result["status"] != "ok":
                    errors.append(
                        {
                            "dataset_id": result["dataset_id"],
                            "error": result.get("error"),
                        }
                    )
                    summaries.append(result["summary"])
                    continue
                for row in result["rows"]:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    total_rows += 1
                summaries.append(result["summary"])
        else:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=int(args.workers)
            ) as executor:
                futures = [
                    executor.submit(_build_one_dataset, payload)
                    for payload in payloads
                ]
                for idx, future in enumerate(
                    concurrent.futures.as_completed(futures), start=1
                ):
                    result = future.result()
                    if result["status"] != "ok":
                        errors.append(
                            {
                                "dataset_id": result["dataset_id"],
                                "error": result.get("error"),
                            }
                        )
                        summaries.append(result["summary"])
                    else:
                        for row in result["rows"]:
                            handle.write(json.dumps(row, sort_keys=True) + "\n")
                            total_rows += 1
                        summaries.append(result["summary"])
                    if int(args.progress_every) > 0 and (
                        idx % int(args.progress_every) == 0 or idx == len(futures)
                    ):
                        print(
                            f"[{split_name}] completed {idx}/{len(futures)} datasets; "
                            f"rows={total_rows}; errors={len(errors)}",
                            flush=True,
                        )

    unique_counts = [
        int(item["unique_topology_count"])
        for item in summaries
        if item.get("unique_topology_count") is not None
    ]
    input_tokens = [
        int(item["selected_input_tokens"])
        for item in summaries
        if item.get("selected_input_tokens") is not None
    ]
    return {
        "split": split_name,
        "dataset_count_requested": int(len(dataset_ids)),
        "dataset_count_ok": int(len(summaries) - len(errors)),
        "dataset_count_error": int(len(errors)),
        "row_count": int(total_rows),
        "bank_path": str(bank_path),
        "errors": errors,
        "unique_topology_count_summary": _summary_stats(unique_counts),
        "selected_input_token_summary": _summary_stats(input_tokens),
        "datasets": sorted(summaries, key=lambda item: str(item.get("dataset_id"))),
    }


def _summary_stats(values: List[int]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "min": None,
            "p25": None,
            "median": None,
            "mean": None,
            "p75": None,
            "p90": None,
            "max": None,
        }
    values = sorted(values)

    def quantile(q: float) -> float:
        if len(values) == 1:
            return float(values[0])
        pos = q * (len(values) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return float(values[lo])
        frac = pos - lo
        return float(values[lo] * (1.0 - frac) + values[hi] * frac)

    return {
        "min": float(values[0]),
        "p25": quantile(0.25),
        "median": quantile(0.5),
        "mean": float(sum(values)) / float(len(values)),
        "p75": quantile(0.75),
        "p90": quantile(0.9),
        "max": float(values[-1]),
    }


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build fixed random-start/posterior-target banks for fixed-size "
            "OrthoMaM posterior subsets."
        )
    )
    parser.add_argument(
        "--source-config",
        type=Path,
        default=REPO_ROOT
        / "configs"
        / "local_orthomam10leaf_train80_randomposterior_samebatch_livephyla_rawfull_e2e_100m_aggressive_onelr_lr1e3_20260514.yaml",
    )
    parser.add_argument("--nexus-root", type=Path, default=DEFAULT_NEXUS_ROOT)
    parser.add_argument("--mrbayes-root", type=Path, default=DEFAULT_MRBAYES_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bank-name", default=None)
    parser.add_argument("--num-leaves", type=int, default=29)
    parser.add_argument("--cases-per-dataset", type=int, default=210)
    parser.add_argument("--max-input-tokens", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--max-train-datasets", type=int, default=0)
    parser.add_argument("--max-test-datasets", type=int, default=0)
    parser.add_argument(
        "--split",
        choices=["train", "test", "both"],
        default="both",
    )
    parser.add_argument(
        "--ensure-all-topologies-if-possible",
        action="store_true",
        default=True,
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    if int(args.num_leaves) < 3:
        raise ValueError("--num-leaves must be at least 3")
    if int(args.cases_per_dataset) <= 0:
        raise ValueError("--cases-per-dataset must be positive")

    all_ids = _ids_from_config(args.source_config)
    train_ids, test_ids = _split_ids(all_ids, int(args.split_seed))
    if int(args.max_train_datasets) > 0:
        train_ids = train_ids[: int(args.max_train_datasets)]
    if int(args.max_test_datasets) > 0:
        test_ids = test_ids[: int(args.max_test_datasets)]

    bank_name = args.bank_name or (
        f"orthomam{int(args.num_leaves)}leaf_fixedpair"
        f"_c{int(args.cases_per_dataset)}_seed{int(args.seed)}"
    )
    output_dir = args.output_root / bank_name
    output_dir.mkdir(parents=True, exist_ok=True)

    split_summaries = []
    if args.split in {"train", "both"}:
        split_summaries.append(
            _write_split_bank(
                split_name="train",
                dataset_ids=train_ids,
                args=args,
                bank_name=bank_name,
                output_dir=output_dir,
            )
        )
    if args.split in {"test", "both"}:
        split_summaries.append(
            _write_split_bank(
                split_name="test",
                dataset_ids=test_ids,
                args=args,
                bank_name=bank_name,
                output_dir=output_dir,
            )
        )

    manifest = {
        "bank_name": bank_name,
        "output_dir": str(output_dir),
        "source_config": str(args.source_config),
        "nexus_root": str(args.nexus_root),
        "mrbayes_root": str(args.mrbayes_root),
        "num_leaves": int(args.num_leaves),
        "cases_per_dataset": int(args.cases_per_dataset),
        "max_input_tokens": int(args.max_input_tokens),
        "seed": int(args.seed),
        "split_seed": int(args.split_seed),
        "total_source_dataset_count": int(len(all_ids)),
        "train_dataset_count_requested": int(len(train_ids)),
        "test_dataset_count_requested": int(len(test_ids)),
        "splits": split_summaries,
    }
    manifest_path = output_dir / f"{bank_name}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest_path": str(manifest_path), **manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
