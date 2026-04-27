#!/usr/bin/env python3
"""Build fixed-path PhylaFlow bank artifacts for one real phylaflow_datasets case.

The DS fixed-path recipe consumes JSON start/target trees plus one combined
velocity-anchor JSON. This wrapper creates the same artifact shape from the
real MrBayes run directories under /home/yektefai/phylaflow_datasets.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.full_sanity_fixedpair_20260401.make_singlepath_parity_case import (  # noqa: E402
    _build_anchor_payloads,
)
from data.dataset import TreeDataset  # noqa: E402
from utils.bhv_utils import return_tree_boundary_merge_paths  # noqa: E402
from utils.metric_utils import canonicalize_topology_newick  # noqa: E402

DEFAULT_NEXUS_ROOT = Path("/home/yektefai/phylaflow_datasets/nexus")
DEFAULT_MRBAYES_ROOT = Path("/home/yektefai/phylaflow_datasets/runs")
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "analysis" / "full_sanity_fixedpair_20260401" / "real_fixed_path_banks"
)

ASDSF_RE = re.compile(r"AvgStdDev\(s\)")


def _tree_with_semicolon(tree: str) -> str:
    tree = tree.strip()
    if not tree.endswith(";"):
        tree += ";"
    return tree


def _split_mcmc_line(line: str) -> list[str]:
    parts = line.rstrip("\n").split("\t")
    if len(parts) <= 1:
        parts = line.split()
    return parts


def _read_final_asdsf_values(run_dir: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for mcmc_path in sorted(run_dir.glob("*.mcmc")):
        header: list[str] | None = None
        last_row: list[str] | None = None
        with mcmc_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("["):
                    continue
                parts = _split_mcmc_line(raw_line)
                if ASDSF_RE.search(raw_line):
                    header = parts
                    continue
                if header is not None and parts:
                    last_row = parts
        if header is None or last_row is None:
            continue
        try:
            idx = header.index("AvgStdDev(s)")
            final_asdsf = float(last_row[idx])
        except (ValueError, IndexError):
            continue
        values.append({"path": str(mcmc_path), "final_asdsf": final_asdsf})
    return values


def _check_asdsf(
    mrbayes_root: Path,
    dataset_id: str,
    threshold: float,
    skip_check: bool,
) -> dict[str, Any]:
    run_dir = mrbayes_root / dataset_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"MrBayes run directory not found: {run_dir}")

    values = _read_final_asdsf_values(run_dir)
    summary: dict[str, Any] = {
        "threshold": threshold,
        "values": values,
        "min": None,
        "max": None,
        "passed": skip_check,
        "skipped": skip_check,
    }
    if not values:
        if skip_check:
            return summary
        raise RuntimeError(f"No readable final AvgStdDev(s) values found in {run_dir}")

    final_values = [float(item["final_asdsf"]) for item in values]
    summary["min"] = min(final_values)
    summary["max"] = max(final_values)
    summary["passed"] = max(final_values) <= threshold
    if not skip_check and not summary["passed"]:
        raise RuntimeError(
            f"{dataset_id} failed ASDSF threshold: max final ASDSF "
            f"{summary['max']:.6g} > {threshold:.6g}"
        )
    return summary


def _load_posterior_trees(
    dataset: TreeDataset,
    max_posterior_trees: int | None,
) -> tuple[list[str], dict[str, Any]]:
    posterior_trees = [_tree_with_semicolon(tree) for tree in dataset.return_posterior_trees(0)]
    raw_count = len(posterior_trees)
    if raw_count == 0:
        raise RuntimeError("No posterior trees were loaded after burn-in")

    if max_posterior_trees is not None and raw_count > max_posterior_trees:
        # Deterministic stride cap keeps the posterior range represented without
        # loading all huge real-data runs into each downstream sampling loop.
        stride = raw_count / float(max_posterior_trees)
        capped = [posterior_trees[int(i * stride)] for i in range(max_posterior_trees)]
        posterior_trees = capped

    return posterior_trees, {
        "raw_post_burnin_count": raw_count,
        "used_count": len(posterior_trees),
        "max_posterior_trees": max_posterior_trees,
    }


def _build_target_schedule(
    posterior_trees: list[str],
    num_cases: int | None,
    seed: int,
    mode: str,
    support_multiplier: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    if mode == "posterior-samples":
        if num_cases is None:
            raise RuntimeError("--num-cases is required with --target-schedule posterior-samples")
        indices = list(range(len(posterior_trees)))
        rng.shuffle(indices)
        schedule = [
            {"posterior_index": int(indices[case_idx % len(indices)])}
            for case_idx in range(num_cases)
        ]
        return schedule, {
            "mode": mode,
            "num_cases": int(num_cases),
            "posterior_sample_count": len(posterior_trees),
            "unique_topology_count": None,
            "full_support": None,
        }

    topology_to_indices: dict[str, list[int]] = {}
    for idx, tree in enumerate(posterior_trees):
        key = canonicalize_topology_newick(str(tree).strip())
        topology_to_indices.setdefault(str(key), []).append(int(idx))

    items = []
    total_count = len(posterior_trees)
    for topo_key, indices in sorted(topology_to_indices.items(), key=lambda item: item[1][0]):
        items.append(
            {
                "topology_key": str(topo_key),
                "posterior_index": int(indices[0]),
                "topology_count": int(len(indices)),
                "topology_probability": float(len(indices)) / float(total_count),
            }
        )

    if mode == "unique-topologies":
        if num_cases is None:
            num_cases = len(items)
        if num_cases > len(items):
            raise RuntimeError(
                f"Requested {num_cases} cases but only {len(items)} unique target topologies exist."
            )
        rng.shuffle(items)
        schedule = [dict(item) for item in items[:num_cases]]
        return schedule, {
            "mode": mode,
            "num_cases": int(num_cases),
            "posterior_sample_count": len(posterior_trees),
            "unique_topology_count": len(items),
            "full_support": num_cases == len(items),
        }

    if mode != "weighted-topologies":
        raise ValueError(f"Unknown target schedule mode: {mode}")

    if num_cases is None:
        num_cases = len(items) * int(support_multiplier)

    base_alloc = [0 for _ in items]
    full_support_requested = num_cases >= len(items)
    if full_support_requested:
        base_alloc = [1 for _ in items]

    extra_cases = int(num_cases) - int(sum(base_alloc))
    scaled_counts = [
        (float(item["topology_count"]) / float(total_count)) * extra_cases
        for item in items
    ]
    allocated = [
        int(base) + int(math.floor(count))
        for base, count in zip(base_alloc, scaled_counts)
    ]
    remaining = int(num_cases) - int(sum(allocated))
    if remaining > 0:
        remainder_order = sorted(
            range(len(items)),
            key=lambda idx: (
                scaled_counts[idx] - math.floor(scaled_counts[idx]),
                items[idx]["topology_count"],
            ),
            reverse=True,
        )
        for idx in remainder_order[:remaining]:
            allocated[idx] += 1

    schedule: list[dict[str, Any]] = []
    represented = 0
    for item, alloc in zip(items, allocated):
        if int(alloc) <= 0:
            continue
        represented += 1
        for _ in range(int(alloc)):
            schedule.append(dict(item))
    if len(schedule) != int(num_cases):
        raise RuntimeError(
            f"Weighted topology allocation created {len(schedule)} cases, expected {num_cases}."
        )
    rng.shuffle(schedule)
    return schedule, {
        "mode": mode,
        "num_cases": int(num_cases),
        "support_multiplier": int(support_multiplier) if num_cases is not None else None,
        "posterior_sample_count": len(posterior_trees),
        "unique_topology_count": len(items),
        "represented_topology_count": int(represented),
        "full_support": int(represented) == len(items),
        "full_support_requested": bool(full_support_requested),
    }


def _pick_fixed_pair(
    dataset: TreeDataset,
    posterior_trees: list[str],
    target_index: int,
    case_idx: int,
    seed: int,
    min_boundary_paths: int,
    max_start_tries_per_target: int,
    used_pairs: set[tuple[str, str]],
) -> dict[str, Any]:
    target_tree = posterior_trees[target_index]
    for start_try in range(max_start_tries_per_target):
        random.seed(seed + case_idx * 1_000_003 + start_try)
        start_tree = _tree_with_semicolon(dataset.sample_random_tree(target_tree))
        pair_key = (start_tree, target_tree)
        if pair_key in used_pairs:
            continue
        boundary_paths = return_tree_boundary_merge_paths(start_tree, target_tree)
        if len(boundary_paths) < min_boundary_paths:
            continue
        used_pairs.add(pair_key)
        return {
            "start_tree": start_tree,
            "target_tree": target_tree,
            "posterior_index": target_index,
            "boundary_path_count": len(boundary_paths),
            "start_try": start_try,
        }
    raise RuntimeError(
        "Could not find a valid start/target pair. Lower --min-boundary-paths, "
        "raise --max-start-tries-per-target, or choose a larger dataset."
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _build_case_artifacts(
    args: argparse.Namespace,
    dataset: TreeDataset,
    posterior_trees: list[str],
    output_dir: Path,
    bank_name: str,
) -> dict[str, Any]:
    target_schedule, target_summary = _build_target_schedule(
        posterior_trees=posterior_trees,
        num_cases=None if args.num_cases is None else int(args.num_cases),
        seed=int(args.pair_seed),
        mode=str(args.target_schedule),
        support_multiplier=int(args.support_multiplier),
    )
    num_cases = int(target_summary["num_cases"])
    used_pairs: set[tuple[str, str]] = set()
    width = max(2, len(str(max(num_cases - 1, 0))))

    start_paths: list[str] = []
    target_paths: list[str] = []
    all_anchors: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []

    for case_idx in range(num_cases):
        case_name = f"{bank_name}_case{case_idx:0{width}d}"
        target_entry = dict(target_schedule[case_idx])
        pair = _pick_fixed_pair(
            dataset=dataset,
            posterior_trees=posterior_trees,
            target_index=int(target_entry["posterior_index"]),
            case_idx=case_idx,
            seed=args.pair_seed,
            min_boundary_paths=args.min_boundary_paths,
            max_start_tries_per_target=args.max_start_tries_per_target,
            used_pairs=used_pairs,
        )
        anchor_payload = _build_anchor_payloads(
            start_tree=pair["start_tree"],
            target_tree=pair["target_tree"],
            bank_group_key=case_name,
            o0_count=args.o0_anchor_count,
            a1_count=args.a1_anchor_count,
            o2_count=args.o2_anchor_count,
            full_path_count=args.full_path_anchor_count,
        )
        anchors = list(anchor_payload["anchors"])
        boundary_path_count = int(anchor_payload["boundary_path_count"])
        num_leaves = int(anchor_payload["num_leaves"])

        start_path = output_dir / f"{case_name}_start.json"
        target_path = output_dir / f"{case_name}_target.json"
        start_payload = {
            "tree": pair["start_tree"],
            "group_key": case_name,
            "dataset_id": args.dataset_id,
            "posterior_index": pair["posterior_index"],
            "case_index": case_idx,
        }
        target_payload = {
            "tree": pair["target_tree"],
            "group_key": case_name,
            "dataset_id": args.dataset_id,
            "posterior_index": pair["posterior_index"],
            "case_index": case_idx,
        }
        _write_json(start_path, start_payload)
        _write_json(target_path, target_payload)

        start_paths.append(str(start_path))
        target_paths.append(str(target_path))
        all_anchors.extend(anchors)
        cases.append(
            {
                "case_index": case_idx,
                "group_key": case_name,
                "start_path": str(start_path),
                "target_path": str(target_path),
                "posterior_index": pair["posterior_index"],
                "boundary_path_count": boundary_path_count,
                "num_leaves": num_leaves,
                "anchor_count": len(anchors),
                "start_try": pair["start_try"],
                "topology_key": target_entry.get("topology_key"),
                "topology_count": target_entry.get("topology_count"),
                "topology_probability": target_entry.get("topology_probability"),
            }
        )

    anchor_path = output_dir / f"{bank_name}_velocity_anchors.json"
    _write_json(anchor_path, all_anchors)

    return {
        "start_paths": start_paths,
        "target_paths": target_paths,
        "anchor_path": str(anchor_path),
        "anchor_count": len(all_anchors),
        "num_cases": int(num_cases),
        "target_schedule": target_summary,
        "cases": cases,
    }


def _default_bank_name(dataset_id: str, num_cases: int | None, target_schedule: str) -> str:
    clean_dataset_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_id)
    clean_schedule = re.sub(r"[^A-Za-z0-9_.-]+", "_", target_schedule)
    suffix = "auto" if num_cases is None else f"n{num_cases}"
    return f"real_{clean_dataset_id}_{clean_schedule}_fixedpath_{suffix}"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed-path bank JSONs for one real phylaflow_datasets run."
    )
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--nexus-root", type=Path, default=DEFAULT_NEXUS_ROOT)
    parser.add_argument("--mrbayes-root", type=Path, default=DEFAULT_MRBAYES_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bank-name", default=None)
    parser.add_argument(
        "--num-cases",
        type=int,
        default=None,
        help=(
            "Number of bank cases. If omitted, unique-topologies uses all unique "
            "posterior topologies and weighted-topologies uses unique_count * "
            "--support-multiplier."
        ),
    )
    parser.add_argument("--support-multiplier", type=int, default=3)
    parser.add_argument("--pair-seed", type=int, default=271828)
    parser.add_argument(
        "--target-schedule",
        choices=["weighted-topologies", "unique-topologies", "posterior-samples"],
        default="unique-topologies",
        help=(
            "How target posterior trees are scheduled. unique-topologies emits one "
            "case per unique posterior topology by default. weighted-topologies "
            "matches the DS full-support/proportional allocation when num-cases is "
            "at least the unique topology count."
        ),
    )
    parser.add_argument("--asdsf-threshold", type=float, default=0.05)
    parser.add_argument("--skip-asdsf-check", action="store_true")
    parser.add_argument(
        "--max-posterior-trees",
        type=int,
        default=None,
        help=(
            "Debug-only deterministic cap on post-burn-in posterior trees. "
            "Default is no cap, matching the DS preprocessing behavior."
        ),
    )
    parser.add_argument("--min-boundary-paths", type=int, default=3)
    parser.add_argument("--max-start-tries-per-target", type=int, default=200)
    parser.add_argument("--o0-anchor-count", type=int, default=16)
    parser.add_argument("--a1-anchor-count", type=int, default=8)
    parser.add_argument("--o2-anchor-count", type=int, default=8)
    parser.add_argument("--full-path-anchor-count", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.num_cases is not None and args.num_cases <= 0:
        raise ValueError("--num-cases must be positive")
    if args.support_multiplier <= 0:
        raise ValueError("--support-multiplier must be positive")
    if args.max_posterior_trees is not None and args.max_posterior_trees <= 0:
        args.max_posterior_trees = None

    nexus_root = args.nexus_root.expanduser().resolve()
    mrbayes_root = args.mrbayes_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    bank_name = args.bank_name or _default_bank_name(
        args.dataset_id,
        args.num_cases,
        args.target_schedule,
    )
    output_dir = output_root / bank_name

    asdsf_summary = _check_asdsf(
        mrbayes_root=mrbayes_root,
        dataset_id=args.dataset_id,
        threshold=args.asdsf_threshold,
        skip_check=args.skip_asdsf_check,
    )

    dataset = TreeDataset(
        nexus_root=nexus_root,
        mrbayes_root=mrbayes_root,
        filter_ids=[args.dataset_id],
    )
    if len(dataset) != 1:
        raise RuntimeError(f"Expected one dataset for {args.dataset_id}, found {len(dataset)}")

    posterior_trees, posterior_summary = _load_posterior_trees(
        dataset=dataset,
        max_posterior_trees=args.max_posterior_trees,
    )
    artifact_summary = _build_case_artifacts(
        args=args,
        dataset=dataset,
        posterior_trees=posterior_trees,
        output_dir=output_dir,
        bank_name=bank_name,
    )

    manifest = {
        "bank_name": bank_name,
        "dataset_id": args.dataset_id,
        "nexus_root": str(nexus_root),
        "mrbayes_root": str(mrbayes_root),
        "output_dir": str(output_dir),
        "asdsf": asdsf_summary,
        "posterior": posterior_summary,
        "num_cases": int(artifact_summary["num_cases"]),
        "pair_seed": args.pair_seed,
        "min_boundary_paths": args.min_boundary_paths,
        "max_start_tries_per_target": args.max_start_tries_per_target,
        "anchor_settings": {
            "o0_anchor_count": args.o0_anchor_count,
            "a1_anchor_count": args.a1_anchor_count,
            "o2_anchor_count": args.o2_anchor_count,
            "full_path_anchor_count": args.full_path_anchor_count,
        },
        "training_config_fields": {
            "data.overfit_fixed_pair_start_tree_json_paths": artifact_summary["start_paths"],
            "data.overfit_fixed_pair_target_tree_json_paths": artifact_summary["target_paths"],
            "data.overfit_full_path_control_extra_velocity_samples_json_path": artifact_summary[
                "anchor_path"
            ],
            "data.overfit_virtual_epoch_size": int(artifact_summary["num_cases"]),
            "trainer.sample_metrics_num_pairs": int(artifact_summary["num_cases"]),
            "model.first_hit_head_num_cases": int(artifact_summary["num_cases"]),
        },
        **artifact_summary,
    }
    manifest_path = output_dir / f"{bank_name}_manifest.json"
    _write_json(manifest_path, manifest)

    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "bank_name": bank_name,
                "dataset_id": args.dataset_id,
                "num_cases": int(artifact_summary["num_cases"]),
                "anchor_count": artifact_summary["anchor_count"],
                "posterior_used_count": posterior_summary["used_count"],
                "asdsf_max": asdsf_summary["max"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
