#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

ROOT = Path("/home/yektefai/PhylaFlow")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ete3 import Tree as EteTree  # noqa: E402

from analysis.full_sanity_fixedpair_20260401.probe_current_mainline_ckpt import (  # noqa: E402
    build_dataset,
    build_module,
    load_config,
    set_seed,
)
from analysis.full_sanity_fixedpair_20260401.remap_posterior_to_harness_lexindex import (  # noqa: E402
    build_harness_lexicographic_ordering_map,
    build_numeric_to_harness_lexicographic_ordering_map,
    load_posterior_trees,
    remap_tree_with_ordering_map,
)
from utils.metric_utils import (  # noqa: E402
    calculate_norm_rf,
    canonicalize_topology_newick,
    kl_divergence_topological_distributions,
    kl_divergence_tree_topology_distributions,
    topk_posterior_tree_recall,
)


def _scalar_timepoint(value) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.item())
        return float(value.flatten()[0].item())
    if isinstance(value, (list, tuple)):
        first = value[0]
        if isinstance(first, torch.Tensor):
            return float(first.item())
        return float(first)
    return float(value)


def _weighted_support(
    trees: Iterable[str],
) -> Tuple[collections.Counter, Dict[str, str]]:
    counts = collections.Counter()
    representatives: Dict[str, str] = {}
    for tree in trees:
        key = canonicalize_topology_newick(str(tree))
        counts[key] += 1
        representatives.setdefault(key, str(tree))
    return counts, representatives


def _tree_leaf_names(newick: str) -> List[str]:
    return [leaf.name for leaf in EteTree(newick, format=1).iter_leaves()]


def _support_rf_metrics(
    tree_newick: str,
    support_counts: collections.Counter,
    support_representatives: Dict[str, str],
) -> Dict[str, float]:
    total = float(sum(support_counts.values()))
    if total <= 0 or not support_representatives:
        return {
            "in_support": 0.0,
            "min_rf_norm": float("nan"),
            "weighted_mean_rf_norm": float("nan"),
        }

    tree_key = canonicalize_topology_newick(tree_newick)
    rf_pairs = [
        (
            float(calculate_norm_rf(tree_newick, support_representatives[key])),
            float(count),
        )
        for key, count in support_counts.items()
    ]
    min_rf = min(rf for rf, _ in rf_pairs)
    weighted_mean_rf = sum(rf * count for rf, count in rf_pairs) / total
    return {
        "in_support": 1.0 if tree_key in support_counts else 0.0,
        "min_rf_norm": float(min_rf),
        "weighted_mean_rf_norm": float(weighted_mean_rf),
    }


def _metric_block(posterior_trees: List[str], sampled_trees: List[str], num_leaves: int):
    payload = {}
    payload.update(
        kl_divergence_topological_distributions(
            posterior_trees,
            sampled_trees,
            num_leaves=num_leaves,
        )
    )
    payload.update(kl_divergence_tree_topology_distributions(posterior_trees, sampled_trees))
    payload.update(topk_posterior_tree_recall(posterior_trees, sampled_trees))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-id", default="DS1")
    parser.add_argument(
        "--short-root",
        default="/home/yektefai/30272299/short_run_data_DS1-8",
    )
    parser.add_argument(
        "--golden-root",
        default="/home/yektefai/30272299/golden_run_data_DS1-8",
    )
    parser.add_argument("--num-trials", type=int, default=16)
    parser.add_argument("--trprobs-sample-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--bootstrap-module",
        default=(
            "analysis.full_sanity_fixedpair_20260401."
            "run_current_main_harness_predsim_overrun_hybridbank"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    logging.getLogger("run.TrainingModule").setLevel(logging.WARNING)
    importlib.import_module(str(args.bootstrap_module))

    config = load_config(Path(args.config))
    set_seed(int(args.seed))
    dataset = build_dataset(config)
    module = build_module(config, dataset, Path(args.checkpoint), str(args.device))
    train_dataset = dataset.dataset_train

    short_raw = load_posterior_trees(
        posterior_root=str(args.short_root),
        dataset_id=str(args.dataset_id),
        trprobs_sample_count_per_file=int(args.trprobs_sample_count),
    )
    golden_raw = load_posterior_trees(
        posterior_root=str(args.golden_root),
        dataset_id=str(args.dataset_id),
        trprobs_sample_count_per_file=int(args.trprobs_sample_count),
    )
    reference_tree = short_raw[0] if short_raw else golden_raw[0]
    posterior_raw_to_lex = build_harness_lexicographic_ordering_map(reference_tree)
    numeric_to_lex = build_numeric_to_harness_lexicographic_ordering_map(reference_tree)

    short_lex = [
        remap_tree_with_ordering_map(tree, posterior_raw_to_lex) for tree in short_raw
    ]
    golden_lex = [
        remap_tree_with_ordering_map(tree, posterior_raw_to_lex) for tree in golden_raw
    ]
    short_support_counts, short_support_representatives = _weighted_support(short_lex)
    golden_support_counts, golden_support_representatives = _weighted_support(golden_lex)
    num_leaves = len(EteTree(short_lex[0] if short_lex else golden_lex[0], format=1).get_leaves())

    results: List[dict] = []
    with torch.no_grad():
        for trial in range(int(args.num_trials)):
            harness_pair = None
            if (
                getattr(train_dataset, "overfit_fixed_pair", False)
                and bool(
                    getattr(
                        module,
                        "sampling_random_fixed_pair_bank_use_at_sampling",
                        False,
                    )
                )
                and hasattr(module, "_get_harness_sampling_pair")
            ):
                harness_pair = module._get_harness_sampling_pair(train=True)

            if harness_pair is not None:
                start_tree_numeric = str(harness_pair["start_tree"])
                target_tree_numeric = str(harness_pair["target_tree"])
                start_tree_for_sampling = start_tree_numeric
                target_tree_for_sampling = target_tree_numeric
                start_tree_lex = remap_tree_with_ordering_map(
                    start_tree_numeric,
                    posterior_raw_to_lex,
                )
                target_tree_lex = remap_tree_with_ordering_map(
                    target_tree_numeric,
                    posterior_raw_to_lex,
                )
            else:
                item = train_dataset[0]
                start_tree_numeric = str(item["newick_tree"])
                target_tree_numeric = str(item["target_tree"])
                start_tree_for_sampling = remap_tree_with_ordering_map(
                    start_tree_numeric,
                    numeric_to_lex,
                )
                target_tree_for_sampling = remap_tree_with_ordering_map(
                    target_tree_numeric,
                    numeric_to_lex,
                )
                start_tree_lex = remap_tree_with_ordering_map(start_tree_numeric, numeric_to_lex)
                target_tree_lex = remap_tree_with_ordering_map(
                    target_tree_numeric,
                    numeric_to_lex,
                )

            sample_kwargs = module._build_harness_sample_kwargs(
                {
                    "start_tree": start_tree_for_sampling,
                    "target_tree": target_tree_for_sampling,
                },
                train=True,
            )
            sampled_trees, timepoint, _, _, _, trace = module.sample(
                [start_tree_for_sampling],
                **sample_kwargs,
            )
            sampled_tree_raw = str(sampled_trees[0])
            if harness_pair is not None:
                sampled_labels = _tree_leaf_names(sampled_tree_raw)
                if all(label in posterior_raw_to_lex for label in sampled_labels):
                    sampled_tree = remap_tree_with_ordering_map(
                        sampled_tree_raw,
                        posterior_raw_to_lex,
                    )
                else:
                    sampled_tree = remap_tree_with_ordering_map(
                        sampled_tree_raw,
                        numeric_to_lex,
                    )
            else:
                sampled_tree = str(sampled_tree_raw)

            target_short_metrics = _support_rf_metrics(
                target_tree_lex,
                short_support_counts,
                short_support_representatives,
            )
            sampled_short_metrics = _support_rf_metrics(
                sampled_tree,
                short_support_counts,
                short_support_representatives,
            )
            sampled_golden_metrics = _support_rf_metrics(
                sampled_tree,
                golden_support_counts,
                golden_support_representatives,
            )

            results.append(
                {
                    "trial": int(trial + 1),
                    "start_rf_norm": float(calculate_norm_rf(start_tree_lex, target_tree_lex)),
                    "final_rf_norm": float(calculate_norm_rf(sampled_tree, target_tree_lex)),
                    "timepoint": _scalar_timepoint(timepoint),
                    "num_velocity_states": int(len(trace.get("velocity", []))),
                    "num_ar_states": int(len(trace.get("autoregressive", []))),
                    "target_short_in_support": float(target_short_metrics["in_support"]),
                    "target_short_min_rf_norm": float(target_short_metrics["min_rf_norm"]),
                    "sampled_short_in_support": float(sampled_short_metrics["in_support"]),
                    "sampled_short_min_rf_norm": float(sampled_short_metrics["min_rf_norm"]),
                    "sampled_short_weighted_mean_rf_norm": float(
                        sampled_short_metrics["weighted_mean_rf_norm"]
                    ),
                    "sampled_golden_in_support": float(sampled_golden_metrics["in_support"]),
                    "sampled_golden_min_rf_norm": float(sampled_golden_metrics["min_rf_norm"]),
                    "sampled_golden_weighted_mean_rf_norm": float(
                        sampled_golden_metrics["weighted_mean_rf_norm"]
                    ),
                    "start_tree_lex": start_tree_lex,
                    "target_tree_lex": target_tree_lex,
                    "sampled_tree": sampled_tree,
                }
            )

    sampled_trees = [row["sampled_tree"] for row in results]
    pairconditioned_sample_vs_short = _metric_block(
        short_lex,
        sampled_trees,
        num_leaves=num_leaves,
    )
    pairconditioned_sample_vs_golden = _metric_block(
        golden_lex,
        sampled_trees,
        num_leaves=num_leaves,
    )

    payload = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "bootstrap_module": str(args.bootstrap_module),
        "dataset_id": str(args.dataset_id),
        "device": str(args.device),
        "num_leaves": int(num_leaves),
        "num_trials": int(len(results)),
        "pairconditioned_sample_vs_short": pairconditioned_sample_vs_short,
        "pairconditioned_sample_vs_golden": pairconditioned_sample_vs_golden,
        "results": results,
        "mean_start_rf_norm": float(
            sum(row["start_rf_norm"] for row in results) / max(len(results), 1)
        ),
        "mean_final_rf_norm": float(
            sum(row["final_rf_norm"] for row in results) / max(len(results), 1)
        ),
        "best_final_rf_norm": float(min(row["final_rf_norm"] for row in results)),
        "worst_final_rf_norm": float(max(row["final_rf_norm"] for row in results)),
        "target_short_support_rate": float(
            sum(row["target_short_in_support"] for row in results) / max(len(results), 1)
        ),
        "sampled_short_support_rate": float(
            sum(row["sampled_short_in_support"] for row in results) / max(len(results), 1)
        ),
        "sampled_golden_support_rate": float(
            sum(row["sampled_golden_in_support"] for row in results) / max(len(results), 1)
        ),
        "mean_sampled_short_min_rf_norm": float(
            sum(row["sampled_short_min_rf_norm"] for row in results) / max(len(results), 1)
        ),
        "mean_sampled_short_weighted_mean_rf_norm": float(
            sum(row["sampled_short_weighted_mean_rf_norm"] for row in results)
            / max(len(results), 1)
        ),
        "mean_sampled_golden_min_rf_norm": float(
            sum(row["sampled_golden_min_rf_norm"] for row in results) / max(len(results), 1)
        ),
        "mean_sampled_golden_weighted_mean_rf_norm": float(
            sum(row["sampled_golden_weighted_mean_rf_norm"] for row in results)
            / max(len(results), 1)
        ),
        "short_unique_topologies": int(len(short_support_counts)),
        "golden_unique_topologies": int(len(golden_support_counts)),
    }

    output_path = Path(args.output).resolve()
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
