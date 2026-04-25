#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

ROOT = Path("/home/yektefai/PhylaFlow")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.full_sanity_fixedpair_20260401.probe_current_mainline_ckpt import (  # noqa: E402
    build_module,
    load_config,
    set_seed,
)
from analysis.full_sanity_fixedpair_20260401.remap_posterior_to_harness_lexindex import (  # noqa: E402
    build_harness_lexicographic_ordering_map,
    build_numeric_to_harness_lexicographic_ordering_map,
    remap_tree_with_ordering_map,
)
from data.dataset import PhylaDataModule, TreeDataset  # noqa: E402
from ete3 import Tree as EteTree  # noqa: E402
from utils.metric_utils import (  # noqa: E402
    calculate_norm_rf,
    kl_divergence_topological_distributions,
    kl_divergence_tree_topology_distributions,
    topk_posterior_tree_recall,
)


def _numeric_name_sort_key(name: str):
    try:
        return (0, int(str(name)))
    except Exception:
        return (1, str(name))


def _remap_tree_with_map(tree_newick: str, seq_ordering_map: Dict[str, str]) -> str:
    tree = EteTree(tree_newick, format=1)
    for leaf in tree.get_leaves():
        mapped = seq_ordering_map.get(str(leaf.name))
        if mapped is None:
            raise ValueError(f"Leaf '{leaf.name}' missing from sequence ordering map.")
        leaf.name = str(mapped)
    return tree.write(format=1)


def _choose_best_checkpoint(config: dict, metrics_path: Optional[str]) -> tuple[Path, dict]:
    checkpoint_dir = Path(config["trainer"]["checkpoint_dir"])
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")
    if not metrics_path:
        raise ValueError("Need a metrics trace path to auto-select the best checkpoint.")

    metrics_file = Path(metrics_path)
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics trace not found: {metrics_file}")

    best_row = None
    best_ckpt = None
    for line in metrics_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        step = int(row.get("global_step", -1))
        if step < 0:
            continue
        direct = sorted(checkpoint_dir.rglob(f"*step={step:06d}.ckpt"))
        versioned = sorted(checkpoint_dir.rglob(f"*step={step:06d}-v*.ckpt"))
        matches = [path for path in direct if "-v" not in path.name] or direct or versioned
        if not matches:
            continue
        rf = float(row.get("rf_norm", float("inf")))
        if best_row is None or rf < float(best_row["rf_norm"]):
            best_row = row
            best_ckpt = matches[0]

    if best_row is None or best_ckpt is None:
        raise ValueError("Could not match any metrics rows to saved checkpoints.")
    return best_ckpt, best_row


def _build_posterior_dataset(root: str, dataset_id: str, trprobs_sample_count_per_file: int):
    return TreeDataset(
        nexus_root="unused",
        mrbayes_root="unused",
        posterior_trprobs_root=root,
        posterior_dataset_id=dataset_id,
        trprobs_sample_count_per_file=trprobs_sample_count_per_file,
    )


def _build_current_dataset(config: dict, dataset_id: str):
    dataset = PhylaDataModule(config, train_ids=[str(dataset_id)], test_ids=[str(dataset_id)])
    dataset.setup(stage=None)
    return dataset


def _normalize_tree_list(
    trees: List[str],
    seq_ordering_map: Dict[str, str],
) -> List[str]:
    return [_remap_tree_with_map(tree, seq_ordering_map) for tree in trees]


def _remap_tree_list_with_ordering_map(
    trees: List[str],
    ordering_map: Dict[str, str],
) -> List[str]:
    return [remap_tree_with_ordering_map(tree, ordering_map) for tree in trees]


def _sample_model_trees(
    module,
    train_dataset,
    *,
    dataset_id: str,
    num_samples: int,
    batch_size: int,
    seed: int,
    max_events: int,
) -> List[str]:
    reference_item = train_dataset[0]
    reference_tree = str(reference_item["target_tree"])
    n_leaves = len(EteTree(reference_tree, format=1).get_leaves())
    name_mapping = train_dataset.return_nexus_number_to_name(dataset_id)

    pair = {
        "start_tree": reference_tree,
        "target_tree": reference_tree,
        "n_leaves": n_leaves,
        "max_events": int(max_events),
        "name_mapping": name_mapping,
    }
    sample_kwargs = module._build_harness_sample_kwargs(
        pair,
        train=True,
        target_trees=None,
        return_trace=False,
        max_events=int(max_events),
    )

    rng = random.Random(int(seed))
    final_trees: List[str] = []
    with torch.no_grad():
        while len(final_trees) < int(num_samples):
            current_batch = min(int(batch_size), int(num_samples) - len(final_trees))
            start_trees = []
            for _ in range(current_batch):
                random.seed(rng.randint(0, 2**31 - 1))
                start_trees.append(train_dataset.sample_random_tree(reference_tree))
            sampled_trees, _, _, _, _ = module.sample(start_trees, **sample_kwargs)
            final_trees.extend(str(tree) for tree in sampled_trees)
    return final_trees


def _metric_block(posterior_trees: List[str], sampled_trees: List[str], num_leaves: int):
    payload = {}
    payload.update(
        kl_divergence_topological_distributions(
            posterior_trees, sampled_trees, num_leaves=num_leaves
        )
    )
    payload.update(kl_divergence_tree_topology_distributions(posterior_trees, sampled_trees))
    payload.update(topk_posterior_tree_recall(posterior_trees, sampled_trees))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset-id", default="DS1")
    parser.add_argument(
        "--golden-root",
        default="/home/yektefai/30272299/golden_run_data_DS1-8",
    )
    parser.add_argument(
        "--short-root",
        default="/home/yektefai/30272299/short_run_data_DS1-8",
    )
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-events", type=int, default=256)
    parser.add_argument("--trprobs-sample-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--tree-indexing-mode",
        choices=("dataset_numeric", "harness_lex"),
        default="dataset_numeric",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    logging.getLogger("run.TrainingModule").setLevel(logging.WARNING)

    config = load_config(Path(args.config))
    config.setdefault("trainer", {})
    config["trainer"]["sampling_predsim_overrun_use_at_sampling"] = False
    config["trainer"]["sampling_actual_event_boundary_use_at_sampling"] = False
    config["trainer"]["sampling_actual_event_boundary_include_predicted_first_hit"] = False
    config["trainer"]["sampling_random_fixed_pair_bank_use_at_sampling"] = False

    chosen_checkpoint: Path
    best_row = None
    if args.checkpoint:
        chosen_checkpoint = Path(args.checkpoint).resolve()
    else:
        chosen_checkpoint, best_row = _choose_best_checkpoint(
            config,
            config["trainer"].get("sample_metrics_trace_path"),
        )

    set_seed(int(args.seed))
    dataset = _build_current_dataset(config, str(args.dataset_id))
    train_dataset = dataset.dataset_train
    module = build_module(config, dataset, chosen_checkpoint, str(args.device))
    module.sampling_disable_inner_logging = True
    module.sampling_cache_autoregressive_state = True
    module.sampling_cache_polytomy_groups = True
    module.sampling_cache_tri_mask = True

    short_item = train_dataset[0]
    seq_ordering_map = {
        str(key): str(value)
        for key, value in short_item["seq_ordering_map"].items()
    }
    num_leaves = len(EteTree(short_item["target_tree"], format=1).get_leaves())

    golden_dataset = _build_posterior_dataset(
        str(args.golden_root),
        str(args.dataset_id),
        int(args.trprobs_sample_count),
    )
    short_dataset = _build_posterior_dataset(
        str(args.short_root),
        str(args.dataset_id),
        int(args.trprobs_sample_count),
    )

    golden_raw = golden_dataset.return_posterior_trees(str(args.dataset_id))
    short_raw = short_dataset.return_posterior_trees(str(args.dataset_id))
    if args.tree_indexing_mode == "dataset_numeric":
        golden_trees = _normalize_tree_list(golden_raw, seq_ordering_map)
        short_trees = _normalize_tree_list(short_raw, seq_ordering_map)
        sampled_tree_ordering_map = None
    else:
        reference_tree = golden_raw[0] if golden_raw else short_raw[0]
        posterior_tree_ordering_map = build_harness_lexicographic_ordering_map(
            reference_tree
        )
        sampled_tree_ordering_map = (
            build_numeric_to_harness_lexicographic_ordering_map(reference_tree)
        )
        golden_trees = _remap_tree_list_with_ordering_map(
            golden_raw,
            posterior_tree_ordering_map,
        )
        short_trees = _remap_tree_list_with_ordering_map(
            short_raw,
            posterior_tree_ordering_map,
        )

    sampled_trees = _sample_model_trees(
        module,
        train_dataset,
        dataset_id=str(args.dataset_id),
        num_samples=int(args.num_samples),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        max_events=int(args.max_events),
    )
    if sampled_tree_ordering_map is not None:
        sampled_trees = _remap_tree_list_with_ordering_map(
            sampled_trees,
            sampled_tree_ordering_map,
        )

    sample_vs_golden = _metric_block(golden_trees, sampled_trees, num_leaves=num_leaves)
    short_vs_golden = _metric_block(golden_trees, short_trees, num_leaves=num_leaves)

    payload = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(chosen_checkpoint),
        "best_metrics_row": best_row,
        "dataset_id": str(args.dataset_id),
        "device": str(args.device),
        "tree_indexing_mode": str(args.tree_indexing_mode),
        "num_leaves": int(num_leaves),
        "num_sampled_trees": int(len(sampled_trees)),
        "num_golden_trees": int(len(golden_trees)),
        "num_short_trees": int(len(short_trees)),
        "sampling_mode": {
            "sampling_predsim_overrun_use_at_sampling": False,
            "sampling_actual_event_boundary_use_at_sampling": False,
            "sampling_actual_event_boundary_include_predicted_first_hit": False,
            "sampling_random_fixed_pair_bank_use_at_sampling": False,
            "max_events": int(args.max_events),
        },
        "sample_vs_golden": sample_vs_golden,
        "short_vs_golden": short_vs_golden,
        "sampled_tree_examples": sampled_trees[:5],
        "golden_tree_examples": golden_trees[:5],
        "short_tree_examples": short_trees[:5],
        "mean_sample_vs_random_golden_rf_norm": float(
            sum(
                calculate_norm_rf(sampled_trees[i], golden_trees[i % len(golden_trees)])
                for i in range(len(sampled_trees))
            )
            / max(len(sampled_trees), 1)
        ),
    }

    output_path = Path(args.output).resolve()
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
