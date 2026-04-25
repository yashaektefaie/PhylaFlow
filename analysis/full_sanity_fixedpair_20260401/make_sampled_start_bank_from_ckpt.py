#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from ete3 import Tree as EteTree

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.full_sanity_fixedpair_20260401.probe_current_mainline_ckpt import (  # noqa: E402
    build_dataset,
    build_module,
    load_config,
    set_seed,
)
from utils.metric_utils import calculate_norm_rf  # noqa: E402


def _labels_from_group_key(group_key: Optional[str]) -> Optional[List[str]]:
    if not group_key:
        return None
    prefix = "labels:"
    if not str(group_key).startswith(prefix):
        return None
    return [item for item in str(group_key)[len(prefix) :].split(",") if item]


def _sample_pair(train_dataset) -> Dict[str, Any]:
    pair = train_dataset.sample_overfit_fixed_pair_bank_pair()
    if pair is not None:
        return pair

    start_bank = list(
        getattr(train_dataset, "overfit_fixed_pair_start_tree_newick_bank", []) or []
    )
    target_bank = list(
        getattr(train_dataset, "overfit_fixed_pair_target_tree_newick_bank", []) or []
    )
    if not target_bank:
        target_bank = list(train_dataset.return_posterior_trees(0))
    if not start_bank or not target_bank:
        raise ValueError("Could not sample an overfit fixed-pair bank pair.")
    start_tree = random.choice(start_bank)
    target_tree = random.choice(target_bank)
    return {
        "base_random_tree": str(start_tree),
        "random_tree": str(start_tree),
        "effective_target_tree": str(target_tree),
        "bank_group_key": f"n{len(EteTree(str(target_tree), format=1).get_leaves())}",
    }


def _mean(values: List[float]) -> Optional[float]:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else None


def _median(values: List[float]) -> Optional[float]:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else None


def _quantile(values: List[float], q: float) -> Optional[float]:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q)) if values else None


def _write_start_bank(
    *,
    rows: List[Dict[str, Any]],
    start_dir: Path,
    checkpoint: Path,
    config_path: Path,
    seed: int,
) -> None:
    start_dir.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(rows):
        group_key = str(row.get("bank_group_key") or f"n{row['n_leaves']}")
        payload = {
            "start_tree": row["sampled_tree"],
            "bank_group_key": group_key,
            "selected_original_labels": _labels_from_group_key(group_key),
            "source_checkpoint": str(checkpoint.resolve()),
            "source_config": str(config_path.resolve()),
            "source_pair_index": int(row["pair_index"]),
            "source_start_rf_norm": float(row["start_rf_norm"]),
            "source_sampled_rf_norm": float(row["sampled_rf_norm"]),
            "source_num_velocity_states": int(row["num_velocity_states"]),
            "source_num_ar_states": int(row["num_ar_states"]),
            "source_start_tree": row["start_tree"],
            "source_target_tree": row["target_tree"],
            "seed": int(seed),
        }
        (start_dir / f"start_{idx:04d}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )


def _write_derived_config(
    *,
    base_config: Dict[str, Any],
    output_config: Path,
    run_tag: str,
    start_dir: Path,
    analysis_dir: Path,
) -> None:
    config = copy.deepcopy(base_config)
    trainer = config.setdefault("trainer", {})
    data = config.setdefault("data", {})

    trainer["record"] = True
    trainer.setdefault("wandb_project", "phylaflow_overfit")
    trainer["checkpoint_dir"] = (
        f"./checkpoints/full_sanity_fixedpair_20260401/{run_tag}"
    )
    trainer["sample_metrics_trace_path"] = str(
        analysis_dir / f"{run_tag}_metrics.jsonl"
    )
    trainer["dynamic_start_bank_trace_path"] = str(
        analysis_dir / f"{run_tag}_updates.jsonl"
    )
    trainer["dynamic_start_bank_artifact_dir"] = str(
        analysis_dir / f"{run_tag}_artifacts"
    )

    data["overfit_fixed_pair_start_tree_json_path"] = None
    data["overfit_fixed_pair_start_tree_json_paths"] = None
    data["overfit_fixed_pair_start_tree_json_dir"] = str(start_dir)
    data["overfit_fixed_pair_group_by_json_metadata"] = bool(
        data.get("overfit_fixed_pair_group_by_json_metadata", False)
    )

    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(yaml.safe_dump(config, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-starts", type=int, default=64)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--bank-tag", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument(
        "--analysis-dir",
        default=str(ROOT / "analysis/full_sanity_fixedpair_20260401"),
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    analysis_dir = Path(args.analysis_dir).resolve()
    start_dir = analysis_dir / f"{args.bank_tag}_start_bank_{int(args.num_starts)}"
    summary_path = analysis_dir / f"{args.bank_tag}_bank_{int(args.num_starts)}_summary.json"
    output_config = Path(args.output_config).resolve()

    config = load_config(config_path)
    set_seed(int(args.seed))
    dataset = build_dataset(config)
    train_dataset = dataset.dataset_train
    module = build_module(config, dataset, checkpoint, str(args.device))

    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for pair_idx in range(int(args.num_starts)):
            pair_info = _sample_pair(train_dataset)
            start_tree = str(pair_info["random_tree"])
            target_tree = str(pair_info["effective_target_tree"])
            n_leaves = len(EteTree(start_tree, format=1).get_leaves())
            pair = {
                "start_tree": start_tree,
                "target_tree": target_tree,
                "n_leaves": int(n_leaves),
                "max_events": 1024,
                "name_mapping": (
                    train_dataset.return_nexus_number_to_name(0)
                    if hasattr(train_dataset, "return_nexus_number_to_name")
                    else None
                ),
            }
            sample_kwargs = module._build_harness_sample_kwargs(pair, train=True)
            trees, _, _, _, _, trace = module.sample([start_tree], **sample_kwargs)
            sampled_tree = str(trees[0])
            rows.append(
                {
                    "pair_index": int(pair_idx),
                    "n_leaves": int(n_leaves),
                    "bank_group_key": str(pair_info.get("bank_group_key") or f"n{n_leaves}"),
                    "start_rf_norm": float(calculate_norm_rf(start_tree, target_tree)),
                    "sampled_rf_norm": float(calculate_norm_rf(sampled_tree, target_tree)),
                    "num_velocity_states": int(len((trace or {}).get("velocity", []) or [])),
                    "num_ar_states": int(len((trace or {}).get("autoregressive", []) or [])),
                    "start_tree": start_tree,
                    "target_tree": target_tree,
                    "sampled_tree": sampled_tree,
                }
            )
            print(
                json.dumps(
                    {
                        "pair_index": int(pair_idx),
                        "sampled_rf_norm": rows[-1]["sampled_rf_norm"],
                        "num_velocity_states": rows[-1]["num_velocity_states"],
                        "num_ar_states": rows[-1]["num_ar_states"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    _write_start_bank(
        rows=rows,
        start_dir=start_dir,
        checkpoint=checkpoint,
        config_path=config_path,
        seed=int(args.seed),
    )
    _write_derived_config(
        base_config=config,
        output_config=output_config,
        run_tag=str(args.run_tag),
        start_dir=start_dir,
        analysis_dir=analysis_dir,
    )

    sampled_values = [float(row["sampled_rf_norm"]) for row in rows]
    start_values = [float(row["start_rf_norm"]) for row in rows]
    velocity_counts = [int(row["num_velocity_states"]) for row in rows]
    ar_counts = [int(row["num_ar_states"]) for row in rows]
    summary = {
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "output_config": str(output_config),
        "start_dir": str(start_dir),
        "num_starts": int(args.num_starts),
        "seed": int(args.seed),
        "device": str(args.device),
        "mean_source_start_rf_norm": _mean(start_values),
        "mean_source_sampled_rf_norm": _mean(sampled_values),
        "median_source_sampled_rf_norm": _median(sampled_values),
        "best_source_sampled_rf_norm": min(sampled_values) if sampled_values else None,
        "worst_source_sampled_rf_norm": max(sampled_values) if sampled_values else None,
        "p10_source_sampled_rf_norm": _quantile(sampled_values, 0.10),
        "p90_source_sampled_rf_norm": _quantile(sampled_values, 0.90),
        "mean_num_velocity_states": _mean([float(x) for x in velocity_counts]),
        "mean_num_ar_states": _mean([float(x) for x in ar_counts]),
        "rows": [
            {
                "pair_index": int(row["pair_index"]),
                "start_rf_norm": float(row["start_rf_norm"]),
                "sampled_rf_norm": float(row["sampled_rf_norm"]),
                "num_velocity_states": int(row["num_velocity_states"]),
                "num_ar_states": int(row["num_ar_states"]),
                "bank_group_key": str(row["bank_group_key"]),
            }
            for row in rows
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
